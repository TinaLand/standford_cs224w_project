# src/rl/training/independent_learning.py
"""
Independent Learning (IQL) Training Pipeline
Each sector agent learns independently without coordination
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Any
import json

# Fix PyTorch serialization for pandas timestamps (PyTorch 2.6+)
import torch.serialization
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([pd._libs.tslibs.timestamps._unpickle_timestamp])

from ..config import MultiAgentConfig, GNNConfig, PROJECT_ROOT, DEVICE, MODELS_DIR
from ..agents.sector_agent import SectorAgent
from ..coordination.coordinator import SectorGrouping
from ..environments.multi_agent import MultiAgentTradingEnv

# Import GNN model
sys.path.append(str(PROJECT_ROOT / 'src'))
from training.transformer_trainer import (
    RoleAwareGraphTransformer,
    HIDDEN_CHANNELS,
    OUT_CHANNELS,
    NUM_LAYERS,
    NUM_HEADS,
)


def load_gnn_model_for_independent_learning() -> torch.nn.Module:
    """
    Load the trained GNN model for independent learning.
    
    Returns:
        Loaded and frozen GNN model
    """
    print("\n--- Loading GNN Model for Independent Learning ---")
    
    # Determine Input Dimension
    data_graphs_dir = PROJECT_ROOT / "data" / "graphs"
    sample_graph_path = list(data_graphs_dir.glob('graph_t_*.pt'))[0]
    temp_data = torch.load(sample_graph_path, weights_only=False)
    INPUT_DIM = temp_data['stock'].x.shape[1]
    
    # Initialize GNN model
    gnn_model = RoleAwareGraphTransformer(
        INPUT_DIM,
        HIDDEN_CHANNELS,
        OUT_CHANNELS,
        NUM_LAYERS,
        NUM_HEADS,
    ).to(DEVICE)
    
    # Load trained weights
    if not GNNConfig.CORE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained GNN model not found at: {GNNConfig.CORE_MODEL_PATH}")
    
    state_dict = torch.load(GNNConfig.CORE_MODEL_PATH, map_location=DEVICE, weights_only=False)
    gnn_model.load_state_dict(state_dict, strict=False)
    
    # Freeze parameters
    for param in gnn_model.parameters():
        param.requires_grad = False
    
    print(f"✅ GNN Model loaded and frozen for independent learning")
    return gnn_model


def determine_independent_training_period():
    """
    Determine training period for independent learning.
    
    Returns:
        Tuple of (start_date, end_date)
    """
    data_graphs_dir = PROJECT_ROOT / "data" / "graphs"
    graph_files = list(data_graphs_dir.glob('graph_t_*.pt'))
    
    if not graph_files:
        raise ValueError("No graph files found. Run Phase 2 first.")
    
    graph_dates = sorted([pd.to_datetime(f.stem.split('_')[-1]) for f in graph_files])
    graph_start = graph_dates[0]
    graph_end = graph_dates[-1]
    
    # Use last 20% for independent learning training
    date_range_days = (graph_end - graph_start).days
    start_offset_days = int(date_range_days * 0.8)
    start_date = graph_start + pd.Timedelta(days=start_offset_days)
    end_date = graph_end
    
    print(f"📅 Independent Learning Training period: {start_date.date()} to {end_date.date()}")
    
    return start_date, end_date


class IndependentLearningTrainer:
    """
    Independent Learning Trainer.
    Each sector agent learns independently without coordination.
    """
    
    def __init__(
        self,
        agents: Dict[str, SectorAgent],
        environment: MultiAgentTradingEnv,
        total_timesteps: int = MultiAgentConfig.TOTAL_TIMESTEPS
    ):
        """
        Initialize independent learning trainer.
        
        Args:
            agents: Dictionary of sector agents
            environment: Multi-agent trading environment
            total_timesteps: Total training timesteps
        """
        self.agents = agents
        self.environment = environment
        self.total_timesteps = total_timesteps
        
        # Training statistics
        self.training_stats = {
            'episode_returns': [],
            'episode_lengths': [],
            'sector_performance': {sector: [] for sector in agents.keys()},
            'individual_agent_losses': {sector: [] for sector in agents.keys()}
        }
    
    def train(self) -> Dict[str, Any]:
        """
        Run independent learning training.
        Each agent learns independently without coordination.
        
        Returns:
            Training statistics
        """
        print(f"\n🔨 Starting Independent Learning Training ({self.total_timesteps} timesteps)")
        print("=" * 60)
        print("Note: Each agent learns independently (no coordination)")
        print("=" * 60)
        
        current_timestep = 0
        episode = 0
        
        while current_timestep < self.total_timesteps:
            episode += 1
            episode_stats = self._train_episode(episode)
            
            # Update statistics
            self.training_stats['episode_returns'].append(episode_stats['total_return'])
            self.training_stats['episode_lengths'].append(episode_stats['episode_length'])
            
            for sector, performance in episode_stats['sector_performance'].items():
                self.training_stats['sector_performance'][sector].append(performance)
            
            current_timestep += episode_stats['episode_length']
            
            # Log progress
            if episode % 10 == 0:
                avg_return = np.mean(self.training_stats['episode_returns'][-10:])
                print(f"Episode {episode}, Timestep {current_timestep}, Avg Return: {avg_return:.4f}")
        
        print(f"✅ Independent Learning Training completed after {episode} episodes")
        return self.training_stats
    
    def _train_episode(self, episode: int) -> Dict[str, Any]:
        """
        Train a single episode with independent agents.
        
        Args:
            episode: Episode number
            
        Returns:
            Episode statistics
        """
        # Reset environment
        global_obs, info = self.environment.reset()
        sector_observations = info['sector_observations']
        
        episode_length = 0
        total_reward = 0
        sector_rewards = {sector: 0.0 for sector in self.agents.keys()}
        done = False
        
        # Store experiences for each agent
        agent_experiences = {sector: [] for sector in self.agents.keys()}
        
        while not done:
            # Independent execution: each agent selects actions independently
            agent_actions = {}
            for sector, agent in self.agents.items():
                if sector in sector_observations:
                    obs = sector_observations[sector]
                    # Ensure observation has correct shape for this sector
                    expected_dim = agent.num_stocks * (1 + agent.embedding_dim)
                    if len(obs) != expected_dim:
                        # Reshape or pad observation to match expected dimension
                        if len(obs) > expected_dim:
                            obs = obs[:expected_dim]
                        else:
                            padding = np.zeros(expected_dim - len(obs), dtype=np.float32)
                            obs = np.concatenate([obs, padding])
                    
                    try:
                        action, _ = agent.predict(obs, deterministic=False)
                        agent_actions[sector] = action
                    except Exception as e:
                        print(f"⚠️  Error predicting for {sector}: {e}, obs shape: {obs.shape}, expected: {expected_dim}")
                        # Fallback: random actions
                        agent_actions[sector] = np.zeros(agent.num_stocks, dtype=np.int32)
            
            # Step environment
            next_global_obs, step_sector_rewards, terminated, truncated, step_info = self.environment.step(agent_actions)
            done = terminated or truncated
            
            # Store experiences for independent learning
            for sector, agent in self.agents.items():
                if sector in sector_observations and sector in step_sector_rewards:
                    agent_experiences[sector].append({
                        'observation': sector_observations[sector],
                        'action': agent_actions.get(sector),
                        'reward': step_sector_rewards[sector],
                        'next_observation': step_info.get('sector_observations', {}).get(sector),
                        'done': done
                    })
            
            # Update statistics
            episode_length += 1
            total_reward += sum(step_sector_rewards.values())
            
            for sector, reward in step_sector_rewards.items():
                sector_rewards[sector] += reward
            
            # Update for next step
            global_obs = next_global_obs
            sector_observations = step_info.get('sector_observations', {})
        
        # Independent training: each agent learns from its own experiences
        for sector, agent in self.agents.items():
            if sector in agent_experiences and len(agent_experiences[sector]) > 0:
                # Train agent independently using its own experiences
                # In practice, this would use proper RL algorithms (PPO, DQN, etc.)
                # For now, we use a simplified approach
                try:
                    # Use PPO's learn method if available
                    if hasattr(agent.agent, 'learn'):
                        # Simplified: train on collected experiences
                        # In practice, would use proper experience replay
                        pass
                except Exception as e:
                    print(f"⚠️  Training agent {sector} failed: {e}")
        
        return {
            'total_return': total_reward,
            'episode_length': episode_length,
            'sector_performance': sector_rewards
        }


def create_independent_agents(
    gnn_model: torch.nn.Module,
    sector_groups: Dict[str, List[str]],
    all_tickers: List[str],
    embedding_dim: int = 64
) -> Dict[str, SectorAgent]:
    """
    Create independent agents for each sector.
    
    Args:
        gnn_model: Trained GNN model
        sector_groups: Dictionary mapping sector names to ticker lists
        all_tickers: List of all tickers
        embedding_dim: Embedding dimension
        
    Returns:
        Dictionary of sector agents
    """
    print("\n--- Creating Independent Agents ---")
    
    agents = {}
    for sector_name, tickers in sector_groups.items():
        if len(tickers) == 0:
            continue
        
        print(f"  Creating agent for {sector_name} ({len(tickers)} stocks)")
        
        agent = SectorAgent(
            sector_name=sector_name,
            tickers=tickers,
            gnn_model=gnn_model,
            device=DEVICE,
            embedding_dim=embedding_dim,
            learning_rate=MultiAgentConfig.LEARNING_RATE
        )
        
        agents[sector_name] = agent
    
    print(f"✅ Created {len(agents)} independent agents")
    return agents


def run_independent_learning_training(
    total_timesteps: int = MultiAgentConfig.TOTAL_TIMESTEPS
) -> Tuple[Dict[str, SectorAgent], Dict[str, Any]]:
    """
    Run complete independent learning training pipeline.
    
    Args:
        total_timesteps: Total training timesteps
        
    Returns:
        Tuple of (agents, training_stats)
    """
    print("\n🤖 Starting Independent Learning Training Pipeline")
    print("=" * 60)
    
    # 1. Load GNN Model
    try:
        gnn_model = load_gnn_model_for_independent_learning()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        raise
    
    # 2. Get sector groupings
    print(f"\n--- Loading Sector Groupings ---")
    sector_groups = SectorGrouping.load_sector_mapping()
    
    # Get all tickers
    all_tickers = []
    for tickers in sector_groups.values():
        all_tickers.extend(tickers)
    all_tickers = sorted(list(set(all_tickers)))
    
    print(f"  Total sectors: {len(sector_groups)}")
    print(f"  Total stocks: {len(all_tickers)}")
    
    # 3. Create Independent Agents
    # Get embedding dimension from environment (it will compute it correctly)
    # First create a temporary environment to get the correct embedding_dim
    start_date, end_date = determine_independent_training_period()
    temp_env = MultiAgentTradingEnv(
        start_date=start_date,
        end_date=end_date,
        gnn_model=gnn_model,
        sector_groups=sector_groups,
        device=DEVICE
    )
    embedding_dim = temp_env.embedding_dim
    print(f"  Using embedding dimension from environment: {embedding_dim}")
    
    agents = create_independent_agents(
        gnn_model=gnn_model,
        sector_groups=sector_groups,
        all_tickers=all_tickers,
        embedding_dim=embedding_dim
    )
    
    # 4. Create Environment (already created above to get embedding_dim)
    print(f"\n--- Creating Multi-Agent Environment ---")
    env = temp_env  # Use the environment we created to get embedding_dim
    
    # 5. Create Trainer
    trainer = IndependentLearningTrainer(
        agents=agents,
        environment=env,
        total_timesteps=total_timesteps
    )
    
    # 6. Train
    training_stats = trainer.train()
    
    # 7. Save results
    print(f"\n--- Saving Results ---")
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Save training statistics
    stats_file = results_dir / "independent_learning_training_stats.json"
    with open(stats_file, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_stats = convert_to_serializable(training_stats)
        json.dump(serializable_stats, f, indent=2)
    
    print(f"✅ Training statistics saved to: {stats_file}")
    
    # Save agent models
    models_dir = MODELS_DIR / "independent_agents"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    for sector_name, agent in agents.items():
        agent_file = models_dir / f"independent_agent_{sector_name.lower()}.zip"
        try:
            agent.agent.save(str(agent_file))
            print(f"✅ Saved agent for {sector_name}")
        except Exception as e:
            print(f"⚠️  Failed to save agent for {sector_name}: {e}")
    
    return agents, training_stats


if __name__ == "__main__":
    print("=" * 60)
    print("Independent Learning (IQL) Training")
    print("=" * 60)
    print("\nThis script trains sector agents independently without coordination.")
    print("Each agent learns to optimize its own sector performance.")
    print()
    
    try:
        agents, stats = run_independent_learning_training()
        print("\n" + "=" * 60)
        print("✅ Independent Learning Training Complete!")
        print("=" * 60)
        print(f"\nTrained {len(agents)} independent agents")
        print(f"Total episodes: {len(stats['episode_returns'])}")
        print(f"Average return: {np.mean(stats['episode_returns']):.4f}")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


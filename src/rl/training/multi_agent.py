# src/rl/training/multi_agent.py
"""
Multi-Agent RL Training Pipeline
Enhanced version with performance optimizations and better structure
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
from ..coordination import create_multi_agent_system, MultiAgentCoordinator
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


def load_gnn_model_for_multi_agent() -> torch.nn.Module:
    """
    Load the trained GNN model for multi-agent RL training.
    
    Returns:
        Loaded and frozen GNN model
    """
    print("\n--- Loading GNN Model for Multi-Agent RL ---")
    
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
    
    print(f"✅ GNN Model loaded and frozen for multi-agent training")
    return gnn_model


def determine_multi_agent_training_period():
    """
    Determine training period for multi-agent system.
    
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
    
    # Use last 20% for multi-agent training
    date_range_days = (graph_end - graph_start).days
    start_offset_days = int(date_range_days * 0.8)
    start_date = graph_start + pd.Timedelta(days=start_offset_days)
    end_date = graph_end
    
    print(f"📅 Multi-Agent Training period: {start_date.date()} to {end_date.date()}")
    
    return start_date, end_date


class MultiAgentTrainer:
    """
    Multi-Agent RL Trainer with CTDE implementation.
    Handles coordination between sector agents and centralized training.
    """
    
    def __init__(
        self,
        coordinator: MultiAgentCoordinator,
        environment: MultiAgentTradingEnv,
        total_timesteps: int = MultiAgentConfig.TOTAL_TIMESTEPS
    ):
        """
        Initialize multi-agent trainer.
        
        Args:
            coordinator: Multi-agent coordinator
            environment: Multi-agent trading environment
            total_timesteps: Total training timesteps
        """
        self.coordinator = coordinator
        self.environment = environment
        self.total_timesteps = total_timesteps
        
        # Training statistics
        self.training_stats = {
            'episode_returns': [],
            'episode_lengths': [],
            'sector_performance': {sector: [] for sector in coordinator.agents.keys()},
            'coordination_losses': []
        }
    
    def train(self) -> Dict[str, Any]:
        """
        Run multi-agent training with CTDE.
        
        Returns:
            Training statistics
        """
        print(f"\n🔨 Starting Multi-Agent Training ({self.total_timesteps} timesteps)")
        
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
        
        print(f"✅ Multi-Agent Training completed after {episode} episodes")
        return self.training_stats
    
    def _train_episode(self, episode: int) -> Dict[str, Any]:
        """
        Train a single episode with all agents.
        
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
        sector_rewards = {sector: 0.0 for sector in self.coordinator.agents.keys()}
        done = False
        
        while not done:
            # Decentralized execution: each agent selects actions
            agent_actions = self.coordinator.get_agent_actions(
                sector_observations, 
                deterministic=False
            )
            
            # Step environment
            next_global_obs, step_sector_rewards, terminated, truncated, step_info = self.environment.step(agent_actions)
            done = terminated or truncated
            
            # Update statistics
            episode_length += 1
            total_reward += sum(step_sector_rewards.values())
            
            for sector, reward in step_sector_rewards.items():
                sector_rewards[sector] += reward
            
            # Centralized training (simplified - in practice would use replay buffer)
            if MultiAgentConfig.CENTRALIZED_TRAINING:
                self._centralized_training_step(
                    sector_observations,
                    agent_actions,
                    step_sector_rewards,
                    step_info.get('sector_observations', {}),
                    done
                )
            
            # Update for next step
            global_obs = next_global_obs
            sector_observations = step_info.get('sector_observations', {})
        
        return {
            'total_return': total_reward,
            'episode_length': episode_length,
            'sector_performance': sector_rewards
        }
    
    def _centralized_training_step(
        self,
        observations: Dict[str, np.ndarray],
        actions: Dict[str, np.ndarray],
        rewards: Dict[str, float],
        next_observations: Dict[str, np.ndarray],
        done: bool
    ):
        """
        Perform centralized training step.
        
        Args:
            observations: Current sector observations
            actions: Actions taken by each agent
            rewards: Rewards received by each agent
            next_observations: Next sector observations
            done: Episode termination flag
        """
        # Simplified centralized training
        # In practice, this would use experience replay and proper CTDE algorithms
        
        # Convert rewards to global Q-values
        q_values = {sector: reward for sector, reward in rewards.items()}
        
        # Create dummy global state for mixing
        global_state = torch.zeros(MultiAgentConfig.GLOBAL_STATE_DIM, device=DEVICE)
        
        # Compute global Q-value
        try:
            global_q = self.coordinator.compute_global_q_value(q_values, global_state)
            
            # Simple TD target (would be more sophisticated in practice)
            target_q = torch.tensor([sum(rewards.values())], device=DEVICE)
            
            # Train mixing network
            loss = self.coordinator.train_mixing_network(q_values, global_state, target_q)
            self.training_stats['coordination_losses'].append(loss)
            
        except Exception as e:
            # Handle training errors gracefully
            print(f"⚠️  Centralized training step failed: {e}")


def run_multi_agent_training(
    total_timesteps: int = MultiAgentConfig.TOTAL_TIMESTEPS,
    learning_rate: float = MultiAgentConfig.LEARNING_RATE
) -> Tuple[MultiAgentCoordinator, Dict[str, Any]]:
    """
    Run complete multi-agent RL training pipeline.
    
    Args:
        total_timesteps: Total training timesteps
        learning_rate: Learning rate for coordination
        
    Returns:
        Tuple of (coordinator, training_stats)
    """
    print("\n🤖 Starting Multi-Agent RL Training Pipeline")
    print("=" * 60)
    
    # 1. Load GNN Model
    try:
        gnn_model = load_gnn_model_for_multi_agent()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        raise
    
    # 2. Create Multi-Agent System
    print(f"\n--- Creating Multi-Agent System ---")
    
    # Get embedding dimension from GNN model
    embedding_dim = 64  # Default fallback
    try:
        # Test the GNN model to get actual embedding dimension
        data_graphs_dir = PROJECT_ROOT / "data" / "graphs"
        sample_graph_path = list(data_graphs_dir.glob('graph_t_*.pt'))[0]
        sample_data = torch.load(sample_graph_path, weights_only=False).to(DEVICE)
        num_nodes = sample_data['stock'].x.shape[0]
        date_tensor = torch.zeros(num_nodes, device=DEVICE)
        
        with torch.no_grad():
            embeddings = gnn_model.get_embeddings(sample_data, date_tensor)
            embedding_dim = embeddings.shape[1]
            
        print(f"✅ Detected embedding dimension: {embedding_dim}")
    except Exception as e:
        print(f"⚠️  Could not detect embedding dimension, using default: {embedding_dim}")
    
    # Get actual tickers from the data
    actual_tickers = None
    try:
        sample_data = torch.load(sample_graph_path, weights_only=False)
        if hasattr(sample_data, 'tickers'):
            actual_tickers = sample_data.tickers
            print(f"✅ Using actual tickers from data: {len(actual_tickers)} stocks")
    except Exception as e:
        print(f"⚠️  Could not get actual tickers from data: {e}")
    
    coordinator = create_multi_agent_system(
        gnn_model=gnn_model,
        device=DEVICE,
        embedding_dim=embedding_dim,
        learning_rate=learning_rate,
        actual_tickers=actual_tickers
    )
    
    # 3. Determine Training Period
    start_date, end_date = determine_multi_agent_training_period()
    
    # 4. Create Multi-Agent Environment
    print(f"\n--- Creating Multi-Agent Environment ---")
    environment = MultiAgentTradingEnv(
        start_date=start_date,
        end_date=end_date,
        gnn_model=gnn_model,
        sector_groups=coordinator.sector_groups,
        device=DEVICE
    )
    
    # 5. Create Trainer and Train
    print(f"\n--- Starting Training ---")
    trainer = MultiAgentTrainer(coordinator, environment, total_timesteps)
    training_stats = trainer.train()
    
    # 6. Save Results (Original location)
    save_path = MODELS_DIR / "multi_agent_results"
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save coordinator
    coordinator.save(save_path)
    
    # Save training statistics
    stats_file = save_path / "training_stats.json"
    with open(stats_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_stats = {}
        for key, value in training_stats.items():
            if isinstance(value, dict):
                serializable_stats[key] = {k: list(v) if isinstance(v, np.ndarray) else v for k, v in value.items()}
            elif isinstance(value, np.ndarray):
                serializable_stats[key] = value.tolist()
            else:
                serializable_stats[key] = value
        
        json.dump(serializable_stats, f, indent=2)
    
    print(f"\n✅ Multi-Agent Training Results saved to: {save_path}")
    
    # 6b. Save Results to /results Directory
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create comprehensive multi-agent results JSON
    multi_agent_results = {
        "model_type": "multi_agent_ctde",
        "training_config": {
            "total_timesteps": total_timesteps,
            "learning_rate": learning_rate,
            "num_agents": len(coordinator.agents),
            "centralized_training": True,
            "decentralized_execution": True
        },
        "sector_configuration": {
            sector_name: {
                "num_stocks": agent.num_stocks,
                "tickers": coordinator.sector_groups.get(sector_name, [])
            }
            for sector_name, agent in coordinator.agents.items()
        },
        "training_period": {
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d")
        },
        "training_statistics": serializable_stats,
        "coordinator_path": str(save_path / "multi_agent_coordinator"),
        "status": "completed"
    }
    
    # Save multi-agent results JSON to /results
    results_json_path = results_dir / "multi_agent_results.json"
    with open(results_json_path, 'w') as f:
        json.dump(multi_agent_results, f, indent=2)
    print(f"✅ Multi-Agent Results JSON saved to: {results_json_path}")
    
    # 7. Print Summary
    print(f"\n📊 Training Summary:")
    print(f"   Episodes: {len(training_stats['episode_returns'])}")
    print(f"   Average Return: {np.mean(training_stats['episode_returns']):.4f}")
    print(f"   Total Timesteps: {sum(training_stats['episode_lengths'])}")
    
    for sector in coordinator.agents.keys():
        sector_perf = training_stats['sector_performance'][sector]
        if sector_perf:
            print(f"   {sector} Avg Performance: {np.mean(sector_perf):.4f}")
    
    print("\n✅ Multi-Agent RL Training Complete!")
    
    return coordinator, training_stats


def main():
    """Main entry point for multi-agent training."""
    try:
        coordinator, stats = run_multi_agent_training()
        return coordinator, stats
    except Exception as e:
        print(f"\n❌ Multi-Agent Training Failed: {e}")
        raise


if __name__ == '__main__':
    main()
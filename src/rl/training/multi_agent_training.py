# phase7_multi_agent_training.py
"""
Phase 7: Multi-Agent RL Training
Implements Cooperative Multi-Agent RL with CTDE for stock trading.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import sys
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from src.models.multi_agent.coordinator import (
    MultiAgentCoordinator,
    SectorGrouping,
    MultiAgentTradingEnv
)
from src.rl.integration import load_gnn_model_for_rl
from src.rl.environment import StockTradingEnv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training Configuration
TOTAL_TIMESTEPS = 10000  # Can be increased
LEARNING_RATE = 1e-5
START_DATE = pd.to_datetime('2023-01-01')
END_DATE = pd.to_datetime('2024-12-31')


class MultiAgentTrainingCallback(BaseCallback):
    """Callback for monitoring multi-agent training."""
    
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        # Log episode info if done
        if 'episode' in self.locals.get('infos', [{}])[0]:
            episode_info = self.locals['infos'][0]['episode']
            if episode_info:
                self.episode_rewards.append(episode_info['r'])
                self.episode_lengths.append(episode_info['l'])
        return True


def create_sector_environments(
    gnn_model,
    sector_groups: Dict[str, List[str]],
    all_tickers: List[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp
) -> Dict[str, StockTradingEnv]:
    """
    Create individual environments for each sector.
    
    Note: This is a simplified version. In practice, you'd need to
    create environments that only expose stocks from each sector.
    """
    envs = {}
    
    for sector_name, tickers in sector_groups.items():
        # Filter to only this sector's tickers
        valid_tickers = [t for t in tickers if t in all_tickers]
        
        if len(valid_tickers) > 0:
            # Create environment (simplified - uses full env but filters actions)
            env = StockTradingEnv(start_date, end_date, gnn_model, DEVICE)
            envs[sector_name] = env
    
    return envs


def train_multi_agent_system(
    coordinator: MultiAgentCoordinator,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    total_timesteps: int = 10000
) -> Dict[str, any]:
    """
    Train multi-agent system using CTDE (Centralized Training, Decentralized Execution).
    
    Training Strategy:
    1. Each agent trains independently on its sector
    2. Mixing network learns to combine Q-values
    3. Global reward is shared for centralized training
    """
    print("\n" + "="*60)
    print("🤖 Training Multi-Agent RL System")
    print("="*60)
    
    # Create environments for each sector
    sector_groups = coordinator.sector_groups
    all_tickers = coordinator.all_tickers
    
    print(f"\n📊 Training Configuration:")
    print(f"   - Number of agents: {len(coordinator.agents)}")
    print(f"   - Total stocks: {len(all_tickers)}")
    print(f"   - Timesteps: {total_timesteps}")
    print(f"   - Date range: {start_date.date()} to {end_date.date()}")
    
    # Training statistics
    training_stats = {
        'sector_performance': {},
        'global_performance': {},
        'mixing_network_loss': []
    }
    
    # Train each agent independently (simplified approach)
    print("\n🔨 Training individual agents...")
    
    for sector_name, agent in coordinator.agents.items():
        print(f"\n   Training {sector_name} agent ({agent.num_stocks} stocks)...")
        
        # Create environment for this sector
        # Note: In practice, you'd create a filtered environment
        # For now, we'll use the full environment but only train on relevant actions
        env = StockTradingEnv(start_date, end_date, coordinator.gnn_model, DEVICE)
        
        # Create callback
        callback = MultiAgentTrainingCallback(verbose=1)
        
        try:
            # Train agent
            # Disable progress bar to avoid dependency issues
            agent.agent.learn(
                total_timesteps=total_timesteps // len(coordinator.agents),
                callback=callback,
                progress_bar=False  # Disabled to avoid rich dependency issues
            )
            
            agent.is_trained = True
            
            # Record performance
            training_stats['sector_performance'][sector_name] = {
                'mean_reward': np.mean(callback.episode_rewards) if callback.episode_rewards else 0,
                'mean_length': np.mean(callback.episode_lengths) if callback.episode_lengths else 0
            }
            
            print(f"   ✅ {sector_name} agent trained")
            
        except Exception as e:
            print(f"   ⚠️  Error training {sector_name} agent: {e}")
            training_stats['sector_performance'][sector_name] = {'error': str(e)}
    
    # Train mixing network (simplified - would need proper Q-value collection)
    print("\n🔨 Training mixing network...")
    print("   (Note: Full mixing network training requires Q-value collection)")
    print("   (This is a simplified implementation)")
    
    print("\n✅ Multi-agent training completed!")
    
    return training_stats


def evaluate_multi_agent_system(
    coordinator: MultiAgentCoordinator,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    n_episodes: int = 5
) -> Dict[str, float]:
    """
    Evaluate multi-agent system performance.
    
    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "="*60)
    print("📊 Evaluating Multi-Agent System")
    print("="*60)
    
    # Use MultiAgentTradingEnv for proper observation splitting
    multi_env = MultiAgentTradingEnv(
        start_date=start_date,
        end_date=end_date,
        gnn_model=coordinator.gnn_model,
        sector_groups=coordinator.sector_groups,
        all_tickers=coordinator.all_tickers,
        device=DEVICE
    )
    
    episode_returns = []
    episode_sharpe = []
    episode_max_dd = []
    
    for episode in range(n_episodes):
        obs, info = multi_env.reset()
        done = False
        portfolio_values = [multi_env.base_env.portfolio_value]
        step_count = 0
        max_steps = min(50, multi_env.base_env.max_steps)  # Limit steps per episode for faster evaluation
        
        while not done and step_count < max_steps:
            # Split global observation into sector-specific observations
            sector_obs = multi_env.get_sector_observations(obs)
            
            # Get actions from all agents
            actions_dict = coordinator.get_agent_actions(sector_obs, deterministic=True)
            
            # Step environment with merged actions
            obs, reward, terminated, truncated, info = multi_env.step(actions_dict)
            done = terminated or truncated
            
            portfolio_values.append(multi_env.base_env.portfolio_value)
            step_count += 1
        
        # Calculate metrics
        returns = np.array(portfolio_values)
        total_return = (returns[-1] / returns[0]) - 1
        
        # Sharpe ratio (simplified)
        daily_returns = np.diff(returns) / returns[:-1]
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
        
        # Max drawdown
        cumulative = np.maximum.accumulate(returns)
        drawdown = (cumulative - returns) / cumulative
        max_dd = np.max(drawdown)
        
        episode_returns.append(total_return)
        episode_sharpe.append(sharpe)
        episode_max_dd.append(max_dd)
    
    results = {
        'mean_return': np.mean(episode_returns),
        'std_return': np.std(episode_returns),
        'mean_sharpe': np.mean(episode_sharpe),
        'mean_max_dd': np.mean(episode_max_dd),
        'n_episodes': n_episodes
    }
    
    print(f"\n📈 Evaluation Results:")
    print(f"   Mean Return: {results['mean_return']*100:.2f}%")
    print(f"   Mean Sharpe: {results['mean_sharpe']:.2f}")
    print(f"   Mean Max DD: {results['mean_max_dd']*100:.2f}%")
    
    return results


def main():
    """Main training and evaluation pipeline."""
    print("🚀 Phase 7: Multi-Agent RL Training")
    print("="*60)
    
    # Load GNN model
    print("\n📁 Loading GNN model...")
    try:
        gnn_model = load_gnn_model_for_rl()
        gnn_model.eval()
        print("✅ GNN model loaded")
    except Exception as e:
        print(f"❌ Error loading GNN model: {e}")
        print("   Please ensure Phase 4 model is trained first.")
        return
    
    # Get all tickers from sector groups BEFORE creating coordinator
    print("\n🤖 Creating multi-agent system...")
    sector_groups = SectorGrouping.load_sector_mapping()
    
    # Extract all tickers from sector groups
    all_tickers = []
    for tickers in sector_groups.values():
        all_tickers.extend(tickers)
    all_tickers = sorted(list(set(all_tickers)))
    
    print(f"📊 Loaded {len(sector_groups)} sectors with {len(all_tickers)} total tickers")
    
    # Create multi-agent coordinator with all tickers
    coordinator = MultiAgentCoordinator(
        gnn_model=gnn_model,
        sector_groups=sector_groups,
        all_tickers=all_tickers,  # Now properly populated
        device=DEVICE,
        learning_rate=LEARNING_RATE
    )
    
    # Train multi-agent system
    training_stats = train_multi_agent_system(
        coordinator=coordinator,
        start_date=START_DATE,
        end_date=END_DATE,
        total_timesteps=TOTAL_TIMESTEPS
    )
    
    # Evaluate
    eval_results = evaluate_multi_agent_system(
        coordinator=coordinator,
        start_date=START_DATE,
        end_date=END_DATE,
        n_episodes=5
    )
    
    # Save results
    import json
    results = {
        'training_stats': training_stats,
        'evaluation_results': eval_results
    }
    
    results_file = RESULTS_DIR / 'multi_agent_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to: {results_file}")
    
    # Save models
    multi_agent_dir = MODELS_DIR / "multi_agent_models"
    multi_agent_dir.mkdir(parents=True, exist_ok=True)
    
    for sector_name, agent in coordinator.agents.items():
        if agent.is_trained:
            agent_path = multi_agent_dir / f"agent_{sector_name}.zip"
            agent.agent.save(agent_path)
            print(f"✅ Saved {sector_name} agent to: {agent_path}")
    
    print("\n🎉 Multi-Agent RL Training Complete!")


if __name__ == "__main__":
    main()


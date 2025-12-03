# scripts/phase5_rl_quick_test.py
"""
Quick test of final RL training (reduced timesteps for faster validation)
"""

import torch
from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / 'scripts'))

from src.rl.training.final_training import FinalStockTradingEnv
from src.rl.integration import load_gnn_model_for_rl
from src.rl.agents.single_agent import StockTradingAgent

# Quick test configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODELS_DIR = PROJECT_ROOT / "models"
RL_SAVE_PATH = MODELS_DIR / "rl_ppo_agent_model_final"
RL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

TOTAL_TIMESTEPS = 5000  # Reduced for quick test
PPO_LEARNING_RATE = 3e-4
REWARD_TYPE = 'risk_adjusted'

START_DATE = pd.to_datetime('2023-01-01')
END_DATE = pd.to_datetime('2024-12-31')


def quick_test():
    """Quick test of final training."""
    print("=" * 80)
    print("🧪 Quick Test: Final RL Training (5000 timesteps)")
    print("=" * 80)
    
    # Load GNN model
    gnn_model = load_gnn_model_for_rl()
    
    # Create environment
    def make_env():
        return FinalStockTradingEnv(
            start_date=START_DATE,
            end_date=END_DATE,
            gnn_model=gnn_model,
            device=DEVICE,
            reward_type=REWARD_TYPE
        )
    
    # Create agent
    agent = StockTradingAgent(
        gnn_model=gnn_model,
        env_factory=make_env,
        device=DEVICE,
        learning_rate=PPO_LEARNING_RATE,
        tensorboard_log=None,
        policy="MlpPolicy",
        verbose=1
    )
    
    # Train
    print("\n--- Training (5000 timesteps, ~2-3 minutes) ---")
    training_stats = agent.train(total_timesteps=TOTAL_TIMESTEPS)
    
    # Save
    save_path = RL_SAVE_PATH / "ppo_stock_agent_final_quick"
    agent.save(save_path)
    print(f"\n✅ Quick test agent saved to: {save_path}")
    
    return agent


if __name__ == '__main__':
    agent = quick_test()


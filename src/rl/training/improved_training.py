# scripts/phase5_rl_improved_training.py
"""
Phase 5: RL Integration with Improved Risk-Adjusted Reward Function

This script trains the RL agent using the improved environment that considers:
- Sharpe Ratio (risk-adjusted returns)
- Max Drawdown penalty
- Volatility penalty

This should help the agent achieve better risk-adjusted performance than Buy-and-Hold.
"""

import torch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / 'scripts'))

from phase5_rl_integration import load_gnn_model_for_rl
from rl_environment_improved import ImprovedStockTradingEnv
from rl_agent import StockTradingAgent

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODELS_DIR = PROJECT_ROOT / "models"
RL_SAVE_PATH = MODELS_DIR / "rl_ppo_agent_model_improved"
RL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

# Training parameters
TOTAL_TIMESTEPS = 10000  # Can increase for better performance
PPO_LEARNING_RATE = 3e-4
REWARD_TYPE = 'risk_adjusted'  # Options: 'simple', 'sharpe', 'drawdown_aware', 'risk_adjusted'

# Backtesting period (same as original)
START_DATE = '2023-01-01'
END_DATE = '2024-12-31'


def run_improved_rl_training():
    """
    Train RL agent with improved risk-adjusted reward function.
    """
    print("=" * 70)
    print("🚀 Phase 5: Improved RL Training with Risk-Adjusted Rewards")
    print("=" * 70)
    print(f"Reward Type: {REWARD_TYPE}")
    print(f"Total Timesteps: {TOTAL_TIMESTEPS}")
    print("=" * 70)
    
    # Load GNN model
    print("\n--- Loading GNN Model ---")
    gnn_model = load_gnn_model_for_rl()
    
    # Create environment factory with improved reward
    def make_env():
        return ImprovedStockTradingEnv(
            start_date=START_DATE,
            end_date=END_DATE,
            gnn_model=gnn_model,
            device=DEVICE,
            reward_type=REWARD_TYPE
        )
    
    # Create RL agent
    print("\n--- Initializing RL Agent (PPO) ---")
    agent = StockTradingAgent(
        gnn_model=gnn_model,
        env_factory=make_env,
        device=DEVICE,
        learning_rate=PPO_LEARNING_RATE,
        tensorboard_log=PROJECT_ROOT / "runs" / "rl_improved",
        policy="MlpPolicy",
        verbose=1
    )
    
    # Train agent
    print("\n--- Training RL Agent ---")
    print("This may take several minutes...")
    training_stats = agent.train(total_timesteps=TOTAL_TIMESTEPS)
    
    # Save agent
    save_path = RL_SAVE_PATH / "ppo_stock_agent_improved"
    agent.save(save_path)
    print(f"\n✅ Improved RL Agent saved to: {save_path}")
    
    # Print training summary
    print("\n" + "=" * 70)
    print("📊 Training Summary")
    print("=" * 70)
    print(f"Total Timesteps: {TOTAL_TIMESTEPS}")
    print(f"Reward Type: {REWARD_TYPE}")
    print(f"Model saved: {save_path}")
    print("\n💡 Next Steps:")
    print("  1. Evaluate the improved agent using phase6_evaluation.py")
    print("  2. Compare with baseline strategies")
    print("  3. Check if Sharpe ratio and Max Drawdown improved")
    
    return agent, training_stats


if __name__ == '__main__':
    agent, stats = run_improved_rl_training()


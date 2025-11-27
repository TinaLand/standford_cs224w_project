# scripts/phase5_rl_final_training.py
"""
Phase 5: Final RL Training with All Improvements

This script combines:
1. Balanced trading environment (faster position building)
2. Risk-adjusted reward function (Sharpe, drawdown, volatility)
3. Full training pipeline

Expected improvements:
- Better returns in uptrends (faster position building)
- Better risk-adjusted returns (Sharpe ratio)
- Lower drawdowns (risk control)
"""

import torch
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / 'scripts'))

from phase5_rl_integration import load_gnn_model_for_rl
from rl_environment_balanced import BalancedStockTradingEnv
from rl_environment_improved import ImprovedStockTradingEnv
from rl_agent import StockTradingAgent

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODELS_DIR = PROJECT_ROOT / "models"
RL_SAVE_PATH = MODELS_DIR / "rl_ppo_agent_model_final"
RL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

# Training parameters
TOTAL_TIMESTEPS = 15000  # Increased for better convergence
PPO_LEARNING_RATE = 3e-4
REWARD_TYPE = 'risk_adjusted'  # Use risk-adjusted rewards

# Backtesting period
import pandas as pd
START_DATE = pd.to_datetime('2023-01-01')
END_DATE = pd.to_datetime('2024-12-31')


class FinalStockTradingEnv(BalancedStockTradingEnv):
    """
    Final environment combining:
    - Balanced trading (dynamic position sizing)
    - Risk-adjusted rewards
    """
    
    def __init__(self, start_date, end_date, gnn_model, device, reward_type='risk_adjusted'):
        # Initialize balanced environment
        super().__init__(start_date, end_date, gnn_model, device)
        
        # Add risk-adjusted reward tracking (from ImprovedStockTradingEnv)
        from collections import deque
        self.reward_type = reward_type
        self.portfolio_history = deque(maxlen=20 + 10)
        self.portfolio_history.append(self.initial_cash)
        self.returns_history = deque(maxlen=20)
        self.peak_value = self.initial_cash
    
    def _calculate_sharpe_ratio(self, returns):
        """Calculate Sharpe ratio from returns."""
        if len(returns) < 2:
            return 0.0
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        if std_return == 0:
            return 0.0
        sharpe = (mean_return - 0.0 / 252) / std_return * np.sqrt(252)
        return sharpe
    
    def _calculate_drawdown(self, current_value):
        """Calculate current drawdown from peak."""
        if current_value > self.peak_value:
            self.peak_value = current_value
        if self.peak_value == 0:
            return 0.0
        drawdown = (self.peak_value - current_value) / self.peak_value
        return drawdown
    
    def _calculate_risk_adjusted_reward(self, portfolio_return, current_value):
        """Calculate risk-adjusted reward."""
        self.portfolio_history.append(current_value)
        self.returns_history.append(portfolio_return)
        
        # Base return
        base_reward = portfolio_return
        
        # Sharpe bonus
        sharpe_bonus = 0.0
        if len(self.returns_history) >= 5:
            returns_array = np.array(self.returns_history)
            sharpe = self._calculate_sharpe_ratio(returns_array)
            sharpe_bonus = 0.2 * np.tanh(sharpe / 2.0)
        
        # Drawdown penalty
        drawdown = self._calculate_drawdown(current_value)
        drawdown_penalty = 0.5 * drawdown
        
        # Volatility penalty
        volatility_penalty = 0.0
        if len(self.returns_history) >= 5:
            returns_array = np.array(self.returns_history)
            volatility = np.std(returns_array)
            volatility_penalty = 0.3 * min(volatility * 10, 1.0)
        
        risk_adjusted_reward = (
            base_reward +
            sharpe_bonus -
            drawdown_penalty -
            volatility_penalty
        )
        
        return risk_adjusted_reward
    
    def step(self, action):
        """Override step to add risk-adjusted rewards."""
        # Get base step from balanced environment
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Get current portfolio value
        current_value = info['portfolio_value']
        
        # Calculate portfolio return
        if len(self.portfolio_history) > 0:
            prev_value = self.portfolio_history[-1]
            portfolio_return = (current_value - prev_value) / prev_value if prev_value > 0 else 0.0
        else:
            portfolio_return = 0.0
        
        # Apply risk-adjusted reward
        if self.reward_type == 'risk_adjusted':
            improved_reward = self._calculate_risk_adjusted_reward(portfolio_return, current_value)
        else:
            improved_reward = reward
        
        # Update info
        info['simple_reward'] = reward
        info['improved_reward'] = improved_reward
        info['portfolio_return'] = portfolio_return
        
        if len(self.returns_history) >= 5:
            returns_array = np.array(self.returns_history)
            info['sharpe_ratio'] = self._calculate_sharpe_ratio(returns_array)
            info['volatility'] = np.std(returns_array)
        
        info['drawdown'] = self._calculate_drawdown(current_value)
        info['peak_value'] = self.peak_value
        
        return obs, improved_reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        """Reset environment and clear history."""
        obs, info = super().reset(seed=seed, options=options)
        
        # Reset tracking
        self.portfolio_history.clear()
        self.portfolio_history.append(self.initial_cash)
        self.returns_history.clear()
        self.peak_value = self.initial_cash
        
        return obs, info


def run_final_rl_training():
    """
    Train RL agent with all improvements.
    """
    print("=" * 80)
    print("🚀 Phase 5: Final RL Training with All Improvements")
    print("=" * 80)
    print("Improvements:")
    print("  1. ✅ Balanced trading environment (dynamic position sizing)")
    print("  2. ✅ Risk-adjusted reward function (Sharpe, drawdown, volatility)")
    print("  3. ✅ Faster position building in uptrends")
    print("=" * 80)
    print(f"Reward Type: {REWARD_TYPE}")
    print(f"Total Timesteps: {TOTAL_TIMESTEPS}")
    print(f"Learning Rate: {PPO_LEARNING_RATE}")
    print("=" * 80)
    
    # Load GNN model
    print("\n--- Loading GNN Model ---")
    gnn_model = load_gnn_model_for_rl()
    
    # Create environment factory
    def make_env():
        return FinalStockTradingEnv(
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
        tensorboard_log=PROJECT_ROOT / "runs" / "rl_final",
        policy="MlpPolicy",
        verbose=1
    )
    
    # Train agent
    print("\n--- Training RL Agent ---")
    print("This may take 10-30 minutes depending on your hardware...")
    print("Progress will be shown below:\n")
    
    training_stats = agent.train(total_timesteps=TOTAL_TIMESTEPS)
    
    # Save agent
    save_path = RL_SAVE_PATH / "ppo_stock_agent_final"
    agent.save(save_path)
    print(f"\n✅ Final RL Agent saved to: {save_path}")
    
    # Print training summary
    print("\n" + "=" * 80)
    print("📊 Training Summary")
    print("=" * 80)
    print(f"Total Timesteps: {TOTAL_TIMESTEPS}")
    print(f"Reward Type: {REWARD_TYPE}")
    print(f"Model saved: {save_path}")
    print("\n💡 Next Steps:")
    print("  1. Evaluate the final agent using phase6_evaluation.py")
    print("  2. Compare with baseline strategies")
    print("  3. Check if Sharpe ratio > 2.18 (beat Buy-and-Hold)")
    print("  4. Verify returns are competitive with Buy-and-Hold")
    
    return agent, training_stats


if __name__ == '__main__':
    import numpy as np
    agent, stats = run_final_rl_training()


# scripts/rl_environment_improved.py
"""
Improved RL Environment with Risk-Adjusted Reward Function

This version implements a reward function that considers:
1. Portfolio returns (with transaction costs)
2. Sharpe Ratio (risk-adjusted returns)
3. Max Drawdown penalty
4. Volatility penalty

This encourages the agent to achieve better risk-adjusted performance than Buy-and-Hold.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from collections import deque
import sys

# Import base environment
sys.path.append(str(Path(__file__).resolve().parent))
from src.rl.environments.single_agent import StockTradingEnv
from src.rl.config import SingleAgentConfig, DATA_GRAPHS_DIR, PROJECT_ROOT
TRANSACTION_COST = SingleAgentConfig.TRANSACTION_COST

# --- Improved Reward Configuration ---
REWARD_TYPE = 'risk_adjusted'  # Options: 'simple', 'sharpe', 'risk_adjusted', 'drawdown_aware'
SHARPE_WINDOW = 20  # Rolling window for Sharpe ratio calculation (trading days)
RISK_FREE_RATE = 0.0  # Risk-free rate (0% for simplicity, can be set to treasury rate)
DRAWDOWN_PENALTY_WEIGHT = 0.5  # Weight for drawdown penalty in reward
VOLATILITY_PENALTY_WEIGHT = 0.3  # Weight for volatility penalty
SHARPE_BONUS_WEIGHT = 0.2  # Weight for Sharpe ratio bonus


class ImprovedStockTradingEnv(StockTradingEnv):
    """
    Enhanced Stock Trading Environment with Risk-Adjusted Rewards.
    
    Key Improvements:
    1. Tracks portfolio value history for risk metrics
    2. Calculates rolling Sharpe ratio
    3. Penalizes drawdowns
    4. Rewards risk-adjusted returns over raw returns
    """
    
    def __init__(self, start_date, end_date, gnn_model, device, reward_type='risk_adjusted'):
        super().__init__(start_date, end_date, gnn_model, device)
        
        self.reward_type = reward_type
        
        # Track portfolio history for risk metrics
        self.portfolio_history = deque(maxlen=SHARPE_WINDOW + 10)  # Keep more for calculation
        self.portfolio_history.append(self.initial_cash)
        
        # Track peak portfolio value for drawdown calculation
        self.peak_value = self.initial_cash
        
        # Track returns for Sharpe calculation
        self.returns_history = deque(maxlen=SHARPE_WINDOW)
        
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """
        Calculate Sharpe Ratio from returns.
        
        Sharpe Ratio = (Mean Return - Risk-Free Rate) / Std(Returns)
        
        Args:
            returns: Array of portfolio returns
        
        Returns:
            Sharpe ratio (annualized if enough data)
        """
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualize (assuming 252 trading days per year)
        # For daily returns, multiply by sqrt(252)
        sharpe = (mean_return - RISK_FREE_RATE / 252) / std_return * np.sqrt(252)
        
        return sharpe
    
    def _calculate_drawdown(self, current_value: float) -> float:
        """
        Calculate current drawdown from peak.
        
        Drawdown = (Peak Value - Current Value) / Peak Value
        
        Args:
            current_value: Current portfolio value
        
        Returns:
            Current drawdown (0.0 to 1.0)
        """
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        if self.peak_value == 0:
            return 0.0
        
        drawdown = (self.peak_value - current_value) / self.peak_value
        return drawdown
    
    def _calculate_risk_adjusted_reward(
        self,
        portfolio_return: float,
        current_value: float
    ) -> float:
        """
        Calculate risk-adjusted reward that encourages:
        1. Higher returns
        2. Lower volatility (higher Sharpe)
        3. Lower drawdowns
        
        Reward = Return + Sharpe_Bonus - Drawdown_Penalty - Volatility_Penalty
        
        Args:
            portfolio_return: Simple portfolio return for this step
            current_value: Current portfolio value
        
        Returns:
            Risk-adjusted reward
        """
        # Update history
        self.portfolio_history.append(current_value)
        self.returns_history.append(portfolio_return)
        
        # 1. Base return component
        base_reward = portfolio_return
        
        # 2. Sharpe ratio bonus (if we have enough history)
        sharpe_bonus = 0.0
        if len(self.returns_history) >= 5:  # Need at least 5 days
            returns_array = np.array(self.returns_history)
            sharpe = self._calculate_sharpe_ratio(returns_array)
            # Normalize Sharpe (typical range: -2 to 5, we want to scale to ~0-1)
            sharpe_bonus = SHARPE_BONUS_WEIGHT * np.tanh(sharpe / 2.0)  # tanh normalization
        
        # 3. Drawdown penalty
        drawdown = self._calculate_drawdown(current_value)
        drawdown_penalty = DRAWDOWN_PENALTY_WEIGHT * drawdown
        
        # 4. Volatility penalty (if we have enough history)
        volatility_penalty = 0.0
        if len(self.returns_history) >= 5:
            returns_array = np.array(self.returns_history)
            volatility = np.std(returns_array)
            # Penalize high volatility (normalize to 0-1 range)
            volatility_penalty = VOLATILITY_PENALTY_WEIGHT * min(volatility * 10, 1.0)
        
        # Combine all components
        risk_adjusted_reward = (
            base_reward +
            sharpe_bonus -
            drawdown_penalty -
            volatility_penalty
        )
        
        return risk_adjusted_reward
    
    def _calculate_sharpe_reward(
        self,
        portfolio_return: float,
        current_value: float
    ) -> float:
        """
        Reward based primarily on Sharpe ratio.
        
        This encourages the agent to maximize risk-adjusted returns.
        """
        self.portfolio_history.append(current_value)
        self.returns_history.append(portfolio_return)
        
        if len(self.returns_history) < 5:
            # Not enough data, use simple return
            return portfolio_return
        
        returns_array = np.array(self.returns_history)
        sharpe = self._calculate_sharpe_ratio(returns_array)
        
        # Reward is proportional to Sharpe ratio
        # Scale to reasonable range (Sharpe typically -2 to 5)
        reward = np.tanh(sharpe / 2.0) * 0.1  # Scale to ~0.1 range
        
        return reward
    
    def _calculate_drawdown_aware_reward(
        self,
        portfolio_return: float,
        current_value: float
    ) -> float:
        """
        Reward that heavily penalizes drawdowns.
        
        This encourages the agent to preserve capital during downturns.
        """
        self.portfolio_history.append(current_value)
        self.returns_history.append(portfolio_return)
        
        # Base return
        base_reward = portfolio_return
        
        # Heavy drawdown penalty
        drawdown = self._calculate_drawdown(current_value)
        drawdown_penalty = 2.0 * drawdown  # Heavy penalty
        
        reward = base_reward - drawdown_penalty
        
        return reward
    
    def step(self, action):
        """
        Override step function to use improved reward calculation.
        """
        # Get base step result
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Get current portfolio value
        current_value = info['portfolio_value']
        
        # Calculate portfolio return for this step
        if len(self.portfolio_history) > 0:
            prev_value = self.portfolio_history[-1]
            portfolio_return = (current_value - prev_value) / prev_value if prev_value > 0 else 0.0
        else:
            portfolio_return = 0.0
        
        # Apply improved reward function based on type
        if self.reward_type == 'simple':
            # Use original simple reward
            improved_reward = reward
        elif self.reward_type == 'sharpe':
            improved_reward = self._calculate_sharpe_reward(portfolio_return, current_value)
        elif self.reward_type == 'drawdown_aware':
            improved_reward = self._calculate_drawdown_aware_reward(portfolio_return, current_value)
        elif self.reward_type == 'risk_adjusted':
            improved_reward = self._calculate_risk_adjusted_reward(portfolio_return, current_value)
        else:
            improved_reward = reward
        
        # Add reward info to info dict
        info['simple_reward'] = reward
        info['improved_reward'] = improved_reward
        info['portfolio_return'] = portfolio_return
        
        # Add risk metrics if available
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
        
        # Reset tracking variables
        self.portfolio_history.clear()
        self.portfolio_history.append(self.initial_cash)
        self.returns_history.clear()
        self.peak_value = self.initial_cash
        
        return obs, info


# --- Example Usage and Testing ---

if __name__ == '__main__':
    """
    Test the improved environment with different reward types.
    """
    from phase5_rl_integration import load_gnn_model_for_rl
    
    print("=" * 70)
    print("🧪 Testing Improved RL Environment with Risk-Adjusted Rewards")
    print("=" * 70)
    
    # Load GNN model
    gnn_model = load_gnn_model_for_rl()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test different reward types
    reward_types = ['simple', 'sharpe', 'drawdown_aware', 'risk_adjusted']
    
    from src.evaluation.evaluation import START_DATE_TEST, END_DATE_TEST
    
    results = []
    
    for reward_type in reward_types:
        print(f"\n{'='*70}")
        print(f"Testing Reward Type: {reward_type}")
        print(f"{'='*70}")
        
        env = ImprovedStockTradingEnv(
            start_date=START_DATE_TEST,
            end_date=END_DATE_TEST,
            gnn_model=gnn_model,
            device=device,
            reward_type=reward_type
        )
        
        # Run a simple test (random actions)
        obs, info = env.reset()
        total_reward = 0
        rewards = []
        sharpe_ratios = []
        drawdowns = []
        
        for step in range(min(100, env.max_steps)):  # Test first 100 steps
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            rewards.append(reward)
            
            if 'sharpe_ratio' in info:
                sharpe_ratios.append(info['sharpe_ratio'])
            if 'drawdown' in info:
                drawdowns.append(info['drawdown'])
            
            if terminated or truncated:
                break
        
        final_value = info['portfolio_value']
        avg_reward = np.mean(rewards) if rewards else 0
        avg_sharpe = np.mean(sharpe_ratios) if sharpe_ratios else 0
        max_drawdown = max(drawdowns) if drawdowns else 0
        
        results.append({
            'reward_type': reward_type,
            'final_value': final_value,
            'total_reward': total_reward,
            'avg_reward': avg_reward,
            'avg_sharpe': avg_sharpe,
            'max_drawdown': max_drawdown
        })
        
        print(f"  Final Value: ${final_value:.2f}")
        print(f"  Total Reward: {total_reward:.4f}")
        print(f"  Avg Reward: {avg_reward:.4f}")
        print(f"  Avg Sharpe: {avg_sharpe:.4f}")
        print(f"  Max Drawdown: {max_drawdown*100:.2f}%")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Reward Type Comparison")
    print("=" * 70)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    print("\n✅ Improved environment tested successfully!")
    print("💡 Use 'risk_adjusted' reward type for training to beat Buy-and-Hold.")


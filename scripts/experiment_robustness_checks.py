#!/usr/bin/env python3
"""
Robustness Checks: Transaction Costs and Slippage Sensitivity

Tests model performance under different transaction cost and slippage assumptions.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, List

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.environments.single_agent import StockTradingEnv
from src.rl.training.single_agent import load_gnn_model_for_rl
import torch
import pandas as pd
import gymnasium as gym

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RobustnessStockTradingEnv(StockTradingEnv):
    """
    Wrapper for StockTradingEnv that allows custom transaction cost and slippage.
    """
    def __init__(self, start_date, end_date, gnn_model, device, 
                 transaction_cost=0.001, slippage=0.0):
        super().__init__(start_date, end_date, gnn_model, device)
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        # Store original transaction cost from config
        from src.rl.config import SingleAgentConfig
        self._original_transaction_cost = SingleAgentConfig.TRANSACTION_COST
    
    def step(self, action):
        """
        Override step to use custom transaction cost and slippage.
        """
        # Temporarily override TRANSACTION_COST in config
        from src.rl.config import SingleAgentConfig
        original_tc = SingleAgentConfig.TRANSACTION_COST
        SingleAgentConfig.TRANSACTION_COST = self.transaction_cost
        
        try:
            # Apply slippage to prices before executing trades
            # This is a simplified approach - in reality, slippage affects execution price
            obs, reward, terminated, truncated, info = super().step(action)
            
            # Apply slippage penalty to portfolio value
            if 'portfolio_value' in info and self.slippage > 0:
                # Reduce portfolio value by slippage amount
                info['portfolio_value'] *= (1 - self.slippage)
            
            return obs, reward, terminated, truncated, info
        finally:
            # Restore original transaction cost
            SingleAgentConfig.TRANSACTION_COST = original_tc


def run_rl_evaluation_with_params(
    transaction_cost: float,
    slippage: float,
    model_path: str = None
) -> Dict[str, float]:
    """
    Run RL evaluation with specific transaction cost and slippage.
    
    Args:
        transaction_cost: Transaction cost as fraction (e.g., 0.001 = 0.1%)
        slippage: Slippage as fraction (e.g., 0.0005 = 0.05%)
        model_path: Path to trained RL model (optional)
    
    Returns:
        Dictionary of performance metrics
    """
    print(f"\n--- Testing: Transaction Cost={transaction_cost:.4f}, Slippage={slippage:.4f} ---")
    
    # Load GNN model
    gnn_model = load_gnn_model_for_rl()
    gnn_model.eval()
    
    # Define test period (use same as evaluation)
    START_DATE_TEST = pd.to_datetime('2023-01-01')
    END_DATE_TEST = pd.to_datetime('2024-12-31')
    
    # Create environment with custom transaction cost and slippage
    env = RobustnessStockTradingEnv(
        start_date=START_DATE_TEST,
        end_date=END_DATE_TEST,
        gnn_model=gnn_model,
        device=DEVICE,
        transaction_cost=transaction_cost,
        slippage=slippage
    )
    
    # Load RL agent if model_path provided
    agent = None
    if model_path and Path(model_path).exists():
        try:
            from stable_baselines3 import PPO
            agent = PPO.load(model_path, env=env)
        except Exception as e:
            print(f"Warning: Could not load agent from {model_path}: {e}")
            print("Using random policy for testing")
    
    # Run evaluation
    obs = env.reset()
    done = False
    portfolio_values = []
    daily_returns = []
    
    while not done:
        if agent is not None:
            action, _ = agent.predict(obs, deterministic=True)
        else:
            # Random policy for baseline
            action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if 'portfolio_value' in info:
            portfolio_values.append(info['portfolio_value'])
        if 'daily_return' in info:
            daily_returns.append(info['daily_return'])
    
    # Calculate metrics
    if len(portfolio_values) < 2:
        return {
            'transaction_cost': transaction_cost,
            'slippage': slippage,
            'cumulative_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'final_portfolio_value': portfolio_values[0] if portfolio_values else 100000.0
        }
    
    portfolio_series = pd.Series(portfolio_values)
    returns = portfolio_series.pct_change().dropna()
    
    # Cumulative return
    cumulative_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
    
    # Sharpe ratio (annualized)
    if len(returns) > 0 and returns.std() > 0:
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
    else:
        sharpe_ratio = 0.0
    
    # Max drawdown
    cumulative_max = portfolio_series.cummax()
    drawdown = (cumulative_max - portfolio_series) / cumulative_max
    max_drawdown = drawdown.max()
    
    results = {
        'transaction_cost': transaction_cost,
        'slippage': slippage,
        'cumulative_return': float(cumulative_return),
        'sharpe_ratio': float(sharpe_ratio),
        'max_drawdown': float(max_drawdown),
        'final_portfolio_value': float(portfolio_values[-1]),
        'num_trading_days': len(portfolio_values)
    }
    
    print(f"   Cumulative Return: {cumulative_return:.4f}")
    print(f"   Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"   Max Drawdown: {max_drawdown:.4f}")
    
    return results


def run_robustness_experiment():
    """
    Run comprehensive robustness checks.
    """
    print("=" * 60)
    print("Robustness Checks: Transaction Costs & Slippage Sensitivity")
    print("=" * 60)
    
    # Test different transaction costs (0% to 0.5%)
    transaction_costs = [0.0, 0.0001, 0.0005, 0.001, 0.002, 0.005]
    
    # Test different slippage values (0% to 0.1%)
    slippage_values = [0.0, 0.0001, 0.0005, 0.001, 0.002, 0.005]
    
    # Try to find trained RL model
    model_path = None
    possible_paths = [
        PROJECT_ROOT / "models" / "ppo_stock_trading.zip",
        PROJECT_ROOT / "models" / "rl_agent.zip",
        PROJECT_ROOT / "models" / "best_rl_model.zip"
    ]
    for path in possible_paths:
        if path.exists():
            model_path = str(path)
            print(f"\nFound RL model: {model_path}")
            break
    
    if model_path is None:
        print("\nWarning: No trained RL model found. Using random policy.")
        print("This will provide baseline performance but not optimal results.")
    
    # Run experiments
    results = []
    
    # 1. Transaction cost sensitivity (with zero slippage)
    print("\n" + "=" * 60)
    print("1. Transaction Cost Sensitivity Analysis")
    print("=" * 60)
    for tc in transaction_costs:
        result = run_rl_evaluation_with_params(
            transaction_cost=tc,
            slippage=0.0,
            model_path=model_path
        )
        results.append(result)
    
    # 2. Slippage sensitivity (with baseline transaction cost)
    print("\n" + "=" * 60)
    print("2. Slippage Sensitivity Analysis")
    print("=" * 60)
    baseline_tc = 0.001  # 0.1% baseline
    for slip in slippage_values:
        result = run_rl_evaluation_with_params(
            transaction_cost=baseline_tc,
            slippage=slip,
            model_path=model_path
        )
        results.append(result)
    
    # 3. Combined sensitivity (both vary)
    print("\n" + "=" * 60)
    print("3. Combined Sensitivity Analysis")
    print("=" * 60)
    combined_configs = [
        (0.0, 0.0),      # No costs
        (0.001, 0.0005),  # Moderate
        (0.002, 0.001),   # High
        (0.005, 0.002),   # Very high
    ]
    for tc, slip in combined_configs:
        result = run_rl_evaluation_with_params(
            transaction_cost=tc,
            slippage=slip,
            model_path=model_path
        )
        results.append(result)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_file = RESULTS_DIR / "robustness_checks_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\n Results saved to: {results_file}")
    
    # Save JSON for detailed analysis
    results_json = RESULTS_DIR / "robustness_checks_results.json"
    with open(results_json, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f" Detailed results saved to: {results_json}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    # Transaction cost impact
    tc_results = [r for r in results if r['slippage'] == 0.0]
    if tc_results:
        print("\nTransaction Cost Impact (Slippage=0):")
        print(f"  Best Return: {max(tc_results, key=lambda x: x['cumulative_return'])['cumulative_return']:.4f}")
        print(f"  Worst Return: {min(tc_results, key=lambda x: x['cumulative_return'])['cumulative_return']:.4f}")
        print(f"  Return Range: {max(tc_results, key=lambda x: x['cumulative_return'])['cumulative_return'] - min(tc_results, key=lambda x: x['cumulative_return'])['cumulative_return']:.4f}")
    
    # Slippage impact
    slip_results = [r for r in results if r['transaction_cost'] == baseline_tc]
    if slip_results:
        print("\nSlippage Impact (Transaction Cost=0.1%):")
        print(f"  Best Return: {max(slip_results, key=lambda x: x['cumulative_return'])['cumulative_return']:.4f}")
        print(f"  Worst Return: {min(slip_results, key=lambda x: x['cumulative_return'])['cumulative_return']:.4f}")
        print(f"  Return Range: {max(slip_results, key=lambda x: x['cumulative_return'])['cumulative_return'] - min(slip_results, key=lambda x: x['cumulative_return'])['cumulative_return']:.4f}")
    
    return results


if __name__ == '__main__':
    try:
        results = run_robustness_experiment()
        print("\n Robustness checks completed successfully!")
    except Exception as e:
        print(f"\n Error during robustness checks: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


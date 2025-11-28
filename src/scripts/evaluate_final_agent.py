# scripts/evaluate_final_agent.py
"""
Evaluate Final RL Agent with All Improvements

Compares the final agent against:
1. Original RL Agent
2. Buy-and-Hold
3. Equal-Weight strategies
"""

import torch
from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / 'scripts'))

from phase5_rl_integration import load_gnn_model_for_rl
from phase6_evaluation import (
    START_DATE_TEST, END_DATE_TEST, RESULTS_DIR,
    calculate_financial_metrics
)
from baseline_strategies import run_all_baseline_strategies
from compare_all_strategies import load_rl_metrics
from rl_agent import StockTradingAgent
import numpy as np


def evaluate_final_agent():
    """
    Evaluate the final improved RL agent.
    """
    print("=" * 80)
    print("📊 Evaluating Final RL Agent with All Improvements")
    print("=" * 80)
    
    # Load GNN model
    print("\n--- Loading GNN Model ---")
    gnn_model = load_gnn_model_for_rl()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load final agent
    print("\n--- Loading Final RL Agent ---")
    from phase5_rl_final_training import FinalStockTradingEnv
    
    def make_final_env():
        return FinalStockTradingEnv(
            start_date=START_DATE_TEST,
            end_date=END_DATE_TEST,
            gnn_model=gnn_model,
            device=device,
            reward_type='risk_adjusted'
        )
    
    final_agent_path = PROJECT_ROOT / "models" / "rl_ppo_agent_model_final" / "ppo_stock_agent_final.zip"
    
    if not final_agent_path.exists():
        print(f"❌ Final agent not found at {final_agent_path}")
        print("   Please run phase5_rl_final_training.py first")
        return None
    
    agent = StockTradingAgent(
        gnn_model=gnn_model,
        env_factory=make_final_env,
        device=device,
        learning_rate=1e-5,
        tensorboard_log=None,
        policy="MlpPolicy",
        verbose=0
    )
    
    agent.load(final_agent_path)
    print(f"✅ Final agent loaded from: {final_agent_path}")
    
    # Evaluate final agent
    print("\n--- Evaluating Final Agent ---")
    
    # Run backtest manually
    env = make_final_env()
    obs, info = env.reset()
    
    portfolio_values = [info.get('portfolio_value', 10000)]
    done = False
    
    print("Running backtest...")
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        portfolio_values.append(info['portfolio_value'])
        
        if env.current_step % 100 == 0:
            print(f"  Step {env.current_step}: Value ${info['portfolio_value']:.2f}")
    
    # Calculate metrics
    total_days = len(portfolio_values) - 1
    final_metrics = calculate_financial_metrics(portfolio_values, total_days)
    final_metrics['final_value'] = portfolio_values[-1]
    final_metrics['total_days'] = total_days
    
    # Load original agent metrics
    print("\n--- Loading Original Agent Metrics ---")
    original_metrics = load_rl_metrics()
    
    # Load baseline strategies
    print("\n--- Loading Baseline Strategies ---")
    baseline_df = run_all_baseline_strategies(
        START_DATE_TEST,
        END_DATE_TEST,
        initial_cash=10000.0
    )
    
    # Create comprehensive comparison
    print("\n" + "=" * 80)
    print("📊 Comprehensive Strategy Comparison")
    print("=" * 80)
    
    comparison_data = []
    
    # Add original RL agent
    if original_metrics:
        comparison_data.append({
            'Strategy': 'RL Agent (Original)',
            'Final_Value': original_metrics['final_value'],
            'Return': original_metrics['Cumulative_Return'] * 100,
            'Sharpe': original_metrics['Sharpe_Ratio'],
            'Max_DD': original_metrics['Max_Drawdown'] * 100
        })
    
    # Add final RL agent
    comparison_data.append({
        'Strategy': 'RL Agent (Final - Improved)',
        'Final_Value': final_metrics['final_value'],
        'Return': final_metrics['Cumulative_Return'] * 100,
        'Sharpe': final_metrics['Sharpe_Ratio'],
        'Max_DD': final_metrics['Max_Drawdown'] * 100
    })
    
    # Add baseline strategies
    for _, row in baseline_df.iterrows():
        comparison_data.append({
            'Strategy': row['strategy'],
            'Final_Value': row['final_value'],
            'Return': row['Cumulative_Return'] * 100,
            'Sharpe': row['Sharpe_Ratio'],
            'Max_DD': row['Max_Drawdown'] * 100
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Sharpe', ascending=False)
    
    print(comparison_df.to_string(index=False))
    
    # Calculate improvements
    if original_metrics:
        print("\n" + "=" * 80)
        print("📈 Final Agent vs Original Agent")
        print("=" * 80)
        
        return_improvement = final_metrics['Cumulative_Return'] * 100 - original_metrics['Cumulative_Return'] * 100
        sharpe_improvement = final_metrics['Sharpe_Ratio'] - original_metrics['Sharpe_Ratio']
        dd_improvement = (final_metrics['Max_Drawdown'] - original_metrics['Max_Drawdown']) * 100
        
        print(f"Return: {final_metrics['Cumulative_Return']*100:.2f}% vs {original_metrics['Cumulative_Return']*100:.2f}% ({return_improvement:+.2f}%)")
        print(f"Sharpe: {final_metrics['Sharpe_Ratio']:.4f} vs {original_metrics['Sharpe_Ratio']:.4f} ({sharpe_improvement:+.4f})")
        print(f"Max DD: {final_metrics['Max_Drawdown']*100:.2f}% vs {original_metrics['Max_Drawdown']*100:.2f}% ({dd_improvement:+.2f}%)")
    
    # Compare with Buy-and-Hold
    buy_hold_row = baseline_df[baseline_df['strategy'] == 'Buy-and-Hold'].iloc[0]
    
    print("\n" + "=" * 80)
    print("🎯 Final Agent vs Buy-and-Hold")
    print("=" * 80)
    
    return_vs_bh = final_metrics['Cumulative_Return'] * 100 - buy_hold_row['Cumulative_Return'] * 100
    sharpe_vs_bh = final_metrics['Sharpe_Ratio'] - buy_hold_row['Sharpe_Ratio']
    dd_vs_bh = (final_metrics['Max_Drawdown'] - buy_hold_row['Max_Drawdown']) * 100
    
    print(f"Return: {final_metrics['Cumulative_Return']*100:.2f}% vs {buy_hold_row['Cumulative_Return']*100:.2f}% ({return_vs_bh:+.2f}%)")
    print(f"Sharpe: {final_metrics['Sharpe_Ratio']:.4f} vs {buy_hold_row['Sharpe_Ratio']:.4f} ({sharpe_vs_bh:+.4f})")
    print(f"Max DD: {final_metrics['Max_Drawdown']*100:.2f}% vs {buy_hold_row['Max_Drawdown']*100:.2f}% ({dd_vs_bh:+.2f}%)")
    
    # Final verdict
    print("\n" + "=" * 80)
    print("✅ Final Verdict")
    print("=" * 80)
    
    if sharpe_vs_bh > 0:
        print("✅ Final Agent beats Buy-and-Hold on Sharpe Ratio (risk-adjusted returns)!")
    else:
        print("⚠️  Final Agent Sharpe ratio still below Buy-and-Hold")
    
    if dd_vs_bh < 0:
        print("✅ Final Agent has lower Max Drawdown (better risk control)!")
    else:
        print("⚠️  Final Agent Max Drawdown is higher")
    
    if return_vs_bh > -20:  # Within 20% of Buy-and-Hold
        print("✅ Final Agent returns are competitive with Buy-and-Hold!")
    else:
        print("⚠️  Final Agent returns still significantly lower than Buy-and-Hold")
    
    # Save results
    comparison_df.to_csv(RESULTS_DIR / 'final_agent_comparison.csv', index=False)
    print(f"\n✅ Comparison results saved to: {RESULTS_DIR / 'final_agent_comparison.csv'}")
    
    return comparison_df


if __name__ == '__main__':
    results_df = evaluate_final_agent()


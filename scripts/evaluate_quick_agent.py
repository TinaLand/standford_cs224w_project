# scripts/evaluate_quick_agent.py
"""
Evaluate the quick test agent to verify improvements
"""

import torch
from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / 'scripts'))

from phase5_rl_integration import load_gnn_model_for_rl
from phase5_rl_final_training import FinalStockTradingEnv
from phase6_evaluation import (
    START_DATE_TEST, END_DATE_TEST, RESULTS_DIR,
    calculate_financial_metrics
)
from baseline_strategies import run_all_baseline_strategies
from rl_agent import StockTradingAgent


def evaluate_quick_agent():
    """Evaluate the quick test agent."""
    print("=" * 80)
    print("📊 Evaluating Quick Test Agent (Improved Environment + Rewards)")
    print("=" * 80)
    
    # Load GNN model
    gnn_model = load_gnn_model_for_rl()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load quick test agent
    quick_agent_path = PROJECT_ROOT / "models" / "rl_ppo_agent_model_final" / "ppo_stock_agent_final_quick.zip"
    
    if not quick_agent_path.exists():
        print(f"❌ Quick test agent not found. Run phase5_rl_quick_test.py first.")
        return None
    
    def make_env():
        return FinalStockTradingEnv(
            start_date=START_DATE_TEST,
            end_date=END_DATE_TEST,
            gnn_model=gnn_model,
            device=device,
            reward_type='risk_adjusted'
        )
    
    agent = StockTradingAgent(
        gnn_model=gnn_model,
        env_factory=make_env,
        device=device,
        learning_rate=1e-5,
        tensorboard_log=None,
        policy="MlpPolicy",
        verbose=0
    )
    
    agent.load(quick_agent_path)
    print(f"✅ Quick test agent loaded")
    
    # Run backtest
    print("\n--- Running Backtest ---")
    env = make_env()
    obs, info = env.reset()
    
    portfolio_values = [info.get('portfolio_value', 10000)]
    done = False
    
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        portfolio_values.append(info['portfolio_value'])
        
        if env.current_step % 100 == 0:
            print(f"  Step {env.current_step}: Value ${info['portfolio_value']:.2f}")
    
    # Calculate metrics
    total_days = len(portfolio_values) - 1
    metrics = calculate_financial_metrics(portfolio_values, total_days)
    
    print(f"\n✅ Backtest Complete")
    print(f"Final Value: ${portfolio_values[-1]:.2f}")
    print(f"Cumulative Return: {metrics['Cumulative_Return']*100:.2f}%")
    print(f"Sharpe Ratio: {metrics['Sharpe_Ratio']:.4f}")
    print(f"Max Drawdown: {metrics['Max_Drawdown']*100:.2f}%")
    
    # Compare with baselines
    print("\n--- Comparing with Baseline Strategies ---")
    baseline_df = run_all_baseline_strategies(
        START_DATE_TEST,
        END_DATE_TEST,
        initial_cash=10000.0
    )
    
    # Load original agent metrics
    from compare_all_strategies import load_rl_metrics
    original_metrics = load_rl_metrics()
    
    # Create comparison
    print("\n" + "=" * 80)
    print("📊 Performance Comparison")
    print("=" * 80)
    
    comparison = []
    
    if original_metrics:
        comparison.append({
            'Strategy': 'RL Agent (Original)',
            'Return': original_metrics['Cumulative_Return'] * 100,
            'Sharpe': original_metrics['Sharpe_Ratio'],
            'Max_DD': original_metrics['Max_Drawdown'] * 100
        })
    
    comparison.append({
        'Strategy': 'RL Agent (Quick Test - Improved)',
        'Return': metrics['Cumulative_Return'] * 100,
        'Sharpe': metrics['Sharpe_Ratio'],
        'Max_DD': metrics['Max_Drawdown'] * 100
    })
    
    for _, row in baseline_df.iterrows():
        comparison.append({
            'Strategy': row['strategy'],
            'Return': row['Cumulative_Return'] * 100,
            'Sharpe': row['Sharpe_Ratio'],
            'Max_DD': row['Max_Drawdown'] * 100
        })
    
    comp_df = pd.DataFrame(comparison)
    comp_df = comp_df.sort_values('Sharpe', ascending=False)
    print(comp_df.to_string(index=False))
    
    # Save
    comp_df.to_csv(RESULTS_DIR / 'quick_agent_comparison.csv', index=False)
    print(f"\n✅ Results saved to: {RESULTS_DIR / 'quick_agent_comparison.csv'}")
    
    return comp_df


if __name__ == '__main__':
    results = evaluate_quick_agent()


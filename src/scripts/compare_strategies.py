# scripts/compare_all_strategies.py
"""
Comprehensive Strategy Comparison

Compares RL Agent performance with baseline strategies (Buy-and-Hold, Equal-Weight).
"""

import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / 'scripts'))

from src.evaluation.evaluation import START_DATE_TEST, END_DATE_TEST, RESULTS_DIR
# Note: run_all_baseline_strategies may need to be implemented or imported from another module


def load_rl_metrics():
    """Load RL agent metrics from final_metrics.csv."""
    rl_metrics_path = RESULTS_DIR / 'final_metrics.csv'
    
    if not rl_metrics_path.exists():
        print("  RL metrics not found. Run phase6_evaluation.py first.")
        return None
    
    rl_df = pd.read_csv(rl_metrics_path)
    if rl_df.empty:
        return None
    
    # Extract RL metrics
    rl_row = rl_df.iloc[0]
    return {
        'strategy': 'RL Agent (PPO)',
        'final_value': rl_row['final_value'],
        'Cumulative_Return': rl_row['Cumulative_Return'],
        'Sharpe_Ratio': rl_row['Sharpe_Ratio'],
        'Max_Drawdown': rl_row['Max_Drawdown']
    }


def main():
    """Main comparison function."""
    print("=" * 70)
    print(" Comprehensive Strategy Comparison: RL vs Baselines")
    print("=" * 70)
    
    # Load RL metrics
    rl_metrics = load_rl_metrics()
    
    # Run baseline strategies
    baseline_df = run_all_baseline_strategies(
        START_DATE_TEST,
        END_DATE_TEST,
        initial_cash=10000.0
    )
    
    # Combine results
    if rl_metrics:
        # Add RL metrics to DataFrame
        rl_row = pd.DataFrame([rl_metrics])
        all_results = pd.concat([rl_row, baseline_df], ignore_index=True)
    else:
        all_results = baseline_df
        print("\n  RL metrics not available. Showing baseline strategies only.")
    
    # Sort by Sharpe Ratio (descending)
    all_results = all_results.sort_values('Sharpe_Ratio', ascending=False)
    
    print("\n" + "=" * 70)
    print(" COMPREHENSIVE STRATEGY COMPARISON")
    print("=" * 70)
    print(all_results.to_string(index=False))
    
    # Calculate relative performance
    if rl_metrics:
        print("\n" + "=" * 70)
        print(" RL Agent vs Best Baseline")
        print("=" * 70)
        
        best_baseline = baseline_df.loc[baseline_df['Sharpe_Ratio'].idxmax()]
        
        print(f"\nBest Baseline Strategy: {best_baseline['strategy']}")
        print(f"  Sharpe Ratio: {best_baseline['Sharpe_Ratio']:.4f}")
        print(f"  Return: {best_baseline['Cumulative_Return']*100:.2f}%")
        print(f"  Max Drawdown: {best_baseline['Max_Drawdown']*100:.2f}%")
        
        print(f"\nRL Agent (PPO):")
        print(f"  Sharpe Ratio: {rl_metrics['Sharpe_Ratio']:.4f}")
        print(f"  Return: {rl_metrics['Cumulative_Return']*100:.2f}%")
        print(f"  Max Drawdown: {rl_metrics['Max_Drawdown']*100:.2f}%")
        
        # Calculate differences
        sharpe_diff = rl_metrics['Sharpe_Ratio'] - best_baseline['Sharpe_Ratio']
        return_diff = (rl_metrics['Cumulative_Return'] - best_baseline['Cumulative_Return']) * 100
        dd_diff = (rl_metrics['Max_Drawdown'] - best_baseline['Max_Drawdown']) * 100
        
        print(f"\nDifference (RL - Best Baseline):")
        print(f"  Sharpe Ratio: {sharpe_diff:+.4f} ({sharpe_diff/best_baseline['Sharpe_Ratio']*100:+.2f}%)")
        print(f"  Return: {return_diff:+.2f}%")
        print(f"  Max Drawdown: {dd_diff:+.2f}%")
    
    # Save comprehensive results
    all_results.to_csv(RESULTS_DIR / 'comprehensive_strategy_comparison.csv', index=False)
    print(f"\n Comprehensive comparison saved to: {RESULTS_DIR / 'comprehensive_strategy_comparison.csv'}")
    
    return all_results


if __name__ == '__main__':
    main()


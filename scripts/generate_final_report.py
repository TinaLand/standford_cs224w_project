# scripts/generate_final_report.py
"""
Generate Final Project Report Summary

This script generates a comprehensive summary of all results and improvements.
"""

import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / 'scripts'))

RESULTS_DIR = PROJECT_ROOT / "results"


def generate_final_report():
    """Generate comprehensive final report."""
    print("=" * 80)
    print("📊 Final Project Report Summary")
    print("=" * 80)
    
    # Load all results
    results = {}
    
    # 1. GNN Node-Level Metrics
    try:
        gnn_metrics = pd.read_csv(RESULTS_DIR / 'gnn_node_metrics.csv')
        results['gnn_metrics'] = gnn_metrics.iloc[0].to_dict()
        print("\n✅ GNN Node-Level Metrics loaded")
    except:
        print("⚠️  GNN metrics not found")
    
    # 2. RL Agent Metrics
    try:
        final_metrics = pd.read_csv(RESULTS_DIR / 'final_metrics.csv')
        results['rl_original'] = final_metrics.iloc[0].to_dict()
        print("✅ Original RL Agent metrics loaded")
    except:
        print("⚠️  Original RL metrics not found")
    
    # 3. Quick Test Agent Metrics
    try:
        quick_metrics = pd.read_csv(RESULTS_DIR / 'quick_agent_comparison.csv')
        results['rl_quick'] = quick_metrics[quick_metrics['Strategy'] == 'RL Agent (Quick Test - Improved)'].iloc[0].to_dict()
        print("✅ Quick Test Agent metrics loaded")
    except:
        print("⚠️  Quick test metrics not found")
    
    # 4. Baseline Strategies
    try:
        baseline_df = pd.read_csv(RESULTS_DIR / 'comprehensive_strategy_comparison.csv')
        results['baselines'] = baseline_df
        print("✅ Baseline strategies loaded")
    except:
        print("⚠️  Baseline strategies not found")
    
    # 5. Phase 3 vs Phase 4 Comparison
    try:
        phase_comparison = pd.read_csv(RESULTS_DIR / 'phase3_vs_phase4_comparison.csv')
        results['phase_comparison'] = phase_comparison
        print("✅ Phase comparison loaded")
    except:
        print("⚠️  Phase comparison not found")
    
    # 6. Ablation Results
    try:
        ablation_df = pd.read_csv(RESULTS_DIR / 'ablation_results.csv')
        results['ablation'] = ablation_df
        print("✅ Ablation results loaded")
    except:
        print("⚠️  Ablation results not found")
    
    # Generate report
    print("\n" + "=" * 80)
    print("📈 KEY PERFORMANCE METRICS")
    print("=" * 80)
    
    # GNN Metrics
    if 'gnn_metrics' in results:
        gnn = results['gnn_metrics']
        print("\n--- GNN Model (Node-Level) ---")
        print(f"  Accuracy: {gnn.get('accuracy', 0)*100:.2f}%")
        print(f"  F1 Score: {gnn.get('f1_score', 0):.4f}")
        print(f"  Precision@Top-10: {gnn.get('Precision@Top-10', 0)*100:.2f}%")
        print(f"  IC Mean: {gnn.get('IC_mean', 0):.4f}")
        print(f"  IC IR: {gnn.get('IC_IR', 0):.4f}")
    
    # RL Agent Comparison
    print("\n--- RL Agent Performance (Portfolio-Level) ---")
    
    if 'rl_original' in results:
        orig = results['rl_original']
        print(f"\n  Original Agent:")
        print(f"    Return: {orig.get('Cumulative_Return', 0)*100:.2f}%")
        print(f"    Sharpe: {orig.get('Sharpe_Ratio', 0):.4f}")
        print(f"    Max DD: {orig.get('Max_Drawdown', 0)*100:.2f}%")
    
    if 'rl_quick' in results:
        quick = results['rl_quick']
        print(f"\n  Improved Agent (Quick Test):")
        print(f"    Return: {quick.get('Return', 0):.2f}%")
        print(f"    Sharpe: {quick.get('Sharpe', 0):.4f}")
        print(f"    Max DD: {quick.get('Max_DD', 0):.2f}%")
        
        if 'rl_original' in results:
            orig = results['rl_original']
            print(f"\n  Improvement:")
            print(f"    Return: {quick.get('Return', 0) - orig.get('Cumulative_Return', 0)*100:+.2f}%")
            print(f"    Sharpe: {quick.get('Sharpe', 0) - orig.get('Sharpe_Ratio', 0):+.4f}")
    
    # Baseline Comparison
    if 'baselines' in results:
        print("\n--- Baseline Strategies ---")
        baseline_df = results['baselines']
        for _, row in baseline_df.iterrows():
            print(f"\n  {row['strategy']}:")
            print(f"    Return: {row.get('Cumulative_Return', 0)*100:.2f}%")
            print(f"    Sharpe: {row.get('Sharpe_Ratio', 0):.4f}")
            print(f"    Max DD: {row.get('Max_Drawdown', 0)*100:.2f}%")
    
    # Phase Comparison
    if 'phase_comparison' in results:
        print("\n--- Phase 3 vs Phase 4 Comparison ---")
        phase_df = results['phase_comparison']
        for _, row in phase_df.iterrows():
            print(f"\n  {row['Model']}:")
            print(f"    Accuracy: {row.get('Accuracy', 0)*100:.2f}%")
            print(f"    F1: {row.get('F1_Score', 0):.4f}")
            print(f"    Precision@Top-10: {row.get('Precision@Top-10', 0)*100:.2f}%")
    
    # Save report
    report_path = RESULTS_DIR / 'final_project_report.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CS224W Project - Final Report Summary\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Key Achievements:\n")
        f.write("- GNN Model trained and evaluated\n")
        f.write("- RL Agent implemented with improvements\n")
        f.write("- Baseline strategies compared\n")
        f.write("- Comprehensive evaluation completed\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Performance Summary\n")
        f.write("=" * 80 + "\n")
        
        if 'rl_quick' in results:
            quick = results['rl_quick']
            f.write(f"\nImproved RL Agent:\n")
            f.write(f"  Return: {quick.get('Return', 0):.2f}%\n")
            f.write(f"  Sharpe: {quick.get('Sharpe', 0):.4f}\n")
            f.write(f"  Max DD: {quick.get('Max_DD', 0):.2f}%\n")
    
    print(f"\n✅ Final report saved to: {report_path}")
    
    return results


if __name__ == '__main__':
    results = generate_final_report()


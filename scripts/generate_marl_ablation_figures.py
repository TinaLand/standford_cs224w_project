#!/usr/bin/env python3
"""
Generate MARL Ablation Study Figures
Creates visualizations for MARL vs Single-Agent vs Independent Learning comparison
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
FIGS_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGS_DIR.mkdir(exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

def load_marl_ablation_data():
    """Load MARL ablation study results."""
    results_file = RESULTS_DIR / "marl_ablation_results.json"
    summary_file = RESULTS_DIR / "marl_ablation_summary.csv"
    
    if not results_file.exists():
        raise FileNotFoundError(f"MARL ablation results not found: {results_file}")
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    summary_df = pd.read_csv(summary_file) if summary_file.exists() else None
    
    return results, summary_df

def create_marl_comparison_chart():
    """Create comparison chart for MARL vs Single-Agent vs Independent Learning."""
    results, summary_df = load_marl_ablation_data()
    
    if summary_df is None:
        print("Warning: Summary CSV not found, creating from JSON data")
        # Create summary from JSON
        methods = []
        avg_returns = []
        sharpe_ratios = []
        
        if 'MARL_QMIX' in results:
            methods.append('MARL (QMIX)')
            avg_returns.append(results['MARL_QMIX']['avg_return'])
            sharpe_ratios.append(results['MARL_QMIX']['training_metrics'].get('sharpe_ratio', 0))
        
        if 'Single_Agent_PPO' in results:
            methods.append('Single-Agent RL')
            avg_returns.append(results['Single_Agent_PPO']['training_metrics']['mean_return'])
            sharpe_ratios.append(0)  # Not calculated
        
        if 'Independent_Learning' in results:
            methods.append('Independent Learning')
            avg_returns.append(results['Independent_Learning']['avg_return'])
            sharpe_ratios.append(results['Independent_Learning']['training_metrics'].get('sharpe_ratio', 0))
        
        summary_df = pd.DataFrame({
            'Method': methods,
            'Avg Return': avg_returns,
            'Sharpe Ratio': sharpe_ratios
        })
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Average Return Comparison
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    bars1 = ax1.bar(summary_df['Method'], summary_df['Avg Return'], color=colors[:len(summary_df)])
    ax1.set_ylabel('Average Return', fontsize=11, fontweight='bold')
    ax1.set_title('Average Return Comparison', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    # Rotate x-axis labels
    ax1.set_xticks(range(len(summary_df)))
    ax1.set_xticklabels(summary_df['Method'], rotation=15, ha='right')
    
    # Plot 2: Sharpe Ratio Comparison
    bars2 = ax2.bar(summary_df['Method'], summary_df['Sharpe Ratio'], color=colors[:len(summary_df)])
    ax2.set_ylabel('Sharpe Ratio', fontsize=11, fontweight='bold')
    ax2.set_title('Sharpe Ratio Comparison', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        if height != 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    # Rotate x-axis labels
    ax2.set_xticks(range(len(summary_df)))
    ax2.set_xticklabels(summary_df['Method'], rotation=15, ha='right')
    
    plt.tight_layout()
    output_path = FIGS_DIR / "figure_marl_ablation_comparison.png"
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path}")
    plt.close()

def create_sector_performance_chart():
    """Create sector-level performance comparison chart."""
    results, _ = load_marl_ablation_data()
    
    # Extract sector performance for MARL and Independent Learning
    marl_sectors = {}
    iql_sectors = {}
    
    if 'MARL_QMIX' in results and 'sector_performance' in results['MARL_QMIX']:
        for sector, perf_list in results['MARL_QMIX']['sector_performance'].items():
            if perf_list and len(perf_list) > 0:
                marl_sectors[sector] = np.mean(perf_list) if isinstance(perf_list, list) else perf_list
    
    if 'Independent_Learning' in results and 'sector_performance' in results['Independent_Learning']:
        for sector, perf_list in results['Independent_Learning']['sector_performance'].items():
            if perf_list and len(perf_list) > 0:
                iql_sectors[sector] = np.mean(perf_list) if isinstance(perf_list, list) else perf_list
    
    if not marl_sectors and not iql_sectors:
        print("Warning: No sector performance data found")
        return
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Prepare data
    all_sectors = sorted(set(list(marl_sectors.keys()) + list(iql_sectors.keys())))
    x = np.arange(len(all_sectors))
    width = 0.35
    
    marl_values = [marl_sectors.get(s, 0) for s in all_sectors]
    iql_values = [iql_sectors.get(s, 0) for s in all_sectors]
    
    # Create bars
    bars1 = ax.bar(x - width/2, marl_values, width, label='MARL (QMIX)', color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x + width/2, iql_values, width, label='Independent Learning', color='#F18F01', alpha=0.8)
    
    # Customize
    ax.set_ylabel('Average Return per Sector', fontsize=11, fontweight='bold')
    ax.set_title('Sector-Level Performance Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_sectors, rotation=15, ha='right')
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if abs(height) > 1:  # Only label if significant
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
    
    plt.tight_layout()
    output_path = FIGS_DIR / "figure_marl_sector_performance.png"
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path}")
    plt.close()

def create_training_curves_comparison():
    """Create training curves comparison for all three methods."""
    results, _ = load_marl_ablation_data()
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Extract episode returns
    if 'MARL_QMIX' in results and 'episode_returns' in results['MARL_QMIX']:
        marl_returns = results['MARL_QMIX']['episode_returns']
        episodes_marl = range(1, len(marl_returns) + 1)
        ax.plot(episodes_marl, marl_returns, label='MARL (QMIX)', color='#2E86AB', linewidth=2, marker='o', markersize=4)
    
    has_data = False
    if 'MARL_QMIX' in results and 'episode_returns' in results['MARL_QMIX']:
        has_data = True
    
    if 'Single_Agent_PPO' in results:
        # Single-agent may have fewer episodes, use evaluation results
        sa_data = results['Single_Agent_PPO']
        if 'evaluation_results' in sa_data:
            mean_return = sa_data['evaluation_results'].get('mean_return', 0)
            n_episodes = sa_data['evaluation_results'].get('n_episodes', 5)
            # Create a flat line for single-agent (fewer episodes)
            ax.plot([1, n_episodes], [mean_return, mean_return], 
                   label='Single-Agent RL', color='#A23B72', linewidth=2, linestyle='--', marker='s', markersize=4)
            has_data = True
    
    if 'Independent_Learning' in results and 'episode_returns' in results['Independent_Learning']:
        iql_returns = results['Independent_Learning']['episode_returns']
        episodes_iql = range(1, len(iql_returns) + 1)
        ax.plot(episodes_iql, iql_returns, label='Independent Learning', color='#F18F01', linewidth=2, marker='^', markersize=4)
        has_data = True
    
    if not has_data:
        print("Warning: No training curve data available")
        return
    
    ax.set_xlabel('Episode', fontsize=11, fontweight='bold')
    ax.set_ylabel('Episode Return', fontsize=11, fontweight='bold')
    ax.set_title('Training Curves: Episode Returns Over Time', fontsize=12, fontweight='bold')
    if has_data:
        ax.legend(loc='best')
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    
    plt.tight_layout()
    output_path = FIGS_DIR / "figure_marl_training_curves.png"
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path}")
    plt.close()

def main():
    """Generate all MARL ablation study figures."""
    print("=" * 60)
    print("Generating MARL Ablation Study Figures")
    print("=" * 60)
    
    try:
        create_marl_comparison_chart()
        create_sector_performance_chart()
        create_training_curves_comparison()
        
        print("\n" + "=" * 60)
        print("✅ All MARL ablation figures generated successfully!")
        print("=" * 60)
        print("\nGenerated figures:")
        print("  1. figure_marl_ablation_comparison.png - Method comparison")
        print("  2. figure_marl_sector_performance.png - Sector-level performance")
        print("  3. figure_marl_training_curves.png - Training curves")
        
    except Exception as e:
        print(f"\n❌ Error generating figures: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()


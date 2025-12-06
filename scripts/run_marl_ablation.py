"""
MARL Ablation Study Script
Compares MARL (QMIX) vs Single-Agent RL vs Independent Learning
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import json
from typing import Dict, Any

RESULTS_DIR = PROJECT_ROOT / "results"

def load_training_results():
    """Load training results from all three methods."""
    
    results = {}
    
    # 1. Load Multi-Agent RL results
    marl_file = RESULTS_DIR / "multi_agent_results.json"
    if marl_file.exists():
        with open(marl_file, 'r') as f:
            marl_data = json.load(f)
            training_stats = marl_data.get('training_statistics', {})
            results['MARL'] = {
                'episode_returns': training_stats.get('episode_returns', []),
                'episode_lengths': training_stats.get('episode_lengths', []),
                'sector_performance': training_stats.get('sector_performance', {}),
                'avg_return': np.mean(training_stats.get('episode_returns', [0])),
                'total_episodes': len(training_stats.get('episode_returns', [])),
                'total_timesteps': sum(training_stats.get('episode_lengths', []))
            }
    else:
        print(f"Warning: {marl_file} not found")
        results['MARL'] = None
    
    # 2. Load Single-Agent RL results
    single_agent_file = RESULTS_DIR / "single_agent_results.json"
    if single_agent_file.exists():
        with open(single_agent_file, 'r') as f:
            single_data = json.load(f)
            eval_results = single_data.get('evaluation_results', {})
            results['Single_Agent'] = {
                'mean_return': eval_results.get('mean_return', 0),
                'std_return': eval_results.get('std_return', 0),
                'n_episodes': eval_results.get('n_episodes', 0),
                'mean_length': eval_results.get('mean_length', 0)
            }
    else:
        print(f"Warning: {single_agent_file} not found")
        results['Single_Agent'] = None
    
    # 3. Load Independent Learning results
    iql_file = RESULTS_DIR / "independent_learning_training_stats.json"
    if iql_file.exists():
        with open(iql_file, 'r') as f:
            iql_data = json.load(f)
            results['Independent_Learning'] = {
                'episode_returns': iql_data.get('episode_returns', []),
                'episode_lengths': iql_data.get('episode_lengths', []),
                'sector_performance': iql_data.get('sector_performance', {}),
                'avg_return': np.mean(iql_data.get('episode_returns', [0])),
                'total_episodes': len(iql_data.get('episode_returns', [])),
                'total_timesteps': sum(iql_data.get('episode_lengths', []))
            }
    else:
        print(f"Warning: {iql_file} not found")
        results['Independent_Learning'] = None
    
    return results

def calculate_metrics(returns):
    """Calculate performance metrics from returns."""
    if not returns or len(returns) == 0:
        return {}
    
    returns_array = np.array(returns)
    
    metrics = {
        'mean_return': float(np.mean(returns_array)),
        'std_return': float(np.std(returns_array)),
        'min_return': float(np.min(returns_array)),
        'max_return': float(np.max(returns_array)),
        'median_return': float(np.median(returns_array)),
        'sharpe_ratio': float(np.mean(returns_array) / np.std(returns_array)) if np.std(returns_array) > 0 else 0.0
    }
    
    return metrics

def run_marl_ablation():
    """Run MARL ablation: compare MARL vs Single-Agent vs Independent"""
    
    print("=" * 60)
    print("MARL Ablation Study")
    print("=" * 60)
    print("Comparing:")
    print("  1. MARL (QMIX) - Multi-Agent RL with coordination")
    print("  2. Single-Agent RL (PPO) - Baseline")
    print("  3. Independent Learning (IQL) - No coordination")
    print()
    
    # Load results
    print("Loading training results...")
    results = load_training_results()
    
    # Create comparison table
    comparison = {}
    
    # MARL
    if results['MARL']:
        marl_metrics = calculate_metrics(results['MARL']['episode_returns'])
        comparison['MARL_QMIX'] = {
            'description': 'Multi-Agent RL with QMIX mixing network',
            'training_metrics': marl_metrics,
            'total_episodes': results['MARL']['total_episodes'],
            'total_timesteps': results['MARL']['total_timesteps'],
            'avg_return': results['MARL']['avg_return'],
            'sector_performance': results['MARL']['sector_performance'],
            'advantages': [
                'Sector specialization',
                'Coordination through QMIX',
                'Scalable to larger stock universes',
                'Interpretable sector-level decisions'
            ]
        }
    
    # Single-Agent
    if results['Single_Agent']:
        comparison['Single_Agent_PPO'] = {
            'description': 'Single PPO agent for all stocks',
            'training_metrics': {
                'mean_return': results['Single_Agent']['mean_return'],
                'std_return': results['Single_Agent']['std_return']
            },
            'n_episodes': results['Single_Agent']['n_episodes'],
            'limitations': [
                'Action space explosion (3^50)',
                'No sector specialization',
                'Harder to interpret',
                'Poor scalability'
            ]
        }
    
    # Independent Learning
    if results['Independent_Learning']:
        iql_metrics = calculate_metrics(results['Independent_Learning']['episode_returns'])
        comparison['Independent_Learning'] = {
            'description': 'Independent Q-Learning per sector (no coordination)',
            'training_metrics': iql_metrics,
            'total_episodes': results['Independent_Learning']['total_episodes'],
            'total_timesteps': results['Independent_Learning']['total_timesteps'],
            'avg_return': results['Independent_Learning']['avg_return'],
            'sector_performance': results['Independent_Learning']['sector_performance'],
            'limitations': [
                'No global coordination',
                'Cannot enforce portfolio constraints',
                'Agents may work against each other',
                'No value decomposition'
            ]
        }
    
    # Print comparison
    print("\n" + "=" * 60)
    print("Training Performance Comparison")
    print("=" * 60)
    
    if 'MARL_QMIX' in comparison:
        marl = comparison['MARL_QMIX']
        print(f"\n1. MARL (QMIX):")
        print(f"   Average Return: {marl['avg_return']:.4f}")
        print(f"   Total Episodes: {marl['total_episodes']}")
        print(f"   Sharpe Ratio: {marl['training_metrics'].get('sharpe_ratio', 0):.4f}")
        if marl['sector_performance']:
            print(f"   Sector Performance:")
            for sector, perf in marl['sector_performance'].items():
                if perf:
                    avg_perf = np.mean(perf) if isinstance(perf, list) else perf
                    print(f"     - {sector}: {avg_perf:.4f}")
    
    if 'Single_Agent_PPO' in comparison:
        sa = comparison['Single_Agent_PPO']
        print(f"\n2. Single-Agent RL (PPO):")
        print(f"   Mean Return: {sa['training_metrics']['mean_return']:.4f}")
        print(f"   Std Return: {sa['training_metrics']['std_return']:.4f}")
        print(f"   Episodes: {sa['n_episodes']}")
    
    if 'Independent_Learning' in comparison:
        iql = comparison['Independent_Learning']
        print(f"\n3. Independent Learning (IQL):")
        print(f"   Average Return: {iql['avg_return']:.4f}")
        print(f"   Total Episodes: {iql['total_episodes']}")
        print(f"   Sharpe Ratio: {iql['training_metrics'].get('sharpe_ratio', 0):.4f}")
        if iql['sector_performance']:
            print(f"   Sector Performance:")
            for sector, perf in iql['sector_performance'].items():
                if perf:
                    avg_perf = np.mean(perf) if isinstance(perf, list) else perf
                    print(f"     - {sector}: {avg_perf:.4f}")
    
    # Create summary table
    print("\n" + "=" * 60)
    print("Summary Comparison")
    print("=" * 60)
    
    summary_data = []
    if 'MARL_QMIX' in comparison:
        summary_data.append({
            'Method': 'MARL (QMIX)',
            'Avg Return': comparison['MARL_QMIX']['avg_return'],
            'Sharpe Ratio': comparison['MARL_QMIX']['training_metrics'].get('sharpe_ratio', 0),
            'Episodes': comparison['MARL_QMIX']['total_episodes']
        })
    if 'Single_Agent_PPO' in comparison:
        summary_data.append({
            'Method': 'Single-Agent RL',
            'Avg Return': comparison['Single_Agent_PPO']['training_metrics']['mean_return'],
            'Sharpe Ratio': 0,  # Not calculated
            'Episodes': comparison['Single_Agent_PPO']['n_episodes']
        })
    if 'Independent_Learning' in comparison:
        summary_data.append({
            'Method': 'Independent Learning',
            'Avg Return': comparison['Independent_Learning']['avg_return'],
            'Sharpe Ratio': comparison['Independent_Learning']['training_metrics'].get('sharpe_ratio', 0),
            'Episodes': comparison['Independent_Learning']['total_episodes']
        })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print("\n" + summary_df.to_string(index=False))
    
    # Save results
    output_file = RESULTS_DIR / 'marl_ablation_results.json'
    with open(output_file, 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    # Save CSV summary
    if summary_data:
        csv_file = RESULTS_DIR / 'marl_ablation_summary.csv'
        summary_df.to_csv(csv_file, index=False)
        print(f"✅ Summary saved to: {csv_file}")
    
    print("\n" + "=" * 60)
    print("MARL Ablation Study Complete")
    print("=" * 60)
    
    # Analysis
    print("\n📊 Key Findings:")
    if 'MARL_QMIX' in comparison and 'Independent_Learning' in comparison:
        marl_return = comparison['MARL_QMIX']['avg_return']
        iql_return = comparison['Independent_Learning']['avg_return']
        if marl_return > iql_return:
            print(f"   - MARL outperforms Independent Learning by {marl_return - iql_return:.2f} in average return")
        else:
            print(f"   - Independent Learning outperforms MARL by {iql_return - marl_return:.2f} in average return")
    
    if 'MARL_QMIX' in comparison and 'Single_Agent_PPO' in comparison:
        marl_return = comparison['MARL_QMIX']['avg_return']
        sa_return = comparison['Single_Agent_PPO']['training_metrics']['mean_return']
        if marl_return > sa_return:
            print(f"   - MARL outperforms Single-Agent RL by {marl_return - sa_return:.2f} in average return")
        else:
            print(f"   - Single-Agent RL outperforms MARL by {sa_return - marl_return:.2f} in average return")

if __name__ == "__main__":
    run_marl_ablation()

# enhancement_multi_agent_analysis.py
"""
Multi-Agent Decision Analysis for Deep Analysis Enhancement
Implements agent disagreement analysis, sector performance breakdown, and mixing network analysis.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import sys

sys.path.append(str(Path(__file__).resolve().parent))

from multi_agent_rl_coordinator import MultiAgentCoordinator, SectorGrouping
from phase5_rl_integration import load_gnn_model_for_rl
from src.rl.environments.single_agent import StockTradingEnv

from src.utils.paths import PROJECT_ROOT, MODELS_DIR, RESULTS_DIR, MODELS_PLOTS_DIR as PLOTS_DIR
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def analyze_agent_disagreements(
    coordinator: MultiAgentCoordinator,
    env: StockTradingEnv,
    dates: List[pd.Timestamp],
    n_samples: int = 100
) -> Dict[str, Any]:
    """
    Analyze when and why agents disagree.
    
    Returns:
        Dictionary with disagreement statistics
    """
    print("\n" + "="*60)
    print("🔍 Analyzing Multi-Agent Disagreements")
    print("="*60)
    
    disagreements = []
    sector_actions_history = defaultdict(list)
    
    for i, date in enumerate(dates[:n_samples]):
        try:
            # Get global observation
            obs, info = env.reset()
            
            # Get observations for each sector (simplified - would need proper splitting)
            sector_obs = {}
            for sector_name in coordinator.agents.keys():
                sector_obs[sector_name] = obs  # Placeholder - needs proper splitting
            
            # Get actions from all agents
            actions_dict = coordinator.get_agent_actions(sector_obs, deterministic=False)
            
            # Store actions
            for sector, actions in actions_dict.items():
                sector_actions_history[sector].append(actions)
            
            # Calculate disagreement metrics
            # Disagreement = variance in action distributions across agents
            all_actions = list(actions_dict.values())
            if len(all_actions) > 1:
                # Calculate action variance
                action_variance = np.var([np.mean(acts) for acts in all_actions])
                disagreements.append({
                    'date': date,
                    'variance': action_variance,
                    'num_agents': len(actions_dict),
                    'actions': actions_dict
                })
        
        except Exception as e:
            if i < 5:  # Only print first few errors
                print(f"  ⚠️  Error on date {date}: {e}")
            continue
    
    # Analyze disagreement patterns
    if disagreements:
        variances = [d['variance'] for d in disagreements]
        avg_variance = np.mean(variances)
        max_variance = np.max(variances)
        min_variance = np.min(variances)
        
        print(f"\n📊 Disagreement Statistics:")
        print(f"   Average Action Variance: {avg_variance:.4f}")
        print(f"   Max Variance: {max_variance:.4f}")
        print(f"   Min Variance: {min_variance:.4f}")
        
        # Find high disagreement periods
        high_disagreement = [d for d in disagreements if d['variance'] > avg_variance * 1.5]
        print(f"   High Disagreement Periods: {len(high_disagreement)}/{len(disagreements)}")
    
    return {
        'disagreements': disagreements,
        'sector_actions_history': dict(sector_actions_history),
        'statistics': {
            'avg_variance': np.mean(variances) if disagreements else 0,
            'max_variance': np.max(variances) if disagreements else 0,
            'high_disagreement_count': len(high_disagreement) if disagreements else 0
        }
    }


def analyze_sector_performance(
    coordinator: MultiAgentCoordinator,
    env: StockTradingEnv,
    dates: List[pd.Timestamp],
    n_episodes: int = 10
) -> Dict[str, Any]:
    """
    Break down performance by sector.
    
    Returns:
        Dictionary with sector-specific performance metrics
    """
    print("\n" + "="*60)
    print("📊 Analyzing Sector-Specific Performance")
    print("="*60)
    
    sector_performance = {}
    
    for sector_name, agent in coordinator.agents.items():
        print(f"\n   Analyzing {sector_name} sector ({agent.num_stocks} stocks)...")
        
        # Track sector-specific metrics
        sector_returns = []
        sector_sharpe = []
        sector_drawdowns = []
        
        for episode in range(n_episodes):
            try:
                obs, info = env.reset()
                done = False
                portfolio_values = [env.portfolio_value]
                
                while not done:
                    # Get sector-specific observation (simplified)
                    sector_obs = {sector_name: obs}  # Placeholder
                    
                    # Get action from this sector's agent
                    actions_dict = coordinator.get_agent_actions(sector_obs, deterministic=True)
                    
                    # Merge actions (simplified)
                    if sector_name in actions_dict:
                        # For now, use only this sector's actions
                        # In full implementation, would merge with other sectors
                        action = actions_dict[sector_name]
                        
                        # Step environment (simplified - would need proper action merging)
                        obs, reward, terminated, truncated, info = env.step(action)
                        done = terminated or truncated
                        
                        portfolio_values.append(env.portfolio_value)
                
                # Calculate metrics for this episode
                if len(portfolio_values) > 1:
                    returns = np.array(portfolio_values)
                    total_return = (returns[-1] / returns[0]) - 1
                    
                    daily_returns = np.diff(returns) / returns[:-1]
                    sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
                    
                    cumulative = np.maximum.accumulate(returns)
                    drawdown = (cumulative - returns) / cumulative
                    max_dd = np.max(drawdown)
                    
                    sector_returns.append(total_return)
                    sector_sharpe.append(sharpe)
                    sector_drawdowns.append(max_dd)
            
            except Exception as e:
                if episode < 3:
                    print(f"     ⚠️  Error in episode {episode}: {e}")
                continue
        
        if sector_returns:
            sector_performance[sector_name] = {
                'mean_return': np.mean(sector_returns),
                'std_return': np.std(sector_returns),
                'mean_sharpe': np.mean(sector_sharpe),
                'mean_max_dd': np.mean(sector_drawdowns),
                'n_episodes': len(sector_returns)
            }
            
            print(f"     Mean Return: {sector_performance[sector_name]['mean_return']*100:.2f}%")
            print(f"     Mean Sharpe: {sector_performance[sector_name]['mean_sharpe']:.2f}")
    
    return sector_performance


def analyze_mixing_network_weights(
    coordinator: MultiAgentCoordinator,
    global_states: List[torch.Tensor],
    q_values_history: List[Dict[str, float]]
) -> Dict[str, Any]:
    """
    Analyze how the mixing network combines Q-values.
    
    Returns:
        Dictionary with mixing network analysis
    """
    print("\n" + "="*60)
    print("🔍 Analyzing Mixing Network Weights")
    print("="*60)
    
    mixing_analysis = {
        'weight_statistics': {},
        'q_value_correlations': {},
        'mixing_patterns': []
    }
    
    if not global_states or not q_values_history:
        print("  ⚠️  No data available for mixing network analysis")
        return mixing_analysis
    
    # Analyze mixing weights over time
    for i, (global_state, q_values) in enumerate(zip(global_states[:50], q_values_history[:50])):
        try:
            # Get mixing network weights (requires access to internal weights)
            # This is a simplified analysis
            q_vals = torch.tensor([q_values.get(sector, 0.0) for sector in coordinator.agents.keys()])
            
            # Calculate statistics
            mixing_analysis['mixing_patterns'].append({
                'step': i,
                'q_mean': q_vals.mean().item(),
                'q_std': q_vals.std().item(),
                'q_max': q_vals.max().item(),
                'q_min': q_vals.min().item()
            })
        except Exception as e:
            continue
    
    if mixing_analysis['mixing_patterns']:
        q_means = [p['q_mean'] for p in mixing_analysis['mixing_patterns']]
        mixing_analysis['weight_statistics'] = {
            'avg_q_mean': np.mean(q_means),
            'std_q_mean': np.std(q_means),
            'q_value_range': (np.min(q_means), np.max(q_means))
        }
        
        print(f"\n📊 Mixing Network Statistics:")
        print(f"   Average Q-value: {mixing_analysis['weight_statistics']['avg_q_mean']:.4f}")
        print(f"   Q-value Range: {mixing_analysis['weight_statistics']['q_value_range']}")
    
    return mixing_analysis


def visualize_multi_agent_analysis(
    disagreement_results: Dict,
    sector_performance: Dict,
    mixing_analysis: Dict,
    output_dir: Path
):
    """Create visualizations for multi-agent analysis."""
    print("\n📊 Creating Multi-Agent Analysis Visualizations...")
    
    # 1. Disagreement Over Time
    if disagreement_results.get('disagreements'):
        fig, ax = plt.subplots(figsize=(12, 6))
        dates = [d['date'] for d in disagreement_results['disagreements']]
        variances = [d['variance'] for d in disagreement_results['disagreements']]
        
        ax.plot(dates, variances, alpha=0.7)
        ax.axhline(y=np.mean(variances), color='r', linestyle='--', label='Mean Variance')
        ax.set_xlabel('Date')
        ax.set_ylabel('Action Variance (Disagreement)')
        ax.set_title('Multi-Agent Disagreement Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'multi_agent_disagreement.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ Saved: multi_agent_disagreement.png")
    
    # 2. Sector Performance Comparison
    if sector_performance:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        sectors = list(sector_performance.keys())
        returns = [sector_performance[s]['mean_return']*100 for s in sectors]
        sharpe = [sector_performance[s]['mean_sharpe'] for s in sectors]
        max_dd = [sector_performance[s]['mean_max_dd']*100 for s in sectors]
        
        axes[0].bar(sectors, returns)
        axes[0].set_ylabel('Return (%)')
        axes[0].set_title('Sector Returns')
        axes[0].tick_params(axis='x', rotation=45)
        
        axes[1].bar(sectors, sharpe)
        axes[1].set_ylabel('Sharpe Ratio')
        axes[1].set_title('Sector Sharpe Ratios')
        axes[1].tick_params(axis='x', rotation=45)
        
        axes[2].bar(sectors, max_dd)
        axes[2].set_ylabel('Max Drawdown (%)')
        axes[2].set_title('Sector Max Drawdowns')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'sector_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ Saved: sector_performance_comparison.png")
    
    # 3. Mixing Network Q-Value Patterns
    if mixing_analysis.get('mixing_patterns'):
        fig, ax = plt.subplots(figsize=(12, 6))
        steps = [p['step'] for p in mixing_analysis['mixing_patterns']]
        q_means = [p['q_mean'] for p in mixing_analysis['mixing_patterns']]
        
        ax.plot(steps, q_means, alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('Mean Q-Value')
        ax.set_title('Mixing Network Q-Value Patterns')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'mixing_network_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ Saved: mixing_network_patterns.png")


def main():
    """Main analysis pipeline."""
    print("🚀 Multi-Agent Decision Analysis")
    print("="*60)
    
    # Load GNN model
    print("\n📁 Loading GNN model...")
    try:
        gnn_model = load_gnn_model_for_rl()
        gnn_model.eval()
        print("✅ GNN model loaded")
    except Exception as e:
        print(f"❌ Error loading GNN model: {e}")
        return
    
    # Create coordinator
    print("\n🤖 Creating multi-agent coordinator...")
    sector_groups = SectorGrouping.load_sector_mapping()
    all_tickers = []
    for tickers in sector_groups.values():
        all_tickers.extend(tickers)
    all_tickers = sorted(list(set(all_tickers)))
    
    coordinator = MultiAgentCoordinator(
        gnn_model=gnn_model,
        sector_groups=sector_groups,
        all_tickers=all_tickers,
        device=DEVICE,
        learning_rate=1e-5
    )
    
    # Create environment
    print("\n🌍 Creating environment...")
    start_date = pd.to_datetime('2023-01-01')
    end_date = pd.to_datetime('2024-12-31')
    env = StockTradingEnv(start_date, end_date, gnn_model, DEVICE)
    
    # Get dates
    dates = env.data_loader['dates'][:100]  # Sample 100 dates
    
    # Run analyses
    print("\n" + "="*60)
    print("Running Multi-Agent Analyses...")
    print("="*60)
    
    # 1. Disagreement Analysis
    disagreement_results = analyze_agent_disagreements(coordinator, env, dates, n_samples=50)
    
    # 2. Sector Performance Analysis
    sector_performance = analyze_sector_performance(coordinator, env, dates, n_episodes=5)
    
    # 3. Mixing Network Analysis (simplified - needs Q-value collection)
    mixing_analysis = analyze_mixing_network_weights(coordinator, [], [])
    
    # Visualize
    visualize_multi_agent_analysis(disagreement_results, sector_performance, mixing_analysis, PLOTS_DIR)
    
    # Save results
    import json
    results = {
        'disagreement_analysis': {
            'statistics': disagreement_results.get('statistics', {}),
            'n_samples': len(disagreement_results.get('disagreements', []))
        },
        'sector_performance': sector_performance,
        'mixing_network_analysis': mixing_analysis.get('weight_statistics', {})
    }
    
    results_file = RESULTS_DIR / 'multi_agent_analysis.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to: {results_file}")
    print("\n🎉 Multi-Agent Analysis Complete!")


if __name__ == "__main__":
    main()


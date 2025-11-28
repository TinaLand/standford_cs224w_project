# enhancement_failure_analysis.py
"""
Failure Analysis for Deep Analysis Enhancement
Implements worst period analysis, error pattern identification, and drawdown analysis.
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

from phase5_rl_integration import load_gnn_model_for_rl
from rl_environment import StockTradingEnv
from rl_agent import StockTradingAgent

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = MODELS_DIR / "plots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def analyze_worst_periods(
    agent: StockTradingAgent,
    env: StockTradingEnv,
    n_episodes: int = 20
) -> Dict[str, Any]:
    """
    Identify worst-performing periods and analyze what went wrong.
    
    Returns:
        Dictionary with worst period analysis
    """
    print("\n" + "="*60)
    print("🔍 Analyzing Worst-Performing Periods")
    print("="*60)
    
    episode_performances = []
    
    for episode in range(n_episodes):
        try:
            obs, info = env.reset()
            done = False
            portfolio_values = [env.portfolio_value]
            actions_history = []
            rewards_history = []
            dates_history = []
            
            step = 0
            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                portfolio_values.append(env.portfolio_value)
                actions_history.append(action)
                rewards_history.append(reward)
                dates_history.append(env.data_loader['dates'][env.current_step] if env.current_step < len(env.data_loader['dates']) else None)
                
                step += 1
                if step > 1000:  # Safety limit
                    break
            
            # Calculate episode metrics
            if len(portfolio_values) > 1:
                returns = np.array(portfolio_values)
                total_return = (returns[-1] / returns[0]) - 1
                
                daily_returns = np.diff(returns) / returns[:-1]
                sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
                
                cumulative = np.maximum.accumulate(returns)
                drawdown = (cumulative - returns) / cumulative
                max_dd = np.max(drawdown)
                
                episode_performances.append({
                    'episode': episode,
                    'return': total_return,
                    'sharpe': sharpe,
                    'max_dd': max_dd,
                    'portfolio_values': portfolio_values,
                    'actions_history': actions_history,
                    'rewards_history': rewards_history,
                    'dates_history': dates_history
                })
        
        except Exception as e:
            if episode < 3:
                print(f"  ⚠️  Error in episode {episode}: {e}")
            continue
    
    # Identify worst periods
    if episode_performances:
        # Sort by return (worst first)
        sorted_by_return = sorted(episode_performances, key=lambda x: x['return'])
        worst_episodes = sorted_by_return[:5]  # Top 5 worst
        
        # Sort by max drawdown (worst first)
        sorted_by_dd = sorted(episode_performances, key=lambda x: x['max_dd'], reverse=True)
        worst_dd_episodes = sorted_by_dd[:5]  # Top 5 worst drawdowns
        
        print(f"\n📊 Worst Period Analysis:")
        print(f"   Total Episodes Analyzed: {len(episode_performances)}")
        print(f"\n   Worst Returns:")
        for i, ep in enumerate(worst_episodes[:3]):
            print(f"     {i+1}. Episode {ep['episode']}: Return={ep['return']*100:.2f}%, Sharpe={ep['sharpe']:.2f}, Max DD={ep['max_dd']*100:.2f}%")
        
        print(f"\n   Worst Drawdowns:")
        for i, ep in enumerate(worst_dd_episodes[:3]):
            print(f"     {i+1}. Episode {ep['episode']}: Max DD={ep['max_dd']*100:.2f}%, Return={ep['return']*100:.2f}%")
        
        return {
            'episode_performances': episode_performances,
            'worst_episodes_by_return': worst_episodes,
            'worst_episodes_by_dd': worst_dd_episodes,
            'statistics': {
                'mean_return': np.mean([ep['return'] for ep in episode_performances]),
                'std_return': np.std([ep['return'] for ep in episode_performances]),
                'worst_return': worst_episodes[0]['return'] if worst_episodes else 0,
                'mean_max_dd': np.mean([ep['max_dd'] for ep in episode_performances]),
                'worst_max_dd': worst_dd_episodes[0]['max_dd'] if worst_dd_episodes else 0
            }
        }
    
    return {}


def analyze_error_patterns(
    gnn_model,
    test_dates: List[pd.Timestamp],
    targets_dict: Dict[pd.Timestamp, torch.Tensor],
    tickers: List[str]
) -> Dict[str, Any]:
    """
    Identify systematic mistakes in GNN predictions.
    
    Returns:
        Dictionary with error pattern analysis
    """
    print("\n" + "="*60)
    print("🔍 Analyzing Error Patterns")
    print("="*60)
    
    from phase4_core_training import load_graph_data
    
    all_predictions = []
    all_targets = []
    false_positives = []
    false_negatives = []
    sector_errors = defaultdict(lambda: {'fp': 0, 'fn': 0, 'total': 0})
    
    # Load sector mapping
    sector_file = PROJECT_ROOT / "data" / "raw" / "static_sector_industry.csv"
    ticker_to_sector = {}
    if sector_file.exists():
        sector_df = pd.read_csv(sector_file)
        for _, row in sector_df.iterrows():
            ticker_to_sector[row['Ticker']] = row.get('Sector', 'Unknown')
    
    gnn_model.eval()
    with torch.no_grad():
        for date in test_dates[:100]:  # Sample 100 dates
            try:
                data = load_graph_data(date, tickers)
                if data is None:
                    continue
                
                data = data.to(DEVICE)
                target = targets_dict.get(date)
                if target is None:
                    continue
                
                # Get predictions
                logits = gnn_model(data)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                # Analyze errors
                for i, (pred, true_label) in enumerate(zip(preds.cpu().numpy(), target.cpu().numpy())):
                    ticker = tickers[i] if i < len(tickers) else f"Stock_{i}"
                    sector = ticker_to_sector.get(ticker, 'Unknown')
                    
                    all_predictions.append(pred)
                    all_targets.append(true_label)
                    sector_errors[sector]['total'] += 1
                    
                    if pred == 1 and true_label == 0:
                        false_positives.append({'ticker': ticker, 'sector': sector, 'date': date})
                        sector_errors[sector]['fp'] += 1
                    elif pred == 0 and true_label == 1:
                        false_negatives.append({'ticker': ticker, 'sector': sector, 'date': date})
                        sector_errors[sector]['fn'] += 1
            
            except Exception as e:
                continue
    
    # Calculate error statistics
    if all_predictions:
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        accuracy = np.mean(all_predictions == all_targets)
        fp_rate = len(false_positives) / len(all_predictions)
        fn_rate = len(false_negatives) / len(all_predictions)
        
        print(f"\n📊 Error Pattern Statistics:")
        print(f"   Overall Accuracy: {accuracy*100:.2f}%")
        print(f"   False Positive Rate: {fp_rate*100:.2f}%")
        print(f"   False Negative Rate: {fn_rate*100:.2f}%")
        print(f"   Total False Positives: {len(false_positives)}")
        print(f"   Total False Negatives: {len(false_negatives)}")
        
        print(f"\n   Sector-Specific Errors:")
        for sector, errors in sector_errors.items():
            if errors['total'] > 0:
                fp_pct = errors['fp'] / errors['total'] * 100
                fn_pct = errors['fn'] / errors['total'] * 100
                print(f"     {sector}: FP={fp_pct:.1f}%, FN={fn_pct:.1f}% (n={errors['total']})")
        
        return {
            'overall_accuracy': accuracy,
            'false_positive_rate': fp_rate,
            'false_negative_rate': fn_rate,
            'false_positives': false_positives[:50],  # Sample
            'false_negatives': false_negatives[:50],  # Sample
            'sector_errors': dict(sector_errors),
            'error_statistics': {
                'total_predictions': len(all_predictions),
                'total_errors': len(false_positives) + len(false_negatives),
                'error_rate': (len(false_positives) + len(false_negatives)) / len(all_predictions)
            }
        }
    
    return {}


def analyze_drawdown_periods(
    portfolio_values: List[float],
    dates: List[pd.Timestamp]
) -> Dict[str, Any]:
    """
    Analyze what happened during maximum drawdown periods.
    
    Returns:
        Dictionary with drawdown period analysis
    """
    print("\n" + "="*60)
    print("🔍 Analyzing Drawdown Periods")
    print("="*60)
    
    if len(portfolio_values) < 2:
        return {}
    
    returns = np.array(portfolio_values)
    cumulative = np.maximum.accumulate(returns)
    drawdown = (cumulative - returns) / cumulative
    
    # Find drawdown periods
    drawdown_periods = []
    in_drawdown = False
    drawdown_start = None
    drawdown_start_idx = None
    
    for i, dd in enumerate(drawdown):
        if dd > 0.05:  # Drawdown > 5%
            if not in_drawdown:
                in_drawdown = True
                drawdown_start = dates[i] if i < len(dates) else None
                drawdown_start_idx = i
        else:
            if in_drawdown:
                in_drawdown = False
                drawdown_periods.append({
                    'start': drawdown_start,
                    'end': dates[i-1] if i > 0 and i-1 < len(dates) else None,
                    'start_idx': drawdown_start_idx,
                    'end_idx': i-1,
                    'max_dd': np.max(drawdown[drawdown_start_idx:i]),
                    'duration': i - drawdown_start_idx
                })
    
    # Find maximum drawdown period
    if drawdown_periods:
        max_dd_period = max(drawdown_periods, key=lambda x: x['max_dd'])
        
        print(f"\n📊 Drawdown Analysis:")
        print(f"   Total Drawdown Periods (>5%): {len(drawdown_periods)}")
        print(f"   Maximum Drawdown: {max_dd_period['max_dd']*100:.2f}%")
        print(f"   Max DD Period: {max_dd_period['start']} to {max_dd_period['end']}")
        print(f"   Max DD Duration: {max_dd_period['duration']} days")
        
        return {
            'drawdown_periods': drawdown_periods,
            'max_drawdown_period': max_dd_period,
            'statistics': {
                'total_drawdown_periods': len(drawdown_periods),
                'max_drawdown': max_dd_period['max_dd'],
                'avg_drawdown_duration': np.mean([p['duration'] for p in drawdown_periods]) if drawdown_periods else 0
            }
        }
    
    return {}


def visualize_failure_analysis(
    worst_periods: Dict,
    error_patterns: Dict,
    drawdown_analysis: Dict,
    output_dir: Path
):
    """Create visualizations for failure analysis."""
    print("\n📊 Creating Failure Analysis Visualizations...")
    
    # 1. Worst Periods Portfolio Values
    if worst_periods.get('worst_episodes_by_return'):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for idx, ep in enumerate(worst_periods['worst_episodes_by_return'][:4]):
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            
            portfolio_values = ep['portfolio_values']
            ax.plot(portfolio_values, alpha=0.7)
            ax.set_title(f"Episode {ep['episode']}: Return={ep['return']*100:.2f}%")
            ax.set_xlabel('Step')
            ax.set_ylabel('Portfolio Value')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'worst_periods_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ Saved: worst_periods_analysis.png")
    
    # 2. Error Pattern by Sector
    if error_patterns.get('sector_errors'):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sectors = list(error_patterns['sector_errors'].keys())
        fp_rates = [error_patterns['sector_errors'][s]['fp'] / max(error_patterns['sector_errors'][s]['total'], 1) * 100 
                   for s in sectors]
        fn_rates = [error_patterns['sector_errors'][s]['fn'] / max(error_patterns['sector_errors'][s]['total'], 1) * 100 
                   for s in sectors]
        
        x = np.arange(len(sectors))
        width = 0.35
        
        ax.bar(x - width/2, fp_rates, width, label='False Positive Rate', alpha=0.7)
        ax.bar(x + width/2, fn_rates, width, label='False Negative Rate', alpha=0.7)
        ax.set_xlabel('Sector')
        ax.set_ylabel('Error Rate (%)')
        ax.set_title('Error Patterns by Sector')
        ax.set_xticks(x)
        ax.set_xticklabels(sectors, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'error_patterns_by_sector.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ Saved: error_patterns_by_sector.png")
    
    # 3. Drawdown Periods Timeline
    if drawdown_analysis.get('drawdown_periods'):
        fig, ax = plt.subplots(figsize=(15, 6))
        
        # Plot drawdown periods
        for period in drawdown_analysis['drawdown_periods']:
            if period['start'] and period['end']:
                ax.axvspan(period['start'], period['end'], alpha=0.3, color='red', 
                          label='Drawdown Period' if period == drawdown_analysis['drawdown_periods'][0] else '')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown Period')
        ax.set_title('Drawdown Periods Timeline')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'drawdown_periods_timeline.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ Saved: drawdown_periods_timeline.png")


def main():
    """Main failure analysis pipeline."""
    print("🚀 Failure Analysis")
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
    
    # Load RL agent
    print("\n🤖 Loading RL agent...")
    try:
        agent_path = MODELS_DIR / "rl_ppo_agent_model_final" / "ppo_stock_agent_final.zip"
        if agent_path.exists():
            from stable_baselines3 import PPO
            from stable_baselines3.common.env_util import make_vec_env
            from rl_environment import StockTradingEnv
            
            start_date = pd.to_datetime('2023-01-01')
            end_date = pd.to_datetime('2024-12-31')
            
            def env_factory():
                return StockTradingEnv(start_date, end_date, gnn_model, DEVICE)
            
            vec_env = make_vec_env(env_factory, n_envs=1)
            agent = PPO.load(agent_path, env=vec_env, device="cpu")
            
            # Wrap in StockTradingAgent
            from rl_agent import StockTradingAgent
            trading_agent = StockTradingAgent(gnn_model, env_factory, DEVICE)
            trading_agent.agent = agent
            trading_agent.is_trained = True
            
            print("✅ RL agent loaded")
        else:
            print("⚠️  RL agent not found, skipping agent-based analysis")
            trading_agent = None
    except Exception as e:
        print(f"⚠️  Error loading RL agent: {e}")
        trading_agent = None
    
    # Create environment
    print("\n🌍 Creating environment...")
    start_date = pd.to_datetime('2023-01-01')
    end_date = pd.to_datetime('2024-12-31')
    env = StockTradingEnv(start_date, end_date, gnn_model, DEVICE)
    
    # Run analyses
    print("\n" + "="*60)
    print("Running Failure Analyses...")
    print("="*60)
    
    # 1. Worst Periods Analysis (if agent available)
    worst_periods = {}
    if trading_agent:
        worst_periods = analyze_worst_periods(trading_agent, env, n_episodes=10)
    
    # 2. Error Pattern Analysis
    from phase4_core_training import load_targets, get_train_val_test_dates
    from utils_data import load_data_file
    
    tickers = load_data_file('node_features_X_t_final.csv', 'processed').columns.tolist()[:50]  # Sample
    targets_dict = load_targets()
    train_dates, val_dates, test_dates = get_train_val_test_dates()
    
    error_patterns = analyze_error_patterns(gnn_model, test_dates, targets_dict, tickers)
    
    # 3. Drawdown Analysis (if worst periods available)
    drawdown_analysis = {}
    if worst_periods.get('worst_episodes_by_dd'):
        # Use worst drawdown episode
        worst_dd_ep = worst_periods['worst_episodes_by_dd'][0]
        dates = worst_dd_ep.get('dates_history', [])
        if dates and worst_dd_ep.get('portfolio_values'):
            drawdown_analysis = analyze_drawdown_periods(
                worst_dd_ep['portfolio_values'],
                dates
            )
    
    # Visualize
    visualize_failure_analysis(worst_periods, error_patterns, drawdown_analysis, PLOTS_DIR)
    
    # Save results
    import json
    results = {
        'worst_periods': {
            'statistics': worst_periods.get('statistics', {}),
            'n_episodes': len(worst_periods.get('episode_performances', []))
        },
        'error_patterns': {
            'overall_accuracy': error_patterns.get('overall_accuracy', 0),
            'error_statistics': error_patterns.get('error_statistics', {})
        },
        'drawdown_analysis': {
            'statistics': drawdown_analysis.get('statistics', {})
        }
    }
    
    results_file = RESULTS_DIR / 'failure_analysis.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to: {results_file}")
    print("\n🎉 Failure Analysis Complete!")


if __name__ == "__main__":
    main()


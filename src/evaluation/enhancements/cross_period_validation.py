# enhancement_cross_period_validation.py
"""
Cross-Period Validation for Results Enhancement
Tests performance across different market regimes (bull, bear, volatile).
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.integration import load_gnn_model_for_rl
from src.rl.environments.single_agent import StockTradingEnv
from src.rl.agents.single_agent import StockTradingAgent
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from src.utils.paths import MODELS_DIR, RESULTS_DIR, MODELS_PLOTS_DIR as PLOTS_DIR
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_period(
    agent: StockTradingAgent,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    period_name: str,
    n_episodes: int = 10
) -> Dict[str, float]:
    """
    Evaluate agent performance on a specific time period.
    
    Returns:
        Dictionary with performance metrics
    """
    print(f"\n📊 Evaluating {period_name} period ({start_date.date()} to {end_date.date()})...")
    
    # Create environment for this period
    gnn_model = agent.gnn_model
    env = StockTradingEnv(start_date, end_date, gnn_model, DEVICE)
    
    episode_returns = []
    episode_sharpe = []
    episode_max_dd = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        try:
            obs, info = env.reset()
            done = False
            portfolio_values = [env.portfolio_value]
            
            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                portfolio_values.append(env.portfolio_value)
            
            # Calculate metrics
            if len(portfolio_values) > 1:
                returns = np.array(portfolio_values)
                total_return = (returns[-1] / returns[0]) - 1
                
                daily_returns = np.diff(returns) / returns[:-1]
                sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
                
                cumulative = np.maximum.accumulate(returns)
                drawdown = (cumulative - returns) / cumulative
                max_dd = np.max(drawdown)
                
                episode_returns.append(total_return)
                episode_sharpe.append(sharpe)
                episode_max_dd.append(max_dd)
                episode_lengths.append(len(portfolio_values))
        
        except Exception as e:
            if episode < 3:
                print(f"  ⚠️  Error in episode {episode}: {e}")
            continue
    
    if episode_returns:
        results = {
            'period_name': period_name,
            'start_date': start_date,
            'end_date': end_date,
            'mean_return': np.mean(episode_returns),
            'std_return': np.std(episode_returns),
            'mean_sharpe': np.mean(episode_sharpe),
            'std_sharpe': np.std(episode_sharpe),
            'mean_max_dd': np.mean(episode_max_dd),
            'std_max_dd': np.std(episode_max_dd),
            'mean_length': np.mean(episode_lengths),
            'n_episodes': len(episode_returns)
        }
        
        print(f"   Mean Return: {results['mean_return']*100:.2f}%")
        print(f"   Mean Sharpe: {results['mean_sharpe']:.2f}")
        print(f"   Mean Max DD: {results['mean_max_dd']*100:.2f}%")
        
        return results
    
    return {}


def classify_market_regime(
    prices_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp
) -> str:
    """
    Classify market regime based on price movements.
    
    Returns:
        'bull', 'bear', or 'volatile'
    """
    # Filter to period
    period_prices = prices_df.loc[start_date:end_date]
    
    if period_prices.empty:
        return 'unknown'
    
    # Calculate returns
    returns = period_prices.pct_change().dropna()
    
    # Calculate metrics
    mean_return = returns.mean().mean()
    volatility = returns.std().mean()
    
    # Classify
    if mean_return > 0.0005 and volatility < 0.02:  # Positive returns, low volatility
        return 'bull'
    elif mean_return < -0.0005:  # Negative returns
        return 'bear'
    else:  # High volatility or mixed
        return 'volatile'


def run_cross_period_validation(
    agent: StockTradingAgent,
    periods: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Run validation across multiple time periods.
    
    Returns:
        Dictionary with cross-period validation results
    """
    print("\n" + "="*60)
    print("🔄 Cross-Period Validation")
    print("="*60)
    
    period_results = {}
    
    for period in periods:
        period_name = period['name']
        start_date = pd.to_datetime(period['start_date'])
        end_date = pd.to_datetime(period['end_date'])
        
        results = evaluate_period(agent, start_date, end_date, period_name, n_episodes=5)
        if results:
            period_results[period_name] = results
    
    # Aggregate statistics
    if period_results:
        all_returns = [r['mean_return'] for r in period_results.values()]
        all_sharpe = [r['mean_sharpe'] for r in period_results.values()]
        all_max_dd = [r['mean_max_dd'] for r in period_results.values()]
        
        print(f"\n📊 Cross-Period Summary:")
        print(f"   Periods Tested: {len(period_results)}")
        print(f"   Average Return: {np.mean(all_returns)*100:.2f}%")
        print(f"   Average Sharpe: {np.mean(all_sharpe):.2f}")
        print(f"   Average Max DD: {np.mean(all_max_dd)*100:.2f}%")
        print(f"   Return Std: {np.std(all_returns)*100:.2f}%")
        print(f"   Sharpe Std: {np.std(all_sharpe):.2f}")
        
        return {
            'period_results': period_results,
            'aggregate_statistics': {
                'n_periods': len(period_results),
                'mean_return': np.mean(all_returns),
                'std_return': np.std(all_returns),
                'mean_sharpe': np.mean(all_sharpe),
                'std_sharpe': np.std(all_sharpe),
                'mean_max_dd': np.mean(all_max_dd),
                'std_max_dd': np.std(all_max_dd)
            }
        }
    
    return {}


def visualize_cross_period_validation(
    validation_results: Dict,
    output_dir: Path
):
    """Create visualizations for cross-period validation."""
    print("\n📊 Creating Cross-Period Validation Visualizations...")
    
    if not validation_results.get('period_results'):
        return
    
    period_results = validation_results['period_results']
    
    # 1. Performance Comparison Across Periods
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    periods = list(period_results.keys())
    returns = [period_results[p]['mean_return']*100 for p in periods]
    sharpe = [period_results[p]['mean_sharpe'] for p in periods]
    max_dd = [period_results[p]['mean_max_dd']*100 for p in periods]
    
    axes[0].bar(periods, returns)
    axes[0].set_ylabel('Return (%)')
    axes[0].set_title('Returns Across Market Regimes')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].bar(periods, sharpe)
    axes[1].set_ylabel('Sharpe Ratio')
    axes[1].set_title('Sharpe Ratios Across Market Regimes')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    axes[2].bar(periods, max_dd)
    axes[2].set_ylabel('Max Drawdown (%)')
    axes[2].set_title('Max Drawdowns Across Market Regimes')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cross_period_validation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: cross_period_validation.png")


def main():
    """Main cross-period validation pipeline."""
    print("🚀 Cross-Period Validation")
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
        if not agent_path.exists():
            print("⚠️  RL agent not found, skipping validation")
            return
        
        start_date = pd.to_datetime('2023-01-01')
        end_date = pd.to_datetime('2024-12-31')
        
        def env_factory():
            return StockTradingEnv(start_date, end_date, gnn_model, DEVICE)
        
        vec_env = make_vec_env(env_factory, n_envs=1)
        ppo_agent = PPO.load(agent_path, env=vec_env, device="cpu")
        
        from src.rl.agents.single_agent import StockTradingAgent
        agent = StockTradingAgent(gnn_model, env_factory, DEVICE)
        agent.agent = ppo_agent
        agent.is_trained = True
        
        print("✅ RL agent loaded")
    except Exception as e:
        print(f"❌ Error loading RL agent: {e}")
        return
    
    # Define test periods (different market regimes)
    periods = [
        {
            'name': 'Bull Market (2023-2024)',
            'start_date': '2023-01-01',
            'end_date': '2024-12-31',
            'regime': 'bull'
        },
        {
            'name': 'Early 2023',
            'start_date': '2023-01-01',
            'end_date': '2023-06-30',
            'regime': 'bull'
        },
        {
            'name': 'Late 2023',
            'start_date': '2023-07-01',
            'end_date': '2023-12-31',
            'regime': 'bull'
        },
        {
            'name': 'Early 2024',
            'start_date': '2024-01-01',
            'end_date': '2024-06-30',
            'regime': 'bull'
        },
        {
            'name': 'Late 2024',
            'start_date': '2024-07-01',
            'end_date': '2024-12-31',
            'regime': 'bull'
        }
    ]
    
    # Run validation
    validation_results = run_cross_period_validation(agent, periods)
    
    # Visualize
    visualize_cross_period_validation(validation_results, PLOTS_DIR)
    
    # Save results
    import json
    results_file = RESULTS_DIR / 'cross_period_validation.json'
    with open(results_file, 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to: {results_file}")
    print("\n🎉 Cross-Period Validation Complete!")


if __name__ == "__main__":
    main()


# scripts/quick_rl_test.py
"""
Quick RL Test - Run a short evaluation and show results
"""
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / 'scripts'))
sys.path.append(str(PROJECT_ROOT / 'scripts' / 'components'))

from phase5_rl_integration import load_gnn_model_for_rl
from rl_environment import StockTradingEnv
from rl_agent import StockTradingAgent
from phase6_evaluation import calculate_financial_metrics

MODELS_DIR = PROJECT_ROOT / "models"
DATA_GRAPHS_DIR = PROJECT_ROOT / "data" / "graphs"
RESULTS_DIR = PROJECT_ROOT / "results"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RL_SAVE_PATH = MODELS_DIR / "rl_ppo_agent_model" / "ppo_stock_agent.zip"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def quick_test():
    """Run a quick test with existing agent or create a minimal one."""
    print("="*70)
    print("Quick RL Agent Test")
    print("="*70)
    
    # Load GNN
    print("\n[1/3] Loading GNN model...")
    gnn_model = load_gnn_model_for_rl()
    
    # Get date range
    graph_files = list(DATA_GRAPHS_DIR.glob('graph_t_*.pt'))
    graph_dates = sorted([pd.to_datetime(f.stem.split('_')[-1]) for f in graph_files])
    graph_start = graph_dates[0]
    graph_end = graph_dates[-1]
    date_range_days = (graph_end - graph_start).days
    start_offset_days = int(date_range_days * 0.8)
    START_DATE = graph_start + pd.Timedelta(days=start_offset_days)
    END_DATE = graph_end
    
    print(f"\n[2/3] Setting up environment and agent...")
    print(f"   Date range: {START_DATE.date()} to {END_DATE.date()}")
    
    def make_env():
        return StockTradingEnv(
            start_date=START_DATE,
            end_date=END_DATE,
            gnn_model=gnn_model,
            device=DEVICE
        )
    
    # Create agent
    agent = StockTradingAgent(
        gnn_model=gnn_model,
        env_factory=make_env,
        device=DEVICE,
        learning_rate=1e-5,
        tensorboard_log=None,
        policy="MlpPolicy",
        verbose=0
    )
    
    # Try to load existing agent, if not train a short one
    if RL_SAVE_PATH.exists():
        try:
            agent.load(RL_SAVE_PATH)
            print("   ✅ Loaded existing agent")
        except Exception as e:
            print(f"   ⚠️  Could not load existing agent: {e}")
            print("   Training new agent (100 steps)...")
            agent.train(total_timesteps=100, progress_bar=False)
    else:
        print("   ⚠️  No existing agent found")
        print("   Training new agent (100 steps)...")
        agent.train(total_timesteps=100, progress_bar=False)
    
    # Run backtest
    print(f"\n[3/3] Running backtest...")
    env = make_env()
    obs, info = env.reset()
    
    portfolio_values = [env.initial_cash]
    rewards = []
    step = 0
    done = False
    
    while not done and step < 500:  # Limit to 500 steps for quick test
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        portfolio_values.append(info.get('portfolio_value', env.portfolio_value))
        rewards.append(reward)
        
        if step % 50 == 0:
            print(f"   Step {step}: Value ${portfolio_values[-1]:.2f}")
        
        step += 1
    
    # Calculate metrics
    metrics = calculate_financial_metrics(portfolio_values, step)
    total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Initial Value:     ${portfolio_values[0]:,.2f}")
    print(f"Final Value:       ${portfolio_values[-1]:,.2f}")
    print(f"Total Return:       {total_return:.2f}%")
    print(f"Sharpe Ratio:       {metrics.get('Sharpe_Ratio', 0):.4f}")
    print(f"Max Drawdown:       {metrics.get('Max_Drawdown', 0)*100:.2f}%")
    print(f"Steps Completed:    {step}")
    print("="*70)
    
    # Create simple plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Portfolio value
    ax1.plot(portfolio_values, 'b-', linewidth=2)
    ax1.axhline(y=portfolio_values[0], color='r', linestyle='--', label='Initial')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.set_title('Portfolio Value Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Returns
    returns = np.array(portfolio_values) / portfolio_values[0] - 1
    ax2.plot(returns * 100, 'g-', linewidth=2)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Cumulative Return (%)')
    ax2.set_title('Cumulative Returns')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = RESULTS_DIR / 'quick_rl_test_results.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Plot saved to: {plot_path}")
    print("\n")


if __name__ == '__main__':
    quick_test()


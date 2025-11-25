# scripts/test_dynamic_viz.py
"""
Test version of dynamic visualization - runs without GUI for testing
"""
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / 'scripts'))
sys.path.append(str(PROJECT_ROOT / 'scripts' / 'components'))

from phase5_rl_integration import load_gnn_model_for_rl
from rl_environment import StockTradingEnv
from rl_agent import StockTradingAgent

MODELS_DIR = PROJECT_ROOT / "models"
DATA_GRAPHS_DIR = PROJECT_ROOT / "data" / "graphs"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RL_SAVE_PATH = MODELS_DIR / "rl_ppo_agent_model" / "ppo_stock_agent.zip"

def test_dynamic_viz():
    """Test the dynamic visualization setup without GUI."""
    print("="*70)
    print("Testing Dynamic Visualization Setup")
    print("="*70)
    
    # Step 1: Load GNN
    print("\n[1/4] Loading GNN model...")
    try:
        gnn_model = load_gnn_model_for_rl()
        print("✅ GNN model loaded")
    except Exception as e:
        print(f"❌ Failed to load GNN: {e}")
        return
    
    # Step 2: Get date range
    print("\n[2/4] Setting up date range...")
    graph_files = list(DATA_GRAPHS_DIR.glob('graph_t_*.pt'))
    if not graph_files:
        print("❌ No graph files found")
        return
    
    graph_dates = sorted([pd.to_datetime(f.stem.split('_')[-1]) for f in graph_files])
    graph_start = graph_dates[0]
    graph_end = graph_dates[-1]
    date_range_days = (graph_end - graph_start).days
    start_offset_days = int(date_range_days * 0.8)
    START_DATE = graph_start + pd.Timedelta(days=start_offset_days)
    END_DATE = graph_end
    
    print(f"✅ Date range: {START_DATE.date()} to {END_DATE.date()}")
    
    # Step 3: Create environment
    print("\n[3/4] Creating environment...")
    try:
        def make_env():
            return StockTradingEnv(
                start_date=START_DATE,
                end_date=END_DATE,
                gnn_model=gnn_model,
                device=DEVICE
            )
        env = make_env()
        print("✅ Environment created")
    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Create and test agent
    print("\n[4/4] Setting up agent...")
    try:
        agent = StockTradingAgent(
            gnn_model=gnn_model,
            env_factory=make_env,
            device=DEVICE,
            learning_rate=1e-5,
            tensorboard_log=None,
            policy="MlpPolicy",
            verbose=0
        )
        
        if RL_SAVE_PATH.exists():
            try:
                agent.load(RL_SAVE_PATH)
                print("✅ Agent loaded from file")
            except:
                print("⚠️  Could not load agent, training new one (50 steps)...")
                agent.train(total_timesteps=50, progress_bar=False)
        else:
            print("⚠️  No agent found, training new one (50 steps)...")
            agent.train(total_timesteps=50, progress_bar=False)
        
        # Test a few steps
        print("\n🧪 Testing agent with 10 steps...")
        obs, info = env.reset()
        portfolio_values = [env.initial_cash]
        
        for step in range(10):
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            portfolio_values.append(info.get('portfolio_value', env.portfolio_value))
            print(f"  Step {step}: Value ${portfolio_values[-1]:.2f}, Reward: {reward:.6f}")
        
        print(f"\n✅ Test successful!")
        print(f"   Initial: ${portfolio_values[0]:.2f}")
        print(f"   Final:   ${portfolio_values[-1]:.2f}")
        print(f"   Return:  {(portfolio_values[-1]/portfolio_values[0]-1)*100:.2f}%")
        
        print("\n" + "="*70)
        print("✅ All components working! Ready for dynamic visualization.")
        print("="*70)
        print("\nTo run with GUI:")
        print("  python scripts/dynamic_rl_visualization.py")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_dynamic_viz()


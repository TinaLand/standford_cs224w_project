# scripts/phase5_rl_integration.py

import torch
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import sys

# Try to import tensorboard, make it optional
try:
    import tensorboard
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

# Fix PyTorch serialization for pandas timestamps (PyTorch 2.6+)
import torch.serialization
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([pd._libs.tslibs.timestamps._unpickle_timestamp])

# Add necessary paths for local imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / 'scripts')) # To import rl_environment
sys.path.append(str(PROJECT_ROOT / 'scripts' / 'components')) # For GNN model components

# --- Configuration (Shared) ---
MODELS_DIR = PROJECT_ROOT / "models"
DATA_GRAPHS_DIR = PROJECT_ROOT / "data" / "graphs"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# File Paths
CORE_GNN_MODEL_PATH = MODELS_DIR / 'core_transformer_model.pt'
RL_LOG_PATH = PROJECT_ROOT / "logs" / "rl_logs"
RL_SAVE_PATH = MODELS_DIR / "rl_ppo_agent_model"

# RL Hyperparameters
TOTAL_TIMESTEPS = 10000  # Total steps for the agent to learn (High value for finance)
PPO_LEARNING_RATE = 1e-5
NUM_ENVS = 1              # Number of parallel environments

# Ensure output directories exist
RL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
RL_LOG_PATH.mkdir(parents=True, exist_ok=True)

# NOTE: The GNN model definition and the environment definition 
# must be imported from their respective files.
from phase4_core_training import RoleAwareGraphTransformer 
from rl_environment import StockTradingEnv 


def load_gnn_model_for_rl():
    """
    Loads the trained Core GNN model, freezes its weights, and prepares it 
    for generating embeddings (features) for the RL agent's observation space.
    """
    print("\n--- 1. Loading Trained Core GNN Model ---")
    
    # 1. Determine Input Dimension: Load a sample graph to get feature size
    sample_graph_path = list(DATA_GRAPHS_DIR.glob('graph_t_*.pt'))[0]
    temp_data = torch.load(sample_graph_path, weights_only=False)
    INPUT_DIM = temp_data['stock'].x.shape[1]

    # 2. Initialize GNN model structure (Must match Phase 4)
    gnn_model = RoleAwareGraphTransformer(
        INPUT_DIM, 256, 2, 2, 4 # Use same dimensions as Phase 4 training
    ).to(DEVICE)
    
    # 3. Load trained weights
    if not CORE_GNN_MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained GNN model not found at: {CORE_GNN_MODEL_PATH}")
        
    gnn_model.load_state_dict(torch.load(CORE_GNN_MODEL_PATH, map_location=DEVICE, weights_only=False))
    
    # 4. Freeze GNN parameters
    for param in gnn_model.parameters():
        param.requires_grad = False
        
    print(f"✅ Core GNN Model loaded and frozen (Input Dim: {INPUT_DIM}).")
    return gnn_model


def run_rl_pipeline():
    """Runs the RL training pipeline using PPO."""
    
    # 1. Load GNN Model
    try:
        gnn_model = load_gnn_model_for_rl()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}. Ensure phase4_core_training.py ran successfully.")
        return

    # 2. Define Backtesting Period (Use dynamic dates corresponding to test/evaluation split)
    # Get actual date range from available graph files (not just price data)
    import pandas as pd
    
    # Get graph file dates (these are what we actually have)
    graph_files = list(DATA_GRAPHS_DIR.glob('graph_t_*.pt'))
    if not graph_files:
        raise ValueError("No graph files found. Run Phase 2 first.")
    
    graph_dates = sorted([pd.to_datetime(f.stem.split('_')[-1]) for f in graph_files])
    graph_start = graph_dates[0]
    graph_end = graph_dates[-1]
    
    # Use last 20% of graph data for backtesting (test set)
    date_range_days = (graph_end - graph_start).days
    start_offset_days = int(date_range_days * 0.8)
    START_DATE = graph_start + pd.Timedelta(days=start_offset_days)
    END_DATE = graph_end
    
    print(f"📅 Backtesting period: {START_DATE.date()} to {END_DATE.date()}")
    print(f"   (Graph files available: {graph_start.date()} to {graph_end.date()})")
    print(f"   (Using last {int((date_range_days - start_offset_days) / date_range_days * 100)}% of graph data)")
    
    # 3. Setup Environment
    # stable-baselines3 requires a function to create the environment
    def make_env():
        return StockTradingEnv(
            start_date=START_DATE, 
            end_date=END_DATE, 
            gnn_model=gnn_model, 
            device=DEVICE
        )

    # SB3 requires vectorized environments
    vec_env = make_vec_env(make_env, n_envs=NUM_ENVS)
    
    # 4. RL Agent Setup (PPO)
    print(f"\n--- 2. Setting up PPO Agent (Total Timesteps: {TOTAL_TIMESTEPS}) ---")
    
    # Configure tensorboard logging (optional)
    if TENSORBOARD_AVAILABLE:
        print("✅ TensorBoard logging enabled")
        tensorboard_log_path = RL_LOG_PATH
    else:
        print("⚠️  TensorBoard not available, logging disabled")
        print("   Install with: pip install tensorboard")
        tensorboard_log_path = None
    
    # We use MlpPolicy, which is standard for Gym's discrete action space
    # PPO with MlpPolicy performs better on CPU according to stable-baselines3
    ppo_kwargs = {
        "policy": "MlpPolicy",
        "env": vec_env,
        "verbose": 1,
        "learning_rate": PPO_LEARNING_RATE,
        "device": "cpu",  # Use CPU for better MLP performance
    }
    
    # Only add tensorboard_log if tensorboard is available
    if tensorboard_log_path:
        ppo_kwargs["tensorboard_log"] = tensorboard_log_path
    
    model = PPO(**ppo_kwargs)
    
    # 5. Training
    print("\n🔨 Starting PPO RL Training (GNN-driven features)...")
    
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
    except Exception as e:
        print(f"❌ RL Training failed: {e}")
        return
    
    # 6. Save Agent
    model.save(RL_SAVE_PATH / "ppo_stock_agent")
    print(f"\n✅ RL Agent trained and saved to: {RL_SAVE_PATH / 'ppo_stock_agent.zip'}")

    print("\n🎯 Phase 5: RL Integration and Training Complete! Ready for Phase 6: Evaluation.")

if __name__ == '__main__':
    run_rl_pipeline()
# scripts/phase5_rl_integration.py

import torch
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
import sys

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
# Use enhanced environment with risk-adjusted rewards
try:
    from rl_environment_enhanced import StockTradingEnvEnhanced as StockTradingEnv
except ImportError:
    from rl_environment import StockTradingEnv
    print("⚠️  Using basic RL environment. Install enhanced version for risk-adjusted rewards.") 


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
    # Placeholder: Assuming the test set starts after a certain date (e.g., 85% split)
    # In a full project, these dates should be read from the Phase 4 training logs.
    START_DATE = pd.to_datetime('2022-01-01')
    END_DATE = pd.to_datetime('2024-12-31')
    
    # 3. Setup Environment
    # stable-baselines3 requires a function to create the environment
    def make_env():
        # Use enhanced environment with risk-adjusted rewards and constraints
        try:
            from rl_environment_enhanced import StockTradingEnvEnhanced
            return StockTradingEnvEnhanced(
                start_date=START_DATE, 
                end_date=END_DATE, 
                gnn_model=gnn_model, 
                device=DEVICE,
                initial_cash=10000.0,
                max_position_pct=0.1,  # Max 10% per stock
                enable_slippage=True,
                enable_risk_penalty=True
            )
        except ImportError:
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
    
    # We use MlpPolicy, which is standard for Gym's discrete action space
    # PPO with MlpPolicy performs better on CPU according to stable-baselines3
    model = PPO(
        "MlpPolicy", 
        vec_env, 
        verbose=1, 
        learning_rate=PPO_LEARNING_RATE, 
        device='cpu',  # Use CPU for better MLP performance
        tensorboard_log=RL_LOG_PATH
    )
    
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
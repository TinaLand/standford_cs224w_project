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
# NOTE: This file lives in `src/rl/`, so the project root is three levels up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / 'scripts')) # To import rl_environment (legacy, kept for backward compatibility)
sys.path.append(str(PROJECT_ROOT / 'scripts' / 'components')) # For GNN model components (legacy)

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
from src.training.transformer_trainer import (
    RoleAwareGraphTransformer,
    HIDDEN_CHANNELS,
    OUT_CHANNELS,
    NUM_LAYERS,
    NUM_HEADS,
)
# Legacy imports - use new structure
from .environments.single_agent import StockTradingEnv
from .agents.single_agent import StockTradingAgent 


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

    # 2. Initialize GNN model structure (MUST match Phase 4 training hyperparameters)
    # Use the same HIDDEN_CHANNELS / NUM_LAYERS / NUM_HEADS as in transformer_trainer.py
    gnn_model = RoleAwareGraphTransformer(
        INPUT_DIM,
        HIDDEN_CHANNELS,
        OUT_CHANNELS,
        NUM_LAYERS,
        NUM_HEADS,
    ).to(DEVICE)
    
    # 3. Load trained weights
    if not CORE_GNN_MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained GNN model not found at: {CORE_GNN_MODEL_PATH}")

    # Use strict=False to be robust to minor architecture/logging changes while
    # ensuring that matching layers are correctly loaded.
    state_dict = torch.load(CORE_GNN_MODEL_PATH, map_location=DEVICE, weights_only=False)
    missing, unexpected = gnn_model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print("⚠️  Warning when loading GNN state_dict for RL:")
        if missing:
            print(f"   Missing keys: {missing}")
        if unexpected:
            print(f"   Unexpected keys: {unexpected}")
    
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
    
    # 3. Setup Environment Factory
    def make_env():
        return StockTradingEnv(
            start_date=START_DATE, 
            end_date=END_DATE, 
            gnn_model=gnn_model, 
            device=DEVICE
        )
    
    # 4. Create RL Agent
    print(f"\n--- 2. Creating RL Agent (Total Timesteps: {TOTAL_TIMESTEPS}) ---")
    
    # Configure tensorboard logging (optional)
    tensorboard_log_path = RL_LOG_PATH if TENSORBOARD_AVAILABLE else None
    if TENSORBOARD_AVAILABLE:
        print("✅ TensorBoard logging enabled")
    else:
        print("⚠️  TensorBoard not available, logging disabled")
    
    # Create agent using the wrapper class
    agent = StockTradingAgent(
        gnn_model=gnn_model,
        env_factory=make_env,
        device=DEVICE,
        learning_rate=PPO_LEARNING_RATE,
        tensorboard_log=tensorboard_log_path,
        policy="MlpPolicy",
        verbose=1
    )
    
    # 5. Train Agent
    try:
        training_stats = agent.train(total_timesteps=TOTAL_TIMESTEPS)
        print(f"\n📊 Training Statistics: {training_stats}")
    except Exception as e:
        print(f"❌ RL Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 6. Save Agent
    agent.save(RL_SAVE_PATH / "ppo_stock_agent")
    print(f"\n✅ RL Agent trained and saved to: {RL_SAVE_PATH / 'ppo_stock_agent.zip'}")
    
    return agent

    print("\n🎯 Phase 5: RL Integration and Training Complete! Ready for Phase 6: Evaluation.")

if __name__ == '__main__':
    run_rl_pipeline()
# src/rl/training/single_agent.py
"""
Single-Agent RL Training Pipeline
Enhanced version with improved structure and configuration
"""

import torch
import pandas as pd
from pathlib import Path
import sys
from typing import Optional

# Fix PyTorch serialization for pandas timestamps and PyTorch Geometric (PyTorch 2.6+)
import torch.serialization
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([pd._libs.tslibs.timestamps._unpickle_timestamp])
    # Add PyTorch Geometric BaseStorage for graph loading
    try:
        from torch_geometric.data.storage import BaseStorage
        torch.serialization.add_safe_globals([BaseStorage])
    except ImportError:
        pass

from ..config import SingleAgentConfig, GNNConfig, PROJECT_ROOT, DEVICE, RL_LOG_PATH, RL_SAVE_PATH
from ..agents.single_agent import StockTradingAgent
from ..environments.single_agent import StockTradingEnv

# Import GNN model
sys.path.append(str(PROJECT_ROOT / 'src'))
from training.transformer_trainer import (
    RoleAwareGraphTransformer,
    HIDDEN_CHANNELS,
    OUT_CHANNELS,
    NUM_LAYERS,
    NUM_HEADS,
)

# Try to import tensorboard
try:
    import tensorboard
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


def load_gnn_model_for_rl() -> torch.nn.Module:
    """
    Load the trained GNN model for RL training.
    
    Returns:
        Loaded and frozen GNN model
    """
    print("\n--- Loading Trained GNN Model for RL ---")
    
    # 1. Determine Input Dimension
    data_graphs_dir = PROJECT_ROOT / "data" / "graphs"
    sample_graph_path = list(data_graphs_dir.glob('graph_t_*.pt'))[0]
    temp_data = torch.load(sample_graph_path, weights_only=False)
    INPUT_DIM = temp_data['stock'].x.shape[1]
    
    # 2. Initialize GNN model structure
    gnn_model = RoleAwareGraphTransformer(
        INPUT_DIM,
        HIDDEN_CHANNELS,
        OUT_CHANNELS,
        NUM_LAYERS,
        NUM_HEADS,
    ).to(DEVICE)
    
    # 3. Load trained weights
    if not GNNConfig.CORE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained GNN model not found at: {GNNConfig.CORE_MODEL_PATH}")
    
    state_dict = torch.load(GNNConfig.CORE_MODEL_PATH, map_location=DEVICE, weights_only=False)
    missing, unexpected = gnn_model.load_state_dict(state_dict, strict=False)
    
    if missing or unexpected:
        print("  Warning when loading GNN state_dict for RL:")
        if missing:
            print(f"   Missing keys: {missing}")
        if unexpected:
            print(f"   Unexpected keys: {unexpected}")
    
    # 4. Freeze GNN parameters
    for param in gnn_model.parameters():
        param.requires_grad = False
        
    print(f" GNN Model loaded and frozen (Input Dim: {INPUT_DIM})")
    return gnn_model


def determine_training_period():
    """
    Determine training period based on available graph data.
    
    Returns:
        Tuple of (start_date, end_date)
    """
    data_graphs_dir = PROJECT_ROOT / "data" / "graphs"
    graph_files = list(data_graphs_dir.glob('graph_t_*.pt'))
    
    if not graph_files:
        raise ValueError("No graph files found. Run Phase 2 first.")
    
    # Get date range from graph files
    graph_dates = sorted([pd.to_datetime(f.stem.split('_')[-1]) for f in graph_files])
    graph_start = graph_dates[0]
    graph_end = graph_dates[-1]
    
    # Use last 20% of graph data for RL training (test set)
    date_range_days = (graph_end - graph_start).days
    start_offset_days = int(date_range_days * 0.8)
    start_date = graph_start + pd.Timedelta(days=start_offset_days)
    end_date = graph_end
    
    print(f" RL Training period: {start_date.date()} to {end_date.date()}")
    print(f"   (Graph files available: {graph_start.date()} to {graph_end.date()})")
    print(f"   (Using last {int((date_range_days - start_offset_days) / date_range_days * 100)}% of data)")
    
    return start_date, end_date


def create_trading_environment(gnn_model: torch.nn.Module, start_date: pd.Timestamp, end_date: pd.Timestamp):
    """
    Create trading environment factory.
    
    Args:
        gnn_model: Loaded GNN model
        start_date: Training start date
        end_date: Training end date
        
    Returns:
        Environment factory function
    """
    def make_env():
        return StockTradingEnv(
            start_date=start_date,
            end_date=end_date,
            gnn_model=gnn_model,
            device=DEVICE
        )
    
    return make_env


def run_single_agent_training(
    total_timesteps: int = SingleAgentConfig.TOTAL_TIMESTEPS,
    learning_rate: float = SingleAgentConfig.LEARNING_RATE,
    verbose: int = SingleAgentConfig.VERBOSE
) -> StockTradingAgent:
    """
    Run single-agent RL training pipeline.
    
    Args:
        total_timesteps: Total training timesteps
        learning_rate: Learning rate for PPO
        verbose: Verbosity level
        
    Returns:
        Trained StockTradingAgent
    """
    print("\n Starting Single-Agent RL Training Pipeline")
    print("=" * 60)
    
    # 1. Load GNN Model
    try:
        gnn_model = load_gnn_model_for_rl()
    except FileNotFoundError as e:
        print(f" Error: {e}")
        print("   Ensure Phase 4 training completed successfully.")
        raise
    
    # 2. Determine Training Period
    start_date, end_date = determine_training_period()
    
    # 3. Create Environment Factory
    env_factory = create_trading_environment(gnn_model, start_date, end_date)
    
    # 4. Setup TensorBoard Logging
    tensorboard_log_path = RL_LOG_PATH if TENSORBOARD_AVAILABLE else None
    if TENSORBOARD_AVAILABLE:
        print(" TensorBoard logging enabled")
    else:
        print("  TensorBoard not available, logging disabled")
    
    # 5. Create RL Agent
    print(f"\n--- Creating Single-Agent RL Agent ---")
    agent = StockTradingAgent(
        gnn_model=gnn_model,
        env_factory=env_factory,
        device=DEVICE,
        learning_rate=learning_rate,
        tensorboard_log=tensorboard_log_path,
        policy=SingleAgentConfig.POLICY,
        verbose=verbose,
        agent_id="single_ppo_agent"
    )
    
    # 6. Train Agent
    print(f"\n--- Training Agent ({total_timesteps} timesteps) ---")
    try:
        training_stats = agent.train(
            total_timesteps=total_timesteps,
            progress_bar=False  # Disable progress bar to avoid tqdm/rich dependency
        )
        print(f"\n Training Statistics: {training_stats}")
    except Exception as e:
        print(f" Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # 7. Save Agent (Original location)
    save_path = RL_SAVE_PATH / "ppo_stock_agent"
    agent.save(save_path)
    print(f"\n Agent saved to: {save_path}.zip")
    
    # 8. Copy Agent to Results Directory
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_agent_path = results_dir / "ppo_stock_agent"
    agent.save(results_agent_path)
    print(f" Agent also saved to results: {results_agent_path}.zip")
    
    # 9. Generate Results JSON
    def convert_to_json_serializable(obj):
        """Convert numpy/torch types to JSON-serializable Python types."""
        import numpy as np
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().item() if obj.numel() == 1 else obj.cpu().tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)  # Fallback to string representation
    
    results_json = {
        "model_type": "single_agent_ppo",
        "training_config": {
            "total_timesteps": total_timesteps,
            "learning_rate": learning_rate,
            "policy": SingleAgentConfig.POLICY,
            "verbose": verbose
        },
        "training_statistics": convert_to_json_serializable(training_stats),
        "model_path": str(results_agent_path) + ".zip",
        "training_period": {
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d")
        },
        "status": "completed"
    }
    
    # 10. Quick Evaluation
    print(f"\n--- Quick Evaluation ---")
    try:
        test_env = env_factory()
        eval_results = agent.evaluate(test_env, n_episodes=5, deterministic=True)
        print(f" Evaluation Results: {eval_results}")
        results_json["evaluation_results"] = convert_to_json_serializable(eval_results)
    except Exception as e:
        print(f"  Evaluation failed: {e}")
        results_json["evaluation_results"] = {"error": str(e)}
    
    # 11. Save Results JSON
    import json
    results_json_path = results_dir / "single_agent_results.json"
    with open(results_json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f" Results JSON saved to: {results_json_path}")
    
    print("\n Single-Agent RL Training Complete!")
    return agent


def main():
    """Main entry point for single-agent training."""
    try:
        agent = run_single_agent_training()
        return agent
    except Exception as e:
        print(f"\n Single-Agent Training Failed: {e}")
        raise


if __name__ == '__main__':
    main()
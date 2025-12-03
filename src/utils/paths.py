"""
Centralized path configuration for the project.

This module provides a single source of truth for all project paths,
eliminating duplicate path definitions across the codebase.
"""

from pathlib import Path

# Project root directory (3 levels up from src/utils/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_EDGES_DIR = DATA_DIR / "edges"
DATA_GRAPHS_DIR = DATA_DIR / "graphs"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
MODELS_PLOTS_DIR = MODELS_DIR / "plots"

# Results directories
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_PLOTS_DIR = RESULTS_DIR / "plots"

# Log directories
LOGS_DIR = PROJECT_ROOT / "logs"
RL_LOG_PATH = LOGS_DIR / "rl_logs"

# Run directories
RUNS_DIR = PROJECT_ROOT / "runs"
TENSORBOARD_DIR = RUNS_DIR

# Figures directory
FIGURES_DIR = PROJECT_ROOT / "figures"

# Common data files
OHLCV_RAW_FILE = DATA_RAW_DIR / "stock_prices_ohlcv_raw.csv"
SECTOR_FILE = DATA_RAW_DIR / "static_sector_industry.csv"
NODE_FEATURES_FILE = DATA_PROCESSED_DIR / "node_features_X_t_final.csv"

# Model files
CORE_TRANSFORMER_MODEL = MODELS_DIR / "core_transformer_model.pt"
BASELINE_GCN_MODEL = MODELS_DIR / "baseline_gcn_model.pt"
RL_AGENT_MODEL_DIR = MODELS_DIR / "rl_ppo_agent_model"
RL_AGENT_MODEL_FILE = RL_AGENT_MODEL_DIR / "ppo_stock_agent.zip"

# Ensure directories exist
for directory in [
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    DATA_EDGES_DIR,
    DATA_GRAPHS_DIR,
    MODELS_DIR,
    MODELS_CHECKPOINTS_DIR,
    MODELS_PLOTS_DIR,
    RESULTS_DIR,
    RESULTS_PLOTS_DIR,
    LOGS_DIR,
    RL_LOG_PATH,
    FIGURES_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "DATA_RAW_DIR",
    "DATA_PROCESSED_DIR",
    "DATA_EDGES_DIR",
    "DATA_GRAPHS_DIR",
    "MODELS_DIR",
    "MODELS_CHECKPOINTS_DIR",
    "MODELS_PLOTS_DIR",
    "RESULTS_DIR",
    "RESULTS_PLOTS_DIR",
    "LOGS_DIR",
    "RL_LOG_PATH",
    "RUNS_DIR",
    "TENSORBOARD_DIR",
    "FIGURES_DIR",
    "OHLCV_RAW_FILE",
    "SECTOR_FILE",
    "NODE_FEATURES_FILE",
    "CORE_TRANSFORMER_MODEL",
    "BASELINE_GCN_MODEL",
    "RL_AGENT_MODEL_DIR",
    "RL_AGENT_MODEL_FILE",
]


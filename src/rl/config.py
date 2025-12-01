# src/rl/config.py
"""
Centralized RL Configuration
Contains all hyperparameters and settings for single-agent and multi-agent RL
"""

from pathlib import Path
import torch

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_GRAPHS_DIR = PROJECT_ROOT / "data" / "graphs"
RL_LOG_PATH = PROJECT_ROOT / "logs" / "rl_logs"
RL_SAVE_PATH = MODELS_DIR / "rl_ppo_agent_model"

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Single-Agent RL Configuration
class SingleAgentConfig:
    # PPO Hyperparameters
    LEARNING_RATE = 1e-5
    TOTAL_TIMESTEPS = 10000
    POLICY = "MlpPolicy"
    VERBOSE = 1
    
    # Environment Parameters
    TRANSACTION_COST = 0.001  # 0.1% per trade
    MAX_POSITION_SIZE = 0.2   # 20% per stock
    MIN_CASH_RESERVE = 0.05   # 5% minimum cash
    MAX_LEVERAGE = 1.0        # No leverage
    
    # Reward Configuration
    USE_SHARPE_REWARD = True
    SHARPE_WINDOW = 20        # Rolling window for Sharpe calculation
    RISK_FREE_RATE = 0.02     # Annual risk-free rate

# Multi-Agent RL Configuration
class MultiAgentConfig:
    # CTDE Parameters
    LEARNING_RATE = 1e-5
    TOTAL_TIMESTEPS = 15000
    NUM_ENVS = 1
    
    # Mixing Network Parameters
    MIXING_HIDDEN_DIM = 64
    GLOBAL_STATE_DIM = 256
    
    # Sector Configuration
    USE_DYNAMIC_SECTORS = True
    MIN_STOCKS_PER_SECTOR = 1  # Lowered to accommodate real data distribution
    
    # Training Parameters
    CENTRALIZED_TRAINING = True
    DECENTRALIZED_EXECUTION = True
    
    # Balanced Multi-Agent Configuration (10 stocks per sector)
    # Optimized for stable RL training with equal agent workloads
    DEFAULT_SECTORS = {
        'Technology': [
            'AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMZN', 'AVGO', 'ADBE', 'CSCO', 'CRM'
        ],
        'Financials': [
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'COF', 'SCHW', 'BLK'
        ],
        'Healthcare': [
            'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN'
        ],
        'Consumer Discretionary': [
            'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'TJX', 'DG', 'ROST', 'BBY'
        ],
        'Energy': [
            'XOM', 'CVX', 'SLB', 'EOG', 'COP', 'MPC', 'VLO', 'PSX', 'HAL', 'OXY'
        ]
    }

# GNN Model Configuration
class GNNConfig:
    CORE_MODEL_PATH = MODELS_DIR / 'core_transformer_model.pt'
    HIDDEN_CHANNELS = 256
    OUT_CHANNELS = 64
    NUM_LAYERS = 3
    NUM_HEADS = 4

# Ensure directories exist
RL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
RL_LOG_PATH.mkdir(parents=True, exist_ok=True)
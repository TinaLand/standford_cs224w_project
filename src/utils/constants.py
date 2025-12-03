"""
Centralized configuration constants and hyperparameters.

This module provides a single source of truth for common configuration values
used across training, evaluation, and RL modules.
"""

import torch

# ============================================================================
# Device Configuration
# ============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# Training Hyperparameters
# ============================================================================

# Model Architecture
HIDDEN_CHANNELS = 512           # Hidden size (increased from 256 for better capacity)
NUM_LAYERS = 3                  # Number of GNN layers (increased from 2 for deeper model)
NUM_HEADS = 8                   # Number of attention heads
OUT_CHANNELS = 2                # Binary classification: Up/Down
PE_DIM = 32                     # Positional embedding dimension
TIME_PE_DIM = 16                # Time positional encoding dimension

# Training Configuration
LEARNING_RATE = 0.0008          # Learning rate (adjusted for stable training)
NUM_EPOCHS = 40                 # Number of training epochs
BATCH_SIZE = 32                 # Batch size for training
LOOKAHEAD_DAYS = 5              # Target: 5-day-ahead return sign

# Loss Function Configuration
LOSS_TYPE = 'focal'             # Options: 'standard', 'weighted', 'focal'
FOCAL_ALPHA = 0.85              # Weight for minority class (increased for Down/Flat class)
FOCAL_GAMMA = 3.0               # Focusing parameter (increased for harder focus)
USE_CLASS_WEIGHTS = True        # Enable class weights for additional balancing

# Multi-task Learning Configuration
ENABLE_MULTI_TASK = True        # Enable classification + regression
REG_LOSS_WEIGHT = 0.5           # Weight for regression loss in total loss

# Time-Aware Modeling Configuration
ENABLE_TIME_AWARE = True        # Enable time-aware positional encoding

# Training Optimizations
ENABLE_AMP = True               # Automatic Mixed Precision
ENABLE_MINI_BATCH = False       # Mini-batch training for large graphs
DROPOUT_RATE = 0.3              # Dropout rate for regularization
WEIGHT_DECAY = 1e-5             # L2 regularization
GRAD_CLIP_NORM = 1.0            # Gradient clipping norm

# Early Stopping Configuration
ENABLE_EARLY_STOPPING = True
EARLY_STOP_PATIENCE = 12        # Increased from 10 to allow more training epochs
EARLY_STOP_MIN_DELTA = 0.0001   # Minimum improvement threshold

# Learning Rate Scheduler Configuration
ENABLE_LR_SCHEDULER = True
LR_SCHEDULER_PATIENCE = 4       # Increased from 3 for more stable training
LR_SCHEDULER_FACTOR = 0.5       # Multiply LR by this factor
LR_SCHEDULER_MIN_LR = 1e-6      # Minimum learning rate

# ============================================================================
# Evaluation Configuration
# ============================================================================

ENABLE_ROC_AUC = True           # Calculate ROC-AUC score
ENABLE_CONFUSION_MATRIX = True  # Generate confusion matrix plots

# ============================================================================
# RL Configuration
# ============================================================================

# Transaction Costs
TRANSACTION_COST = 0.001        # 0.1% per trade

# Initial Portfolio
INITIAL_BALANCE = 100000.0      # Initial cash balance

# ============================================================================
# Data Configuration
# ============================================================================

# Stock Selection
NUM_STOCKS = 50                 # Number of stocks in dataset
START_DATE = '2015-01-01'       # Start date for data collection
END_DATE = '2024-12-31'         # End date for data collection

# Feature Engineering
ROLLING_WINDOW = 20             # Rolling window for correlations
CORRELATION_THRESHOLD = 0.3     # Minimum correlation for edge creation

# ============================================================================
# Export All Constants
# ============================================================================

__all__ = [
    # Device
    "DEVICE",
    
    # Model Architecture
    "HIDDEN_CHANNELS",
    "NUM_LAYERS",
    "NUM_HEADS",
    "OUT_CHANNELS",
    "PE_DIM",
    "TIME_PE_DIM",
    
    # Training
    "LEARNING_RATE",
    "NUM_EPOCHS",
    "BATCH_SIZE",
    "LOOKAHEAD_DAYS",
    
    # Loss Function
    "LOSS_TYPE",
    "FOCAL_ALPHA",
    "FOCAL_GAMMA",
    "USE_CLASS_WEIGHTS",
    
    # Multi-task
    "ENABLE_MULTI_TASK",
    "REG_LOSS_WEIGHT",
    
    # Time-aware
    "ENABLE_TIME_AWARE",
    
    # Optimizations
    "ENABLE_AMP",
    "ENABLE_MINI_BATCH",
    "DROPOUT_RATE",
    "WEIGHT_DECAY",
    "GRAD_CLIP_NORM",
    
    # Early Stopping
    "ENABLE_EARLY_STOPPING",
    "EARLY_STOP_PATIENCE",
    "EARLY_STOP_MIN_DELTA",
    
    # LR Scheduler
    "ENABLE_LR_SCHEDULER",
    "LR_SCHEDULER_PATIENCE",
    "LR_SCHEDULER_FACTOR",
    "LR_SCHEDULER_MIN_LR",
    
    # Evaluation
    "ENABLE_ROC_AUC",
    "ENABLE_CONFUSION_MATRIX",
    
    # RL
    "TRANSACTION_COST",
    "INITIAL_BALANCE",
    
    # Data
    "NUM_STOCKS",
    "START_DATE",
    "END_DATE",
    "ROLLING_WINDOW",
    "CORRELATION_THRESHOLD",
]


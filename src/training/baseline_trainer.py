# phase3_baseline_training.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, HeteroConv
from torch_geometric.data import Data, HeteroData
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, roc_curve
from datetime import datetime
import time
import matplotlib.pyplot as plt
import seaborn as sns

# FIX: Allow PyTorch to safely load torch-geometric objects (PyTorch >= 2.6)
import torch.serialization
from torch_geometric.data.storage import BaseStorage, NodeStorage, EdgeStorage, GlobalStorage
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([BaseStorage, NodeStorage, EdgeStorage, GlobalStorage])

# TensorBoard support
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print(" TensorBoard not available. Install with: pip install tensorboard")

# --- Configuration ---
# Import centralized paths and utilities
from src.utils.paths import PROJECT_ROOT, DATA_GRAPHS_DIR, MODELS_DIR, OHLCV_RAW_FILE, TENSORBOARD_DIR
from src.utils.graph_loader import load_graph_data

# Hyperparameters
HIDDEN_DIM = 128  # Increased from 64 (more capacity for 15 features)
OUT_DIM = 2  # Binary classification: Up (1) or Down/Flat (0)
NUM_EPOCHS = 40  # Increased for better convergence
LEARNING_RATE = 0.0005  # Balanced (not too high, not too low)
LOOKAHEAD_DAYS = 5 #  5-day-ahead stock return sign [cite: 29]

# Class Imbalance Handling Configuration
# Options: 'standard' (no weighting), 'weighted' (class weights), 'focal' (focal loss)
LOSS_TYPE = 'focal'  # Using focal loss for imbalance handling
FOCAL_ALPHA = 0.75    # Increased from 0.5 to give more weight to minority class (Down/Flat)
FOCAL_GAMMA = 2.0      # Standard focusing parameter

# Checkpoint Configuration
ENABLE_CHECKPOINTING = True     # Save full checkpoints (model, optimizer, epoch, metrics)
RESUME_FROM_CHECKPOINT = False  # Resume training from last checkpoint
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
SAVE_CHECKPOINT_EVERY = 5       # Save checkpoint every N epochs (in addition to best)

# Early Stopping Configuration
ENABLE_EARLY_STOPPING = True    # Stop training if validation F1 doesn't improve
EARLY_STOP_PATIENCE = 10        # Increased from 5 to allow more training epochs
EARLY_STOP_MIN_DELTA = 0.0001   # Minimum improvement to be considered as improvement

# Learning Rate Scheduler Configuration
ENABLE_LR_SCHEDULER = True      # Enable learning rate scheduling
LR_SCHEDULER_TYPE = 'plateau'   # Options: 'plateau', 'step', 'exponential'
LR_SCHEDULER_PATIENCE = 3       # Epochs to wait before reducing LR (for plateau)
LR_SCHEDULER_FACTOR = 0.5       # Multiply LR by this factor when reducing
LR_SCHEDULER_MIN_LR = 1e-6      # Minimum learning rate

# TensorBoard & Metrics Logging Configuration
ENABLE_TENSORBOARD = True       # Enable TensorBoard logging
# TENSORBOARD_DIR is now imported from src.utils.paths
ENABLE_ROC_AUC = True          # Calculate ROC-AUC score
ENABLE_CONFUSION_MATRIX = True  # Generate confusion matrix plots
PLOTS_DIR = MODELS_DIR / "plots"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Helper Utilities ---

def _read_time_series_csv(path: Path) -> pd.DataFrame:
    """
    Robust CSV reader that accepts files with or without an explicit 'Date' column.
    Ensures the returned DataFrame has a DatetimeIndex named 'Date'.
    """
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Time-series CSV at {path} is empty. Ensure Phase 1 outputs exist.")
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
    else:
        first_col = df.columns[0]
        df[first_col] = pd.to_datetime(df[first_col])
        df = df.set_index(first_col)
        df.index.name = 'Date'
    return df

# --- 1. Model Definition (Simple GAT Baseline) ---

class BaselineGNN(torch.nn.Module):
    """
    A simple GAT model for node-level classification.
    We will simplify and use a single edge type for the baseline.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # Using GATConv as a slightly more advanced baseline than GCN
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, dropout=0.3)
        self.lin1 = torch.nn.Linear(hidden_channels * 4, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # x: Node features [N, F_in]
        # edge_index: Adjacency list [2, E]
        
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv1(x, edge_index).relu()
        
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return x # Output: [N, 2] (logits for Up/Down)

# --- 2. Loss Functions for Class Imbalance ---

class FocalLoss(torch.nn.Module):
    """
    Focal Loss for addressing class imbalance in classification tasks.
    
    Mathematical Formulation:
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    where:
    - p_t is the model's estimated probability for the true class
    - α_t is a weighting factor for class t (balances class frequencies)
    - γ (gamma) is the focusing parameter (reduces loss for well-classified examples)
    
    Intuition:
    - When γ = 0, focal loss is equivalent to cross-entropy
    - When γ > 0, the loss focuses more on hard-to-classify examples
    - The (1 - p_t)^γ term down-weights easy examples (where p_t is high)
    - This helps the model focus on minority class and hard examples
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    Paper: https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (float): Weighting factor for positive class (0-1).
                          For binary classification, alpha balances pos/neg classes.
                          Default 0.25 means negative class gets 0.75 weight.
            gamma (float): Focusing parameter. Higher values give more focus to hard examples.
                          Typical values: 2.0 (default), range [0, 5]
            reduction (str): Specifies reduction to apply: 'none' | 'mean' | 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model predictions (logits), shape [N, num_classes]
            targets: Ground truth class labels, shape [N]
        
        Returns:
            Focal loss value (scalar if reduction='mean', tensor if reduction='none')
        """
        # Step 1: Convert logits to probabilities using softmax
        # Softmax: p_i = exp(logit_i) / sum(exp(logit_j))
        p = F.softmax(inputs, dim=1)
        
        # Step 2: Create one-hot encoding of targets
        # Example: if target=1, one_hot = [0, 1] for binary classification
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Step 3: Get the probability of the true class for each example
        # p_t = p[i, true_class_i] for each example i
        p_t = p.gather(1, targets.view(-1, 1)).squeeze(1)
        
        # Step 4: Calculate focal loss components
        # (1 - p_t)^gamma: modulating factor that reduces loss for well-classified examples
        # When p_t is high (confident correct prediction), (1-p_t) is small -> loss is down-weighted
        # When p_t is low (uncertain/wrong prediction), (1-p_t) is large -> loss is emphasized
        focal_weight = (1 - p_t) ** self.gamma
        
        # Step 5: Apply alpha weighting for class balance
        # For binary: alpha for class 1, (1-alpha) for class 0
        alpha_t = torch.where(targets == 1, 
                             torch.tensor(self.alpha, device=inputs.device),
                             torch.tensor(1 - self.alpha, device=inputs.device))
        
        # Step 6: Combine all components
        # Final focal loss = alpha_t * (1-p_t)^gamma * CE_loss
        focal_loss = alpha_t * focal_weight * ce_loss
        
        # Step 7: Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss

def compute_class_weights(targets_dict, train_dates):
    """
    Computes class weights for handling imbalanced datasets.
    
    Mathematical Formula:
    w_i = n_total / (n_classes * n_i)
    
    where:
    - n_total is the total number of samples
    - n_classes is the number of classes (2 for binary)
    - n_i is the number of samples in class i
    
    Intuition:
    - Minority class gets higher weight -> more penalty when misclassified
    - Majority class gets lower weight -> less penalty when misclassified
    - This balances the contribution of each class to the loss
    
    Example:
    If we have 1000 negative samples (class 0) and 200 positive samples (class 1):
    - w_0 = 1200 / (2 * 1000) = 0.6  (majority class: lower weight)
    - w_1 = 1200 / (2 * 200) = 3.0   (minority class: higher weight)
    
    Args:
        targets_dict: Dictionary mapping dates to target tensors
        train_dates: List of training dates
    
    Returns:
        class_weights: Tensor of shape [num_classes] with weights for each class
        class_counts: Dictionary with count of each class
    """
    print("\n  Computing class weights for imbalanced data...")
    
    # Step 1: Collect all training labels
    all_train_labels = []
    for date in train_dates:
        target = targets_dict.get(date)
        if target is not None:
            all_train_labels.append(target.numpy())
    
    # Concatenate all labels into a single array
    all_train_labels = np.concatenate(all_train_labels)
    
    # Step 2: Count samples per class
    unique_classes, class_counts = np.unique(all_train_labels, return_counts=True)
    class_count_dict = dict(zip(unique_classes.astype(int), class_counts))
    
    # Step 3: Calculate total samples and number of classes
    n_total = len(all_train_labels)
    n_classes = len(unique_classes)
    
    # Step 4: Compute weights using the formula: n_total / (n_classes * n_i)
    class_weights = torch.zeros(OUT_DIM, dtype=torch.float32)
    for cls_idx in range(OUT_DIM):
        if cls_idx in class_count_dict:
            # Standard sklearn-style class weight formula
            class_weights[cls_idx] = n_total / (n_classes * class_count_dict[cls_idx])
        else:
            # If class doesn't appear in training data, assign weight of 1.0
            class_weights[cls_idx] = 1.0
    
    # Step 5: Print class distribution statistics
    print(f"    Training Set Class Distribution:")
    for cls_idx in range(OUT_DIM):
        count = class_count_dict.get(cls_idx, 0)
        percentage = (count / n_total) * 100
        weight = class_weights[cls_idx].item()
        class_name = "Positive (Up)" if cls_idx == 1 else "Negative (Down/Flat)"
        print(f"      Class {cls_idx} ({class_name}): {count:6d} samples ({percentage:5.2f}%) | Weight: {weight:.4f}")
    
    # Step 6: Calculate imbalance ratio for reporting
    if len(class_count_dict) == 2:
        imbalance_ratio = max(class_counts) / min(class_counts)
        print(f"     Imbalance Ratio: {imbalance_ratio:.2f}:1")
        if imbalance_ratio > 2.0:
            print(f"    Significant imbalance detected. Using {LOSS_TYPE} loss is recommended.")
    
    return class_weights, class_count_dict

def save_checkpoint(epoch, model, optimizer, metrics, checkpoint_dir, is_best=False):
    """
    Save a complete training checkpoint.
    
    A checkpoint includes all information needed to resume training:
    - Model weights (state_dict): Learned parameters of the neural network
    - Optimizer state (state_dict): Momentum, learning rates, etc.
    - Current epoch: Which training iteration we're on
    - Training metrics: Loss history, accuracy, F1 scores over time
    
    This allows you to:
    1. Resume training if interrupted
    2. Roll back to earlier epochs if overfitting occurs
    3. Compare different training runs
    4. Debug training issues by examining saved metrics
    
    Args:
        epoch (int): Current training epoch number
        model (nn.Module): The neural network model
        optimizer (Optimizer): PyTorch optimizer (Adam, SGD, etc.)
        metrics (dict): Dictionary containing training history
            Expected keys: 'train_loss', 'val_acc', 'val_f1', 'epoch_times'
        checkpoint_dir (Path): Directory to save checkpoints
        is_best (bool): Whether this is the best model so far
    
    Saved Files:
        - checkpoint_epoch_N.pt: Regular checkpoint at epoch N
        - checkpoint_best.pt: Best model (highest validation F1)
        - checkpoint_latest.pt: Most recent checkpoint (for easy resuming)
    """
    checkpoint = {
        # Model architecture weights
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        
        # Optimizer state (important for momentum-based optimizers like Adam)
        'optimizer_state_dict': optimizer.state_dict(),
        
        # Training metrics history
        'metrics': metrics,
        
        # Configuration for verification when loading
        'config': {
            'hidden_dim': HIDDEN_DIM,
            'out_dim': OUT_DIM,
            'learning_rate': LEARNING_RATE,
            'loss_type': LOSS_TYPE,
        },
        
        # Timestamp for tracking
        'timestamp': datetime.now().isoformat(),
    }
    
    # Save regular epoch checkpoint
    checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pt'
    torch.save(checkpoint, checkpoint_path)
    
    # Save as latest checkpoint (for easy resuming)
    latest_path = checkpoint_dir / 'checkpoint_latest.pt'
    torch.save(checkpoint, latest_path)
    
    # Save as best checkpoint if this is the best model
    if is_best:
        best_path = checkpoint_dir / 'checkpoint_best.pt'
        torch.save(checkpoint, best_path)
        print(f"   Saved BEST checkpoint: {best_path.name}")
    
    print(f"   Saved checkpoint: {checkpoint_path.name}")

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Load a checkpoint and restore training state.
    
    This function restores:
    - Model weights to the saved state
    - Optimizer state (learning rates, momentum, etc.)
    - Epoch number to resume from
    - Training metrics history
    
    Mathematical Note:
    When using optimizers with momentum (like Adam), it's crucial to restore
    the optimizer state. Adam maintains two moving averages:
    - m_t: First moment (mean) of gradients
    - v_t: Second moment (variance) of gradients
    
    Without restoring these, training may be unstable after resuming.
    
    Args:
        checkpoint_path (Path): Path to checkpoint file
        model (nn.Module): Model to load weights into
        optimizer (Optimizer, optional): Optimizer to restore state
    
    Returns:
        tuple: (start_epoch, metrics_dict)
            - start_epoch: Epoch to resume training from (checkpoint_epoch + 1)
            - metrics_dict: Dictionary of training history
    """
    if not checkpoint_path.exists():
        print(f" Checkpoint not found: {checkpoint_path}")
        return 0, {
            'train_loss': [],
            'val_acc': [],
            'val_f1': [],
            'epoch_times': []
        }
    
    print(f" Loading checkpoint from: {checkpoint_path.name}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    # Restore model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"   Model weights restored")
    
    # Restore optimizer state (if optimizer provided)
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"   Optimizer state restored")
    
    # Get training state
    start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
    metrics = checkpoint.get('metrics', {
        'train_loss': [],
        'val_acc': [],
        'val_f1': [],
        'epoch_times': []
    })
    
    # Display checkpoint info
    print(f"   Checkpoint Info:")
    print(f"     - Saved at epoch: {checkpoint['epoch']}")
    print(f"     - Resuming from epoch: {start_epoch}")
    print(f"     - Training history: {len(metrics.get('train_loss', []))} epochs")
    
    if 'config' in checkpoint:
        config = checkpoint['config']
        print(f"     - Loss type: {config.get('loss_type', 'unknown')}")
    
    if 'timestamp' in checkpoint:
        print(f"     - Saved at: {checkpoint['timestamp']}")
    
    # Check for best metrics
    if metrics.get('val_f1'):
        best_f1 = max(metrics['val_f1'])
        print(f"     - Best validation F1 so far: {best_f1:.4f}")
    
    return start_epoch, metrics

def cleanup_old_checkpoints(checkpoint_dir, keep_last_n=5, keep_best=True):
    """
    Clean up old checkpoints to save disk space.
    
    Strategy:
    - Always keep the best checkpoint (highest validation F1)
    - Always keep the latest checkpoint (for resuming)
    - Keep only the last N regular checkpoints
    - Delete older checkpoints to save space
    
    Args:
        checkpoint_dir (Path): Directory containing checkpoints
        keep_last_n (int): Number of recent checkpoints to keep
        keep_best (bool): Whether to preserve the best checkpoint
    """
    # Get all epoch checkpoints (excluding best and latest)
    checkpoint_files = sorted(
        checkpoint_dir.glob('checkpoint_epoch_*.pt'),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    
    # Keep only the last N checkpoints
    if len(checkpoint_files) > keep_last_n:
        to_delete = checkpoint_files[keep_last_n:]
        for checkpoint_file in to_delete:
            try:
                checkpoint_file.unlink()
                print(f"   Deleted old checkpoint: {checkpoint_file.name}")
            except Exception as e:
                print(f"   Could not delete {checkpoint_file.name}: {e}")

def plot_confusion_matrix(y_true, y_pred, epoch, split='test', save_dir=None):
    """
    Generate and save confusion matrix plot.
    
    A confusion matrix shows the performance of a classification model by comparing:
    - True Positives (TP): Correctly predicted positive class
    - True Negatives (TN): Correctly predicted negative class  
    - False Positives (FP): Incorrectly predicted positive (Type I error)
    - False Negatives (FN): Incorrectly predicted negative (Type II error)
    
    Matrix layout:
                    Predicted
                    Neg   Pos
    Actual  Neg  [  TN    FP ]
            Pos  [  FN    TP ]
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        epoch: Current epoch number (for filename)
        split: Dataset split name ('train', 'val', 'test')
        save_dir: Directory to save plot
    """
    if save_dir is None:
        save_dir = PLOTS_DIR
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Down/Flat (0)', 'Up (1)'],
                yticklabels=['Down/Flat (0)', 'Up (1)'])
    plt.title(f'Confusion Matrix - {split.capitalize()} Set (Epoch {epoch})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Add performance metrics as text
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1:.3f}'
    plt.text(2.5, 0.5, metrics_text, fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Save plot
    save_path = save_dir / f'confusion_matrix_{split}_epoch_{epoch:03d}.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return cm, save_path

def find_optimal_threshold(y_true, y_prob):
    """
    Find optimal classification threshold using ROC curve.
    
    Uses Youden's J statistic: J = TPR - FPR
    Optimal threshold maximizes J (best balance between sensitivity and specificity).
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_prob: Predicted probabilities for positive class
    
    Returns:
        optimal_threshold: Threshold that maximizes Youden's J statistic
    """
    try:
        if len(np.unique(y_true)) < 2:
            return 0.5  # Default threshold if only one class
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        
        # Calculate Youden's J statistic: J = TPR - FPR
        j_scores = tpr - fpr
        
        # Find threshold that maximizes J
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        return optimal_threshold
    except Exception as e:
        print(f"   Could not find optimal threshold: {e}, using default 0.5")
        return 0.5

def calculate_roc_auc(y_true, y_prob):
    """
    Calculate ROC-AUC score.
    
    ROC-AUC (Receiver Operating Characteristic - Area Under Curve) measures:
    - The model's ability to distinguish between classes
    - Range: 0 to 1 (higher is better)
    - 0.5 = random guessing
    - 1.0 = perfect classification
    
    Mathematical Definition:
    ROC curve plots True Positive Rate (TPR) vs False Positive Rate (FPR)
    TPR = TP / (TP + FN)  [Also called Recall or Sensitivity]
    FPR = FP / (FP + TN)  [1 - Specificity]
    
    AUC = ∫ TPR(FPR) d(FPR)  [Area under the ROC curve]
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_prob: Predicted probabilities for positive class
    
    Returns:
        roc_auc: ROC-AUC score (float)
    """
    try:
        # Check if we have both classes in y_true
        if len(np.unique(y_true)) < 2:
            print("   Only one class present in y_true. ROC-AUC not defined.")
            return None
        
        # Calculate ROC-AUC
        roc_auc = roc_auc_score(y_true, y_prob)
        return roc_auc
    except Exception as e:
        print(f"   Could not calculate ROC-AUC: {e}")
        return None

# --- 3. Data Preparation ---

# load_graph_data is now imported from src.utils.graph_loader

def create_target_labels(tickers, dates, lookahead_days):
    """
    Calculates the 5-day ahead return sign (y_{i, t+5}) for all stocks and dates.
    [cite_start]Target: 1 if return > 0, 0 otherwise[cite: 30].
    """
    print(f"\n Calculating {lookahead_days}-day ahead return signs...")
    
    # 1. Load OHLCV data to calculate forward returns
    ohlcv_df = _read_time_series_csv(OHLCV_RAW_FILE)
    
    # Extract only Close prices and restructure
    close_cols = [col for col in ohlcv_df.columns if col.startswith('Close_')]
    close_prices = ohlcv_df[close_cols].copy()
    
    # 2. Calculate forward returns for all stocks
    # Shift the prices back by LOOKAHEAD_DAYS and calculate return
    # R_t+5 = (P_t+5 - P_t) / P_t
    forward_returns_df = (close_prices.shift(-lookahead_days) - close_prices) / close_prices
    
    # 3. Convert to binary labels: 1 (positive return), 0 (otherwise)
    # [cite_start]The proposal uses: 1 if > 0, 0 otherwise [cite: 30]
    target_labels = (forward_returns_df > 0).astype(int)
    
    # 4. Map targets to the graph index (Trading Days)
    targets_dict = {}
    for date in dates:
        # Convert datetime to pd.Timestamp for proper comparison with DatetimeIndex
        date_ts = pd.Timestamp(date)
        if date_ts in target_labels.index:
            # Extract target vector for the trading date
            target_vector = []
            for ticker in tickers:
                col_name = f'Close_{ticker}'
                if col_name in target_labels.columns:
                    target_vector.append(target_labels.loc[date_ts, col_name])
                else:
                    target_vector.append(0) # Safety zero if ticker data is missing
            targets_dict[date] = torch.tensor(target_vector, dtype=torch.long)
            
    print(f" Targets calculated for {len(targets_dict)} trading days.")
    return targets_dict, target_labels.index.unique() # Also return valid trading dates

# --- 3. Training and Evaluation ---

def train(model, optimizer, data, target, criterion):
    """
    Single training step for the baseline GNN.
    
    Args:
        model: The GNN model to train
        optimizer: PyTorch optimizer
        data: Graph data (HeteroData object)
        target: Ground truth labels
        criterion: Loss function (can be standard CE, weighted CE, or focal loss)
    
    Returns:
        loss_value: Scalar loss value for logging
        predictions: Predicted class labels
    """
    model.train()
    optimizer.zero_grad()
    
    # For baseline, we simplify and only use the 'rolling_correlation' edge index
    # (assuming it's the most time-dependent and dominant edge in the HeteroData)
    
    # Extract features and edge index for the BASELINE (simplified Data object logic)
    # NOTE: This assumes that the HeteroData object from Phase 2 has a 'stock' node type 
    # and a 'rolling_correlation' edge type.
    
    # We must flatten HeteroData to Data for the simple GAT/GCN baseline
    try:
        x = data['stock'].x
        # Concatenate edge_index from ALL edge types for simplicity in the baseline
        # In a real scenario, you'd select the most important edge, e.g., 'rolling_correlation'
        
        edge_index_list = []
        for metadata in data.edge_index_dict.keys():
            edge_index_list.append(data[metadata].edge_index)
        
        # If no edges, use self-loops or skip
        if not edge_index_list:
             return 0, 0
        
        # Concatenate all edge indices
        edge_index = torch.cat(edge_index_list, dim=1)
        
    except (AttributeError, KeyError) as e:
        print(f"Skipping training step: Data object structure error: {e}")
        return 0, 0
        
    out = model(x.to(DEVICE), edge_index.to(DEVICE))
    
    # Use the provided criterion (loss function) instead of hardcoded cross_entropy
    # This allows us to switch between standard CE, weighted CE, and focal loss
    loss = criterion(out, target.to(DEVICE))
    
    loss.backward()
    optimizer.step()
    
    # Return loss and predicted labels (for evaluation)
    # Note: During training, we still use argmax for simplicity
    # Threshold optimization is applied during validation/test
    return loss.item(), out.argmax(dim=1)

def evaluate(model, data, target, return_probs=False):
    """
    Single evaluation step.
    
    Args:
        model: The neural network model
        data: Graph data (HeteroData)
        target: Ground truth labels
        return_probs: If True, also return probabilities for ROC-AUC
    
    Returns:
        acc: Accuracy score
        f1: F1 score
        out: Model output logits
        probs: (Optional) Probabilities for positive class
    """
    model.eval()
    
    # Same data extraction logic as in train()
    try:
        x = data['stock'].x
        edge_index_list = [data[metadata].edge_index for metadata in data.edge_index_dict.keys()]
        if not edge_index_list:
             return (0, 0, 0, None) if return_probs else (0, 0, 0)
        edge_index = torch.cat(edge_index_list, dim=1)
    except (AttributeError, KeyError):
        return (0, 0, 0, None) if return_probs else (0, 0, 0)
        
    with torch.no_grad():
        out = model(x.to(DEVICE), edge_index.to(DEVICE))
    
    # Convert true labels and predictions to numpy arrays
    y_true = target.cpu().numpy()
    
    # Get probabilities
    probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()  # Probability of positive class
    
    # Use threshold instead of argmax for better class balance
    # Default threshold is 0.5, but can be optimized
    threshold = 0.5  # Will be optimized in validation/test phase
    y_pred = (probs >= threshold).astype(int)
    
    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    
    if return_probs:
        return acc, f1, out, probs
    
    return acc, f1, out

def run_training_pipeline():
    """Main function to run the time-series training loop."""
    
    # 1. Setup
    # Load the first available graph to determine dimensions and tickers
    graph_files = sorted(list(DATA_GRAPHS_DIR.glob('graph_t_*.pt')))
    if not graph_files:
        print(" CRITICAL: No graph files found. Run Phase 2 first.")
        return
    sample_date = pd.to_datetime(graph_files[0].stem.split('_')[-1])
    sample_graph = load_graph_data(sample_date)
    # Use robust check for HeteroData contents
    if sample_graph is None or not hasattr(sample_graph, 'node_types') or 'stock' not in sample_graph.node_types:
        print(" CRITICAL: Cannot load a sample graph. Phase 2 output files are invalid. STOPPING.")
        return

    # Assuming the features are stored as [N, F] tensor under 'stock'
    INPUT_DIM = sample_graph['stock'].x.shape[1]
    tickers = list(getattr(sample_graph, 'tickers', []))
    if not tickers:
        print(" CRITICAL: Sample graph missing tickers metadata. STOPPING.")
        return
    
    # 2. Get Targets and Date Range
    # Discover all graph dates first (files already listed above)
    graph_dates = [pd.to_datetime(f.stem.split('_')[-1]) for f in graph_files]
    # Load raw data and calculate targets on graph_dates
    targets_dict, _ = create_target_labels(tickers, graph_dates, LOOKAHEAD_DAYS)
    
    # Align training dates: intersect dates that have both a graph AND a target
    training_dates = sorted(list(set(targets_dict.keys()).intersection(graph_dates)))
    
    if not training_dates:
        print(" CRITICAL: No overlapping dates found between graphs and targets. STOPPING.")
        return
        
    # 3. Split Data: Simple time-based split
    TRAIN_END_DATE = training_dates[int(len(training_dates) * 0.70)]
    VAL_END_DATE = training_dates[int(len(training_dates) * 0.85)]
    
    train_dates = [d for d in training_dates if d <= TRAIN_END_DATE]
    val_dates = [d for d in training_dates if TRAIN_END_DATE < d <= VAL_END_DATE]
    test_dates = [d for d in training_dates if d > VAL_END_DATE]
    
    print(f"\n Data Split (Trading Days):")
    print(f"   - Train: {len(train_dates)} days (End: {TRAIN_END_DATE.date()})")
    print(f"   - Val:   {len(val_dates)} days (End: {VAL_END_DATE.date()})")
    print(f"   - Test:  {len(test_dates)} days")
    
    # 4. Setup Loss Function (with class imbalance handling)
    print(f"\n Loss Function Configuration: {LOSS_TYPE}")
    
    if LOSS_TYPE == 'weighted':
        # Compute class weights from training data
        class_weights, class_counts = compute_class_weights(targets_dict, train_dates)
        class_weights = class_weights.to(DEVICE)
        
        # Create weighted cross-entropy loss
        # The weight parameter assigns a manual rescaling weight to each class
        # loss = -w[class] * log(p[class])
        criterion = lambda pred, target: F.cross_entropy(pred, target, weight=class_weights)
        print(f"    Using Weighted Cross-Entropy with computed class weights")
        
    elif LOSS_TYPE == 'focal':
        # Use Focal Loss to handle class imbalance
        criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
        print(f"    Using Focal Loss (alpha={FOCAL_ALPHA}, gamma={FOCAL_GAMMA})")
        print(f"    Focal loss automatically focuses on hard-to-classify examples")
        
    else:  # 'standard'
        # Standard cross-entropy without any class balancing
        criterion = lambda pred, target: F.cross_entropy(pred, target)
        print(f"     Using Standard Cross-Entropy (no class balancing)")
        print(f"    Consider using 'weighted' or 'focal' if classes are imbalanced")
    
    # 5. Model and Optimizer Setup
    model = BaselineGNN(INPUT_DIM, HIDDEN_DIM, OUT_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 5b. Learning Rate Scheduler Setup
    scheduler = None
    if ENABLE_LR_SCHEDULER:
        if LR_SCHEDULER_TYPE == 'plateau':
            # ReduceLROnPlateau: Reduce LR when validation metric plateaus
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='max',  # Maximize validation F1
                factor=LR_SCHEDULER_FACTOR,
                patience=LR_SCHEDULER_PATIENCE,
                min_lr=LR_SCHEDULER_MIN_LR
            )
            print(f"\n Learning Rate Scheduler: ReduceLROnPlateau")
            print(f"   - Patience: {LR_SCHEDULER_PATIENCE} epochs")
            print(f"   - Factor: {LR_SCHEDULER_FACTOR}")
            print(f"   - Min LR: {LR_SCHEDULER_MIN_LR}")
        
        elif LR_SCHEDULER_TYPE == 'step':
            # StepLR: Reduce LR every N epochs
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=LR_SCHEDULER_PATIENCE,
                gamma=LR_SCHEDULER_FACTOR
            )
            print(f"\n Learning Rate Scheduler: StepLR")
            print(f"   - Step size: {LR_SCHEDULER_PATIENCE} epochs")
            print(f"   - Gamma: {LR_SCHEDULER_FACTOR}")
        
        elif LR_SCHEDULER_TYPE == 'exponential':
            # ExponentialLR: Exponentially decay LR
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=LR_SCHEDULER_FACTOR
            )
            print(f"\n Learning Rate Scheduler: ExponentialLR")
            print(f"   - Gamma: {LR_SCHEDULER_FACTOR}")
    
    # 5c. Early Stopping Setup
    early_stop_counter = 0
    early_stop_triggered = False
    if ENABLE_EARLY_STOPPING:
        print(f"\n⏱ Early Stopping Enabled:")
        print(f"   - Patience: {EARLY_STOP_PATIENCE} epochs")
        print(f"   - Min delta: {EARLY_STOP_MIN_DELTA}")
    
    # 6. Initialize Training Metrics and State
    # Metrics dictionary stores the entire training history
    metrics = {
        'train_loss': [],      # Training loss per epoch
        'val_acc': [],         # Validation accuracy per epoch
        'val_f1': [],          # Validation F1-score per epoch
        'val_roc_auc': [],     # Validation ROC-AUC per epoch (if enabled)
        'epoch_times': [],     # Time taken per epoch (seconds)
    }
    
    start_epoch = 1  # Default: start from epoch 1
    best_val_f1 = 0.0
    
    # 6b. Initialize TensorBoard Writer
    writer = None
    if ENABLE_TENSORBOARD and TENSORBOARD_AVAILABLE:
        # Create unique run name with timestamp
        run_name = f"baseline_{LOSS_TYPE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        writer = SummaryWriter(log_dir=TENSORBOARD_DIR / run_name)
        print(f"\n TensorBoard initialized: {run_name}")
        print(f"   Run: tensorboard --logdir={TENSORBOARD_DIR}")
        
        # Log hyperparameters
        hparams = {
            'hidden_dim': HIDDEN_DIM,
            'learning_rate': LEARNING_RATE,
            'loss_type': LOSS_TYPE,
            'early_stopping': ENABLE_EARLY_STOPPING,
            'lr_scheduler': ENABLE_LR_SCHEDULER,
        }
        writer.add_text('Hyperparameters', str(hparams), 0)
    
    # 7. Resume from Checkpoint (if enabled)
    if RESUME_FROM_CHECKPOINT and ENABLE_CHECKPOINTING:
        checkpoint_path = CHECKPOINT_DIR / 'checkpoint_latest.pt'
        if checkpoint_path.exists():
            print("\n Resuming from checkpoint...")
            start_epoch, metrics = load_checkpoint(checkpoint_path, model, optimizer)
            
            # Restore best validation F1 from metrics history
            if metrics.get('val_f1'):
                best_val_f1 = max(metrics['val_f1'])
                print(f"   Restored best validation F1: {best_val_f1:.4f}")
        else:
            print(f"\n No checkpoint found at {checkpoint_path}")
            print("  Starting training from scratch...")
    
    # 8. Training Loop (Transductive/Sequential)
    print("\n Starting Sequential Training...")
    print(f"   Training from epoch {start_epoch} to {NUM_EPOCHS}")
    
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        epoch_start_time = time.time()  # Track epoch duration
        total_loss = 0
        
        # --- Train Phase ---
        for date in tqdm(train_dates, desc=f"Epoch {epoch} Training"):
            data = load_graph_data(date)
            target = targets_dict.get(date)
            if data and target is not None:
                loss, _ = train(model, optimizer, data, target, criterion)
                total_loss += loss
        
        avg_loss = total_loss / len(train_dates)

        # --- Validation Phase ---
        val_accs, val_f1s = [], []
        all_val_true, all_val_pred, all_val_probs = [], [], []
        
        for date in val_dates:
            data = load_graph_data(date)
            target = targets_dict.get(date)
            if data and target is not None:
                # Get predictions and probabilities
                acc, f1, _, probs = evaluate(model, data, target, return_probs=True)
                val_accs.append(acc)
                val_f1s.append(f1)
                
                # Collect for ROC-AUC and optimal threshold finding
                if probs is not None:
                    all_val_true.extend(target.cpu().numpy())
                    all_val_probs.extend(probs)
        
        # Find optimal threshold on validation set
        optimal_threshold = 0.5  # Default
        if len(all_val_true) > 0 and len(all_val_probs) > 0:
            optimal_threshold = find_optimal_threshold(
                np.array(all_val_true),
                np.array(all_val_probs)
            )
            if epoch == start_epoch or epoch % 5 == 0:  # Print every 5 epochs or first epoch
                print(f"    Optimal threshold (validation): {optimal_threshold:.4f}")
        
        # Re-evaluate with optimal threshold
        val_accs_opt, val_f1s_opt = [], []
        for date in val_dates:
            data = load_graph_data(date)
            target = targets_dict.get(date)
            if data and target is not None:
                # Get probabilities
                _, _, out, probs = evaluate(model, data, target, return_probs=True)
                if probs is not None:
                    y_true = target.cpu().numpy()
                    y_pred_opt = (probs >= optimal_threshold).astype(int)
                    acc_opt = accuracy_score(y_true, y_pred_opt)
                    f1_opt = f1_score(y_true, y_pred_opt, average='binary', zero_division=0)
                    val_accs_opt.append(acc_opt)
                    val_f1s_opt.append(f1_opt)
                    all_val_pred.extend(y_pred_opt)
        
        avg_val_acc = np.mean(val_accs_opt) if val_accs_opt else np.mean(val_accs)
        avg_val_f1 = np.mean(val_f1s_opt) if val_f1s_opt else np.mean(val_f1s)
        
        # Calculate ROC-AUC if enabled
        avg_val_roc_auc = None
        if ENABLE_ROC_AUC and len(all_val_true) > 0:
            avg_val_roc_auc = calculate_roc_auc(
                np.array(all_val_true),
                np.array(all_val_probs)
            )
        
        # Calculate epoch duration
        epoch_duration = time.time() - epoch_start_time
        
        # Store metrics for this epoch
        metrics['train_loss'].append(avg_loss)
        metrics['val_acc'].append(avg_val_acc)
        metrics['val_f1'].append(avg_val_f1)
        metrics['val_roc_auc'].append(avg_val_roc_auc if avg_val_roc_auc else 0.0)
        metrics['epoch_times'].append(epoch_duration)

        # Get current learning rate for display
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch summary
        print_str = f"Epoch {epoch:02d} | Train Loss: {avg_loss:.4f} | Val Acc: {avg_val_acc:.4f} | Val F1: {avg_val_f1:.4f}"
        if avg_val_roc_auc is not None:
            print_str += f" | ROC-AUC: {avg_val_roc_auc:.4f}"
        print_str += f" | Time: {epoch_duration:.1f}s | LR: {current_lr:.2e}"
        print(print_str)
        
        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar('Loss/train', avg_loss, epoch)
            writer.add_scalar('Accuracy/val', avg_val_acc, epoch)
            writer.add_scalar('F1/val', avg_val_f1, epoch)
            if avg_val_roc_auc is not None:
                writer.add_scalar('ROC-AUC/val', avg_val_roc_auc, epoch)
            writer.add_scalar('Learning_Rate', current_lr, epoch)

        # Determine if this is the best model so far (with min_delta threshold)
        improvement = avg_val_f1 - best_val_f1
        is_best = improvement > EARLY_STOP_MIN_DELTA if ENABLE_EARLY_STOPPING else avg_val_f1 > best_val_f1
        
        # Save best model based on F1 score (important metric for imbalanced finance data)
        if is_best:
            best_val_f1 = avg_val_f1
            torch.save(model.state_dict(), MODELS_DIR / 'baseline_gcn_model.pt')
            print(f"   New Best Model! F1: {best_val_f1:.4f}")
        
        # Save checkpoint (full training state)
        if ENABLE_CHECKPOINTING:
            # Save checkpoint at regular intervals or if it's the best model
            should_save = is_best or (epoch % SAVE_CHECKPOINT_EVERY == 0)
            
            if should_save:
                save_checkpoint(
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    metrics=metrics,
                    checkpoint_dir=CHECKPOINT_DIR,
                    is_best=is_best
                )
            
            # Periodically clean up old checkpoints to save disk space
            if epoch % (SAVE_CHECKPOINT_EVERY * 2) == 0:
                cleanup_old_checkpoints(CHECKPOINT_DIR, keep_last_n=5)
        
        # Update Learning Rate Scheduler
        if ENABLE_LR_SCHEDULER and scheduler is not None:
            old_lr = optimizer.param_groups[0]['lr']
            
            if LR_SCHEDULER_TYPE == 'plateau':
                # ReduceLROnPlateau needs the validation metric
                scheduler.step(avg_val_f1)
            else:
                # Other schedulers just step
                scheduler.step()
            
            # Check if LR was reduced
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < old_lr:
                print(f"   Learning rate reduced: {old_lr:.2e} → {new_lr:.2e}")
        
        # Early Stopping Check
        if ENABLE_EARLY_STOPPING:
            if is_best:
                # Reset counter if we found a better model
                early_stop_counter = 0
            else:
                # Increment counter if no improvement
                early_stop_counter += 1
                print(f"  ⏱ No improvement for {early_stop_counter}/{EARLY_STOP_PATIENCE} epochs")
                
                if early_stop_counter >= EARLY_STOP_PATIENCE:
                    print(f"\n Early stopping triggered after {epoch} epochs")
                    print(f"   Best validation F1: {best_val_f1:.4f}")
                    early_stop_triggered = True
                    break  # Exit training loop
    
    # 9. Training Complete - Save Final Summary
    print("\n" + "=" * 60)
    if early_stop_triggered:
        print(" Training Stopped Early!")
    else:
        print(" Training Complete!")
    print("=" * 60)
    
    if ENABLE_CHECKPOINTING:
        print(f"\n Training Summary:")
        print(f"   Total epochs trained: {len(metrics['train_loss'])}")
        print(f"   Best validation F1: {best_val_f1:.4f}")
        print(f"   Total training time: {sum(metrics['epoch_times']):.1f}s ({sum(metrics['epoch_times'])/60:.1f}min)")
        print(f"   Average time per epoch: {np.mean(metrics['epoch_times']):.1f}s")
        
        # Display training curve summary
        if len(metrics['train_loss']) > 0:
            print(f"\n   Training Loss: {metrics['train_loss'][0]:.4f} → {metrics['train_loss'][-1]:.4f}")
            print(f"   Validation Acc: {metrics['val_acc'][0]:.4f} → {metrics['val_acc'][-1]:.4f}")
            print(f"   Validation F1:  {metrics['val_f1'][0]:.4f} → {metrics['val_f1'][-1]:.4f}")
        
        print(f"\n Checkpoints saved in: {CHECKPOINT_DIR}")
        print(f"   - Best model: checkpoint_best.pt")
        print(f"   - Latest state: checkpoint_latest.pt")
    
    # 10. Testing (Final Evaluation)
    print("\n" + "=" * 60)
    print(" Final Testing Phase")
    print("=" * 60)
    
    model.load_state_dict(torch.load(MODELS_DIR / 'baseline_gcn_model.pt', weights_only=False))
    
    # Find optimal threshold on validation set (use last validation run)
    print("\n Finding optimal classification threshold on validation set...")
    all_val_true_final, all_val_probs_final = [], []
    for date in val_dates:
        data = load_graph_data(date)
        target = targets_dict.get(date)
        if data and target is not None:
            _, _, _, probs = evaluate(model, data, target, return_probs=True)
            if probs is not None:
                all_val_true_final.extend(target.cpu().numpy())
                all_val_probs_final.extend(probs)
    
    optimal_threshold = 0.5  # Default
    if len(all_val_true_final) > 0 and len(all_val_probs_final) > 0:
        optimal_threshold = find_optimal_threshold(
            np.array(all_val_true_final),
            np.array(all_val_probs_final)
        )
        print(f"    Optimal threshold: {optimal_threshold:.4f} (default: 0.5)")
    else:
        print(f"     Could not find optimal threshold, using default: 0.5")
    
    # Test with optimal threshold
    test_accs, test_f1s = [], []
    all_test_true, all_test_pred, all_test_probs = [], [], []
    
    for date in test_dates:
        data = load_graph_data(date)
        target = targets_dict.get(date)
        if data and target is not None:
            _, _, out, probs = evaluate(model, data, target, return_probs=True)
            
            # Use optimal threshold for predictions
            if probs is not None:
                y_true = target.cpu().numpy()
                y_pred_opt = (probs >= optimal_threshold).astype(int)
                acc_opt = accuracy_score(y_true, y_pred_opt)
                f1_opt = f1_score(y_true, y_pred_opt, average='binary', zero_division=0)
                test_accs.append(acc_opt)
                test_f1s.append(f1_opt)
                
                # Collect for final metrics
                all_test_true.extend(y_true)
                all_test_pred.extend(y_pred_opt)
                all_test_probs.extend(probs)
    
    # Calculate final test metrics
    avg_test_acc = np.mean(test_accs)
    avg_test_f1 = np.mean(test_f1s)
    
    print("\n" + "=" * 60)
    print(f" Final Test Results (Averaged over {len(test_dates)} days):")
    print(f"   - Test Accuracy: {avg_test_acc:.4f}")
    print(f"   - Test F1 Score: {avg_test_f1:.4f}")
    
    # Calculate and display ROC-AUC
    if ENABLE_ROC_AUC and len(all_test_true) > 0:
        test_roc_auc = calculate_roc_auc(
            np.array(all_test_true),
            np.array(all_test_probs)
        )
        if test_roc_auc is not None:
            print(f"   - Test ROC-AUC: {test_roc_auc:.4f}")
            
            # Log to TensorBoard
            if writer is not None:
                writer.add_scalar('ROC-AUC/test', test_roc_auc, 0)
    
    # Generate confusion matrix
    if ENABLE_CONFUSION_MATRIX and len(all_test_true) > 0:
        print(f"\n Generating confusion matrix...")
        cm, cm_path = plot_confusion_matrix(
            np.array(all_test_true),
            np.array(all_test_pred),
            epoch=len(metrics['train_loss']),
            split='test',
            save_dir=PLOTS_DIR
        )
        print(f"    Confusion matrix saved: {cm_path.name}")
        
        # Log confusion matrix to TensorBoard
        if writer is not None:
            fig = plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Down/Flat (0)', 'Up (1)'],
                       yticklabels=['Down/Flat (0)', 'Up (1)'])
            plt.title('Confusion Matrix - Test Set')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            writer.add_figure('Confusion_Matrix/test', fig, 0)
            plt.close(fig)
        
        # Print classification report
        print(f"\n Classification Report:")
        print(classification_report(
            all_test_true,
            all_test_pred,
            target_names=['Down/Flat (0)', 'Up (1)'],
            digits=4
        ))
    
    print("=" * 60)
    
    # Close TensorBoard writer
    if writer is not None:
        writer.close()
        print(f"\n TensorBoard logs saved to: {TENSORBOARD_DIR}")
        print(f"   View with: tensorboard --logdir={TENSORBOARD_DIR}")


if __name__ == '__main__':
    run_training_pipeline()
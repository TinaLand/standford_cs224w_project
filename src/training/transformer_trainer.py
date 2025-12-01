# phase4_core_training.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv
from torch_geometric.data import HeteroData
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
import os
import sys
from torch.nn.utils import clip_grad_norm_

# FIX: Allow PyTorch to safely load torch-geometric objects (PyTorch >= 2.6)
import torch.serialization
from torch_geometric.data.storage import BaseStorage, NodeStorage, EdgeStorage, GlobalStorage
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([BaseStorage, NodeStorage, EdgeStorage, GlobalStorage])

# --- Configuration & Hyperparameters (from Proposal) ---
# NOTE: This file lives in `src/training/`, so the project root is three levels up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_GRAPHS_DIR = PROJECT_ROOT / "data" / "graphs"
MODELS_DIR = PROJECT_ROOT / "models"
OHLCV_RAW_FILE = PROJECT_ROOT / "data" / "raw" / "stock_prices_ohlcv_raw.csv"

# Hyperparameters (IMPROVED for better performance)
LOOKAHEAD_DAYS = 5              # Target: 5-day-ahead return sign
HIDDEN_CHANNELS = 512           # Hidden size (INCREASED from 256 to 512 for better capacity)
NUM_LAYERS = 3                  # Number of layers (INCREASED from 2 to 3 for deeper model)
NUM_HEADS = 8                   # Number of attention heads (INCREASED from 4 to 8 for better attention)
OUT_CHANNELS = 2                # Binary classification: Up/Down
LEARNING_RATE = 0.0008          # Learning rate (ADJUSTED: slightly reduced for more stable training)
NUM_EPOCHS = 40                 # Increased epochs (from 30 to 40) for more training
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class Imbalance Handling Configuration
LOSS_TYPE = 'focal'  # Options: 'standard', 'focal'
# IMPROVED: Adjusted for better Down/Flat class handling
# Down/Flat class has very low recall (1.86%), need stronger focus
FOCAL_ALPHA = 0.85   # Weight for minority class (INCREASED from 0.80 to 0.85 for Down/Flat class)
FOCAL_GAMMA = 3.0    # Focusing parameter (INCREASED from 2.5 to 3.0 for harder focus on hard examples)
# Use class weights in addition to focal loss for better balance
USE_CLASS_WEIGHTS = True  # Enable class weights for additional balancing

# Multi-task Learning Configuration (Classification + Regression on returns)
ENABLE_MULTI_TASK = True
REG_LOSS_WEIGHT = 0.5          # Weight for regression loss in total loss

# Time-Aware Modeling Configuration
ENABLE_TIME_AWARE = True        # Enable time-aware positional encoding
TIME_PE_DIM = 16                # Dimension for time positional encoding

# Early Stopping Configuration
ENABLE_EARLY_STOPPING = True
EARLY_STOP_PATIENCE = 12        # Increased from 10 to 12 to allow more training epochs
EARLY_STOP_MIN_DELTA = 0.0001   # Minimum improvement threshold

# Learning Rate Scheduler Configuration
ENABLE_LR_SCHEDULER = True
LR_SCHEDULER_PATIENCE = 4       # Increased from 3 to 4 for more stable training
LR_SCHEDULER_FACTOR = 0.5       # Multiply LR by this factor
LR_SCHEDULER_MIN_LR = 1e-6      # Minimum learning rate

# Additional Training Improvements
DROPOUT_RATE = 0.3              # Dropout rate for regularization (NEW)
WEIGHT_DECAY = 1e-5             # L2 regularization (NEW)
GRAD_CLIP_NORM = 1.0            # Gradient clipping norm (NEW)

# --- Helper Utilities ---

class FocalLoss(torch.nn.Module):
    """
    Focal Loss for addressing class imbalance in classification tasks.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    IMPROVED: Can optionally use class weights for additional balancing.
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean', weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight  # Class weights for additional balancing
    
    def forward(self, inputs, targets):
        # Use weighted cross-entropy if class weights provided
        if self.weight is not None:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        p = F.softmax(inputs, dim=1)
        p_t = p.gather(1, targets.view(-1, 1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = torch.where(targets == 1, 
                             torch.tensor(self.alpha, device=inputs.device),
                             torch.tensor(1 - self.alpha, device=inputs.device))
        focal_loss = alpha_t * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def find_optimal_threshold(y_true, y_prob):
    """
    Find optimal classification threshold using F1 score (better for imbalanced data).
    IMPROVED: Uses F1 score instead of Youden's J for better class balance.
    """
    try:
        if len(np.unique(y_true)) < 2:
            return 0.5
        
        # Check if all probabilities are the same
        if np.std(y_prob) < 1e-6:
            return 0.5
        
        # Try multiple methods and pick the best
        best_threshold = 0.5
        best_f1 = 0.0
        
        # Method 1: F1 score optimization (better for imbalanced data)
        thresholds_to_try = np.linspace(0.3, 0.7, 50)  # Focus on reasonable range
        for threshold in thresholds_to_try:
            y_pred = (y_prob >= threshold).astype(int)
            if len(np.unique(y_pred)) < 2:  # All predictions same
                continue
            try:
                f1 = f1_score(y_true, y_pred, average='macro')  # Use macro F1 for class balance
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            except:
                continue
        
        # Method 2: ROC curve with Youden's J (fallback)
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            valid_mask = np.isfinite(thresholds)
            if np.any(valid_mask):
                valid_thresholds = thresholds[valid_mask]
                valid_fpr = fpr[valid_mask]
                valid_tpr = tpr[valid_mask]
                j_scores = valid_tpr - valid_fpr
                optimal_idx = np.argmax(j_scores)
                roc_threshold = valid_thresholds[optimal_idx]
                
                # Check if ROC threshold gives better F1
                if np.isfinite(roc_threshold) and 0 <= roc_threshold <= 1:
                    y_pred_roc = (y_prob >= roc_threshold).astype(int)
                    if len(np.unique(y_pred_roc)) >= 2:
                        f1_roc = f1_score(y_true, y_pred_roc, average='macro')
                        if f1_roc > best_f1:
                            best_f1 = f1_roc
                            best_threshold = roc_threshold
        except:
            pass
        
        # Ensure threshold is in valid range [0, 1]
        if not np.isfinite(best_threshold) or best_threshold < 0 or best_threshold > 1:
            return 0.5
        
        return float(best_threshold)
    except Exception as e:
        print(f"  ⚠️ Could not find optimal threshold: {e}, using default 0.5")
        return 0.5

def _read_time_series_csv(path: Path) -> pd.DataFrame:
    """
    Robust CSV reader that ensures a DatetimeIndex named 'Date'.
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

# Mini-batch training settings
ENABLE_MINI_BATCH = False      # Enable neighbor sampling for large graphs (requires torch-sparse, pyg-lib)
BATCH_SIZE = 128               # Number of target nodes per batch
NUM_NEIGHBORS = [15, 10]       # Neighbors to sample per layer
ENABLE_AMP = torch.cuda.is_available()  # Enable automatic mixed precision if CUDA available
GRAD_CLIP_MAX_NORM = GRAD_CLIP_NORM  # Use the new constant
MODEL_SAVE_NAME = 'core_transformer_model.pt'

# Ensure model directory exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --- 1. Model Definition: Core Role-Aware Graph Transformer ---

# Import the enhanced components
from src.models.components.pearl_embedding import PEARLPositionalEmbedding
from src.models.components.transformer_layer import RelationAwareGATv2Conv, RelationAwareAggregator

# Import PyTorch Geometric's neighbor sampling (requires sparse libraries)
try:
    from torch_geometric.loader import NeighborLoader
    MINI_BATCH_AVAILABLE = True
except ImportError:
    MINI_BATCH_AVAILABLE = False
    NeighborLoader = None

class TimePositionalEncoding(torch.nn.Module):
    """
    Time-aware positional encoding based on trading date.
    Encodes temporal information (day of week, month, year, etc.) into embeddings.
    """
    def __init__(self, pe_dim):
        super().__init__()
        self.pe_dim = pe_dim
        
        # Learnable embeddings for different time features
        # Day of week (0-6) -> 7 categories
        self.day_of_week_emb = torch.nn.Embedding(7, pe_dim // 4)
        # Month (1-12) -> 12 categories
        self.month_emb = torch.nn.Embedding(12, pe_dim // 4)
        # Quarter (1-4) -> 4 categories
        self.quarter_emb = torch.nn.Embedding(4, pe_dim // 4)
        # Year (normalized) -> continuous, use sinusoidal encoding
        self.year_proj = torch.nn.Linear(1, pe_dim // 4)
        
    def forward(self, date_tensor):
        """
        Args:
            date_tensor: Tensor of shape [N] with timestamps (days since epoch or similar)
        Returns:
            time_pe: Tensor of shape [N, pe_dim] with time positional encodings
        """
        # Convert to datetime if needed (assuming date_tensor is days since epoch)
        # For simplicity, we'll use a normalized timestamp
        # In practice, you'd extract actual date features
        
        # Normalize timestamp to [0, 1] range (assuming dates are in reasonable range)
        # This is a placeholder - in real implementation, extract actual date features
        normalized_time = date_tensor.float() / 10000.0  # Normalize
        
        # Extract time features (simplified - in practice use actual date parsing)
        # For now, use sinusoidal encoding for continuous time
        time_features = []
        
        # Sinusoidal encoding for continuous time
        div_term = torch.exp(torch.arange(0, self.pe_dim, 2, device=date_tensor.device).float() * 
                           -(torch.log(torch.tensor(10000.0, device=date_tensor.device)) / self.pe_dim))
        time_features.append(torch.sin(normalized_time.unsqueeze(-1) * div_term))
        time_features.append(torch.cos(normalized_time.unsqueeze(-1) * div_term))
        
        # Concatenate all time features
        time_pe = torch.cat(time_features, dim=-1)
        
        # Ensure output dimension matches pe_dim
        if time_pe.shape[-1] != self.pe_dim:
            # Project to correct dimension
            if not hasattr(self, 'time_proj'):
                self.time_proj = torch.nn.Linear(time_pe.shape[-1], self.pe_dim).to(time_pe.device)
            time_pe = self.time_proj(time_pe)
        
        return time_pe

class RoleAwareGraphTransformer(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, num_heads, enable_time_aware=True):
        super().__init__()
        
        # PEARL related dimension
        self.PE_DIM = 32
        self.enable_time_aware = enable_time_aware
        
        # Time-aware positional encoding
        if enable_time_aware:
            self.TIME_PE_DIM = 16
            self.time_pe = TimePositionalEncoding(self.TIME_PE_DIM)
            total_in_dim = in_dim + self.PE_DIM + self.TIME_PE_DIM
        else:
            total_in_dim = in_dim + self.PE_DIM 
        
        # (A) PEARL Positional Embedding Block
        self.pearl_embedding = PEARLPositionalEmbedding(in_dim, self.PE_DIM)
        
        # (B) Graph Transformer Layers (HeteroConv)
        # Defined based on Phase 2 graph construction
        metadata = [
            ('stock', 'sector_industry', 'stock'),
            ('stock', 'competitor', 'stock'),
            ('stock', 'supply_chain', 'stock'),
            ('stock', 'rolling_correlation', 'stock'),
            ('stock', 'fund_similarity', 'stock')
        ]
        
        # Relation-aware convolution layers
        self.convs = torch.nn.ModuleList()
        self.relation_aggregators = torch.nn.ModuleList()
        
        for i in range(num_layers):
            in_d = total_in_dim if i == 0 else hidden_dim
            
            # Create relation-aware convolutions for each edge type
            conv = HeteroConv({
                edge_type: RelationAwareGATv2Conv(
                    in_d, 
                    hidden_dim // num_heads, 
                    heads=num_heads, 
                    dropout=DROPOUT_RATE,
                    edge_type=edge_type
                )
                for edge_type in metadata
            }, aggr='sum')
            
            # Add relation-aware aggregator
            aggregator = RelationAwareAggregator(hidden_dim, metadata)
            
            self.convs.append(conv)
            self.relation_aggregators.append(aggregator)
            
        # (C) Output Classifier (classification head)
        self.lin_out = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, out_dim)
        )

        # (D) Regression head for multi-task learning (predict future returns)
        # Single scalar per node: 5-day ahead return
        self.lin_reg = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, data, date_tensor=None):
        """
        Args:
            data: HeteroData graph object
            date_tensor: Optional tensor of shape [N] with timestamps for time-aware encoding
        """
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        
        # 1. Generate/Concatenate PEARL Embeddings
        x = x_dict['stock']
        pearl_pe = self.pearl_embedding(x, edge_index_dict)
        
        # 2. Add time-aware positional encoding if enabled
        if self.enable_time_aware and date_tensor is not None:
            time_pe = self.time_pe(date_tensor)
            x_with_pe = torch.cat([x, pearl_pe, time_pe], dim=1)
        else:
            x_with_pe = torch.cat([x, pearl_pe], dim=1)
        
        x_dict['stock'] = x_with_pe
        
        # 2. Relation-aware Graph Transformer Layers
        for layer_idx, (conv, aggregator) in enumerate(zip(self.convs, self.relation_aggregators)):
            # Apply convolution to get outputs for each relation
            conv_output = conv(x_dict, edge_index_dict)
            
            # Collect outputs from different relations for aggregation
            relation_outputs = {}
            for edge_type in edge_index_dict.keys():
                if 'stock' in conv_output:
                    relation_outputs[edge_type] = conv_output['stock']
            
            # Apply relation-aware aggregation if we have relation outputs
            if relation_outputs and len(relation_outputs) > 1:
                x_dict['stock'] = aggregator(relation_outputs)
            else:
                x_dict = conv_output
            
            # Apply activations and dropout
            x_dict['stock'] = x_dict['stock'].relu()
            x_dict['stock'] = F.dropout(x_dict['stock'], p=DROPOUT_RATE, training=self.training)
        
        # 3. Node-Level Output
        out = self.lin_out(x_dict['stock'])
        return out
    
    def get_embeddings(self, data, date_tensor=None):
        """
        Extracts node embeddings from the last graph transformer layer
        before the final classification head.
        Used for RL state representation.
        """
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        
        # 1. Generate/Concatenate PEARL Embeddings
        x = x_dict['stock']
        pearl_pe = self.pearl_embedding(x, edge_index_dict)
        
        # 2. Add time-aware positional encoding if enabled
        if self.enable_time_aware and date_tensor is not None:
            time_pe = self.time_pe(date_tensor)
            x_with_pe = torch.cat([x, pearl_pe, time_pe], dim=1)
        else:
            x_with_pe = torch.cat([x, pearl_pe], dim=1)
        
        x_dict['stock'] = x_with_pe
        
        # 2. Relation-aware Graph Transformer Layers
        for layer_idx, (conv, aggregator) in enumerate(zip(self.convs, self.relation_aggregators)):
            conv_output = conv(x_dict, edge_index_dict)
            
            relation_outputs = {}
            for edge_type in edge_index_dict.keys():
                if 'stock' in conv_output:
                    relation_outputs[edge_type] = conv_output['stock']
            
            if relation_outputs and len(relation_outputs) > 1:
                x_dict['stock'] = aggregator(relation_outputs)
            else:
                x_dict = conv_output
            
            x_dict['stock'] = x_dict['stock'].relu()
            x_dict['stock'] = F.dropout(x_dict['stock'], p=DROPOUT_RATE, training=self.training)
        
        # Return embeddings before final linear layer
        return x_dict['stock']

    def forward_regression(self, data, date_tensor=None):
        """
        Regression forward pass using the same backbone.
        Returns a vector of shape [N] with predicted future returns.
        """
        embeddings = self.get_embeddings(data, date_tensor=date_tensor)
        reg_out = self.lin_reg(embeddings).squeeze(-1)
        return reg_out

# --- 2. Data Utilities (Copied from Phase 3 for consistency) ---

def load_graph_data(date):
    """Loads a single graph snapshot for the given date."""
    date_str = date.strftime('%Y%m%d')
    filepath = DATA_GRAPHS_DIR / f'graph_t_{date_str}.pt'
    
    if not filepath.exists():
        return None
    
    try:
        # Load the PyG graph object with weights_only=False for PyTorch 2.6+
        data = torch.load(filepath, weights_only=False)
        return data
    except Exception as e:
        # print(f"❌ Error loading graph {filepath.name}: {e}") # Keep this silent during loop
        return None

def create_target_labels(tickers, dates, lookahead_days):
    """
    Calculates the 5-day ahead return **sign** (classification label)
    and **continuous return** (regression target) for all stocks and dates.
    """
    print(f"\n🏷️ Calculating {lookahead_days}-day ahead return signs and returns...")
    
    ohlcv_df = _read_time_series_csv(OHLCV_RAW_FILE)
    
    close_cols = [col for col in ohlcv_df.columns if col.startswith('Close_')]
    close_prices = ohlcv_df[close_cols].copy()
    
    forward_returns_df = (close_prices.shift(-lookahead_days) - close_prices) / close_prices
    
    # Classification labels: 1 if return > 0, else 0
    target_labels = (forward_returns_df > 0).astype(int)
    
    targets_class_dict = {}
    targets_reg_dict = {}
    for date in dates:
        date_str = date.strftime('%Y-%m-%d')
        if date_str in target_labels.index:
            class_vec = []
            reg_vec = []
            for ticker in tickers:
                col_name = f'Close_{ticker}'
                if col_name in target_labels.columns and col_name in forward_returns_df.columns:
                    class_vec.append(target_labels.loc[date_str, col_name])
                    reg_vec.append(forward_returns_df.loc[date_str, col_name])
                else:
                    class_vec.append(0)
                    reg_vec.append(0.0)
            targets_class_dict[date] = torch.tensor(class_vec, dtype=torch.long)
            targets_reg_dict[date] = torch.tensor(reg_vec, dtype=torch.float32)
            
    print(f"✅ Targets calculated for {len(targets_class_dict)} trading days.")
    return targets_class_dict, targets_reg_dict

# --- 3. Training and Evaluation Functions ---

def _backward_step(loss, model, optimizer, scaler=None, max_grad_norm=None):
    """Handles backward pass with optional AMP scaler and gradient clipping."""
    if scaler is not None and scaler.is_enabled():
        scaler.scale(loss).backward()
        if max_grad_norm and max_grad_norm > 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if max_grad_norm and max_grad_norm > 0:
            clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()


def train(model, optimizer, data, targets_class, targets_reg=None, date_tensor=None, scaler=None, amp_enabled=False, max_grad_norm=None, class_weights=None):
    """Single training step for the Core GNN (HeteroData input).
    
    If ENABLE_MULTI_TASK and targets_reg is provided, uses a combined loss:
        L = L_class + REG_LOSS_WEIGHT * L_reg
    where L_reg is Smooth L1 loss on future returns.
    
    Args:
        date_tensor: Optional tensor of shape [N] with timestamps for time-aware encoding
        class_weights: Optional tensor of class weights for balancing
    """
    model.train()
    optimizer.zero_grad()
    
    # Ensure edge indices are long
    for metadata in data.edge_index_dict.keys():
        data[metadata].edge_index = data[metadata].edge_index.to(torch.long)
        
    with torch.cuda.amp.autocast(enabled=amp_enabled):
        data_device = data.to(DEVICE)
        if date_tensor is not None:
            date_tensor = date_tensor.to(DEVICE)
        out = model(data_device, date_tensor=date_tensor)

        # --- Classification loss ---
        if LOSS_TYPE == 'focal':
            # Use class weights in Focal Loss if available
            weight_arg = class_weights if USE_CLASS_WEIGHTS and class_weights is not None else None
            criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, weight=weight_arg)
            loss_class = criterion(out, targets_class.to(DEVICE))
        else:
            # Use class weights if available (for standard cross-entropy)
            weight_arg = class_weights if USE_CLASS_WEIGHTS and class_weights is not None else None
            loss_class = F.cross_entropy(out, targets_class.to(DEVICE), weight=weight_arg)

        # --- Optional regression loss (multi-task learning) ---
        if ENABLE_MULTI_TASK and targets_reg is not None:
            preds_reg = model.forward_regression(data_device, date_tensor=date_tensor)
            # Ensure shapes match
            reg_targets = targets_reg.to(DEVICE).view_as(preds_reg)
            loss_reg = F.smooth_l1_loss(preds_reg, reg_targets)
            loss = loss_class + REG_LOSS_WEIGHT * loss_reg
        else:
            loss = loss_class
    
    _backward_step(loss, model, optimizer, scaler=scaler, max_grad_norm=max_grad_norm)
    
    return loss.item(), out.detach().argmax(dim=1)


def evaluate(model, data, targets, return_probs=False, threshold=0.5, date_tensor=None):
    """Single evaluation step.
    
    Args:
        date_tensor: Optional tensor of shape [N] with timestamps for time-aware encoding
    """
    model.eval()
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=ENABLE_AMP):
            data_device = data.to(DEVICE)
            if date_tensor is not None:
                date_tensor = date_tensor.to(DEVICE)
            out = model(data_device, date_tensor=date_tensor)
    
    y_true = targets.cpu().numpy()
    probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()  # Probability of positive class
    
    # Use threshold instead of argmax for better class balance
    y_pred = (probs >= threshold).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    
    if return_probs:
        return acc, f1, out, probs
    return acc, f1


def train_with_sampling(model, optimizer, data, targets, loader, scaler=None, amp_enabled=False, max_grad_norm=None):
    """Training step with neighbor sampling for large graphs."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in loader:
        optimizer.zero_grad()
        batch = batch.to(DEVICE)
        
        # Get batch size for target nodes
        batch_size = batch['stock'].batch_size if hasattr(batch['stock'], 'batch_size') else batch['stock'].x.size(0)
        
        # Forward pass
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            out = model(batch)
        
        # Extract predictions for target nodes (first batch_size nodes)
        batch_out = out[:batch_size]
        batch_targets = batch['stock'].y[:batch_size] if hasattr(batch['stock'], 'y') else targets[:batch_size]
        
        # Use Focal Loss if enabled, otherwise standard cross-entropy
        # Note: class_weights not passed to train_with_sampling for simplicity
        # (mini-batch training is less common, full-batch is primary method)
        if LOSS_TYPE == 'focal':
            criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, weight=None)  # No class weights in mini-batch
            loss = criterion(batch_out, batch_targets)
        else:
            loss = F.cross_entropy(batch_out, batch_targets)
        _backward_step(loss, model, optimizer, scaler=scaler, max_grad_norm=max_grad_norm)
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1), 0.0


def evaluate_with_sampling(model, data, targets, loader):
    """Evaluation step with neighbor sampling for large graphs."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            
            # Get batch size for target nodes
            batch_size = batch['stock'].batch_size if hasattr(batch['stock'], 'batch_size') else batch['stock'].x.size(0)
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=ENABLE_AMP):
                out = model(batch)
            
            # Extract predictions for target nodes
            batch_out = out[:batch_size]
            batch_targets = batch['stock'].y[:batch_size] if hasattr(batch['stock'], 'y') else targets[:batch_size]
            
            preds = batch_out.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(batch_targets.cpu())
    
    if not all_preds:
        return 0.0, 0.0
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    acc = (all_preds == all_targets).float().mean().item()
    
    from sklearn.metrics import f1_score
    f1 = f1_score(all_targets.numpy(), all_preds.numpy(), average='binary', zero_division=0)
    
    return acc, f1

def create_neighbor_loader(data, targets, batch_size, num_neighbors, shuffle=True):
    """Create NeighborLoader for mini-batch training."""
    if 'stock' not in data.x_dict:
        raise ValueError("No 'stock' nodes found in data")
    
    num_nodes = data['stock'].x.size(0)
    target_nodes = torch.arange(num_nodes)
    
    # Add targets to data
    data['stock'].y = targets
    
    loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=('stock', target_nodes),
        num_workers=0,
        shuffle=shuffle,
        drop_last=False
    )
    
    return loader

def run_training_pipeline():
    """Main function to run the time-series training loop."""
    print("🚀 Starting Phase 4: Core GNN Training")
    print("=" * 50)
    
    # 1. Setup and Data Alignment
    
    # Get all graph dates from the directory first
    graph_files = sorted(list(DATA_GRAPHS_DIR.glob('graph_t_*.pt')))
    if not graph_files:
        print("❌ CRITICAL: No graph files found. Run Phase 2 first.")
        return
    
    # Determine input dimensions by loading the first available graph file
    sample_date = pd.to_datetime(graph_files[0].stem.split('_')[-1])
    temp_data = load_graph_data(sample_date)
    
    # Use robust check for HeteroData contents
    if temp_data is None or not hasattr(temp_data, 'node_types') or 'stock' not in temp_data.node_types:
        print("❌ CRITICAL: Cannot load a sample graph. Phase 2 output files are invalid. STOPPING.")
        return

    # Assuming the features are stored as [N, F] tensor under 'stock'
    INPUT_DIM = temp_data['stock'].x.shape[1]
    tickers = list(getattr(temp_data, 'tickers', []))
    if not tickers:
        print("⚠️ Sample graph missing tickers metadata. Using placeholder ticker list.")
        # Fallback: infer from node count (53 stocks from Phase 2 logs)
        tickers = [f'TICKER_{i}' for i in range(temp_data['stock'].x.shape[0])]
    
    graph_dates = [pd.to_datetime(f.stem.split('_')[-1]) for f in graph_files]
    
    # 2. Get Targets and Final Training Dates (classification + regression)
    
    targets_class_dict, targets_reg_dict = create_target_labels(tickers, graph_dates, LOOKAHEAD_DAYS)
    
    # Align training dates: intersect dates that have both a graph AND a target
    training_dates = sorted(list(set(targets_class_dict.keys()).intersection(graph_dates)))
    
    if not training_dates:
        print("❌ CRITICAL: No overlapping dates found between graphs and targets. STOPPING.")
        return
        
    # 3. Split Data: Simple time-based split
    split_70 = training_dates[int(len(training_dates) * 0.70)]
    split_85 = training_dates[int(len(training_dates) * 0.85)]
    
    train_dates = [d for d in training_dates if d <= split_70]
    val_dates = [d for d in training_dates if split_70 < d <= split_85]
    test_dates = [d for d in training_dates if d > split_85]
    
    print(f"\n📊 Data Split (Trading Days):")
    print(f"   - Input Dim: {INPUT_DIM}")
    print(f"   - Train: {len(train_dates)} days (End: {split_70.date()})")
    print(f"   - Val:   {len(val_dates)} days (End: {split_85.date()})")
    print(f"   - Test:  {len(test_dates)} days")
    
    # 4. Model Setup
    model = RoleAwareGraphTransformer(
        INPUT_DIM, HIDDEN_CHANNELS, OUT_CHANNELS, NUM_LAYERS, NUM_HEADS,
        enable_time_aware=ENABLE_TIME_AWARE
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)  # Added weight decay for regularization
    scaler = torch.cuda.amp.GradScaler(enabled=ENABLE_AMP)
    model_path = MODELS_DIR / MODEL_SAVE_NAME
    
    # Calculate class weights for better imbalance handling
    class_weights = None
    if USE_CLASS_WEIGHTS:
        # Calculate class distribution from training data
        all_train_targets = []
        for date in train_dates[:min(100, len(train_dates))]:  # Sample first 100 days for efficiency
            target = targets_class_dict.get(date)
            if target is not None:
                all_train_targets.extend(target.cpu().numpy())
        
        if len(all_train_targets) > 0:
            from collections import Counter
            class_counts = Counter(all_train_targets)
            total_samples = sum(class_counts.values())
            
            # Calculate inverse frequency weights
            num_classes = 2
            class_weights_list = []
            for i in range(num_classes):
                count = class_counts.get(i, 1)  # Avoid division by zero
                weight = total_samples / (num_classes * count)
                class_weights_list.append(weight)
            
            class_weights = torch.tensor(class_weights_list, dtype=torch.float32).to(DEVICE)
            print(f"📊 Class weights calculated: Down/Flat={class_weights[0]:.3f}, Up={class_weights[1]:.3f}")
            print(f"   Class distribution: Down/Flat={class_counts.get(0, 0)}, Up={class_counts.get(1, 0)}")
        else:
            print("⚠️  Could not calculate class weights, using default")
    
    # Print loss function configuration
    if LOSS_TYPE == 'focal':
        print(f"✅ Using Focal Loss (alpha={FOCAL_ALPHA}, gamma={FOCAL_GAMMA}) for class imbalance handling")
        if USE_CLASS_WEIGHTS and class_weights is not None:
            print(f"   + Class weights enabled for additional balancing")
    else:
        print(f"⚠️  Using Standard Cross-Entropy (consider 'focal' for imbalanced data)")
    
    # Learning Rate Scheduler
    if ENABLE_LR_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',  # Maximize F1 score
            factor=LR_SCHEDULER_FACTOR,
            patience=LR_SCHEDULER_PATIENCE,
            min_lr=LR_SCHEDULER_MIN_LR
        )
        print(f"📉 LR Scheduler enabled: ReduceLROnPlateau (patience={LR_SCHEDULER_PATIENCE}, factor={LR_SCHEDULER_FACTOR})")
    else:
        scheduler = None
    
    # 4.1. Mini-batch Training Setup (if enabled and available)
    use_mini_batch = ENABLE_MINI_BATCH and MINI_BATCH_AVAILABLE
    if use_mini_batch:
        print(f"🚀 Mini-batch training enabled: batch_size={BATCH_SIZE}, neighbors={NUM_NEIGHBORS}")
    else:
        if ENABLE_MINI_BATCH and not MINI_BATCH_AVAILABLE:
            print("⚠️  Mini-batch training requested but sparse libraries not available")
            print("   Install: pip install torch-sparse pyg-lib")
        print("🚀 Full-batch training enabled")
    
    if scaler.is_enabled():
        print("⚡ AMP Enabled: Using torch.cuda.amp for mixed precision training")
    else:
        print("ℹ️ AMP Disabled: Running in float32")
    
    if GRAD_CLIP_MAX_NORM and GRAD_CLIP_MAX_NORM > 0:
        print(f"✂️ Gradient clipping active (max norm = {GRAD_CLIP_MAX_NORM})")
    else:
        print("ℹ️ Gradient clipping disabled")
    
    # 5. Training Loop
    print("\n🔨 Starting Sequential Training...")
    best_val_f1 = 0.0
    epochs_without_improvement = 0
    training_history = {
        'train_loss': [],
        'val_f1': [],
        'learning_rate': []
    }
    
    for epoch in range(1, NUM_EPOCHS + 1):
        total_loss = 0
        num_train_batches = 0
        
        # --- Train Phase ---
        for date in tqdm(train_dates, desc=f"Epoch {epoch}/{NUM_EPOCHS} Training"):
            data = load_graph_data(date)
            target_class = targets_class_dict.get(date)
            target_reg = targets_reg_dict.get(date) if ENABLE_MULTI_TASK else None
            if data and target_class is not None:
                # Create time tensor for time-aware encoding
                num_nodes = data['stock'].x.size(0)
                if ENABLE_TIME_AWARE:
                    # Convert date to timestamp (days since epoch or reference date)
                    # Use a reference date (e.g., 2015-01-01) for normalization
                    reference_date = pd.Timestamp('2015-01-01')
                    days_since_ref = (date - reference_date).days
                    date_tensor = torch.full((num_nodes,), days_since_ref, dtype=torch.float32)
                else:
                    date_tensor = None
                
                if use_mini_batch:
                    loader = create_neighbor_loader(data, target_class, BATCH_SIZE, NUM_NEIGHBORS, shuffle=True)
                    loss, _ = train_with_sampling(
                        model,
                        optimizer,
                        data,
                        target_class,
                        loader,
                        scaler=scaler,
                        amp_enabled=scaler.is_enabled(),
                        max_grad_norm=GRAD_CLIP_NORM
                    )
                else:
                    loss, _ = train(
                        model,
                        optimizer,
                        data,
                        target_class,
                        targets_reg=target_reg,
                        date_tensor=date_tensor,
                        scaler=scaler,
                        amp_enabled=scaler.is_enabled(),
                        max_grad_norm=GRAD_CLIP_NORM,
                        class_weights=class_weights
                    )
                total_loss += loss
                num_train_batches += 1
        
        avg_loss = total_loss / max(num_train_batches, 1)

        # --- Validation Phase ---
        val_f1s = []
        all_val_true, all_val_probs = [], []
        for date in tqdm(val_dates, desc=f"Epoch {epoch}/{NUM_EPOCHS} Validation", leave=False):
            data = load_graph_data(date)
            target = targets_class_dict.get(date)
            if data and target is not None:
                # Create time tensor for time-aware encoding
                num_nodes = data['stock'].x.size(0)
                if ENABLE_TIME_AWARE:
                    reference_date = pd.Timestamp('2015-01-01')
                    days_since_ref = (date - reference_date).days
                    date_tensor = torch.full((num_nodes,), days_since_ref, dtype=torch.float32)
                else:
                    date_tensor = None
                
                if use_mini_batch:
                    loader = create_neighbor_loader(data, target, BATCH_SIZE * 2, NUM_NEIGHBORS, shuffle=False)
                    _, f1 = evaluate_with_sampling(model, data, target, loader)
                    # For mini-batch, we can't easily get probabilities, so use default threshold
                    val_f1s.append(f1)
                else:
                    _, f1, _, probs = evaluate(model, data, target, return_probs=True, date_tensor=date_tensor)
                    val_f1s.append(f1)
                    if probs is not None:
                        all_val_true.extend(target.cpu().numpy())
                        all_val_probs.extend(probs)
        
        # Find optimal threshold on validation set (every 5 epochs or first epoch)
        optimal_threshold = 0.5  # Default
        if len(all_val_true) > 0 and len(all_val_probs) > 0 and (epoch == 1 or epoch % 5 == 0):
            optimal_threshold = find_optimal_threshold(
                np.array(all_val_true),
                np.array(all_val_probs)
            )
            if epoch == 1 or epoch % 5 == 0:
                print(f"   📊 Optimal threshold (validation): {optimal_threshold:.4f}")
        
        # Re-evaluate with optimal threshold for better F1 score
        if len(all_val_true) > 0 and len(all_val_probs) > 0 and not use_mini_batch:
            val_f1s_opt = []
            for date in val_dates:
                data = load_graph_data(date)
                target = targets_class_dict.get(date)
                if data and target is not None:
                    # Create time tensor
                    num_nodes = data['stock'].x.size(0)
                    if ENABLE_TIME_AWARE:
                        reference_date = pd.Timestamp('2015-01-01')
                        days_since_ref = (date - reference_date).days
                        date_tensor = torch.full((num_nodes,), days_since_ref, dtype=torch.float32)
                    else:
                        date_tensor = None
                    _, _, _, probs = evaluate(model, data, target, return_probs=True, date_tensor=date_tensor)
                    if probs is not None:
                        y_true = target.cpu().numpy()
                        y_pred_opt = (probs >= optimal_threshold).astype(int)
                        f1_opt = f1_score(y_true, y_pred_opt, average='binary', zero_division=0)
                        val_f1s_opt.append(f1_opt)
            if val_f1s_opt:
                avg_val_f1 = float(np.mean(val_f1s_opt))
            else:
                avg_val_f1 = float(np.mean(val_f1s)) if val_f1s else 0.0
        else:
            avg_val_f1 = float(np.mean(val_f1s)) if val_f1s else 0.0
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step(avg_val_f1)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < current_lr:
                print(f"  📉 LR reduced: {current_lr:.2e} -> {new_lr:.2e}")

        # Record history
        training_history['train_loss'].append(avg_loss)
        training_history['val_f1'].append(avg_val_f1)
        training_history['learning_rate'].append(current_lr)

        print(f"Epoch {epoch:02d}/{NUM_EPOCHS} | Train Loss: {avg_loss:.4f} | Val F1: {avg_val_f1:.4f} | LR: {current_lr:.2e}")

        # Save best model based on F1 score
        improvement = avg_val_f1 - best_val_f1
        if improvement > EARLY_STOP_MIN_DELTA:
            best_val_f1 = avg_val_f1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_path)
            print(f"  ✅ Model Saved (New Best F1: {best_val_f1:.4f}, Improvement: {improvement:.4f})")
        else:
            epochs_without_improvement += 1
            if improvement > 0:
                print(f"  ⚠️  No significant improvement (delta: {improvement:.4f} < {EARLY_STOP_MIN_DELTA})")
        
        # Early stopping check
        if ENABLE_EARLY_STOPPING and epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(f"\n🛑 Early stopping triggered after {epoch} epochs (no improvement for {EARLY_STOP_PATIENCE} epochs)")
            print(f"   Best validation F1: {best_val_f1:.4f}")
            break
    
    # 6. Testing (Final Evaluation)
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, weights_only=False))
    else:
        print("⚠️  Warning: Best model checkpoint not found. Using final weights.")
    
    # Find optimal threshold on validation set
    print("\n🔍 Finding optimal classification threshold on validation set...")
    all_val_true_final, all_val_probs_final = [], []
    for date in val_dates:
        data = load_graph_data(date)
        target = targets_class_dict.get(date)
        if data and target is not None and not use_mini_batch:
            # Create time tensor
            num_nodes = data['stock'].x.size(0)
            if ENABLE_TIME_AWARE:
                reference_date = pd.Timestamp('2015-01-01')
                days_since_ref = (date - reference_date).days
                date_tensor = torch.full((num_nodes,), days_since_ref, dtype=torch.float32)
            else:
                date_tensor = None
            _, _, _, probs = evaluate(model, data, target, return_probs=True, date_tensor=date_tensor)
            if probs is not None:
                all_val_true_final.extend(target.cpu().numpy())
                all_val_probs_final.extend(probs)
    
    optimal_threshold = 0.5  # Default
    if len(all_val_true_final) > 0 and len(all_val_probs_final) > 0:
        computed_threshold = find_optimal_threshold(
            np.array(all_val_true_final),
            np.array(all_val_probs_final)
        )
        # Ensure threshold is valid (not inf/nan)
        if np.isfinite(computed_threshold) and 0 <= computed_threshold <= 1:
            optimal_threshold = computed_threshold
            print(f"   ✅ Optimal threshold: {optimal_threshold:.4f}")
        else:
            print(f"   ⚠️  Computed threshold invalid ({computed_threshold}), using default: 0.5")
    else:
        print(f"   ⚠️  Could not find optimal threshold, using default: 0.5")
    
    # Test with optimal threshold
    test_accs, test_f1s = [], []
    all_test_true, all_test_pred = [], []
    
    for date in test_dates:
        data = load_graph_data(date)
        target = targets_class_dict.get(date)
        if data and target is not None:
            # Create time tensor
            num_nodes = data['stock'].x.size(0)
            if ENABLE_TIME_AWARE:
                reference_date = pd.Timestamp('2015-01-01')
                days_since_ref = (date - reference_date).days
                date_tensor = torch.full((num_nodes,), days_since_ref, dtype=torch.float32)
            else:
                date_tensor = None
            
            if use_mini_batch:
                loader = create_neighbor_loader(data, target, BATCH_SIZE * 2, NUM_NEIGHBORS, shuffle=False)
                acc, f1 = evaluate_with_sampling(model, data, target, loader)
                test_accs.append(acc)
                test_f1s.append(f1)
            else:
                _, _, out, probs = evaluate(model, data, target, return_probs=True, date_tensor=date_tensor)
                if probs is not None:
                    y_true = target.cpu().numpy()
                    y_pred_opt = (probs >= optimal_threshold).astype(int)
                    acc_opt = accuracy_score(y_true, y_pred_opt)
                    f1_opt = f1_score(y_true, y_pred_opt, average='binary', zero_division=0)
                    test_accs.append(acc_opt)
                    test_f1s.append(f1_opt)
                    all_test_true.extend(y_true)
                    all_test_pred.extend(y_pred_opt)
            
    mean_test_acc = float(np.mean(test_accs)) if test_accs else 0.0
    mean_test_f1 = float(np.mean(test_f1s)) if test_f1s else 0.0
    
    # Print classification report if we have predictions
    if len(all_test_true) > 0 and len(all_test_pred) > 0:
        print(f"\n📋 Classification Report:")
        print(classification_report(
            all_test_true,
            all_test_pred,
            target_names=['Down/Flat (0)', 'Up (1)'],
            digits=4
        ))

    print("\n" + "=" * 50)
    print(f"🚀 Final Test Results (Core GNN, Averaged over {len(test_dates)} days):")
    print(f"   - Test Accuracy: {mean_test_acc:.4f}")
    print(f"   - Test F1 Score: {mean_test_f1:.4f}")
    print(f"   - Baseline F1: 0.6725 (Goal: Exceed Baseline)")
    print("=" * 50)
    
    # Now the Core Model is trained and ready for RL integration
    print("\n🎯 Core Model Trained. Ready for Phase 5: RL Integration.")

    return {
        "best_val_f1": float(best_val_f1),
        "test_accuracy": mean_test_acc,
        "test_f1": mean_test_f1,
        "model_checkpoint": str(model_path.resolve())
    }


if __name__ == '__main__':
    run_training_pipeline()
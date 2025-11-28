# phase4_core_training.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv
from torch_geometric.data import HeteroData
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import os
import sys
from torch.nn.utils import clip_grad_norm_

# FIX: Allow PyTorch to safely load torch-geometric objects (PyTorch >= 2.6)
import torch.serialization
from torch_geometric.data.storage import BaseStorage, NodeStorage, EdgeStorage, GlobalStorage
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([BaseStorage, NodeStorage, EdgeStorage, GlobalStorage])

# Add components directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent / 'components'))

# --- Configuration & Hyperparameters (from Proposal) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_GRAPHS_DIR = PROJECT_ROOT / "data" / "graphs"
MODELS_DIR = PROJECT_ROOT / "models"
OHLCV_RAW_FILE = PROJECT_ROOT / "data" / "raw" / "stock_prices_ohlcv_raw.csv"

# Hyperparameters (based on proposal)
LOOKAHEAD_DAYS = 5              # Target: 5-day-ahead return sign
HIDDEN_CHANNELS = 256           # Hidden size
NUM_LAYERS = 2                  # Number of layers
NUM_HEADS = 4                   # Number of attention heads
OUT_CHANNELS = 2                # Binary classification: Up/Down
LEARNING_RATE = 0.0005
NUM_EPOCHS = 30                 # Increased epochs for complex model (with early stopping)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Early Stopping Configuration
ENABLE_EARLY_STOPPING = True
EARLY_STOP_PATIENCE = 5         # Stop if no improvement for 5 epochs
EARLY_STOP_MIN_DELTA = 0.0001   # Minimum improvement threshold

# Learning Rate Scheduler Configuration
ENABLE_LR_SCHEDULER = True
LR_SCHEDULER_PATIENCE = 3       # Reduce LR after 3 epochs without improvement
LR_SCHEDULER_FACTOR = 0.5       # Multiply LR by this factor
LR_SCHEDULER_MIN_LR = 1e-6      # Minimum learning rate

# --- Helper Utilities ---

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
GRAD_CLIP_MAX_NORM = 1.0       # Gradient clipping to stabilize training
MODEL_SAVE_NAME = 'core_transformer_model.pt'

# Ensure model directory exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --- 1. Model Definition: Core Role-Aware Graph Transformer ---

# Import the enhanced components
from components.pearl_embedding import PEARLPositionalEmbedding
from components.transformer_layer import RelationAwareGATv2Conv, RelationAwareAggregator

# Import PyTorch Geometric's neighbor sampling (requires sparse libraries)
try:
    from torch_geometric.loader import NeighborLoader
    MINI_BATCH_AVAILABLE = True
except ImportError:
    MINI_BATCH_AVAILABLE = False
    NeighborLoader = None

class RoleAwareGraphTransformer(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, num_heads):
        super().__init__()
        
        # PEARL related dimension
        self.PE_DIM = 32
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
                    dropout=0.3,
                    edge_type=edge_type
                )
                for edge_type in metadata
            }, aggr='sum')
            
            # Add relation-aware aggregator
            aggregator = RelationAwareAggregator(hidden_dim, metadata)
            
            self.convs.append(conv)
            self.relation_aggregators.append(aggregator)
            
        # (C) Output Classifier
        self.lin_out = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, out_dim)
        )

    def forward(self, data):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        
        # 1. Generate/Concatenate PEARL Embeddings
        x = x_dict['stock']
        pearl_pe = self.pearl_embedding(x, edge_index_dict)
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
            x_dict['stock'] = F.dropout(x_dict['stock'], p=0.4, training=self.training)
        
        # 3. Node-Level Output
        out = self.lin_out(x_dict['stock'])
        return out
    
    def get_embeddings(self, data):
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
            x_dict['stock'] = F.dropout(x_dict['stock'], p=0.4, training=self.training)
        
        # Return embeddings before final linear layer
        return x_dict['stock']

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
    """Calculates the 5-day ahead return sign (y_{i, t+5}) for all stocks and dates."""
    print(f"\n🏷️ Calculating {lookahead_days}-day ahead return signs...")
    
    ohlcv_df = _read_time_series_csv(OHLCV_RAW_FILE)
    
    close_cols = [col for col in ohlcv_df.columns if col.startswith('Close_')]
    close_prices = ohlcv_df[close_cols].copy()
    
    forward_returns_df = (close_prices.shift(-lookahead_days) - close_prices) / close_prices
    
    target_labels = (forward_returns_df > 0).astype(int)
    
    targets_dict = {}
    for date in dates:
        date_str = date.strftime('%Y-%m-%d')
        if date_str in target_labels.index:
            target_vector = []
            for ticker in tickers:
                col_name = f'Close_{ticker}'
                if col_name in target_labels.columns:
                    target_vector.append(target_labels.loc[date_str, col_name])
                else:
                    target_vector.append(0)
            targets_dict[date] = torch.tensor(target_vector, dtype=torch.long)
            
    print(f"✅ Targets calculated for {len(targets_dict)} trading days.")
    return targets_dict

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


def train(model, optimizer, data, targets, scaler=None, amp_enabled=False, max_grad_norm=None):
    """Single training step for the Core GNN (HeteroData input)."""
    model.train()
    optimizer.zero_grad()
    
    # Ensure edge indices are long
    for metadata in data.edge_index_dict.keys():
        data[metadata].edge_index = data[metadata].edge_index.to(torch.long)
        
    with torch.cuda.amp.autocast(enabled=amp_enabled):
        out = model(data.to(DEVICE))
        loss = F.cross_entropy(out, targets.to(DEVICE))
    
    _backward_step(loss, model, optimizer, scaler=scaler, max_grad_norm=max_grad_norm)
    
    return loss.item(), out.detach().argmax(dim=1)


def evaluate(model, data, targets):
    """Single evaluation step."""
    model.eval()
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=ENABLE_AMP):
            out = model(data.to(DEVICE))
    
    y_true = targets.cpu().numpy()
    y_pred = out.argmax(dim=1).cpu().numpy()
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    
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
    
    # 2. Get Targets and Final Training Dates
    
    targets_dict = create_target_labels(tickers, graph_dates, LOOKAHEAD_DAYS)
    
    # Align training dates: intersect dates that have both a graph AND a target
    training_dates = sorted(list(set(targets_dict.keys()).intersection(graph_dates)))
    
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
    model = RoleAwareGraphTransformer(INPUT_DIM, HIDDEN_CHANNELS, OUT_CHANNELS, NUM_LAYERS, NUM_HEADS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler(enabled=ENABLE_AMP)
    model_path = MODELS_DIR / MODEL_SAVE_NAME
    
    # Learning Rate Scheduler
    if ENABLE_LR_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',  # Maximize F1 score
            factor=LR_SCHEDULER_FACTOR,
            patience=LR_SCHEDULER_PATIENCE,
            min_lr=LR_SCHEDULER_MIN_LR,
            verbose=True
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
            target = targets_dict.get(date)
            if data and target is not None:
                if use_mini_batch:
                    loader = create_neighbor_loader(data, target, BATCH_SIZE, NUM_NEIGHBORS, shuffle=True)
                    loss, _ = train_with_sampling(
                        model,
                        optimizer,
                        data,
                        target,
                        loader,
                        scaler=scaler,
                        amp_enabled=scaler.is_enabled(),
                        max_grad_norm=GRAD_CLIP_MAX_NORM
                    )
                else:
                    loss, _ = train(
                        model,
                        optimizer,
                        data,
                        target,
                        scaler=scaler,
                        amp_enabled=scaler.is_enabled(),
                        max_grad_norm=GRAD_CLIP_MAX_NORM
                    )
                total_loss += loss
                num_train_batches += 1
        
        avg_loss = total_loss / max(num_train_batches, 1)

        # --- Validation Phase ---
        val_f1s = []
        for date in tqdm(val_dates, desc=f"Epoch {epoch}/{NUM_EPOCHS} Validation", leave=False):
            data = load_graph_data(date)
            target = targets_dict.get(date)
            if data and target is not None:
                if use_mini_batch:
                    loader = create_neighbor_loader(data, target, BATCH_SIZE * 2, NUM_NEIGHBORS, shuffle=False)
                    _, f1 = evaluate_with_sampling(model, data, target, loader)
                else:
                    _, f1 = evaluate(model, data, target)
                val_f1s.append(f1)
        
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
    test_accs, test_f1s = [], []
    
    for date in test_dates:
        data = load_graph_data(date)
        target = targets_dict.get(date)
        if data and target is not None:
            if use_mini_batch:
                loader = create_neighbor_loader(data, target, BATCH_SIZE * 2, NUM_NEIGHBORS, shuffle=False)
                acc, f1 = evaluate_with_sampling(model, data, target, loader)
            else:
                acc, f1 = evaluate(model, data, target)
            test_accs.append(acc)
            test_f1s.append(f1)
            
    mean_test_acc = float(np.mean(test_accs)) if test_accs else 0.0
    mean_test_f1 = float(np.mean(test_f1s)) if test_f1s else 0.0

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
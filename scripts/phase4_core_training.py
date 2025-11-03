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
NUM_EPOCHS = 1                 # Increased epochs for complex model
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ensure model directory exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --- 1. Model Definition: Core Role-Aware Graph Transformer ---

# Import the enhanced components
from components.pearl_embedding import PEARLPositionalEmbedding
from components.transformer_layer import RelationAwareGATv2Conv, RelationAwareAggregator

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
    
    ohlcv_df = pd.read_csv(OHLCV_RAW_FILE, index_col='Date', parse_dates=True)
    
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

def train(model, optimizer, data, targets):
    """Single training step for the Core GNN (HeteroData input)."""
    model.train()
    optimizer.zero_grad()
    
    # Ensure edge indices are long
    for metadata in data.edge_index_dict.keys():
        data[metadata].edge_index = data[metadata].edge_index.to(torch.long)
        
    out = model(data.to(DEVICE))
    loss = F.cross_entropy(out, targets.to(DEVICE))
    
    loss.backward()
    optimizer.step()
    
    return loss.item(), out.argmax(dim=1)

def evaluate(model, data, targets):
    """Single evaluation step."""
    model.eval()
    
    with torch.no_grad():
        out = model(data.to(DEVICE))
    
    y_true = targets.cpu().numpy()
    y_pred = out.argmax(dim=1).cpu().numpy()
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    
    return acc, f1

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
    model_path = MODELS_DIR / 'core_transformer_model.pt'
    
    # 5. Training Loop
    print("\n🔨 Starting Sequential Training...")
    best_val_f1 = 0.0
    
    for epoch in range(1, NUM_EPOCHS + 1):
        total_loss = 0
        
        # --- Train Phase ---
        for date in tqdm(train_dates, desc=f"Epoch {epoch} Training"):
            data = load_graph_data(date)
            target = targets_dict.get(date)
            if data and target is not None:
                loss, _ = train(model, optimizer, data, target)
                total_loss += loss
        
        avg_loss = total_loss / len(train_dates)

        # --- Validation Phase ---
        val_f1s = []
        for date in val_dates:
            data = load_graph_data(date)
            target = targets_dict.get(date)
            if data and target is not None:
                _, f1 = evaluate(model, data, target)
                val_f1s.append(f1)
        
        avg_val_f1 = np.mean(val_f1s)

        print(f"Epoch {epoch:02d} | Train Loss: {avg_loss:.4f} | Val F1: {avg_val_f1:.4f}")

        # Save best model based on F1 score
        if avg_val_f1 > best_val_f1:
            best_val_f1 = avg_val_f1
            torch.save(model.state_dict(), model_path)
            print(f"  --> Model Saved (New Best F1: {best_val_f1:.4f})")
    
    # 6. Testing (Final Evaluation)
    model.load_state_dict(torch.load(model_path, weights_only=False))
    test_accs, test_f1s = [], []
    
    for date in test_dates:
        data = load_graph_data(date)
        target = targets_dict.get(date)
        if data and target is not None:
            acc, f1 = evaluate(model, data, target)
            test_accs.append(acc)
            test_f1s.append(f1)
            
    print("\n" + "=" * 50)
    print(f"🚀 Final Test Results (Core GNN, Averaged over {len(test_dates)} days):")
    print(f"   - Test Accuracy: {np.mean(test_accs):.4f}")
    print(f"   - Test F1 Score: {np.mean(test_f1s):.4f}")
    print(f"   - Baseline F1: 0.6725 (Goal: Exceed Baseline)")
    print("=" * 50)
    
    # Now the Core Model is trained and ready for RL integration
    print("\n🎯 Core Model Trained. Ready for Phase 5: RL Integration.")


if __name__ == '__main__':
    run_training_pipeline()
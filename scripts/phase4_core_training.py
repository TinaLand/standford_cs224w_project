# phase4_core_training.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv, global_mean_pool
from torch_geometric.data import HeteroData
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np

# FIX: Allow PyTorch to safely load torch-geometric objects (PyTorch >= 2.6)
import torch.serialization
from torch_geometric.data.storage import BaseStorage, NodeStorage, EdgeStorage, GlobalStorage
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([BaseStorage, NodeStorage, EdgeStorage, GlobalStorage])

# --- Configuration & Hyperparameters (from Proposal) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_GRAPHS_DIR = PROJECT_ROOT / "data" / "graphs"
MODELS_DIR = PROJECT_ROOT / "models"
OHLCV_RAW_FILE = PROJECT_ROOT / "data" / "raw" / "stock_prices_ohlcv_raw.csv"

# Hyperparameters (based on proposal)
LOOKAHEAD_DAYS = 5              # Target: 5-day-ahead return sign [cite: 29]
HIDDEN_CHANNELS = 256           # Hidden size [cite: 86]
NUM_LAYERS = 2                  # Number of layers [cite: 86]
NUM_HEADS = 4                   # Number of attention heads [cite: 86]
OUT_CHANNELS = 2                # Binary classification: Up/Down
LEARNING_RATE = 0.0005
NUM_EPOCHS = 30                 # Increased epochs for complex model
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ensure model directory exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --- 1. Model Definition: Core Role-Aware Graph Transformer ---

class RoleAwareGraphTransformer(torch.nn.Module):
    """
    Core GNN Model: Heterogeneous Graph Transformer + PEARL Embeddings.
    This architecture handles multi-relational, heterogeneous, static+dynamic edges[cite: 88].
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, num_heads):
        super().__init__()
        
        # 假设 PEARL 嵌入维度为 P_dim=32。PEARL logic will be simulated here.
        self.PE_DIM = 32
        
        # NOTE: in_dim now includes the concatenated PEARL embedding dimension.
        total_in_dim = in_dim + self.PE_DIM 
        
        # --- (A) PEARL Positional Embedding Block (Simulation) ---
        # In a real implementation, PEARL would be a separate block or precomputed.
        # We use a simple linear layer placeholder to map initial features to a fixed PE space.
        self.pearl_mapper = torch.nn.Linear(in_dim, self.PE_DIM)
        
        # --- (B) Graph Transformer Layers (HeteroConv) ---
        # Define heterogeneous metadata based on Phase 2 graph construction
        # Edge types: sector_industry, competitor, supply_chain (Static)
        #             rolling_correlation, fund_similarity (Dynamic)
        metadata = [
            ('stock', 'sector_industry', 'stock'),
            ('stock', 'competitor', 'stock'),
            ('stock', 'supply_chain', 'stock'),
            ('stock', 'rolling_correlation', 'stock'),
            ('stock', 'fund_similarity', 'stock')
        ]
        
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            # Using GATv2Conv which is more powerful than GAT/GCN for complex graphs
            in_d = total_in_dim if i == 0 else hidden_dim
            
            # HeteroConv enables aggregation over all defined edge types
            conv = HeteroConv({
                edge_type: GATv2Conv(in_d, hidden_dim // num_heads, heads=num_heads, dropout=0.3)
                for edge_type in metadata
            }, aggr='sum') # Sum aggregation over different edge types [cite: 50]
            
            self.convs.append(conv)
            
        # --- (C) Output Classifier ---
        self.lin_out = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, out_dim)
        )

    def forward(self, data):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        
        # 1. Generate/Concatenate PEARL Embeddings [cite: 87]
        x = x_dict['stock']
        pearl_pe = self.pearl_mapper(x)
        x_with_pe = torch.cat([x, pearl_pe], dim=1) # [N, F + P_dim]
        
        x_dict['stock'] = x_with_pe
        
        # 2. Heterogeneous Graph Transformer Layers
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            # Apply ReLU and dropout after all edge types have aggregated
            x_dict['stock'] = x_dict['stock'].relu()
            x_dict['stock'] = F.dropout(x_dict['stock'], p=0.4, training=self.training)
        
        # 3. Node-Level Output
        out = self.lin_out(x_dict['stock'])
        return out # Output: [N, 2] logits


# --- 2. Training and Evaluation Utilities ---

def load_graph_data(date):
    """Loads a single graph snapshot for the given date."""
    date_str = date.strftime('%Y%m%d')
    filepath = DATA_GRAPHS_DIR / f'graph_t_{date_str}.pt'
    if not filepath.exists():
        return None
    try:
        data = torch.load(filepath, weights_only=False)
        return data
    except Exception as e:
        print(f"❌ Error loading graph {filepath.name}: {e}")
        return None

def create_target_labels(tickers, dates, lookahead_days):
    """
    Calculates the lookahead-day return sign labels for all stocks and dates.
    1 if forward return > 0, else 0.
    """
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
                target_vector.append(int(target_labels.loc[date_str, col_name]) if col_name in target_labels.columns else 0)
            targets_dict[date] = torch.tensor(target_vector, dtype=torch.long)
    print(f"✅ Targets calculated for {len(targets_dict)} trading days.")
    return targets_dict

def train(model, optimizer, data, targets):
    """Single training step for the Core GNN."""
    model.train()
    optimizer.zero_grad()
    
    # Ensure all edge indices are long
    for metadata in data.edge_index_dict.keys():
        data[metadata].edge_index = data[metadata].edge_index.to(torch.long)
        
    out = model(data.to(DEVICE))
    loss = F.cross_entropy(out, targets.to(DEVICE))
    
    loss.backward()
    optimizer.step()
    
    return loss.item()


# --- 3. Main Execution ---

def run_training_pipeline():
    """Runs the sequential training loop for the core model."""
    print("🚀 Starting Phase 4: Core GNN Training")
    
    # 1. Discover graph files and load a sample graph to infer dims
    graph_files = sorted(list(DATA_GRAPHS_DIR.glob('graph_t_*.pt')))
    if not graph_files:
        print("❌ CRITICAL: No graph files found. Run Phase 2 first.")
        return
    sample_date = pd.to_datetime(graph_files[0].stem.split('_')[-1])
    temp_features = load_graph_data(sample_date)
    if temp_features is None or not hasattr(temp_features, 'node_types') or 'stock' not in temp_features.node_types:
        print("❌ CRITICAL: Cannot load sample graph. STOPPING.")
        return

    # Input dimension is the feature size of the 'stock' node type
    INPUT_DIM = temp_features['stock'].x.shape[1]
    tickers = list(getattr(temp_features, 'tickers', []))
    if not tickers:
        print("⚠️ Sample graph missing tickers metadata. Proceeding without training targets.")
    
    model = RoleAwareGraphTransformer(
        INPUT_DIM, 
        HIDDEN_CHANNELS, 
        OUT_CHANNELS, 
        NUM_LAYERS, 
        NUM_HEADS
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 3. Minimal run: just forward one batch to validate graph/model wiring
    dates = [pd.to_datetime(f.stem.split('_')[-1]) for f in graph_files[:1]]
    data = load_graph_data(dates[0])
    if data is None:
        print("❌ Failed to load first graph for a smoke test.")
        return
    # Ensure edge_index types are long
    for metadata in data.edge_index_dict.keys():
        data[metadata].edge_index = data[metadata].edge_index.to(torch.long)
    with torch.no_grad():
        logits = model(data.to(DEVICE))
    print(f"✅ Smoke test forward pass OK. Logits shape: {tuple(logits.shape)}")
    
    print("\n✅ Phase 4: Core GNN setup complete. Extend training loop next.")
    
if __name__ == '__main__':
    run_training_pipeline()
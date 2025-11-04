# phase2_graph_construction.py
"""
Phase 2: Graph Construction for CS224W Stock RL GNN Project

This script reads Phase 1 outputs (node features, dynamic edge parameters, static connections)
and constructs time-varying graph snapshots G_t = (V, E_t, X_t) using PyTorch Geometric.

Key Features:
- Dynamic edge filtering based on correlation threshold (|ρ_ij| > 0.6)
- Static edge integration (sector, industry, market cap connections)
- Daily graph snapshots saved as PyTorch files
- Memory-efficient batch processing for large time series

Dependencies: torch, torch-geometric, pandas, numpy, pickle
"""

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
import pickle
import os
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_EDGES_DIR = PROJECT_ROOT / "data" / "edges"
DATA_GRAPHS_DIR = PROJECT_ROOT / "data" / "graphs"

# Graph construction parameters
CORRELATION_THRESHOLD = 0.6  # |ρ_ij| > 0.6 for dynamic edges
FUNDAMENTAL_SIMILARITY_THRESHOLD = 0.8  # Fundamental similarity threshold
MIN_EDGE_WEIGHT = 0.1  # Minimum edge weight to include
BATCH_SIZE_DAYS = 30  # Process graphs in batches to manage memory

# Edge attribute normalization parameters
NORMALIZE_EDGE_ATTRS = True  # Enable edge attribute normalization
EDGE_NORMALIZATION_METHOD = 'min_max'  # Options: 'min_max', 'standard', 'robust'

# Ensure output directory exists
DATA_GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

print("🚀 Starting Phase 2: Graph Construction")
print("=" * 50)

# --- Edge Attribute Normalization Functions ---

def normalize_edge_attributes(edge_weights, method='min_max', edge_type='unknown'):
    """
    Normalize edge attributes to improve GNN training.
    
    Args:
        edge_weights (torch.Tensor): Edge weights to normalize
        method (str): Normalization method ('min_max', 'standard', 'robust')
        edge_type (str): Type of edges for logging
    
    Returns:
        torch.Tensor: Normalized edge weights
    """
    if not NORMALIZE_EDGE_ATTRS or len(edge_weights) == 0:
        return edge_weights
    
    original_min, original_max = edge_weights.min().item(), edge_weights.max().item()
    
    if method == 'min_max':
        # Min-Max normalization: scale to [0, 1]
        if original_max > original_min:
            normalized = (edge_weights - original_min) / (original_max - original_min)
        else:
            normalized = torch.zeros_like(edge_weights)
    
    elif method == 'standard':
        # Standard normalization: mean=0, std=1
        mean = edge_weights.mean()
        std = edge_weights.std()
        if std > 0:
            normalized = (edge_weights - mean) / std
        else:
            normalized = edge_weights - mean
    
    elif method == 'robust':
        # Robust normalization using median and IQR
        median = edge_weights.median()
        q75, q25 = torch.quantile(edge_weights, 0.75), torch.quantile(edge_weights, 0.25)
        iqr = q75 - q25
        if iqr > 0:
            normalized = (edge_weights - median) / iqr
        else:
            normalized = edge_weights - median
    
    else:
        normalized = edge_weights
    
    new_min, new_max = normalized.min().item(), normalized.max().item()
    print(f"    {edge_type}: [{original_min:.4f}, {original_max:.4f}] → [{new_min:.4f}, {new_max:.4f}] ({method})")
    
    return normalized

def get_edge_normalization_stats():
    """Pre-compute normalization statistics for all edge types."""
    print("📊 Computing edge normalization statistics...")
    
    stats = {}
    
    # 1. Correlation edges
    try:
        corr_file = DATA_EDGES_DIR / "edges_dynamic_corr_params.csv"
        if corr_file.exists():
            corr_df = pd.read_csv(corr_file)
            corr_values = corr_df['abs_correlation'].values
            stats['correlation'] = {
                'min': float(np.min(corr_values)),
                'max': float(np.max(corr_values)),
                'mean': float(np.mean(corr_values)),
                'std': float(np.std(corr_values))
            }
    except Exception as e:
        print(f"  ⚠️ Could not load correlation stats: {e}")
    
    # 2. Fundamental similarity edges
    try:
        fund_file = DATA_EDGES_DIR / "edges_dynamic_fund_sim_params.csv"
        if fund_file.exists():
            fund_df = pd.read_csv(fund_file)
            fund_values = fund_df['fundamental_similarity'].values
            stats['fundamental'] = {
                'min': float(np.min(fund_values)),
                'max': float(np.max(fund_values)),
                'mean': float(np.mean(fund_values)),
                'std': float(np.std(fund_values))
            }
    except Exception as e:
        print(f"  ⚠️ Could not load fundamental stats: {e}")
    
    return stats

# --- Data Loading Functions ---

def load_node_features():
    """Load consolidated node features (X_t) from Phase 1."""
    print("📁 Loading node features...")
    
    node_features_file = DATA_PROCESSED_DIR / "node_features_X_t_final.csv"
    
    if not node_features_file.exists():
        raise FileNotFoundError(f"Node features file not found: {node_features_file}")
    
    # Load with Date as index
    node_features = pd.read_csv(node_features_file, index_col='Date', parse_dates=True)
    
    # Extract ticker list from column names 
    # Handle different patterns: 'LogRet_1d_AAPL', 'AAPL_Sentiment', 'AAPL_PE_Log', etc.
    tickers = set()
    for col in node_features.columns:
        if 'LogRet_1d_' in col:
            # Pattern: LogRet_1d_TICKER
            ticker = col.split('_')[-1]
            tickers.add(ticker)
        elif col.endswith('_Sentiment'):
            # Pattern: TICKER_Sentiment
            ticker = col.replace('_Sentiment', '')
            tickers.add(ticker)
        elif '_PE_' in col or '_ROE_' in col:
            # Pattern: TICKER_PE_Log, TICKER_ROE_Log
            ticker = col.split('_')[0]
            tickers.add(ticker)
    
    tickers = sorted(list(tickers))
    
    print(f"✅ Loaded node features: {node_features.shape}")
    print(f"   - Date range: {node_features.index.min()} to {node_features.index.max()}")
    print(f"   - Tickers: {len(tickers)}")
    
    return node_features, tickers

def load_dynamic_correlations():
    """Load dynamic correlation parameters from Phase 1 (using .csv format)."""
    print("📈 Loading dynamic correlations...")
    
    # Use .csv format as actually generated by phase1_edge_parameter_calc.py
    corr_file = DATA_EDGES_DIR / "edges_dynamic_corr_params.csv"
    
    if not corr_file.exists():
        raise FileNotFoundError(f"Dynamic correlation file not found: {corr_file}")
    
    # Load correlations from CSV
    correlations = pd.read_csv(corr_file)
    correlations['Date'] = pd.to_datetime(correlations['Date'])
    
    print(f"✅ Loaded dynamic correlations: {correlations.shape}")
    print(f"   - Date range: {correlations['Date'].min()} to {correlations['Date'].max()}")
    print(f"   - Unique pairs: {len(correlations[['ticker1', 'ticker2']].drop_duplicates())}")
    
    return correlations

def load_fundamental_similarities():
    """Load fundamental similarity parameters from Phase 1."""
    print("💼 Loading fundamental similarities...")
    
    fund_sim_file = DATA_EDGES_DIR / "edges_dynamic_fund_sim_params.csv"
    
    if not fund_sim_file.exists():
        print("⚠️ Fundamental similarity file not found, using empty DataFrame")
        return pd.DataFrame()
    
    similarities = pd.read_csv(fund_sim_file)
    
    print(f"✅ Loaded fundamental similarities: {similarities.shape}")
    
    return similarities

def load_static_connections():
    """Load static edge data from the two raw files (FIXED)."""
    print("🏭 Loading static connections...")
    
    # FIX 2: Load the two separate raw files
    DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
    sector_file = DATA_RAW_DIR / "static_sector_industry.csv"
    supply_comp_file = DATA_RAW_DIR / "static_supply_competitor_edges.csv"
    
    if not sector_file.exists() or not supply_comp_file.exists():
        print("⚠️ Required static files not found.")
        return None, None
    
    sector_df = pd.read_csv(sector_file)
    supply_comp_df = pd.read_csv(supply_comp_file)
    
    print(f"✅ Loaded static connections: Sector {sector_df.shape}, Comp/Supply {supply_comp_df.shape}")
    
    return sector_df, supply_comp_df  # Return both dataframes

# --- Graph Construction Functions ---

def pre_calculate_static_edges(tickers, sector_df, supply_comp_df, ticker_to_idx):
    """Calculates all static edges (Sector, Industry, Supply Chain, Competitor) once."""
    print("\n🧱 Pre-calculating Static Edges...")
    
    static_edge_dict = {}
    
    # 1. Sector/Industry Edges
    if sector_df is not None and not sector_df.empty:
        sector_edges = []
        sector_weights = []
        
        for _, row in sector_df.iterrows():
            if 'ticker1' in row and 'ticker2' in row:
                ticker1, ticker2 = row['ticker1'], row['ticker2']
                if ticker1 in ticker_to_idx and ticker2 in ticker_to_idx:
                    idx1, idx2 = ticker_to_idx[ticker1], ticker_to_idx[ticker2]
                    sector_edges.extend([[idx1, idx2], [idx2, idx1]])  # Undirected
                    weight = float(row.get('weight', 1.0))
                    sector_weights.extend([weight, weight])
        
        if sector_edges:
            static_edge_dict['sector_industry'] = (
                torch.tensor(sector_edges, dtype=torch.long).t().contiguous(),
                torch.tensor(sector_weights, dtype=torch.float32).unsqueeze(1)
            )
    
    # 2. Supply Chain/Competitor Edges
    if supply_comp_df is not None and not supply_comp_df.empty:
        supply_edges = []
        supply_weights = []
        
        for _, row in supply_comp_df.iterrows():
            if 'ticker1' in row and 'ticker2' in row:
                ticker1, ticker2 = row['ticker1'], row['ticker2']
                if ticker1 in ticker_to_idx and ticker2 in ticker_to_idx:
                    idx1, idx2 = ticker_to_idx[ticker1], ticker_to_idx[ticker2]
                    supply_edges.extend([[idx1, idx2], [idx2, idx1]])  # Undirected
                    weight = float(row.get('weight', 1.0))
                    supply_weights.extend([weight, weight])
        
        if supply_edges:
            static_edge_dict['supply_competitor'] = (
                torch.tensor(supply_edges, dtype=torch.long).t().contiguous(),
                torch.tensor(supply_weights, dtype=torch.float32).unsqueeze(1)
            )
    
    print(f"✅ Pre-calculated {len(static_edge_dict)} static edge types")
    for edge_type, (edge_index, edge_weight) in static_edge_dict.items():
        print(f"   - {edge_type}: {edge_index.size(1)} edges")
    
    return static_edge_dict

def create_ticker_mapping(tickers):
    """Create mapping from ticker symbols to node indices."""
    ticker_to_idx = {ticker: idx for idx, ticker in enumerate(tickers)}
    idx_to_ticker = {idx: ticker for ticker, idx in ticker_to_idx.items()}
    return ticker_to_idx, idx_to_ticker

def filter_dynamic_edges(correlations_df, date, threshold=CORRELATION_THRESHOLD):
    """Filter dynamic edges for a specific date based on correlation threshold."""
    # Filter by date
    day_correlations = correlations_df[correlations_df['Date'] == date].copy()
    
    if day_correlations.empty:
        return pd.DataFrame()
    
    # Apply correlation threshold
    strong_correlations = day_correlations[
        day_correlations['abs_correlation'] > threshold
    ].copy()
    
    return strong_correlations

def extract_node_features_for_date(node_features_df, date, tickers):
    """
    Extract node feature matrix X_t for a specific date, ensuring consistent dimension.
    The final feature vector for each node must have a uniform length.
    """
    try:
        # Select features for the specific date
        # loc[date] returns a Series where indices are the feature names (e.g., 'LogRet_1d_AAPL')
        day_features_series = node_features_df.loc[date]
    except KeyError:
        # This handles the 'No features found' for non-trading days/missing dates
        return None 
    
    feature_matrix = []
    
    # Get all unique feature prefixes (e.g., 'LogRet_1d', 'RSI_14', 'VIX')
    # We use a set of prefixes to find the number of features per node.
    # Exclude common suffixes that represent the ticker itself.
    
    # 1. Determine the feature dimension (F) based on the first ticker (assuming all are uniform)
    first_ticker = tickers[0]
    
    # Handle two naming conventions:
    # Pattern 1: Feature_TICKER (e.g., LogRet_1d_AAPL)
    # Pattern 2: TICKER_Feature (e.g., AAPL_PE)
    feature_columns_pattern1 = [col for col in node_features_df.columns if col.endswith(f'_{first_ticker}')]
    feature_columns_pattern2 = [col for col in node_features_df.columns if col.startswith(f'{first_ticker}_')]
    
    # Combine both patterns
    feature_columns_for_first_ticker = feature_columns_pattern1 + feature_columns_pattern2
    
    if not feature_columns_for_first_ticker:
        # Fallback if the feature naming convention is inconsistent
        return None

    # 2. Extract and reshape the feature vector (N x F)
    for ticker in tickers:
        ticker_features_vector = []
        
        # Iterate through the standardized feature columns using the first ticker's template
        for col_template in feature_columns_for_first_ticker:
            # Reconstruct the column name for the current ticker based on pattern
            if col_template.endswith(f'_{first_ticker}'):
                # Pattern 1: Feature_TICKER
                feature_prefix = col_template.replace(f'_{first_ticker}', '')
                current_ticker_col = f'{feature_prefix}_{ticker}'
            elif col_template.startswith(f'{first_ticker}_'):
                # Pattern 2: TICKER_Feature
                feature_suffix = col_template.replace(f'{first_ticker}_', '')
                current_ticker_col = f'{ticker}_{feature_suffix}'
            else:
                # Fallback: try as-is
                current_ticker_col = col_template
            
            # Check if the column exists in the daily data
            if current_ticker_col in day_features_series.index:
                ticker_features_vector.append(day_features_series[current_ticker_col])
            else:
                # Safety fallback
                ticker_features_vector.append(0.0) 
        
        feature_matrix.append(ticker_features_vector)
    
    # Convert the final list of lists into a PyTorch Tensor (N x F)
    # Use float32 for consistency with model training
    features_tensor = torch.tensor(feature_matrix, dtype=torch.float32)
    
    # ===== CRITICAL: Normalize node features to prevent gradient issues =====
    # Features can have vastly different scales (e.g., LogRet ~0.01, Price ~50)
    # This leads to gradient explosion/vanishing and prevents learning
    # Apply z-score normalization: (x - mean) / std for each feature dimension
    mean = features_tensor.mean(dim=0, keepdim=True)  # [1, F]
    std = features_tensor.std(dim=0, keepdim=True)    # [1, F]
    
    # Avoid division by zero for constant features
    std = torch.where(std == 0, torch.ones_like(std), std)
    
    features_tensor = (features_tensor - mean) / std
    
    # Replace any resulting NaN/Inf with zeros (safety fallback)
    features_tensor = torch.nan_to_num(features_tensor, nan=0.0, posinf=0.0, neginf=0.0)
    
    return features_tensor

def construct_graph_for_date(date, node_features_df, correlations_df, similarities_df, 
                           static_edge_dict, tickers, ticker_to_idx):
    """Construct a single HETEROGENEOUS graph snapshot for a specific date (FIXED)."""
    
    # Extract node features X_t
    node_features = extract_node_features_for_date(node_features_df, date, tickers)
    
    if node_features is None: # <--- Handles the return from the fixed function
        print(f"⚠️ No features found for date {date.strftime('%Y-%m-%d')}")
        return None
    
    # Initialize HeteroData object
    graph = HeteroData()
    
    # 1. Node Features
    graph['stock'].x = node_features  # shape [N, F]
    
    # 2. Static Edges (Pre-calculated, just add to HeteroData)
    for edge_type, (edge_index, edge_weight) in static_edge_dict.items():
        graph['stock', edge_type, 'stock'].edge_index = edge_index
        # Apply normalization to static edge attributes
        normalized_edge_weight = normalize_edge_attributes(edge_weight, EDGE_NORMALIZATION_METHOD, edge_type)
        graph['stock', edge_type, 'stock'].edge_attr = normalized_edge_weight
    
    # 3. Dynamic Edges (Correlations)
    dynamic_edges = filter_dynamic_edges(correlations_df, date)
    
    corr_edges = []
    corr_weights = []
    
    for _, row in dynamic_edges.iterrows():
        ticker1, ticker2 = row['ticker1'], row['ticker2']
        if ticker1 in ticker_to_idx and ticker2 in ticker_to_idx:
            idx1, idx2 = ticker_to_idx[ticker1], ticker_to_idx[ticker2]
            corr_edges.extend([[idx1, idx2], [idx2, idx1]])  # Undirected
            weight = float(row['abs_correlation'])
            corr_weights.extend([weight, weight])
    
    if corr_edges:
        corr_edge_index = torch.tensor(corr_edges, dtype=torch.long).t().contiguous()
        corr_edge_weight = torch.tensor(corr_weights, dtype=torch.float32).unsqueeze(1)
        
        # Apply normalization to correlation edge attributes
        normalized_corr_weight = normalize_edge_attributes(corr_edge_weight, EDGE_NORMALIZATION_METHOD, 'rolling_correlation')
        
        # Add to HeteroData with explicit type
        graph['stock', 'rolling_correlation', 'stock'].edge_index = corr_edge_index
        graph['stock', 'rolling_correlation', 'stock'].edge_attr = normalized_corr_weight
    
    # 4. Dynamic Edges (Fundamental Similarity - quasi-static)
    fund_sim_edges = []
    fund_sim_weights = []
    
    for _, row in similarities_df.iterrows():
        if row['fundamental_similarity'] > FUNDAMENTAL_SIMILARITY_THRESHOLD:
            ticker1, ticker2 = row['ticker1'], row['ticker2']
            if ticker1 in ticker_to_idx and ticker2 in ticker_to_idx:
                idx1, idx2 = ticker_to_idx[ticker1], ticker_to_idx[ticker2]
                fund_sim_edges.extend([[idx1, idx2], [idx2, idx1]])  # Undirected
                weight = float(row['fundamental_similarity'])
                fund_sim_weights.extend([weight, weight])
    
    if fund_sim_edges:
        fund_sim_edge_index = torch.tensor(fund_sim_edges, dtype=torch.long).t().contiguous()
        fund_sim_edge_weight = torch.tensor(fund_sim_weights, dtype=torch.float32).unsqueeze(1)
        
        # Apply normalization to fundamental similarity edge attributes
        normalized_fund_sim_weight = normalize_edge_attributes(fund_sim_edge_weight, EDGE_NORMALIZATION_METHOD, 'fund_similarity')
        
        # Add to HeteroData
        graph['stock', 'fund_similarity', 'stock'].edge_index = fund_sim_edge_index
        graph['stock', 'fund_similarity', 'stock'].edge_attr = normalized_fund_sim_weight
    
    # Add metadata
    graph.date = date
    graph.tickers = tickers
    
    # Calculate total edges across all edge types
    total_edges = 0
    for edge_type in graph.edge_types:
        if hasattr(graph[edge_type], 'edge_index'):
            total_edges += graph[edge_type].edge_index.size(1)
    graph.num_edges = total_edges
    
    return graph

def save_graph_snapshot(graph, date, output_dir):
    """Save a graph snapshot to disk with verification."""
    date_str = date.strftime('%Y%m%d')
    filename = f"graph_t_{date_str}.pt"
    filepath = output_dir / filename
    
    try:
        # Save using PyTorch
        torch.save(graph, filepath)
        
        # VERIFICATION: Immediately reload and verify the saved object
        try:
            # Fix for PyTorch 2.6+ weights_only security feature
            loaded_graph = torch.load(filepath, weights_only=False)
            
            # Basic integrity checks
            if not hasattr(loaded_graph, 'edge_types'):
                raise ValueError("Loaded graph missing edge_types attribute")
            
            if not hasattr(loaded_graph, 'num_edges'):
                raise ValueError("Loaded graph missing num_edges attribute")
            
            # Check node features exist and have correct shape
            if 'stock' not in loaded_graph.node_types:
                raise ValueError("Loaded graph missing 'stock' node type")
            
            if not hasattr(loaded_graph['stock'], 'x'):
                raise ValueError("Loaded graph missing node features")
            
            node_features = loaded_graph['stock'].x
            if not isinstance(node_features, torch.Tensor):
                raise ValueError("Node features not a PyTorch tensor")
            
            if len(node_features.shape) != 2:
                raise ValueError(f"Node features wrong shape: {node_features.shape}")
            
            # If we get here, the graph is valid
            return filepath
            
        except Exception as e:
            # If verification fails, remove the corrupted file
            if filepath.exists():
                filepath.unlink()
            raise ValueError(f"Graph verification failed: {e}")
            
    except Exception as e:
        raise RuntimeError(f"Failed to save graph for {date_str}: {e}")

# --- Main Execution ---

def main():
    """Main function to orchestrate graph construction."""
    try:
        print(f"📊 Graph Construction Configuration:")
        print(f"   - Correlation threshold: {CORRELATION_THRESHOLD}")
        print(f"   - Fundamental similarity threshold: {FUNDAMENTAL_SIMILARITY_THRESHOLD}")
        print(f"   - Output directory: {DATA_GRAPHS_DIR}")
        print()
        
        # Load all data from Phase 1
        node_features, tickers = load_node_features()
        correlations = load_dynamic_correlations()
        similarities = load_fundamental_similarities()
        sector_df, supply_comp_df = load_static_connections()
        
        # Create ticker mapping
        ticker_to_idx, idx_to_ticker = create_ticker_mapping(tickers)
        print(f"📝 Created ticker mapping for {len(tickers)} stocks")
        
        # Pre-calculate static edges
        static_edge_dict = pre_calculate_static_edges(tickers, sector_df, supply_comp_df, ticker_to_idx)
        
        # FIX: Use only trading days from the loaded feature data
        date_range = node_features.index.unique()  # <--- USE THIS instead of pd.date_range
        
        print(f"\n🗓️ Constructing graphs for date range:")
        print(f"   - Start: {date_range.min()}")
        print(f"   - End: {date_range.max()}")
        print(f"   - Total trading days: {len(date_range)}")
        
        # Construct and save graphs
        successful_graphs = 0
        failed_graphs = 0
        
        print("\n🔨 Starting graph construction...")
        
        for i, date in enumerate(date_range):
            try:
                # Construct graph for this date
                graph = construct_graph_for_date(
                    date, node_features, correlations, similarities, 
                    static_edge_dict, tickers, ticker_to_idx
                )
                
                # Skip if no features found (non-trading day)
                if graph is None:
                    continue
                
                # Save graph snapshot with verification
                try:
                    filepath = save_graph_snapshot(graph, date, DATA_GRAPHS_DIR)
                    successful_graphs += 1
                except Exception as save_error:
                    print(f"❌ Failed to save graph for {date.strftime('%Y-%m-%d')}: {save_error}")
                    failed_graphs += 1
                    continue
                
                # Progress update
                if (i + 1) % 50 == 0 or i == len(date_range) - 1:
                    print(f"   Progress: {i+1}/{len(date_range)} graphs constructed "
                          f"({successful_graphs} successful, {failed_graphs} failed)")
                
                # Detailed log for first few graphs
                if i < 5:
                    print(f"   📅 {date.strftime('%Y-%m-%d')}: "
                          f"{graph.num_nodes} nodes, {graph.num_edges} edges -> {filepath.name}")
                
            except Exception as e:
                failed_graphs += 1
                if failed_graphs <= 5:  # Only show first few errors
                    print(f"   ❌ Failed to construct graph for {date}: {e}")
        
        print(f"\n" + "=" * 50)
        print(f"✅ Phase 2: Graph Construction Complete!")
        print(f"📈 Results Summary:")
        print(f"   - Total graphs constructed: {successful_graphs}")
        print(f"   - Failed constructions: {failed_graphs}")
        print(f"   - Success rate: {successful_graphs/(successful_graphs+failed_graphs)*100:.1f}%")
        print(f"   - Output directory: {DATA_GRAPHS_DIR}")
        print(f"   - Average nodes per graph: {len(tickers)}")
        
        # Display sample graph info
        if successful_graphs > 0:
            sample_files = list(DATA_GRAPHS_DIR.glob("graph_t_*.pt"))[:3]
            print(f"\n📋 Sample graph files:")
            for file in sample_files:
                try:
                    sample_graph = torch.load(file, weights_only=False)
                    print(f"   - {file.name}: {sample_graph.num_nodes} nodes, "
                          f"{sample_graph.num_edges} edges")
                except:
                    print(f"   - {file.name}: (unable to load sample)")
        
        print(f"\n🎯 Ready for Phase 3: GNN Model Training!")
        
    except Exception as e:
        print(f"❌ Error in graph construction pipeline: {e}")
        raise

if __name__ == "__main__":
    main()

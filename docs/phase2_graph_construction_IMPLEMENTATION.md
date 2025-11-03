# Phase 2: Graph Construction - Implementation Guide

## Overview

**File**: `scripts/phase2_graph_construction.py`  
**Purpose**: Build daily graph snapshots from node features and edge parameters  
**Dependencies**: torch, torch-geometric, pandas, numpy  
**Input**: Node features + edge parameters from Phase 1  
**Output**: PyG `HeteroData` graphs (one per trading day)

---

## What Does This File Do?

This script transforms **tabular data** into **graph objects** suitable for GNN training:

**Input**:
- Node features: CSV table (`[Date, Features...]`)
- Edge parameters: CSV tables (`[Date, ticker1, ticker2, correlation]`)

**Output**:
- Graph snapshots: PyG `HeteroData` objects (`graph_t_20200101.pt`)

For each trading day t, creates graph G_t = (V, E_t, X_t):
- **V**: Nodes (stocks)
- **E_t**: Time-varying edges
- **X_t**: Node feature matrix

---

## Why This Design?

### Time-Varying Graphs

**Why Not One Big Graph?**

**Bad Approach**:
```python
# Single static graph
graph = construct_graph(all_dates_combined)
```

**Problems**:
1. Can't model temporal dynamics
2. Mixes past/future information (data leakage)
3. Loses time-varying correlations

**Good Approach** (This File):
```python
# Daily graph snapshots
for date in trading_days:
    graph_t = construct_graph(date)
    save(f'graph_t_{date}.pt')
```

**Benefits**:
1. ✅ Edges change over time (correlation varies)
2. ✅ No data leakage (strict temporal ordering)
3. ✅ Realistic: Market relationships evolve

### HeteroData vs Data

**PyTorch Geometric offers two graph types**:

**Data** (Homogeneous):
```python
data = Data(x=features, edge_index=edges)
# All edges same type
```

**HeteroData** (Heterogeneous - Used Here):
```python
data = HeteroData()
data['stock'].x = features
data['stock', 'correlation', 'stock'].edge_index = corr_edges
data['stock', 'sector', 'stock'].edge_index = sector_edges
# Multiple edge types!
```

**Why HeteroData**:
1. **Multiple Relationships**: Sector, correlation, competitor
2. **Different Weights**: Each edge type learned separately
3. **Interpretability**: Can analyze which edges matter most
4. **Ablation Studies**: Easy to remove specific edge types

---

## Key Components

### 1. Edge Attribute Normalization

```python
def normalize_edge_attributes(edge_weights, method='min_max', edge_type='unknown'):
    """Normalize edge attributes to improve GNN training."""
```

#### Why Normalize Edge Weights?

**Problem**: Different edge types have different scales
- Correlation: [-1, 1]
- Fundamental similarity: [0, 1]
- Static weights: {0.2, 0.6, 0.8, 1.0}

**GNN Message Passing**:
```
m_{ij} = w_{ij} × h_j

If w_{corr} = 0.7, w_{sector} = 1.0:
Message from sector edge is stronger (always)
```

**Solution**: Normalize all edge types to [0, 1]

#### Three Normalization Methods

**Min-Max** (Default):
```
x_norm = (x - min) / (max - min)
```
- Range: [0, 1]
- Preserves relative distances
- Sensitive to outliers

**Standard** (Z-score):
```
x_norm = (x - μ) / σ
```
- Mean: 0, Std: 1
- Better for normally distributed data
- Can have values outside [0, 1]

**Robust** (IQR-based):
```
x_norm = (x - median) / IQR

where IQR = Q3 - Q1
```
- Resistant to outliers
- Good for heavy-tailed distributions

**Why Min-Max for Edges**:
- GNNs often use sigmoid/ReLU activations ([0, 1] friendly)
- Interpretable: 0 = weakest, 1 = strongest
- Consistent across edge types

---

### 2. Ticker Mapping

```python
def create_ticker_mapping(tickers):
    """Create mapping from ticker symbols to node indices."""
```

#### Why Needed?

**Graph Representation**: Nodes are integers
```
Node 0, Node 1, Node 2, ...
```

**Human Representation**: Nodes are tickers
```
AAPL, MSFT, GOOGL, ...
```

**Mapping**:
```python
ticker_to_idx = {'AAPL': 0, 'MSFT': 1, 'GOOGL': 2}
idx_to_ticker = {0: 'AAPL', 1: 'MSFT', 2: 'GOOGL'}
```

**Usage**:
```python
# When building edge_index:
if ticker1 == 'AAPL' and ticker2 == 'MSFT':
    edge = [ticker_to_idx['AAPL'], ticker_to_idx['MSFT']]
    # edge = [0, 1]
```

**Why Bidirectional Mapping**:
- `ticker_to_idx`: Building graphs
- `idx_to_ticker`: Interpreting predictions

---

### 3. Dynamic Edge Filtering

```python
def filter_dynamic_edges(correlations_df, date, threshold=0.6):
    """Filter dynamic edges for a specific date."""
```

#### Threshold Selection

**Correlation Threshold = 0.6**

**Why 0.6?**

**Statistics of Correlations**:
```
Weak:     |ρ| < 0.3
Moderate: 0.3 < |ρ| < 0.7
Strong:   |ρ| > 0.7
```

**Trade-offs**:

**Threshold = 0.3** (low):
- ✅ Dense graph (many edges)
- ✅ More information
- ❌ Noisy edges
- ❌ Slower GNN training

**Threshold = 0.8** (high):
- ✅ Only strong relationships
- ✅ Fast GNN training
- ❌ Sparse graph
- ❌ May miss important signals

**Threshold = 0.6** (chosen):
- Balanced approach
- Filters noise while preserving signal
- Aligns with finance literature (moderate-to-strong)

#### Graph Sparsity Impact

**Example** (50 stocks):
- Complete graph: 1225 possible edges
- Threshold 0.3: ~800 edges (65% dense)
- Threshold 0.6: ~200 edges (16% dense)
- Threshold 0.8: ~50 edges (4% dense)

**GNN Perspective**:
- Too dense: Overly smooth predictions (over-aggregation)
- Too sparse: Information doesn't propagate well
- 10-20% density: Good for most GNN architectures

---

### 4. Node Feature Extraction

```python
def extract_node_features_for_date(node_features_df, date, tickers):
    """Extract node feature matrix X_t for a specific date."""
```

#### Data Transformation Challenge

**Input (Wide CSV)**:
```
Date       | LogRet_1d_AAPL | LogRet_1d_MSFT | RSI_14_AAPL | RSI_14_MSFT
2020-01-01 | 0.012          | 0.008          | 65.3        | 52.1
```

**Needed (GNN Matrix)**:
```
X_t = [
    [0.012, 65.3, ...],  # Stock 0 (AAPL)
    [0.008, 52.1, ...],  # Stock 1 (MSFT)
]
Shape: [N_stocks, N_features]
```

#### Implementation Logic

**Step 1**: Determine feature list from first ticker
```python
first_ticker = tickers[0]  # 'AAPL'
feature_columns_for_first_ticker = [col for col in df.columns 
                                    if col.endswith(f'_{first_ticker}')]
# ['LogRet_1d_AAPL', 'RSI_14_AAPL', ...]
```

**Step 2**: For each ticker, extract same features
```python
for ticker in tickers:
    ticker_features = []
    for col_template in feature_columns_for_first_ticker:
        feature_prefix = col_template.replace(f'_{first_ticker}', '')
        # 'LogRet_1d_AAPL' → 'LogRet_1d'
        
        current_col = f'{feature_prefix}_{ticker}'
        # For MSFT: 'LogRet_1d_MSFT'
        
        ticker_features.append(df[current_col])
```

**Step 3**: Stack into matrix
```python
feature_matrix = torch.tensor(feature_matrix, dtype=torch.float32)
# Shape: [N, F]
```

**Why This Approach**:
- ✅ Guaranteed same dimension for all stocks
- ✅ Handles missing features (fallback to 0)
- ✅ Maintains feature order consistency

---

### 5. Graph Construction

```python
def construct_graph_for_date(date, node_features_df, correlations_df, 
                            similarities_df, static_edge_dict, tickers, ticker_to_idx):
    """Construct a single HETEROGENEOUS graph snapshot."""
```

#### HeteroData Structure

```python
graph = HeteroData()

# 1. Node Features
graph['stock'].x = torch.tensor([...])  # [N, F]

# 2. Multiple Edge Types
graph['stock', 'rolling_correlation', 'stock'].edge_index = torch.tensor([...])
graph['stock', 'rolling_correlation', 'stock'].edge_attr = torch.tensor([...])

graph['stock', 'sector_industry', 'stock'].edge_index = torch.tensor([...])
graph['stock', 'sector_industry', 'stock'].edge_attr = torch.tensor([...])

# 3. Metadata
graph.date = date
graph.tickers = tickers
graph.num_edges = total_edges
```

#### Edge Index Format

**PyG Format**: COO (Coordinate) format
```python
edge_index = torch.tensor([
    [0, 1, 2, 0],  # Source nodes
    [1, 2, 0, 2]   # Target nodes
], dtype=torch.long)

# Represents edges: 0→1, 1→2, 2→0, 0→2
```

**Why COO**:
- Memory efficient for sparse graphs
- Fast for GNN message passing
- Standard PyG format

**Undirected Edges**:
```python
# For edge AAPL ↔ MSFT, add both directions:
corr_edges.extend([[idx1, idx2], [idx2, idx1]])
```

**Why**: GNN message passing needs both directions for undirected graph

#### Edge Attributes

```python
graph['stock', 'rolling_correlation', 'stock'].edge_attr = normalized_weights
# Shape: [num_edges, 1]
```

**Purpose**: Edge weights used in message passing
```
m_{ij} = edge_attr_{ij} × W × h_j
```

Stronger edges contribute more to neighbor aggregation.

---

### 6. Graph Saving with Verification

```python
def save_graph_snapshot(graph, date, output_dir):
    """Save a graph snapshot to disk with verification."""
```

#### Why Verify After Saving?

**Problem**: Corrupted files from:
- Disk full during write
- Process interrupted mid-save
- Serialization bugs

**Solution**: Load immediately after save
```python
torch.save(graph, filepath)

# Verify
loaded_graph = torch.load(filepath, weights_only=False)
assert hasattr(loaded_graph, 'edge_types')
assert loaded_graph['stock'].x.shape[0] == N_stocks
```

**If Verification Fails**:
```python
if filepath.exists():
    filepath.unlink()  # Delete corrupted file
raise ValueError("Graph verification failed")
```

**Benefit**: Detect corruption immediately, not during training

#### PyTorch 2.6+ Compatibility

```python
torch.load(filepath, weights_only=False)
```

**Why `weights_only=False`**:
- PyTorch 2.6+: Security feature to prevent code execution
- HeteroData contains custom objects (not just tensors)
- Need to explicitly allow loading

```python
torch.serialization.add_safe_globals([BaseStorage, NodeStorage, ...])
```

**Registers PyG classes** as safe to load

---

## Graph Statistics Example

**For 50 Stocks on 2020-01-15**:

```
Nodes: 50 (stocks)

Edge Types and Counts:
├─ rolling_correlation: 185 edges (ρ > 0.6)
├─ fund_similarity: 42 edges (sim > 0.8)
├─ sector_industry: 120 edges (same sector/industry)
└─ supply_competitor: 25 edges (simulated relationships)

Total Edges: 372
Graph Density: 372 / 1225 = 30%
```

**Dense Enough**: Information can propagate
**Sparse Enough**: Efficient computation

---

## Memory Management

### Batch Processing (Configured but Not Used)

```python
BATCH_SIZE_DAYS = 30  # Process graphs in batches
```

**If Implemented**:
```python
for batch_start in range(0, len(dates), BATCH_SIZE_DAYS):
    batch_dates = dates[batch_start:batch_start+BATCH_SIZE_DAYS]
    # Process batch
    # Clear memory
    torch.cuda.empty_cache()
```

**Why Useful for Large Datasets**:
- 10 years × 250 days = 2500 graphs
- Each graph ≈ 1-5 MB
- Total: 2.5-12.5 GB (manageable)
- But if 500 stocks or 20 years: need batching

### Current Implementation

**Sequential Processing**:
```python
for date in date_range:
    graph = construct_graph_for_date(date)
    torch.save(graph, filepath)
    # Each graph saved immediately, not kept in memory
```

**Memory Footprint**: ~5-10 MB (one graph at a time)

---

## Edge Type Design

### Why Multiple Edge Types?

**Heterogeneous GNN** learns different aggregation for each type:

```python
# Pseudo-code for heterogeneous message passing
for edge_type in ['correlation', 'sector', 'competitor']:
    messages[edge_type] = W[edge_type] @ neighbor_features
    
final_message = combine(messages)
```

**Benefit**: Model learns:
- Correlation edges: Price movement signals
- Sector edges: Industry trends
- Competitor edges: Market share shifts

### Edge Type Categorization

**Dynamic Edges** (change over time):
- `rolling_correlation`: ρ_{ij}(t) varies daily
- `fund_similarity`: Updates quarterly (quasi-static)

**Static Edges** (constant):
- `sector_industry`: Companies don't change sector often
- `supply_competitor`: Relationships relatively stable

**Why Both**:
- Dynamic: Captures market regime changes
- Static: Captures structural relationships

---

## Graph Metadata

```python
graph.date = date
graph.tickers = tickers
graph.num_edges = total_edges
```

#### Why Store Metadata?

**Use Cases**:

1. **Training Scripts**:
   ```python
   graph = torch.load('graph_t_20200101.pt')
   tickers = graph.tickers  # Know which node is which stock
   ```

2. **Debugging**:
   ```python
   print(f"Graph for {graph.date}: {graph.num_edges} edges")
   ```

3. **Validation**:
   ```python
   assert len(graph.tickers) == graph['stock'].x.shape[0]
   ```

4. **Target Label Alignment**:
   ```python
   for i, ticker in enumerate(graph.tickers):
       target[i] = future_return[ticker]
   ```

---

## Verification System

### Integrity Checks

```python
# 1. Check attributes exist
if not hasattr(loaded_graph, 'edge_types'):
    raise ValueError("Missing edge_types")

# 2. Check node types
if 'stock' not in loaded_graph.node_types:
    raise ValueError("Missing 'stock' node type")

# 3. Check feature shape
if len(node_features.shape) != 2:
    raise ValueError(f"Wrong shape: {node_features.shape}")
```

**Why These Checks**:
1. **edge_types**: HeteroData structure corrupted
2. **'stock' node type**: Schema mismatch
3. **2D shape**: Must be [N, F] matrix

**Alternative**: Trust the save
- ❌ Risk: Corrupt files discovered during training (hours wasted)
- ✅ Verify now: Catch errors immediately (seconds)

---

## Output File Naming

```python
filename = f"graph_t_{date.strftime('%Y%m%d')}.pt"
# Example: graph_t_20200115.pt
```

**Why This Format**:
- **Sortable**: Alphabetical = chronological
- **No ambiguity**: YYYYMMDD unambiguous worldwide
- **Glob-friendly**: `graph_t_*.pt` matches all

**Usage in Training**:
```python
graph_files = sorted(DATA_GRAPHS_DIR.glob('graph_t_*.pt'))
# Automatically in chronological order!
```

---

## Error Handling

### 1. Missing Trading Days

```python
node_features = extract_node_features_for_date(df, date, tickers)
if node_features is None:
    print(f"⚠️ No features for {date}")
    return None
```

**Why**: Non-trading days (weekends, holidays)
- Feature CSV only has trading days
- Attempting to extract non-existent date → None
- Skip graph construction for that day

### 2. Empty Edge Sets

```python
if not corr_edges:
    # No correlation edges for this day
    # Graph still created with other edge types
```

**Why Allow**: Different days have different correlation patterns
- Low volatility day: Few strong correlations
- Crisis day: Many strong correlations
- Graph structure should reflect this

### 3. File Save Failures

```python
try:
    torch.save(graph, filepath)
except Exception as e:
    raise RuntimeError(f"Failed to save: {e}")
```

**Caught at Top Level**:
```python
except Exception as save_error:
    print(f"❌ Failed to save graph for {date}: {save_error}")
    failed_graphs += 1
    continue  # Continue with next date
```

**Result**: Pipeline continues despite individual failures

---

## Performance Statistics

### Computation Time

**For 50 Stocks, 2500 Days**:
- Loading data: ~5 seconds
- Static edge pre-calculation: ~2 seconds
- Graph construction loop: ~10-15 minutes
  - Per graph: ~0.25 seconds
  - Bottleneck: Edge filtering and tensor creation

**If Scaling to 500 Stocks**:
- Edge filtering: O(N²) → 100× slower
- Expected time: ~2-3 hours
- **Optimization needed**: Parallel processing

### Disk Usage

**Per Graph**:
- 50 nodes × 100 features × 4 bytes = 20 KB (features)
- ~300 edges × 2 indices × 8 bytes = 4.8 KB (edge indices)
- ~300 edges × 1 attr × 4 bytes = 1.2 KB (edge attributes)
- Metadata: ~1 KB
- **Total per graph**: ~30-50 KB

**All Graphs**:
- 2500 graphs × 50 KB = 125 MB
- Manageable for any modern system

---

## Integration with Phase 3

**Output** → **Input to `phase3_baseline_training.py`**

```python
# Phase 3 Training Loop:
for date in train_dates:
    graph = torch.load(f'graph_t_{date.strftime("%Y%m%d")}.pt')
    # graph['stock'].x → Node features
    # graph.edge_index_dict → All edge types
    
    predictions = model(graph)
    loss = criterion(predictions, targets)
```

**Key Advantage**: Pre-computed graphs
- Training doesn't rebuild graphs
- Faster experimentation
- Can try different models on same graphs

---

## Summary

**Purpose**: Convert tabular data to GNN-ready graph snapshots  
**Key Innovation**: Time-varying heterogeneous graphs with multiple edge types  
**Output**: 2500+ daily graph snapshots  
**Design**: Verified saves, metadata-rich, production-ready

**This file enables temporal GNN modeling!** 🕸️📈

---

**Last Updated**: 2025-11-02  
**Code Style**: Beginner-friendly with detailed explanations [[memory:3128464]]  
**Language**: English comments [[memory:3128459]]


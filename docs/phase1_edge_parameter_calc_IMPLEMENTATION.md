# Phase 1: Edge Parameter Calculation - Implementation Guide

## Overview

**File**: `scripts/phase1_edge_parameter_calc.py`  
**Purpose**: Calculate dynamic edge parameters for graph construction  
**Dependencies**: pandas, numpy, scikit-learn, scipy  
**Input**: Consolidated node features from feature engineering  
**Output**: Time-varying edge parameters (correlations, similarities)

---

## What Does This File Do?

This script calculates **edge parameters** that determine which stocks are connected in the graph:

1. **Rolling Correlations** - Time-varying edges based on return co-movement
2. **Fundamental Similarity** - Static/quasi-static edges based on company characteristics
3. **Sector Connections** - Categorical edges based on industry classification

These parameters are used in Phase 2 to construct actual graph edges.

---

## Why This Design?

### Separation of Edge Calculation from Graph Construction

**This File (Phase 1c)**: Calculates edge **parameters**
```
For each stock pair (i,j) and date t:
  - Correlation ρ_{ij}(t)
  - Similarity s_{ij}
```

**Phase 2**: Uses parameters to decide which edges to include
```
If ρ_{ij}(t) > threshold:
  - Add edge (i, j) to graph G_t
```

**Why Separate**:
1. **Flexibility**: Try different thresholds without recalculation
2. **Efficiency**: Calculate once, use many times
3. **Debugging**: Inspect edge parameters independently

---

## Key Components

### 1. Rolling Correlation Calculation

```python
def compute_rolling_correlation(technical_data, window=30, min_periods=20):
    """Compute rolling correlation between stock returns over time."""
```

#### Mathematical Foundation

**Pearson Correlation Coefficient**:
```
ρ(X, Y) = Cov(X, Y) / (σ_X × σ_Y)

Expanded:
ρ = Σ((x_i - μ_X)(y_i - μ_Y)) / √(Σ(x_i - μ_X)² × Σ(y_i - μ_Y)²)
```

**Interpretation**:
- ρ = 1: Perfect positive linear relationship
- ρ = 0: No linear relationship  
- ρ = -1: Perfect negative linear relationship

#### Why Rolling Window?

**Static Correlation** (entire history):
```python
corr = df['AAPL'].corr(df['MSFT'])  # Single number
```

**Problem**: Stock relationships **change over time**
- Tech stocks highly correlated during tech boom
- Less correlated during sector rotation
- Financial crisis → all stocks correlate (contagion)

**Rolling Correlation** (30-day window):
```python
for t in dates:
    corr_t = returns[t-29:t].corr()  # Different for each t
```

**Benefit**: Captures **time-varying relationships**

**Example**:
```
AAPL vs MSFT Correlation:
2015: ρ = 0.45 (moderate)
2020: ρ = 0.85 (strong - both work-from-home stocks)
2022: ρ = 0.55 (moderate - different market focus)
```

#### Implementation Optimization

**Naive Approach** (slow):
```python
for t in dates:
    for i in stocks:
        for j in stocks:
            corr[t,i,j] = manual_correlation(returns[i][t-29:t], returns[j][t-29:t])
```

**Time Complexity**: O(T × N² × W) where W=window size

**Pandas Optimization** (fast):
```python
rolling_corr_matrix = returns.rolling(window=30).corr()
```

**Time Complexity**: O(T × N²)  
**Why Faster**: Pandas uses:
- Vectorized NumPy operations
- Efficient correlation algorithms
- Memory-optimized rolling windows

#### Output Format

**Multi-Index Series**:
```
Date        ticker1  ticker2  → correlation
2020-01-30  AAPL     MSFT        0.652
2020-01-30  AAPL     GOOGL       0.718
2020-01-30  MSFT     GOOGL       0.598
2020-01-31  AAPL     MSFT        0.645
...
```

**Why This Format**:
- Each row = one edge at one time point
- Easy to filter by date in Phase 2
- Supports time-varying graph construction

#### Absolute Correlation

```python
'abs_correlation': abs(correlation)
```

**Why Absolute**:

**Problem**: Negative correlation is still a relationship
- AAPL vs airline stock: ρ = -0.7
- Strong relationship (when tech up, airlines down)
- Should create an edge!

**Solution**: Use |ρ| for edge strength
- |ρ| = 0.7 → Strong edge (whether +0.7 or -0.7)
- Sign preserved in `correlation` column for analysis

**GNN Perspective**:
- Edge means "information flow"
- Both positive and negative correlations carry information
- Direction (sign) can be edge feature

---

### 2. Fundamental Similarity

```python
def compute_fundamental_similarity(fundamental_data):
    """Compute similarity based on fundamental metrics."""
```

#### Cosine Similarity Deep Dive

**Formula**:
```
cos(θ) = (A · B) / (||A|| × ||B||)

where:
A · B = a₁b₁ + a₂b₂ + ... + a_nb_n  (dot product)
||A|| = √(a₁² + a₂² + ... + a_n²)    (L2 norm)
```

**Geometric Interpretation**:
```
         B
        /
       /θ
      /______ A

cos(θ) measures angle between vectors
θ = 0°  → cos(θ) = 1 (parallel, identical direction)
θ = 90° → cos(θ) = 0 (perpendicular, unrelated)
```

#### Why Cosine Over Euclidean Distance?

**Euclidean Distance**:
```
d(A, B) = √(Σ(a_i - b_i)²)
```

**Problem**: Scale-dependent
```
Company A: [P/E=20, ROE=0.15]
Company B: [P/E=40, ROE=0.30]  (2× larger in both)

Euclidean: d = √((20-40)² + (0.15-0.30)²) = 20.0
But companies have SAME profile! (Both high growth)
```

**Cosine Similarity**:
```
A_norm = [20, 0.15] → direction: [0.997, 0.075]
B_norm = [40, 0.30] → direction: [0.997, 0.075]

cos(θ) ≈ 1.0  (same direction, recognized as similar!)
```

**Advantage**: Captures **ratio patterns**, not absolute values

#### Fundamental Features Used

```python
potential_features = [
    'market_cap', 'pe_ratio', 'forward_pe', 'peg_ratio',
    'price_to_book', 'roe', 'roa', 'debt_to_equity',
    'current_ratio', 'revenue_growth', 'earnings_growth', 'beta'
]
```

**Why These Features**:
- **Valuation**: P/E, Price-to-Book (expensive vs cheap)
- **Profitability**: ROE, ROA (efficient vs inefficient)
- **Financial Health**: Debt-to-Equity, Current Ratio (safe vs risky)
- **Growth**: Revenue/Earnings growth (growth vs value)
- **Risk**: Beta (volatile vs stable)

**Similarity Interpretation**:
- High similarity → Companies with similar business models
- Different sectors can still be similar (e.g., Amazon and Walmart)

---

### 3. Sector Connections

```python
def compute_sector_similarity(fundamental_data):
    """Compute sector-based connections between stocks."""
```

#### Categorical vs Continuous Edges

**This Script**: Treats sector as **categorical**
```python
if sector1 == sector2:
    weight = 1.0  # Binary: same sector or not
```

**Alternative**: Could use continuous similarity
```python
# Sector embeddings learned from data
sector_embedding = {'Tech': [0.8, 0.2], 'Finance': [0.2, 0.8]}
similarity = cosine(sector_embedding[sector1], sector_embedding[sector2])
```

**Why Categorical for Now**:
- Simple and interpretable
- Sectors are well-defined categories
- Can be enhanced later with learned embeddings

#### Edge Weight Design

```python
connection_type = 'same_sector'
weight = 1.0
```

**Why 1.0**:
- Maximum connection strength
- Sector is strongest economic grouping
- Companies in same sector:
  - Affected by same regulations
  - Same commodity prices
  - Same customer base
  - Compete for same market share

**Different Sector**: weight = 0.1
- Weak connection (still in same economy)
- Captures market-wide effects

---

## Implementation Optimizations

### 1. Data Transformation

**Challenge**: Wide format to long format
```
Wide (from feature engineering):
Date       | LogRet_1d_AAPL | LogRet_1d_MSFT
2020-01-01 | 0.012          | 0.008

Needed for correlation:
Date       | ticker | returns
2020-01-01 | AAPL   | 0.012
2020-01-01 | MSFT   | 0.008
```

**Solution**:
```python
technical_data_list = []
for col in technical_cols:
    ticker = col.split('_')[-1]
    temp_df = consolidated_data[[col]].reset_index()
    temp_df.columns = ['Date', 'returns']
    temp_df['ticker'] = ticker
    technical_data_list.append(temp_df)

technical_data = pd.concat(technical_data_list)
```

### 2. Pairwise Combinations

**Efficiency Consideration**:
```python
ticker_pairs = list(itertools.combinations(valid_tickers, 2))
# N=50 → 1225 pairs
# N=100 → 4950 pairs
```

**Complexity**: O(N²)

**Why Not All Pairs**:
```python
# Could use only nearest neighbors in feature space
# But for 50 stocks, computing all pairs is feasible
```

**Trade-off**:
- Complete graph: Dense connections, more information
- Sparse graph: Faster training, may miss relationships

---

## Error Handling

### 1. Insufficient Data

```python
if len(common_dates) < min_periods:
    continue  # Skip this pair
```

**Why**: Correlation needs minimum sample size
- Too few points → unreliable correlation
- min_periods=20 ensures statistical validity

### 2. Zero Variance

```python
# Pandas .corr() handles zero variance gracefully
# Returns NaN, which we drop
rolling_corr.dropna()
```

**Why**: If stock price doesn't move (σ=0), correlation undefined
- ρ = Cov(X, Y) / (0 × σ_Y) = undefined

### 3. Missing Features

```python
if col_name in latest_fundamentals.index:
    row_data['pe_ratio'] = latest_fundamentals[col_name]
else:
    row_data['pe_ratio'] = 0.0  # Default value
```

**Strategy**: Fill with 0 (post-normalization, 0 means "average")

---

## Performance Considerations

### Memory Usage

**Rolling Correlation**:
```
N stocks × T days × (N-1)/2 pairs ≈ Memory
50 stocks × 2500 days × 1225 pairs = 3M entries × 8 bytes ≈ 24 MB
```

**Manageable** for modern systems

### Computation Time

**Bottleneck**: Rolling correlation
```
Time ≈ O(N² × T × W)
50² × 2500 × 30 ≈ 187.5M operations
```

**Actual Runtime**: ~1-2 minutes (Pandas optimization)

**If Scaling to 500 Stocks**:
- Time ×100 (500²/50²)
- May need optimization (parallel processing)

---

## Summary

**Purpose**: Calculate edge parameters for graph construction  
**Key Outputs**: Correlation time series, similarity matrix, sector connections  
**Design Philosophy**: Separate calculation from usage, optimize for medium-scale data  
**Integration**: Feeds into Phase 2 graph construction

**This file enables dynamic, time-varying graphs!** 📈

---

**Last Updated**: 2025-11-02  
**Code Style**: Beginner-friendly with mathematical explanations [[memory:3128464]]  
**Language**: English comments [[memory:3128459]]


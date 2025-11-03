# Phase 1: Feature Engineering - Implementation Guide

## Overview

**File**: `scripts/phase1_feature_engineering.py`  
**Purpose**: Transform raw data into engineered features for GNN training  
**Dependencies**: pandas, numpy, talib, scikit-learn, scipy  
**Input**: Raw CSV files from phase1_data_collection.py  
**Output**: Consolidated node features (`node_features_X_t_final.csv`) + edge parameters

---

## What Does This File Do?

This script is the **feature engineering pipeline** that transforms raw stock data into:

1. **Technical Indicators** (RSI, MACD, Bollinger Bands, etc.)
2. **Normalized Fundamental Features** (P/E, ROE with log transforms and z-score)
3. **Dynamic Edge Parameters** (rolling correlations, fundamental similarity)
4. **Static Edge Data** (sector/industry connections)

---

## Why This Design?

### Core Philosophy

**Separation of Data Processing Layers**:
```
Raw Data → Feature Engineering → Graph Construction → Model Training
(Phase 1a)  (Phase 1b - THIS FILE) (Phase 2)        (Phase 3/4)
```

**Why Separate**: Different layers serve different purposes:
- **Raw Data**: Minimal processing, reusable
- **Feature Engineering**: Domain knowledge applied
- **Graph Construction**: GNN-specific structure
- **Model Training**: Learning algorithms

---

## Key Components

### 1. Load and Align Data

```python
def load_raw_data():
    """Load raw data files from 'data/raw' directory."""
```

#### What It Does
Loads three types of raw data:
1. OHLCV (stock prices)
2. Fundamentals (P/E, ROE)
3. Sentiment/Macro (VIX, sentiment scores)

#### Why Wide Format is Used

**Data Structure**:
```
Date       | Close_AAPL | Close_MSFT | RSI_14_AAPL | RSI_14_MSFT
2020-01-01 | 75.08      | 160.5      | 65.3        | 52.1
```

**Advantages**:
- All stocks aligned by date (row index)
- Easy to compute cross-stock features (correlations)
- Matches graph node feature matrix format: `[N_stocks, N_features]`

---

### 2. Data Alignment

```python
def align_data(ohlcv_df, fund_df, sentiment_macro_df):
    """Aligns all time series data to the daily trading calendar."""
```

#### The Alignment Challenge

**Problem**: Different data frequencies:
- OHLCV: Daily (trading days)
- Fundamentals: Quarterly
- Sentiment: Daily (but may have gaps)

**Solution**: Forward-fill (ffill)
```python
fund_aligned = fund_df.reindex(date_index).ffill()
```

#### Why Forward-Fill?

**Mathematical Justification**:
- Fundamentals (P/E, ROE) change slowly
- Between updates, using last known value is reasonable
- Alternative (linear interpolation) adds artificial precision

**Example**:
```
Quarterly Fundamental Data:
2020-01-01: P/E = 25.0
2020-04-01: P/E = 26.5

Forward-fill for daily:
2020-01-01: 25.0
2020-01-02: 25.0
...
2020-03-31: 25.0
2020-04-01: 26.5
```

**Why Not Interpolate**:
```
Linear interpolation would give:
2020-02-15: 25.75  ← Artificial precision!
```

Fundamentals don't change linearly between quarters.

---

### 3. Technical Indicators

```python
def calculate_technical_indicators(aligned_data, tickers):
    """Calculate technical indicators for each stock."""
```

#### Implemented Indicators

##### 1. Log Returns (Multiple Horizons)

**Formula**:
```
LogReturn_t = ln(P_t / P_{t-w})
```

**Why Log Returns**:
- **Additive**: `r_{1-2} + r_{2-3} = r_{1-3}`
- **Symmetric**: Up 10% then down 10% ≠ break even
  - Linear: `1.1 × 0.9 = 0.99` (loss)
  - Log: `ln(1.1) + ln(0.9) ≈ 0` (symmetric)
- **Normal Distribution**: Better statistical properties

**Horizons** (1, 5, 20 days):
- 1-day: Momentum/volatility
- 5-day: Weekly pattern
- 20-day: Monthly trend

##### 2. Volatility (30-day Annualized)

**Formula**:
```
σ_{annual} = σ_{daily} × √252
```

**Why √252**:
- 252 = average trading days per year
- If daily returns are i.i.d.:
  - `Var(Year) = 252 × Var(Day)`
  - `Std(Year) = √252 × Std(Day)`

**Mathematical Derivation**:
```
Let r_i be daily returns
Annual return R = Σ r_i

Assuming independence:
Var(R) = Var(Σ r_i) = Σ Var(r_i) = 252 × Var(r)

σ_annual = √Var(R) = √(252 × Var(r)) = √252 × σ_daily
```

##### 3. RSI (Relative Strength Index)

**Formula**:
```
RS = Average Gain / Average Loss (over 14 days)
RSI = 100 - (100 / (1 + RS))
```

**Interpretation**:
- RSI > 70: Overbought (potential reversal down)
- RSI < 30: Oversold (potential reversal up)
- RSI = 50: Neutral

**Why RSI Matters for GNNs**:
- Captures momentum state
- Similar RSI → Similar momentum → Potential edge
- Cross-stock RSI patterns reveal market regimes

##### 4. MACD (Moving Average Convergence Divergence)

**Components**:
```
MACD_line = EMA_12 - EMA_26
Signal_line = EMA_9(MACD_line)
Histogram = MACD_line - Signal_line
```

**Trading Signals**:
- MACD crosses above signal: Bullish
- MACD crosses below signal: Bearish
- Histogram magnitude: Momentum strength

##### 5. Bollinger Bands Width

**Formula**:
```
BB_Width = (Upper_Band - Lower_Band) / Middle_Band

where:
Upper = SMA_20 + 2σ
Middle = SMA_20
Lower = SMA_20 - 2σ
```

**Interpretation**:
- Wide bands: High volatility
- Narrow bands: Low volatility (potential breakout)
- Bands contain ~95% of price moves (2σ rule)

##### 6. ATR (Average True Range)

**Formula**:
```
True Range_t = max(High_t - Low_t, 
                   |High_t - Close_{t-1}|,
                   |Low_t - Close_{t-1}|)

ATR_14 = EMA_14(True Range)
```

**Why ATR**:
- Volatility measure that accounts for gaps
- Higher ATR → Higher risk
- Used for position sizing in trading

---

### 4. Fundamental Normalization

```python
def normalize_fundamentals_and_sentiment(aligned_data, tickers):
    """Standardizes fundamental, sentiment, and macro features."""
```

#### Why Normalize?

**Problem**: Features have different scales:
- P/E Ratio: 5-50
- ROE: 0.01-0.50
- VIX: 10-80

**Solution**: Z-score normalization

**Formula**:
```
z = (x - μ) / σ
```

**Benefits**:
1. **Equal importance** in GNN aggregation
2. **Stable gradients** during training
3. **Interpretability**: z = 2 means "2 std above average"

#### Log Transformation

**Why Log Transform P/E**:
```python
df_to_scale[f'{col}_Log'] = np.log(df_to_scale[col].abs() + 1)
```

**Mathematical Reason**:
- P/E is **multiplicative** not additive
- Company A: P/E = 10, Company B: P/E = 20
- Linear: Difference = 10
- Log: Difference = ln(20) - ln(10) = 0.693
  - Represents: B is 2× more expensive
  - More meaningful for valuation

**Why Add 1**:
- Handles P/E = 0 (undefined otherwise)
- `ln(1) = 0` (neutral point)

#### StandardScaler Mathematics

**Formula**:
```
For each feature column:
1. Calculate: μ = mean, σ = std
2. Transform: x_scaled = (x - μ) / σ
```

**Result**: Features have μ=0, σ=1

**Why Useful for GNNs**:
- Message passing sums features
- Without normalization: large-scale features dominate
- With normalization: all features contribute equally

---

### 5. Dynamic Edge Parameters

#### 5a. Rolling Correlation

```python
rolling_corr_matrix = log_returns_daily.rolling(window=30).corr()
```

**What It Computes**:
For each pair of stocks (i, j) and each date t:
```
ρ_{ij}(t) = Corr(r_i[t-29:t], r_j[t-29:t])
```

**Why 30-day Window**:
- Too short (e.g., 5 days): Noisy, unstable
- Too long (e.g., 90 days): Misses regime changes
- 30 days ≈ 1 month: Balance between stability and responsiveness

**Mathematical Properties**:
```
-1 ≤ ρ ≤ 1

ρ = 1:  Perfect positive correlation
ρ = 0:  No correlation
ρ = -1: Perfect negative correlation
```

**Why Absolute Correlation for Edges**:
```python
rolling_corr_df['abs_correlation'] = abs(correlation)
```

**Reason**: Both positive and negative correlations indicate relationship
- High positive: Stocks move together (same sector)
- High negative: Stocks hedge each other (e.g., oil vs airlines)

#### 5b. Fundamental Similarity

```python
sim = 1 - cosine(v_i, v_j)
```

**Cosine Similarity Formula**:
```
cos(θ) = (A · B) / (||A|| ||B||)

where:
A · B = Σ a_i × b_i (dot product)
||A|| = √(Σ a_i²) (Euclidean norm)
```

**Why Cosine Similarity**:
- **Scale invariant**: Small vs large companies comparable
- **Directional**: Focuses on ratio patterns, not magnitudes
- **Range [0, 1]**: Easy to interpret

**Example**:
```
Stock A: [P/E=20, ROE=0.15]
Stock B: [P/E=40, ROE=0.30]

Scaled to unit vectors:
A_norm = [0.8, 0.6]
B_norm = [0.8, 0.6]  # Same direction!

cos(θ) = 0.8×0.8 + 0.6×0.6 = 1.0 (perfect similarity)
```

Both stocks have same valuation/profitability **profile**.

---

### 6. Static Edge Data

```python
def calculate_static_edge_data(tickers):
    """Calculate static edge relationships."""
```

#### Edge Weighting Strategy

**Sector Connection**: Weight = 0.8
- Strongest relationship
- Same economic forces
- Example: All tech stocks affected by chip shortage

**Industry Connection**: Weight = 0.6
- Moderate relationship
- Similar business models
- Example: Apple (hardware) vs Adobe (software) - both tech, different industries

**Market Cap Tier**: Weight = 0.2
- Weak relationship
- Similar liquidity/volatility profiles
- Example: Large caps tend to be less volatile

**Why Additive**:
```python
edge_weight += 0.8  # Same sector
edge_weight += 0.6  # Same industry
# Total = 1.4 (capped at 1.0)
```

**Rationale**: Multiple connections → stronger edge

---

### 7. Consolidate Node Features

```python
def consolidate_node_features(technical_df, normalized_df, tickers):
    """Consolidate all feature types into single X_t matrix."""
```

#### Final Feature Matrix Structure

**Format**: Wide DataFrame
```
Date       | LogRet_1d_AAPL | RSI_14_AAPL | AAPL_PE_Log | ... 
2020-01-01 | 0.012          | 65.3        | 3.21        | ...
```

**Dimensions**:
- Rows: Trading days (T ≈ 2500)
- Columns: Features × Stocks (F × N ≈ 20 × 50 = 1000)

**Why Inner Join**:
```python
final_X_t = technical_df.join(normalized_df, how='inner')
```

**Reason**: Only keep dates where ALL features exist
- Technical indicators need 50-day lookback
- First 50 days incomplete → drop
- Ensures quality data for GNN training

---

## Data Flow Diagram

```
┌─────────────────────────────────────┐
│   Raw Data Files                    │
│   - OHLCV (daily prices)           │
│   - Fundamentals (quarterly)        │
│   - Sentiment (daily VIX)           │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│   Data Alignment                    │
│   - Reindex to trading days         │
│   - Forward-fill fundamentals       │
│   - Fill missing values             │
└─────────────┬───────────────────────┘
              │
        ┌─────┴─────┐
        │           │
        ▼           ▼
┌──────────────┐ ┌──────────────────┐
│  Technical   │ │  Fundamental     │
│  Indicators  │ │  Normalization   │
│  - RSI       │ │  - Log transform │
│  - MACD      │ │  - Z-score       │
│  - Volatility│ │  - StandardScale │
└──────┬───────┘ └────────┬─────────┘
       │                  │
       └────────┬─────────┘
                │
                ▼
┌─────────────────────────────────────┐
│   Consolidated Node Features        │
│   Output: node_features_X_t_final.csv│
│   Shape: [T_days, F×N_stocks]       │
└─────────────────────────────────────┘

Meanwhile:
┌─────────────────────────────────────┐
│   Edge Parameter Calculation        │
│   - Rolling correlation (30-day)    │
│   - Fundamental similarity (cosine) │
│   - Static connections (sector)     │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│   Edge Parameter Files              │
│   - edges_dynamic_corr_params.pkl   │
│   - edges_dynamic_fund_sim_params.csv│
│   - edges_static_connections.csv    │
└─────────────────────────────────────┘
```

---

## Mathematical Deep Dive

### Why TA-Lib?

**TA-Lib** (Technical Analysis Library) is the industry standard:
- **Correctness**: Tested implementations
- **Performance**: C-based, highly optimized
- **Standardization**: Everyone uses same formulas

**Example: EMA Calculation**

**Naive Implementation** (slow):
```python
for i in range(len(prices)):
    ema[i] = alpha * prices[i] + (1-alpha) * ema[i-1]
```

**TA-Lib** (fast):
- Vectorized operations
- Cache-friendly memory access
- 10-100× faster

### EMA vs SMA

**SMA** (Simple Moving Average):
```
SMA_t = (P_t + P_{t-1} + ... + P_{t-n+1}) / n
```

**EMA** (Exponential Moving Average):
```
EMA_t = α × P_t + (1-α) × EMA_{t-1}

where: α = 2 / (n + 1)
```

**Why EMA is Better**:
1. **More responsive** to recent prices
2. **Smooth transition** (no sudden jumps when old data drops out)
3. **Infinite memory** (all past prices contribute, with decay)

**Weight Decay**:
```
EMA_20:
Today:     weight = 0.095
Yesterday: weight = 0.086
2 days ago: weight = 0.078
...
20 days ago: weight = 0.013 (still contributes!)
```

**SMA_20**:
```
Last 20 days: weight = 0.05 each
Day 21: weight = 0 (suddenly dropped!)
```

---

## Error Handling Strategy

### 1. Missing Tickers

```python
try:
    open_prices = aligned_data[f'Open_{ticker}'].dropna()
except Exception as e:
    print(f"❌ Error calculating for {ticker}: {e}. Skipping.")
```

**Strategy**: Continue with other tickers
**Impact**: Reduces stock count but doesn't crash pipeline

### 2. Insufficient Data

```python
technical_features = technical_features.dropna(how='all')
```

**Why `how='all'`**: Only drop rows where **all** values are NaN
- Preserves rows with partial data
- Later inner join will filter appropriately

### 3. Division by Zero

```python
pe_values.append(max(5.0, pe_value * pe_variation))
roe_values.append(max(0.01, min(0.5, roe_value * roe_variation)))
```

**Bounds Protection**:
- P/E: Min 5.0 (prevents nonsensical negative/zero)
- ROE: Range [0.01, 0.50] (1%-50%, realistic bounds)

---

## Output Files

### 1. `node_features_X_t_final.csv`

**Shape**: `[T, F×N]` where:
- T ≈ 2450 trading days (after dropna)
- F ≈ 20 features per stock
- N ≈ 50 stocks
- Total columns ≈ 1000

**Usage in GNN**:
```python
# Reshape for graph: [N, F]
X_t = df.loc[date].values.reshape(N, F)
```

### 2. `edges_dynamic_corr_params.pkl`

**Format**: Multi-Index Series
```
Date        ticker1  ticker2
2020-01-01  AAPL     MSFT      0.65
2020-01-01  AAPL     GOOGL     0.72
...
```

**Size**: ~1250 pairs × 2450 days ≈ 3M entries

### 3. `edges_static_connections.csv`

**Format**:
```
ticker1, ticker2, static_weight, connection_types, sector1, sector2
AAPL,    MSFT,    1.4,           same_sector,      Tech,    Tech
```

---

## Best Practices

### ✅ 1. Feature Scaling

```python
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
```

**Critical for GNNs**: Without scaling, aggregation biased toward large-scale features

### ✅ 2. Consistent Feature Dimensions

```python
# Ensure all stocks have same feature vector length
for ticker in tickers:
    ticker_features_vector = []
    for feature_prefix in universal_features:
        # Add feature or 0.0 if missing
```

**Why**: GNN requires uniform node feature dimension

### ✅ 3. Temporal Consistency

```python
# Don't use future information
forward_returns = close_prices.shift(-5)  # Correct
backward_returns = close_prices.shift(5)  # Would be cheating!
```

**Data Leakage Prevention**: Never use t+1 information to predict t

---

## Limitations & Future Improvements

### Current Limitations

1. **TA-Lib Dependency**
   - Installation can be tricky
   - Platform-dependent compilation

2. **Memory Usage**
   - Wide format uses more memory
   - 50 stocks × 20 features × 2500 days = manageable
   - 500 stocks would be problematic

3. **Feature Selection**
   - Currently uses all calculated features
   - No automatic feature importance analysis

### Future Improvements

1. **Automated Feature Selection**
   - Mutual information for relevance
   - Remove redundant features (high correlation)

2. **Dynamic Feature Engineering**
   - Add interaction terms
   - Polynomial features for non-linear relationships

3. **Alternative Normalizations**
   - Rank transformation (robust to outliers)
   - Quantile normalization (handles skewed distributions)

---

## Integration with Next Phase

**Output of this file** → **Input to Phase 2 (Graph Construction)**

Phase 2 will:
1. Load `node_features_X_t_final.csv`
2. Load edge parameter files
3. For each date t:
   - Extract X_t from node features
   - Filter edges based on correlation > threshold
   - Build PyG `HeteroData` object
   - Save as `graph_t_YYYYMMDD.pt`

---

## Summary

**Purpose**: Engineer domain-specific features from raw data  
**Key Techniques**: Technical analysis, normalization, correlation, similarity  
**Output**: Ready-to-use node features + edge parameters  
**Design**: Modular, well-documented, production-ready

**This file bridges raw data and GNN-ready graphs!** 🌉

---

**Last Updated**: 2025-11-02  
**Code Style**: Beginner-friendly with mathematical explanations [[memory:3128464]]  
**Language**: English comments [[memory:3128459]]


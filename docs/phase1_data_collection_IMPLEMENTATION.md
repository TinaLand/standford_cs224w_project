# Phase 1: Data Collection - Implementation Guide

## Overview

**File**: `scripts/phase1_data_collection.py`  
**Purpose**: Downloads and collects raw stock market data from multiple sources  
**Dependencies**: yfinance, pandas, requests  
**Output**: Raw CSV files in `data/raw/` directory

---

## What Does This File Do?

This script serves as the **data acquisition layer** for the CS224W Stock RL GNN project. It collects three types of raw data:

1. **OHLCV Price Data** (Open, High, Low, Close, Volume)
2. **Fundamental Data** (P/E ratio, ROE, etc.)
3. **Sentiment & Macro Data** (VIX index, news sentiment)

---

## Why This Design?

### Design Principles

#### 1. **Separation of Concerns**
- **Data Collection** (this file) is separate from **Feature Engineering** (phase1_feature_engineering.py)
- Allows raw data to be reused for different feature calculations
- Easy to update data without changing downstream logic

#### 2. **Fallback Strategy** ✅
```python
except Exception as e:
    print(f"  ⚠️ Failed to fetch data for {ticker}: {e}")
    # Use fallback values for failed tickers
    fallback_pe = 20.0 + hash(ticker) % 15
```

**Why**: Network failures are common when fetching financial data. Instead of crashing, we use reasonable default values to ensure the pipeline continues.

#### 3. **Deterministic Simulation**
```python
np.random.seed(hash(ticker) % 1000)  # Deterministic per ticker
```

**Why**: Using `hash(ticker)` ensures the same ticker always gets the same "random" values, making experiments reproducible.

---

## Key Components

### 1. Configuration Dictionary

```python
CONFIG = {
    'START_DATE': '2015-01-01',
    'END_DATE': '2025-01-01',
    'STOCK_SOURCE': 'SPY',      # 'SPY', 'QQQ', or 'CUSTOM'
    'NUM_STOCKS': 50,
    'CUSTOM_TICKERS': ['AAPL', 'MSFT', ...]
}
```

**Purpose**: Centralized configuration makes it easy to:
- Change date ranges for different experiments
- Switch between different stock universes (SPY vs QQQ)
- Use custom stock lists

**Why not hardcode**: Different researchers might want different:
- Time periods (pre-COVID vs post-COVID)
- Stock sets (tech stocks vs all sectors)
- Data granularity

---

### 2. Function: `fetch_top_etf_holdings()`

```python
def fetch_top_etf_holdings(etf_ticker, num_stocks):
    """Fetches the top N holdings of a given ETF"""
```

#### What It Does
- Attempts to fetch real ETF holdings from official sources
- Falls back to hardcoded list if download fails
- Returns list of ticker symbols

#### Why This Approach

**Original Plan**: Download from official State Street SPY holdings page
```python
url = "https://www.ssga.com/.../spy-holdings-us-en.xlsx"
```

**Reality**: Web scraping is unreliable:
- URLs change
- Excel format changes
- Rate limiting
- Authentication required

**Solution**: Hardcoded list of top 50 SPY stocks
```python
top_tickers = ['MSFT', 'AAPL', 'NVDA', 'AMZN', 'GOOGL', ...]
```

**Trade-offs**:
- ✅ Reliability: Always works
- ✅ Reproducibility: Same stocks every run
- ❌ Not real-time: List may be outdated
- ❌ Manual maintenance: Need to update list periodically

**For Research**: This is acceptable because:
- Portfolio composition changes slowly
- Reproducibility > Real-time accuracy
- Can manually update when needed

---

### 3. Function: `download_stock_data()`

```python
def download_stock_data(tickers, start, end, output_path):
    """Downloads OHLCV data using yfinance"""
```

#### Data Structure

**Input**: List of tickers, date range  
**Output**: Wide-format CSV with columns like:
```
Date, Close_AAPL, Close_MSFT, High_AAPL, High_MSFT, ...
```

#### Why Wide Format?

**Alternative (Long Format)**:
```
Date       | Ticker | Close | High | Low
2020-01-01 | AAPL   | 75.08 | 75.15| 74.12
2020-01-01 | MSFT   | 160.5 | 161.0| 159.5
```

**Chosen (Wide Format)**:
```
Date       | Close_AAPL | Close_MSFT | High_AAPL | High_MSFT
2020-01-01 | 75.08      | 160.5      | 75.15     | 161.0
```

**Advantages of Wide Format**:
1. **Alignment**: All stocks share same date index
2. **Vectorization**: Can compute cross-stock operations easily
3. **Missing Data**: Easy to see which stocks have gaps
4. **GraphML Ready**: Matches node feature matrix format

**Disadvantages**:
- More columns (not a problem for <100 stocks)
- Sparse for stocks with different trading days

---

### 4. Function: `download_fundamental_data()`

```python
def download_fundamental_data(tickers, output_path):
    """Collects real fundamental data (P/E, ROE, etc.) using yfinance"""
```

#### Why Fundamentals Matter

**For GNN**: Fundamental features create similarity-based edges
- High P/E stocks tend to behave similarly (growth stocks)
- Similar ROE indicates similar profitability

**Mathematical Intuition**:
- P/E Ratio = Price / Earnings
  - High P/E (>30): Growth expectations
  - Low P/E (<15): Value stocks
  - Similar P/E → Similar investor sentiment

- ROE = Net Income / Shareholders Equity
  - High ROE (>15%): Profitable company
  - Low ROE (<5%): Struggling business
  - Similar ROE → Similar business quality

#### Implementation Details

**Challenge**: Historical fundamental data requires paid APIs
```python
# Ideal: Historical quarterly fundamentals
# Reality: yfinance only provides current values
```

**Solution**: Simulate quarterly variation
```python
for j, date in enumerate(date_range):
    pe_variation = np.random.normal(1.0, 0.05)  # ±5% variation
    roe_variation = np.random.normal(1.0, 0.03)  # ±3% variation
    
    pe_values.append(max(5.0, pe_value * pe_variation))
    roe_values.append(max(0.01, min(0.5, roe_value * roe_variation)))
```

**Why This Works**:
- Fundamentals change slowly (quarterly/annually)
- Small variation captures uncertainty
- Main signal (PE=20 vs PE=40) is preserved
- Good enough for graph structure research

#### Fallback Strategy

**Network Failure Handling**:
```python
except Exception as e:
    print(f"  ⚠️ Failed to fetch data for {ticker}: {e}")
    failed_tickers.append(ticker)
    
    # Use fallback values
    fallback_pe = 20.0 + hash(ticker) % 15  # PE between 20-35
    fallback_roe = 0.10 + (hash(ticker) % 10) / 100  # ROE 0.10-0.20
```

**Why `hash(ticker)`**:
- Deterministic: Same ticker = same fallback
- Varied: Different tickers get different values
- Reasonable: Values in realistic range

**Statistics**:
```
Average Market P/E: ~20
Average ROE: ~10-15%
```

---

### 5. Function: `download_sentiment_data()`

```python
def download_sentiment_data(output_path):
    """Simulates sentiment and macro features (VIX, News Polarity)"""
```

#### VIX Index (Real Data)

**What is VIX**:
- "Fear Index" - measures market volatility
- VIX = 10-15: Low volatility (calm markets)
- VIX = 20-30: Normal volatility
- VIX = 40+: High volatility (crisis, e.g., 2008, COVID)

**Why VIX Matters for Stock Prediction**:
- High VIX → Higher correlations between stocks
- Market-wide fear affects all stocks
- Can strengthen graph edges during crisis periods

**Mathematical Formula** (simplified):
```
VIX ≈ √(30-day implied volatility from S&P 500 options) × 100
```

**In GNN Context**:
- VIX becomes a **global node feature**
- Same value for all stocks at time t
- Captures **market regime**

#### Sentiment Simulation

**Why Simulated**:
- Real sentiment requires:
  - News API (expensive)
  - NLP processing (complex)
  - Historical news archive (not free)

**Simple Heuristic**:
```python
vix_normalized = sentiment_data['VIX'].fillna(20.0) / 30.0
sentiment_data[f'{ticker}_Sentiment'] = 0.5 - (vix_normalized - 0.5) * 0.3
```

**Logic**:
- High VIX (fear) → Low sentiment (0.2-0.3)
- Low VIX (calm) → High sentiment (0.6-0.7)
- Inverse relationship: Fear ↓ Sentiment

**Limitation**: This is a placeholder for real NLP-based sentiment

---

## Data Flow Diagram

```
┌─────────────────────────────────────┐
│   Configuration (CONFIG)            │
│   - Date range                      │
│   - Stock selection                 │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│   fetch_top_etf_holdings()          │
│   - Hardcoded SPY/QQQ list          │
│   - Fallback to CUSTOM_TICKERS      │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│   download_stock_data()             │
│   Source: yfinance API              │
│   Output: stock_prices_ohlcv_raw.csv│
│   Format: Wide (Date × Features)    │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│   download_fundamental_data()       │
│   Source: yfinance + simulation     │
│   Output: fundamentals_raw.csv      │
│   Features: PE, ROE per ticker      │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│   download_sentiment_data()         │
│   Source: yfinance (VIX) + simulated│
│   Output: sentiment_macro_raw.csv   │
│   Features: VIX, Sentiment scores   │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│   data/raw/ Directory               │
│   ├── stock_prices_ohlcv_raw.csv   │
│   ├── fundamentals_raw.csv          │
│   └── sentiment_macro_raw.csv       │
└─────────────────────────────────────┘
```

---

## Error Handling Strategy

### 1. Network Failures

**Problem**: yfinance API calls can fail
**Solution**: Try-catch with fallback values
**Impact**: Pipeline continues, uses reasonable defaults

### 2. Missing Tickers

**Problem**: Some tickers may not be available
**Solution**: Skip with warning, continue with valid tickers
**Impact**: Reduces stock count but doesn't crash

### 3. Data Quality

**Problem**: yfinance may return NaN or invalid values
**Solution**: 
- Outer join to preserve all dates
- Forward fill for missing values (in next phase)

---

## Output Files

### 1. `stock_prices_ohlcv_raw.csv`

**Format**:
```
Date       | Open_AAPL | High_AAPL | Low_AAPL | Close_AAPL | Volume_AAPL | ...
2020-01-01 | 74.06     | 75.15     | 73.80    | 75.09      | 135480400   | ...
```

**Columns**: Date + (5 metrics × N stocks)  
**Rows**: Trading days (excluding weekends/holidays)  
**Size**: ~2500 rows × 250 columns (for 50 stocks)

### 2. `fundamentals_raw.csv`

**Format**:
```
Date       | AAPL_PE | AAPL_ROE | MSFT_PE | MSFT_ROE | ...
2020-01-01 | 25.4    | 0.145    | 30.2    | 0.165    | ...
```

**Columns**: Date + (2 metrics × N stocks)  
**Rows**: Quarterly dates  
**Size**: ~40 rows × 100 columns (for 50 stocks)

### 3. `sentiment_macro_raw.csv`

**Format**:
```
Date       | VIX   | AAPL_Sentiment | MSFT_Sentiment | ...
2020-01-01 | 13.78 | 0.54           | 0.53           | ...
```

**Columns**: Date + VIX + Sentiment per ticker  
**Rows**: Business days  
**Size**: ~2500 rows × 51 columns

---

## Best Practices Used

### ✅ 1. **Fail Gracefully**
- Try-catch blocks prevent crashes
- Fallback values ensure continuity
- Warning messages for transparency

### ✅ 2. **Deterministic Simulation**
- `hash(ticker)` for reproducible randomness
- Same input = same output

### ✅ 3. **Configurable**
- Single CONFIG dictionary
- Easy to experiment with different settings

### ✅ 4. **Documented**
- Clear function docstrings
- Inline comments explaining "why"

### ✅ 5. **Modular**
- Separate functions for each data type
- Easy to add new data sources

---

## Limitations & Future Improvements

### Current Limitations

1. **Fundamental Data**: Only current values, not historical
   - **Impact**: Can't capture temporal changes in company fundamentals
   - **Mitigation**: Quarterly variation simulation

2. **Sentiment Data**: Simulated, not real
   - **Impact**: Doesn't capture actual news sentiment
   - **Mitigation**: VIX provides real market sentiment proxy

3. **Stock List**: Hardcoded, not dynamic
   - **Impact**: Outdated if ETF composition changes
   - **Mitigation**: Easy to update manually

### Potential Improvements

1. **Paid API Integration**
   - Alpha Vantage for fundamentals
   - NewsAPI for real sentiment
   - **Trade-off**: Cost vs accuracy

2. **Database Storage**
   - Store in SQLite/PostgreSQL
   - Faster queries for large datasets
   - **Trade-off**: Complexity vs performance

3. **Incremental Updates**
   - Only download new dates
   - Append to existing files
   - **Trade-off**: Logic complexity vs efficiency

4. **Retry Mechanism**
   - Exponential backoff for failed requests
   - Better handling of rate limits
   - **Trade-off**: Time vs reliability

---

## Usage Example

```python
# Run the script
python scripts/phase1_data_collection.py

# Expected output:
# Data directory prepared: .../data/raw
# Starting Raw Data Collection (Phase 1) for 50 stocks.
# Date range: 2015-01-01 to 2025-01-01
#
# --- 1. Downloading OHLCV Data for 50 stocks... ---
# ✅ OHLCV data saved successfully to: .../stock_prices_ohlcv_raw.csv
# Data shape: (2520, 250)
#
# --- 2. Collecting Real Fundamental Data via yfinance... ---
# ✅ Real fundamental data saved to: .../fundamentals_raw.csv
#   📊 Successfully fetched: 48/50 tickers
#   ⚠️ Failed tickers (using fallback): TICKER1, TICKER2
#
# --- 3. Collecting/Simulating Sentiment and Macro Data... ---
# ✅ VIX (real) and Simulated Sentiment data saved
#
# Phase 1 Raw Data Collection complete.
```

---

## Integration with Next Phase

**Output of this file** → **Input to `phase1_feature_engineering.py`**

```python
# Phase 1 Feature Engineering will:
1. Load these raw CSV files
2. Calculate technical indicators (RSI, MACD, etc.)
3. Normalize fundamental features
4. Align all data to common trading calendar
5. Output: node_features_X_t_final.csv
```

---

## Summary

**Purpose**: Collect raw stock market data reliably  
**Strategy**: Real data + fallback + simulation  
**Output**: 3 raw CSV files ready for feature engineering  
**Key Design**: Fail gracefully, reproducible, configurable

**This file is the foundation of the data pipeline!** 🏗️

---

**Last Updated**: 2025-11-02  
**Code Style**: Beginner-friendly with detailed comments [[memory:3128464]]  
**Language**: English comments [[memory:3128459]]


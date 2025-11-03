# Phase 1: Static Data Collection - Implementation Guide

## Overview

**File**: `scripts/phase1_static_data_collection.py`  
**Purpose**: Collect static (time-invariant) edge data for graph construction  
**Dependencies**: pandas, numpy  
**Input**: Ticker list from OHLCV data  
**Output**: Static relationship files (sector, supply chain, competitor)

---

## What Does This File Do?

This script simulates **static edge data** that doesn't change over time:

1. **Sector/Industry Classification** - Which companies are in same sector/industry
2. **Supply Chain Relationships** - Customer-supplier connections
3. **Competitor Relationships** - Companies competing in same market

---

## Why Separate from Other Phase 1 Scripts?

### Design Rationale

**Phase 1 Organization**:
- `phase1_data_collection.py`: Time-series data (OHLCV, fundamentals)
- `phase1_feature_engineering.py`: Feature calculations
- `phase1_edge_parameter_calc.py`: Dynamic edge calculations
- **`phase1_static_data_collection.py`**: Static edge data ← THIS FILE

**Why Separate File**:
1. **Different Data Sources**: Static data comes from different APIs/databases
2. **Different Update Frequency**: Updated rarely (annual vs daily)
3. **Optional**: Can run project without static edges (for ablation studies)
4. **Independent Execution**: Can be skipped if only using dynamic edges

---

## Key Components

### 1. Get Ticker List

```python
def get_ticker_list():
    """Reads ticker list from OHLCV raw data file."""
```

#### Why Read from OHLCV File?

**Dependency Chain**:
```
phase1_data_collection.py → Determines which tickers
                           ↓
                OHLCV file stores the tickers
                           ↓
phase1_static_data_collection.py → Uses same tickers
```

**Benefit**: Ensures consistency
- Static edges for exactly the stocks we have price data for
- No missing tickers or mismatches

**Alternative (Bad)**:
```python
tickers = ['AAPL', 'MSFT', ...]  # Hardcoded again
```
- Risk of mismatch if phase1_data_collection.py changes
- Violates DRY (Don't Repeat Yourself) principle

---

### 2. Sector/Industry Classification

```python
def download_sector_industry_data(tickers, output_path):
    """Simulates collection of Sector/Industry classification data."""
```

#### Why Simulated?

**Real Implementation Would Need**:
```python
# Option 1: SEC EDGAR database
sector = query_sec_edgar(ticker)

# Option 2: Yahoo Finance API
info = yf.Ticker(ticker).info
sector = info['sector']

# Option 3: Paid data provider (Bloomberg, FactSet)
sector = bloomberg_api.get_sector(ticker)
```

**Why Simulate Instead**:
1. **SEC EDGAR**: Complex parsing, rate limited
2. **Yahoo Finance**: Inconsistent, sometimes missing
3. **Paid APIs**: Requires subscription

**For Research Project**: Simulation is acceptable
- Demonstrates the methodology
- Can be replaced with real data later
- Main research focus is on **GNN architecture**, not data sourcing

#### Simulation Methodology

```python
sectors = ['Technology', 'Financials', 'Healthcare', 'Energy', 'Consumer Discretionary']
industries_map = {
    'Technology': ['Software', 'Hardware', 'Semiconductors'],
    'Financials': ['Banking', 'Insurance'],
    # ...
}

for ticker in tickers:
    sector = random.choice(sectors)
    industry = random.choice(industries_map[sector])
```

**Why Random**:
- For this assignment/project: Demonstrates concept
- Real implementation would query actual data

**Improvement for Production**:
```python
# Hardcode known sectors for major stocks
known_sectors = {
    'AAPL': {'sector': 'Technology', 'industry': 'Consumer Electronics'},
    'MSFT': {'sector': 'Technology', 'industry': 'Software'},
    'JPM': {'sector': 'Financials', 'industry': 'Banking'},
    # ...
}
```

#### Output Format

```
Ticker, Sector,              Industry
AAPL,   Technology,          Software
MSFT,   Technology,          Software
JPM,    Financials,          Banking
```

**Usage in Phase 2**:
```python
# Stocks in same sector → Create edge
if sector1 == sector2:
    add_edge(i, j, weight=1.0, type='sector')
```

---

### 3. Supply Chain & Competitor Data

```python
def download_supply_chain_competitor_data(tickers, output_path):
    """Simulates supply chain and competitor relationships."""
```

#### Why These Edges Matter

**Economic Theory**:

**Supply Chain**:
```
Customer Company → Supplier Company
(e.g., Apple → Foxconn, though Foxconn not in our stock list)
```

**Impact**: If customer's sales increase → supplier benefits
- Causal relationship (directed edge)
- Predictive power for GNN

**Competitor**:
```
Company A ↔ Company B
(e.g., Coca-Cola ↔ PepsiCo)
```

**Impact**: If competitor gains market share → company loses
- Negative relationship
- Substitute products

**GNN Benefit**: Message passing incorporates:
- Supplier signals propagate to customers
- Competitor performance affects predictions

#### Simulation Strategy

**Supply Chain**:
```python
num_supply_links = int(len(tickers) * 0.5)  # 50% of stocks
for _ in range(num_supply_links):
    customer = random.choice(tickers)
    supplier = random.choice([t for t in tickers if t != customer])
    edges.append({'Ticker1': customer, 'Ticker2': supplier, 
                  'Relation': 'SUPPLY_CHAIN', 'Weight': 1.0})
```

**Why 50%**: Sparse graph
- Real supply chains: Most companies have 3-10 major suppliers
- With 50 stocks: 25 edges is realistic

**Competitor**:
```python
num_comp_links = int(len(tickers) * 0.3)  # 30% of stocks
# Add both directions (undirected relationship)
edges.append({'Ticker1': comp1, 'Ticker2': comp2, 'Relation': 'COMPETITOR'})
edges.append({'Ticker1': comp2, 'Ticker2': comp1, 'Relation': 'COMPETITOR'})
```

**Why Bidirectional**: Competition is symmetric
- If A competes with B, then B competes with A
- Undirected edge in graph theory

#### Real-World Data Sources

**For Production**:
1. **Supply Chain**: 
   - Supplier lists from 10-K SEC filings
   - Bloomberg supply chain data
   - Web scraping company websites

2. **Competitors**:
   - Same GICS industry sub-classification
   - Market share data
   - Product overlap analysis

---

## Output Files

### 1. `static_sector_industry.csv`

**Format**:
```
Ticker, Sector,     Industry
AAPL,   Technology, Hardware
MSFT,   Technology, Software
JPM,    Financials, Banking
```

**Size**: N stocks → N rows  
**Usage**: Phase 2 creates edges for same-sector pairs

### 2. `static_supply_competitor_edges.csv`

**Format**:
```
Ticker1, Ticker2, Relation,      Weight
AAPL,    SUPPLIER, SUPPLY_CHAIN, 1.0
MSFT,    GOOGL,    COMPETITOR,   1.0
GOOGL,   MSFT,     COMPETITOR,   1.0
```

**Size**: ~40-50 edges (for 50 stocks)  
**Usage**: Phase 2 adds these as static edges in HeteroData

---

## Why This Matters for GNNs

### Message Passing with Multiple Edge Types

**Homogeneous GNN** (single edge type):
```
h_i^{(l+1)} = σ(Σ_{j∈N(i)} W^{(l)} h_j^{(l)})
```

**Heterogeneous GNN** (multiple edge types):
```
h_i^{(l+1)} = σ(Σ_{r∈R} Σ_{j∈N_r(i)} W_r^{(l)} h_j^{(l)})

where r ∈ {sector, competitor, correlation, ...}
```

**Benefit**: Different relationships have different weights
- Sector edge: Strong influence
- Competitor edge: May be negative influence
- Correlation edge: Time-varying influence

---

## Integration with Phase 2

### How Static Edges Are Used

**Phase 2 Graph Construction**:
```python
# Load static data
sector_df = pd.read_csv('static_sector_industry.csv')
supply_df = pd.read_csv('static_supply_competitor_edges.csv')

# For each date t:
for row in sector_df:
    if row['Sector1'] == row['Sector2']:
        graph['stock', 'sector', 'stock'].edge_index.append([i, j])

for row in supply_df:
    if row['Relation'] == 'SUPPLY_CHAIN':
        graph['stock', 'supply_chain', 'stock'].edge_index.append([i, j])
```

**Result**: HeteroData graph with multiple edge types

---

## Limitations & Future Work

### Current Limitations

1. **Simulated Data**: Not real supply chain/competitor relationships
2. **Static**: Sectors/competitors can change over time
3. **Incomplete**: Missing some relationship types (partnerships, etc.)

### Future Improvements

1. **Real Data Integration**:
   ```python
   def fetch_real_sector_data(ticker):
       info = yf.Ticker(ticker).info
       return info.get('sector', 'Unknown')
   ```

2. **Time-Varying Sectors**:
   - Companies can change sectors (e.g., Amazon: Retail → Tech)
   - Track sector changes over time

3. **Additional Relationships**:
   - Joint ventures
   - M&A history
   - Patent citations
   - Executive network (shared board members)

4. **Weighted Relationships**:
   ```python
   # Instead of binary 0/1
   competitor_strength = market_share_overlap(company1, company2)
   ```

---

## Best Practices

### ✅ 1. Consistency with Main Ticker List

```python
tickers = get_ticker_list()  # Read from OHLCV file
```

**Benefit**: Guaranteed alignment with price data

### ✅ 2. Bidirectional Edges for Symmetric Relations

```python
# Competitor relationship
edges.append({'Ticker1': comp1, 'Ticker2': comp2})
edges.append({'Ticker1': comp2, 'Ticker2': comp1})
```

**Benefit**: GNN can propagate info in both directions

### ✅ 3. Metadata Preservation

```python
edges.append({
    'ticker1': ticker1,
    'ticker2': ticker2,
    'Relation': 'COMPETITOR',
    'Weight': 1.0,
    'sector1': sector1,  # Extra metadata
    'sector2': sector2
})
```

**Benefit**: Can filter or analyze edges later

---

## Summary

**Purpose**: Provide static relationship data for graph edges  
**Approach**: Simulation with realistic structure  
**Output**: Two CSV files with sector and relationship data  
**Design**: Simple, extensible, integrates with Phase 2

**This file enables heterogeneous graph construction!** 🕸️

---

**Last Updated**: 2025-11-02  
**Code Style**: Beginner-friendly [[memory:3128464]]  
**Language**: English comments [[memory:3128459]]


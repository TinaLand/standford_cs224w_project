# CS224W Stock RL GNN Project

## Project Overview

This project implements a Graph Neural Network (GNN) approach for stock market prediction and reinforcement learning-based trading. The project is structured into phases:

- Phase 1: Data Collection & Feature Engineering
- Phase 2: Graph Construction
- Phase 3: Baseline GNN Training (baseline pipeline wired and runnable)
- Phase 4: Core Transformer Training (model and loop implemented)
- Phase 5: RL Integration (scripts scaffolded)
- Phase 6: Evaluation (scripts scaffolded)

## Project Structure

```
cs224_porject/
├── data/
│   ├── raw/
│   │   ├── stock_prices_ohlcv_raw.csv
│   │   ├── fundamentals_raw.csv
│   │   └── sentiment_macro_raw.csv
│   ├── processed/
│   │   └── node_features_X_t_final.csv     # Consolidated node features X_t
│   └── edges/
│       ├── edges_dynamic_corr_params.csv   # Rolling correlations (rho)
│       ├── edges_dynamic_fund_sim_params.csv # Fundamental similarity
│       └── static_sector_industry.csv      # Static sector/industry edges
├── models/
│   ├── baseline_gcn_model.pt               # Saved baseline model (Phase 3)
│   └── core_transformer_model.pt           # Saved core model (Phase 4)
├── scripts/
│   ├── components/
│   │   ├── pearl_embedding.py              # PEARL embedding components (optional)
│   │   └── transformer_layer.py            # Transformer layer components (optional)
│   ├── phase1_data_collection.py
│   ├── phase1_feature_engineering.py
│   ├── phase1_edge_parameter_calc.py
│   ├── phase2_graph_construction.py        # Builds daily HeteroData graphs
│   ├── phase3_baseline_training.py         # Baseline GNN (GAT) training
│   ├── phase4_core_training.py             # Core role-aware transformer training
│   ├── phase5_rl_integration.py            # RL integration scaffolding
│   ├── phase6_evaluation.py                # Evaluation & visualization
│   ├── rl_environment.py                   # Trading env for RL
│   └── utils_data.py
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.8+ 
- pip or conda package manager

### Installation

1. **Clone the repository** (if using git):
   ```bash
   git clone git@github.com:TinaLand/standford_cs224w_project.git
   cd cs224_porject
   ```

2. **Create and activate virtual environment** (recommended):
   ```bash
   # create virtual env
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running Phase 1: Data + Features

Execute the Phase 1 scripts in order:

1. **Data Collection**:
   ```bash
   cd scripts
   python phase1_data_collection.py
   ```
   - Downloads OHLCV data for selected stocks
   - Collects fundamental data (simulated)
   - Gathers sentiment and VIX data

2. **Feature Engineering**:
   ```bash
   python phase1_feature_engineering.py
   ```
   
   Features include:
   - Calculates technical indicators (RSI, MACD, Bollinger Bands)
   - Normalizes fundamental features
   - Processes sentiment data

3. **Edge Parameter Calculation**:
   ```bash
   python phase1_edge_parameter_calc.py
   ```
   - Computes rolling correlations between stocks
   - Calculates fundamental similarity measures
   - Creates sector-based connection parameters

### Running Phase 2: Graph Construction

From the repo root:
```bash
python scripts/phase2_graph_construction.py
```
- Output: `data/graphs/graph_t_YYYYMMDD.pt` per trading day.
- Graph object: PyG `HeteroData` with node type `'stock'` and edge types:
  - `'rolling_correlation'`, `'fund_similarity'`, `'sector_industry'`, `'supply_competitor'` (if present)

### Running Phase 3: Baseline GNN Training
```bash
python scripts/phase3_baseline_training.py
```
Notes for PyTorch >= 2.6:
- The script already sets `torch.serialization.add_safe_globals([...])` to allowlist PyG storages.
- Uses `torch.load(..., weights_only=False)` when loading graphs.

**Class Imbalance Handling:**
The script supports three loss function options to handle imbalanced datasets:
1. **Standard Cross-Entropy** (`LOSS_TYPE = 'standard'`): No class balancing
2. **Weighted Cross-Entropy** (`LOSS_TYPE = 'weighted'`): Automatically computes class weights
   - Formula: `w_i = n_total / (n_classes * n_i)`
   - Minority class receives higher weight, balancing the loss contribution
3. **Focal Loss** (`LOSS_TYPE = 'focal'`): Focuses on hard-to-classify examples
   - Formula: `FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)`
   - Automatically down-weights well-classified examples
   - Configure with `FOCAL_ALPHA` (default: 0.25) and `FOCAL_GAMMA` (default: 2.0)

To change the loss type, edit `LOSS_TYPE` in `phase3_baseline_training.py` (line 33).

### Running Phase 4: Core Transformer Training
```bash
python scripts/phase4_core_training.py
```
What it does:
- Loads first available `HeteroData` to infer input dim.
- Runs sequential train/val/test split and saves `models/core_transformer_model.pt`.

### Running Phase 5: RL Integration (scaffold)
```bash
python scripts/phase5_rl_integration.py
```
Requires the core model; integrates with `scripts/rl_environment.py`.

### Running Phase 6: Evaluation
```bash
python scripts/phase6_evaluation.py
```
Generates evaluation metrics and (optionally) plots.

## Data Description

### Stock Selection
- **Default**: Top 50 holdings from SPY ETF
- **Configurable**: Modify `CONFIG` in `phase1_data_collection.py`
- **Date Range**: 2015-2025 (configurable)

### Features Generated

#### Technical Features
- **Price-based**: Returns, log returns, volatility
- **Momentum**: RSI (14-day, 30-day), Stochastic oscillator
- **Trend**: MACD, Moving averages (SMA, EMA)
- **Volatility**: Bollinger Bands, ATR
- **Volume**: Volume ratios and moving averages

#### Fundamental Features
- **Valuation**: P/E ratio, Price-to-Book, PEG ratio
- **Profitability**: ROE, ROA, profit margins
- **Financial Health**: Debt-to-Equity, Current ratio
- **Growth**: Revenue growth, earnings growth
- **Risk**: Beta coefficient

#### Sentiment & Macro Features
- **Market Sentiment**: VIX (fear index), rolling averages
- **News Sentiment**: Aggregated sentiment scores (simulated)
- **Macro Indicators**: Various economic indicators

### Edge Parameters (Phase 1/2)
- **Dynamic Correlations**: 30-day rolling correlations between stock returns
- **Fundamental Similarity**: Cosine similarity of fundamental metrics
- **Sector Connections**: Industry and sector-based relationships

## Configuration

Modify settings in `scripts/phase1_data_collection.py`:

```python
CONFIG = {
    'START_DATE': '2015-01-01',    # Data start date
    'END_DATE': '2025-01-01',      # Data end date
    'STOCK_SOURCE': 'SPY',         # 'SPY', 'QQQ', or 'CUSTOM'
    'NUM_STOCKS': 50,              # Number of stocks to include
    'CUSTOM_TICKERS': ['AAPL', 'MSFT', ...]  # Custom ticker list
}
```

## Usage Examples

### Loading Processed Data
```python
from scripts.utils_data import load_data_file

# Load technical features
technical_data = load_data_file('features_technical.csv')

# Load correlation parameters
correlations = load_data_file('edges_dynamic_corr_params.csv', directory='edges')
```

### Data Validation
```python
from scripts.utils_data import validate_ticker_data

# Validate data quality
results = validate_ticker_data(
    technical_data, 
    required_columns=['ticker', 'date', 'returns'],
    min_observations=100
)
```

## Phases & Progress

- **Phase 1**: Data + Features 
- **Phase 2**: Graph Construction 
- **Phase 3**: Baseline Training
- **Phase 4**: Core Transformer
- **Phase 5**: RL Integration
- **Phase 6**: Evaluation

## Open TODOs

- **Phase 2 – Graph Construction**
  - [x] Persist ticker list metadata into each `HeteroData` graph for training scripts. (graph attribute: `tickers`)
  - [x] Add static edge types for `supply_chain` and `competitor` if available; ensure schemas match `('stock', edge_type, 'stock')`.
  - [x] Store and normalize `edge_attr` tensors for dynamic edges (e.g., correlation magnitude, similarity score).
  - [x] Add integrity checker to validate saved graphs (load right after save with `weights_only=False`).

- **Phase 3 – Baseline Training**
  - [x] Temporal time-based split (70/15/15) to avoid leakage.
  - [x] Handle class imbalance (class weights or focal loss).
  - [ ] Save full checkpoints (model, optimizer, epoch, metrics) and resume support.
  - [ ] Add early stopping and learning rate scheduler.
  - [ ] Log metrics to TensorBoard and add ROC-AUC, confusion matrix reporting.

- **Phase 4 – Core Transformer**
  - [ ] Replace simulated PEARL with the component in `scripts/components/pearl_embedding.py`.
  - [ ] Support edge-type–specific attention parameters and relation-aware aggregation.
  - [ ] Enable neighbor sampling / mini-batch training for large graphs.
  - [ ] Add mixed precision (AMP) and gradient clipping for stability.
  - [ ] Provide hyperparameter sweep script (grid or Optuna).

- **Phase 5 – RL Integration**
  - [ ] Finalize `rl_environment.py` reward shaping; include transaction costs and risk penalties.
  - [ ] Integrate SB3 (PPO/A2C) training loop using model embeddings as state.
  - [ ] Add portfolio constraints, position sizing, and risk metrics (max drawdown, Sharpe).
  - [ ] Implement backtesting with slippage and latency modeling.

- **Phase 6 – Evaluation**
  - [ ] Produce plots: equity curve, drawdown, rolling Sharpe, turnover, exposure.
  - [ ] Run ablations for edge types and feature groups.
  - [ ] Compare against baselines (buy-and-hold, sector ETF, equal-weight).

- **Data & Infrastructure**
  - [ ] Centralize configuration in YAML (paths, tickers, thresholds, seeds); load in all scripts.
  - [ ] Add robust logging (structured logs) and progress bars across phases.
  - [ ] Unit tests for data loaders, feature builders, and graph constructors; add CI workflow.
  - [ ] Add `Dockerfile`/DevContainer for reproducible environment.
  - [x] Ensure large artifacts are excluded from git; regenerate from scripts (history cleaned, `.gitignore` covers data/ and venv/).
  - [ ] Determinism: set global seeds and PyTorch deterministic flags; document reproducibility.
  - [ ] Caching for downloads and intermediate features to speed up reruns.
  - [ ] Document hardware requirements and runtime expectations per phase.

- **Known Issues / Notes**
  - macOS OpenMP shared memory errors: set `export OMP_NUM_THREADS=1` if needed.
  - `torch-scatter` / `torch-sparse` warnings are optional; PyG falls back to pure PyTorch.
  - For PyTorch ≥ 2.6, ensure `torch.serialization.add_safe_globals([...])` and `weights_only=False` when loading graphs/models.

## Proposal-Aligned TODOs (From CS224W Project Proposal)

- **Datasets & Scope**
  - Expand beyond SPY top holdings: include QQQ or sector-specific subsets for robustness.
  - Evaluate different time spans (e.g., pre-/post-COVID) and market regimes.

- **Feature Engineering**
  - Add alternative technical indicators cited in proposal and verify incremental utility.
  - Include macro features (e.g., yields, inflation, unemployment) per proposal scope.
  - Add alternative normalization strategies (z-score vs. robust scaling) and compare.

- **Graph Design**
  - Sweep correlation thresholds (e.g., 0.4–0.8) and evaluate graph density impact.
  - Compare edge construction variants: signed vs. absolute correlation, top-k per node.
  - Evaluate additional static edges (supply chain, competitor) as proposed.
  - Test homogeneous vs. heterogeneous processing pipelines.

- **Models & Baselines**
  - Implement and compare: GCN, GAT, GraphSAGE, HGT/Relational GNN per proposal.
  - Non-graph baselines: Logistic Regression, MLP, LSTM/Transformer on per-stock features.
  - Add loss variants (weighted CE, focal loss) and label smoothing.

- **Training Protocol**
  - Strict temporal CV: rolling or expanding window validation as proposed.
  - Seed control and deterministic flags for reproducibility.
  - Hyperparameter tuning (grid/Optuna) following proposal’s search ranges.

- **Evaluation & Metrics**
  - Statistical significance testing (e.g., block bootstrap) for metric differences.
  - Financial metrics: annualized return, Sharpe/Sortino, max drawdown, turnover, hit rate.
  - Robustness checks: sensitivity to transaction cost assumptions and slippage.

- **Ablation Studies**
  - Remove edge types one-by-one and measure performance drop.
  - Remove feature groups (technical/fundamental/sentiment) per proposal.
  - Vary lookahead horizon (e.g., 1/5/10 days) and measure sensitivity.

- **Interpretability**
  - Attention score analysis: top neighbors/relations contributing to predictions.
  - Per-feature SHAP/attribution on node logits.

- **RL Integration**
  - Use proposal’s action/state design; test PPO/A2C and risk-aware rewards.
  - Portfolio constraints and risk controls per proposal (exposure, leverage caps).
  - Walk-forward backtesting aligned with temporal splits.

- **Engineering & Reproducibility**
  - Comprehensive experiment logging and run configs committed.
  - Scripts to fully reproduce figures and tables in the report.
  - Document compute costs and runtime; include a small “quickstart” subset.

## Mathematical Foundation

### Technical Indicators
- **RSI**: $RSI = 100 - \frac{100}{1 + RS}$ where $RS = \frac{\text{Average Gain}}{\text{Average Loss}}$
- **MACD**: $MACD = EMA_{12} - EMA_{26}$
- **Bollinger Bands**: $BB_{upper/lower} = SMA_n \pm k \cdot \sigma_n$

### Edge Weights
- **Correlation**: $\rho_{ij}(t) = \frac{Cov(r_{i,t}, r_{j,t})}{\sigma_{i,t} \sigma_{j,t}}$
- **Cosine Similarity**: $cos(\theta) = \frac{\mathbf{f_i} \cdot \mathbf{f_j}}{|\mathbf{f_i}||\mathbf{f_j}|}$

## Contributing

1. Follow the established code structure and documentation standards
2. Add comprehensive comments explaining mathematical concepts
3. Include unit tests for new functions
4. Update this README when adding new phases

## 📝 Notes

- All functions requiring implementation are marked with 'start code' and 'end code' comments
- Comments and documentation are in English
- Mathematical explanations are provided for educational purposes

## Troubleshooting

### Common Issues

1. **PyTorch Geometric load errors (PyTorch >= 2.6)**:
   - Ensure scripts use `torch.serialization.add_safe_globals([BaseStorage, NodeStorage, EdgeStorage, GlobalStorage])` (already included in Phase 3/4 scripts).
   - Ensure `torch.load(..., weights_only=False)` is used when loading graphs/models.

2. **Virtual environment activation**:
   - Ensure virtual environment is activated before installing packages
   - Windows users: Try different activation commands (PowerShell vs Command Prompt)
   - If activation fails, create a new virtual environment

3. **VIX data download**: 
   - Ensure internet connection and try running again
   - Script includes fallback to dummy data if VIX download fails

4. **Memory issues**: 
   - Reduce the number of stocks in CONFIG if running on limited memory
   - Use smaller date ranges for initial testing

5. **OpenMP / SHM errors on macOS**:
   - Try: `export OMP_NUM_THREADS=1` before running training scripts.
   
6. **Module import errors**:
   - Verify virtual environment is activated
   - Reinstall requirements: `pip install -r requirements.txt`

7. **torch-scatter / torch-sparse warnings**:
   - These extensions are optional; PyG falls back to pure PyTorch. You can ignore the warnings, or install version-matched wheels if needed.

### Getting Help
- Check the validation output from `utils_data.py` functions
- Ensure all data directories exist before running scripts
- Verify date ranges are valid for market data availability  
- Try the simple version first before troubleshooting TA-Lib issues
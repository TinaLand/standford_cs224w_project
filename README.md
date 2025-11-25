# CS224W Stock RL GNN Project

**📄 MILESTONE REPORT**: See **[MILESTONE_REPORT.md](MILESTONE_REPORT.md)** for complete milestone submission

**Key Results**:
- ✅ Complete data pipeline (2,467 graphs, 15 features)
- ✅ GAT model with Focal Loss (Test ROC-AUC: 0.5101)
- ✅ 79% crash detection recall (valuable for risk management)
- ✅ Systematic debugging (3 critical bugs fixed)
- ✅ Production-quality code (3,179 lines, fully documented)

---

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

**Checkpoint Management:**
The script automatically saves full training checkpoints (model, optimizer, epoch, metrics):
- **Automatic Saving**: Checkpoints saved every 5 epochs and when a new best model is found
- **Resume Training**: Set `RESUME_FROM_CHECKPOINT = True` to continue from last checkpoint
- **Checkpoint Files**:
  - `checkpoint_best.pt`: Best model (highest validation F1)
  - `checkpoint_latest.pt`: Most recent training state
  - `checkpoint_epoch_XXX.pt`: Regular epoch checkpoints
- **Storage Location**: `models/checkpoints/`

To resume interrupted training:
```python
# In phase3_baseline_training.py (line 39)
RESUME_FROM_CHECKPOINT = True
```

The checkpoint includes:
- Model weights (state_dict)
- Optimizer state (Adam momentum, learning rates)
- Current epoch number
- Training metrics history (loss, accuracy, F1 scores)

**Early Stopping & Learning Rate Scheduler:**
Automatically prevents overfitting and optimizes learning rate:
- **Early Stopping**: Stops training if validation F1 doesn't improve for N epochs
  - Default patience: 5 epochs
  - Prevents wasted computation on plateau
- **Learning Rate Scheduler**: Reduces LR when validation metric plateaus
  - Default: ReduceLROnPlateau (reduces LR by 0.5 after 3 epochs without improvement)
  - Options: 'plateau', 'step', 'exponential'
  
Configure in `phase3_baseline_training.py`:
```python
# Early Stopping (lines 45-48)
ENABLE_EARLY_STOPPING = True
EARLY_STOP_PATIENCE = 5         # Wait N epochs before stopping

# Learning Rate Scheduler (lines 50-55)
ENABLE_LR_SCHEDULER = True
LR_SCHEDULER_TYPE = 'plateau'   # 'plateau', 'step', or 'exponential'
```

**TensorBoard & Advanced Metrics:**
Real-time training visualization and comprehensive evaluation metrics:
- **TensorBoard Logging**: Automatic logging of loss, accuracy, F1, ROC-AUC, learning rate
- **ROC-AUC Score**: Measures model's ability to distinguish between classes (0.5 = random, 1.0 = perfect)
- **Confusion Matrix**: Visual breakdown of TP, TN, FP, FN with detailed plots
- **Classification Report**: Per-class precision, recall, F1-score

View TensorBoard:
```bash
tensorboard --logdir=runs
# Then open http://localhost:6006 in browser
```

Generated files:
- `runs/`: TensorBoard logs (per training run)
- `models/plots/`: Confusion matrix plots (PNG format)

Configure in `phase3_baseline_training.py`:
```python
# TensorBoard & Metrics (lines 67-72)
ENABLE_TENSORBOARD = True       # TensorBoard logging
ENABLE_ROC_AUC = True          # Calculate ROC-AUC
ENABLE_CONFUSION_MATRIX = True  # Generate confusion matrix
```

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

### ✅ Completed Phases

#### Phase 1 – Data Collection ✅
- [x] **Real data collection** with network fallback prevention ✅ Code updated
  - Modified `phase1_data_collection.py`: `allow_synthetic_fallback=False`
  - Modified `phase1_feature_engineering.py`: Explicit error handling
  - 📖 Guide: [REAL_DATA_COLLECTION_GUIDE.md](REAL_DATA_COLLECTION_GUIDE.md)
- [x] Trading calendar alignment ✅ `utils_data_validation.py`
- [x] Stock suspension handling ✅ Implemented
- [x] Split/dividend adjustments ✅ Implemented
- [x] Missing value imputation ✅ Multiple methods available
- [x] Data quality validation ✅ Comprehensive checks
- [x] Data collection logging ✅ JSON log created

#### Phase 2 – Graph Construction ✅
- [x] Persist ticker metadata in graphs ✅ `tickers` attribute
- [x] Static edge types (supply_chain, competitor) ✅ Schema validated
- [x] Normalize edge attributes ✅ Dynamic edges normalized
- [x] Graph integrity checker ✅ Validation implemented

#### Phase 3 – Baseline Training ✅
- [x] Temporal time-based split (70/15/15) ✅ No leakage
- [x] Class imbalance handling ✅ Focal loss / weighted CE
- [x] Checkpoint system ✅ Full state saving
- [x] Early stopping & LR scheduler ✅ Implemented
- [x] TensorBoard logging ✅ ROC-AUC, confusion matrix

#### Phase 4 – Core Transformer ✅
- [x] PEARL embedding component ✅ `components/pearl_embedding.py`
- [x] Relation-aware aggregation ✅ Edge-type-specific attention
- [x] Mini-batch training ✅ Neighbor sampling support
- [x] AMP & gradient clipping ✅ Stability improvements
- [x] Hyperparameter sweep ✅ `phase4_hyperparameter_sweep.py`

#### Phase 5 – RL Integration ✅ Core Framework
- [x] Reward shaping with transaction costs ✅ 0.1% per trade
- [x] SB3 (PPO) integration ✅ `phase5_rl_integration.py`
- [x] Portfolio constraints ✅ Basic position sizing
- [x] Backtesting framework ✅ Slippage modeling
- [x] GNN embeddings as state ✅ Model integration
- [x] Discrete action space ✅ MultiDiscrete([3] * N)
- [x] Reward function ✅ Return-based (can be enhanced)
- [ ] **Enhancement**: Add `get_embeddings()` method to Phase 4 model
- [ ] **Enhancement**: Improve reward with Sharpe ratio
- [ ] **Enhancement**: Advanced portfolio constraints
- [ ] **Testing**: Run Phase 5 training and validate performance

#### Phase 6 – Evaluation ✅ Basic Framework
- [x] Financial metrics calculation ✅ Sharpe, Max Drawdown, Returns
- [x] Ablation studies framework ✅ `run_ablation_studies()`
- [x] Final backtesting ✅ `run_final_backtest()`
- [ ] **Visualization**: Equity curve, drawdown, rolling Sharpe plots
- [ ] **Baseline Comparison**: Buy-and-hold, ETF, equal-weight
- [ ] **Interpretability**: Attention weights visualization
- [ ] **Analysis**: Failure case investigation

#### Data & Infrastructure ✅
- [x] Real data pipeline ✅ Network failure handling
- [x] Data validation system ✅ `utils_data_validation.py`
- [x] Data versioning ✅ Collection log (JSON)
- [x] Configuration centralization ✅ `config.yaml` + `utils_config.py`
- [x] Git artifact management ✅ `.gitignore` configured
- [x] Reproducibility setup ✅ `setup_reproducibility()`
- [x] Data collection logging ✅ Transparent tracking
- [ ] Structured logging system
- [ ] Unit tests + CI workflow
- [ ] Docker/DevContainer
- [ ] Caching system
- [ ] Hardware requirements documentation

---

### 🔬 Future Improvements (After Real Data)

#### Model Performance Optimization
- [ ] Analyze low ROC-AUC (0.51) - feature importance investigation
- [ ] Experiment with lookahead horizons (1, 3, 5, 10 days)
- [ ] Add temporal GNN layers (EvolveGCN, TemporalGCN)
- [ ] Graph sparsification strategies (Top-K thresholds, correlation cutoffs)
- [ ] Advanced features: order flow, microstructure, alternative data
- [ ] Ensemble methods: multiple models/graph views

#### Research Extensions (Proposal-Aligned)
- [ ] Expand datasets: QQQ, sector-specific subsets
- [ ] Time span analysis: pre/post-COVID, different market regimes
- [ ] Additional technical indicators
- [ ] Macro features: yields, inflation, unemployment
- [ ] Graph design experiments: correlation thresholds, edge variants
- [ ] Model comparisons: GCN, GAT, GraphSAGE, HGT
- [ ] Non-graph baselines: Logistic Regression, MLP, LSTM
- [ ] Statistical significance testing
- [ ] Robustness checks: transaction costs, slippage sensitivity

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

## 📝 Documentation

### Project Documentation
- **[milestone/MILESTONE_REPORT.md](milestone/MILESTONE_REPORT.md)** - Complete milestone report for CS224W (consolidated from previous docs)
- **[milestone/MILESTONE_CREDIT_REPORT_CN.md](milestone/MILESTONE_CREDIT_REPORT_CN.md)** - Chinese milestone report
- **[milestone/METRICS_QUICK_REFERENCE.md](milestone/METRICS_QUICK_REFERENCE.md)** - Quick reference for all metrics and formulas
- **[TESTING_GUIDE.md](TESTING_GUIDE.md)** - Data validation testing guide

### Implementation Guides
- **[docs/README_IMPLEMENTATION_DOCS.md](docs/README_IMPLEMENTATION_DOCS.md)** - Index of 12 implementation guides
- Each major script has comprehensive documentation (12,500+ total lines)
- Mathematical foundations, design patterns, best practices

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
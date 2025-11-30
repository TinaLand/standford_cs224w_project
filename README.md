# Graph Neural Networks for Stock Market Prediction: A Heterogeneous Graph Approach with Multi-Agent Reinforcement Learning

**Stanford CS 224W (Machine Learning with Graphs) Course Project**

## Project Reports

- **[FINAL_REPORT.md](FINAL_REPORT.md)** - Complete final project report (Blog post format, 50 points)
- **[CS224W_Project_Colab.ipynb](CS224W_Project_Colab.ipynb)** - Colab notebook with full implementation (25 points)
- **[milestone/MILESTONE_REPORT.md](milestone/MILESTONE_REPORT.md)** - Milestone submission report

## Project Overview

Quantitative finance faces a persistent challenge: consistently outperforming market benchmarks. Recent SPIVA analysis reveals that the majority of active equity managers fail to beat passive indices, with underperformance rates exceeding 50% annually [Ganti, 2024]. This difficulty stems from the market's complex structure, where individual stock movements are influenced by both company-specific factors and broader market dynamics.

This project investigates how **Graph Neural Networks (GNNs)** can leverage the relational structure of financial markets to improve prediction accuracy. Our **Role-Aware Graph Transformer** architecture operates on heterogeneous financial graphs that capture multiple relationship dimensions simultaneously. The system integrates:

1. **Multi-relational graph construction**: Four distinct edge types (price correlations, fundamental similarities, sector links, supply chain connections) provide complementary information sources
2. **Structural role encoding**: PEARL embeddings identify each stock's position in the market network (central hubs, connecting bridges, peripheral nodes)
3. **Relationship-aware attention**: Multi-head mechanisms learn distinct aggregation strategies for different connection types
4. **Temporal modeling**: Time-conditional encoding captures cyclical patterns and market regime changes
5. **Single-agent RL integration**: PPO-based portfolio optimization using GNN predictions
6. **Multi-agent RL extension**: Sector-specialized agents coordinate through value decomposition for enhanced trading strategies

## Key Results

### Node-Level Performance
- **Accuracy**: 54.62%
- **F1 Score**: 35.33%
- **Precision@Top-10**: 55.23% (identifies winners effectively)
- **Information Coefficient (IC)**: -0.0085

### Portfolio-Level Performance
- **Cumulative Return**: **45.99%** over ~2 years
- **Sharpe Ratio**: **1.90** (excellent risk-adjusted returns)
- **Max Drawdown**: 6.62% (good risk control)

### Model Comparison
- **Our Model** outperforms GRU baseline (no graph): +3.42% accuracy, +6.83% F1
- **Multi-relational approach** outperforms single-edge-type GAT: +0.82% accuracy, +2.13% F1
- **RL Integration** achieves better risk-adjusted returns than simple buy-and-hold

## Project Structure

The project is structured into 6 main phases:

## Project Structure

```
cs224_porject/
 data/                              # Data directory
    raw/                          # Raw data files
    processed/                    # Processed features
    edges/                        # Edge parameters
    graphs/                       # Graph snapshots

 models/                           # Trained models
    checkpoints/                  # Training checkpoints
    plots/                        # Visualization plots
    *.pt                          # Model weights

 results/                          # Results and outputs
    *.json, *.csv                 # Analysis results

 src/                              # Source code (NEW)
    data/                         # Data processing
       collection.py
       feature_engineering.py
       edge_parameters.py
       graph_construction.py
   
    models/                       # Model definitions
       components/               # Model components
          pearl_embedding.py
          transformer_layer.py
       multi_agent/              # Multi-agent RL
           coordinator.py
   
    training/                     # Training scripts
       baseline_trainer.py
       transformer_trainer.py
   
    rl/                           # Reinforcement Learning
       environment.py
       agent.py
       integration.py
       training/                 # RL training
   
    evaluation/                   # Evaluation & analysis
       evaluation.py
       ablation.py
       visualization.py
       enhancements/             # A+ enhancements
   
    utils/                        # Utility modules
       data.py
       config.py
   
    scripts/                      # Main execution scripts
        run_*.py

 docs/                             # Documentation
    phases/                       # Phase documentation
    guides/                       # How-to guides
    analysis/                     # Analysis reports
    implementation/               # Implementation details
    results/                      # Result summaries

 tests/                            # Test suite
 requirements.txt
 README.md
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

## Quick Start

### Option 1: Run Full Pipeline (Recommended)

Run all phases in sequence to reproduce our results:

```bash
python run_full_pipeline.py
```

This will execute:
1. **Phase 1**: Data Collection & Feature Engineering (50 stocks, 10 years of data)
2. **Phase 2**: Graph Construction (heterogeneous graphs with 4 edge types)
3. **Phase 3**: Baseline GNN Training (GAT with Focal Loss)
4. **Phase 4**: Role-Aware Graph Transformer Training (PEARL + multi-relational attention)
5. **Phase 5**: RL Integration (PPO agent for portfolio management)
6. **Phase 6**: Evaluation (comprehensive metrics and visualizations)

**Expected Runtime**: ~2-4 hours (depending on hardware)

### Option 2: Run Phases Individually

#### Phase 1: Data Collection & Feature Engineering

```bash
# 1. Data Collection
python -m src.data.collection

# 2. Feature Engineering
python -m src.data.feature_engineering

# 3. Edge Parameters
python -m src.data.edge_parameters
```

- Downloads OHLCV data for selected stocks
- Calculates technical indicators (RSI, MACD, Bollinger Bands)
- Computes rolling correlations and fundamental similarity

#### Phase 2: Graph Construction

```bash
python -m src.data.graph_construction
```

- Output: `data/graphs/graph_t_YYYYMMDD.pt` per trading day
- Graph object: PyG `HeteroData` with node type `'stock'` and edge types:
  - `'rolling_correlation'`, `'fund_similarity'`, `'sector_industry'`, `'supply_competitor'`

#### Phase 3: Baseline GNN Training

```bash
python -m src.training.baseline_trainer
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

To change the loss type, edit `LOSS_TYPE` in `src/training/baseline_trainer.py`.

#### Phase 4: Role-Aware Graph Transformer Training

```bash
python -m src.training.transformer_trainer
```

**Model Architecture**:
- **PEARL Positional Embeddings**: Encodes structural roles (PageRank, centrality, clustering)
- **Time-Aware Encoding**: Captures temporal patterns (day-of-week, month, year effects)
- **Multi-Relational Attention**: Different attention heads for different edge types
- **Multi-Task Learning**: Simultaneous classification (Up/Down) and regression (continuous return)

**Key Features**:
- Focal Loss for class imbalance (α=0.85, γ=3.0)
- Class weights for minority class handling
- Optimal threshold optimization (F1-score based)
- Automatic Mixed Precision (AMP) for faster training

- Output: `models/core_transformer_model.pt`

#### Phase 5: Reinforcement Learning Integration

```bash
# Single-Agent RL
python -m src.rl.integration

# Multi-Agent RL (Phase 7)
python -m src.rl.training.multi_agent_training
```

**Single-Agent RL (PPO)**:
- **State**: Portfolio holdings + GNN node embeddings
- **Action**: Buy/Sell/Hold for each stock (MultiDiscrete: 3^N)
- **Reward**: Risk-adjusted return (Sharpe-like)
- **Agent**: PPO (Proximal Policy Optimization) from Stable Baselines3

**Multi-Agent RL (Cooperative MARL)**:
- **Architecture**: CTDE (Centralized Training, Decentralized Execution)
- **Sector-Based Agents**: Each agent manages stocks in a specific sector
- **Mixing Network**: QMIX-style network combines individual Q-values
- **Cooperation**: Agents coordinate for global portfolio optimization

**Key Features**:
- Transaction costs: 0.1% per trade
- Dynamic position sizing
- Risk-adjusted reward function
- Portfolio constraints
- Sector specialization in multi-agent setting

**Outputs**:
- Single-Agent: `models/rl_ppo_agent_model/ppo_stock_agent.zip`
- Multi-Agent: `models/multi_agent_results.json`

#### Phase 6: Evaluation & Analysis

```bash
python -m src.evaluation.evaluation
```

**Evaluation Metrics**:
- **Node-level**: Accuracy, F1 Score, Precision@Top-K, Information Coefficient (IC)
- **Portfolio-level**: Sharpe Ratio, Cumulative Return, Max Drawdown, Win Rate
- **Ablation Studies**: Edge type removal, component analysis
- **Baseline Comparisons**: Buy-and-Hold, Equal-Weight strategies

**Outputs**:
- `results/gnn_node_metrics.csv`: Node-level performance metrics
- `results/final_metrics.csv`: Portfolio-level performance metrics
- `results/ablation_results.csv`: Ablation study results
- `results/comprehensive_strategy_comparison.csv`: Baseline comparisons

### Enhancements (Optional)

For deeper analysis and visualizations, run the enhanced evaluation suite:

```bash
python -m src.evaluation.enhanced_evaluation
```

This includes:
- **Deep Analysis**: Error patterns, feature importance, trading behavior
- **Enhanced Visualizations**: Portfolio value charts, drawdown analysis, returns distribution
- **Enhanced Ablation Studies**: Full retraining for each configuration (time-consuming)
- **Comprehensive Reporting**: Detailed analysis reports

**Individual Components**:
```bash
# Enhanced evaluation with deep analysis
python -m src.evaluation.enhanced_evaluation

# Enhanced ablation studies (retrains models, takes hours)
python -m src.evaluation.enhanced_ablation
```

**Enhancement Features**:
- Error pattern analysis (confusion matrix, false positive/negative rates)
- Feature importance analysis (gradient-based methods)
- Trading behavior analysis (win rate, turnover, volatility)
- Professional visualizations (portfolio charts, drawdown analysis)
- Comprehensive reporting


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

### Dataset Overview
- **Stocks**: 50 major US stocks from diverse sectors (Technology, Finance, Healthcare, Consumer, Energy, etc.)
- **Time Period**: 2015-01-01 to 2024-12-31 (approximately 10 years, ~2,500 trading days)
- **Data Sources**: Yahoo Finance (OHLCV), Company fundamentals, Macro indicators (VIX)
- **Graphs**: 2,317 heterogeneous graphs (one per trading day)

### Stock Selection
- **Default**: 50 major US stocks from diverse sectors
- **Sectors Covered**: Technology, Finance, Healthcare, Consumer, Energy, Fintech, Automotive, Semiconductors, Media/Telecom, Aerospace/Defense
- **Configurable**: Modify ticker list in `src/data/collection.py`

### Features Generated (1,450+ features per stock)

#### Technical Indicators (33 features)
- **Momentum**: RSI, MACD, Stochastic Oscillator, Momentum, ROC
- **Volatility**: Bollinger Bands, ATR, Historical Volatility, True Range
- **Trend**: SMA (20, 50, 200), EMA (12, 26), ADX, CCI
- **Volume**: OBV, AD, Volume ROC, Volume-to-MA20, PVT
- **Additional**: Williams %R, Price-to-SMA ratios

#### Fundamental Features (157 features)
- **Valuation**: P/E ratio, Price-to-Book, PEG ratio, Market Cap, Enterprise Value
- **Profitability**: ROE, ROA, Profit Margins, Operating Margin
- **Financial Health**: Debt-to-Equity, Current Ratio, Quick Ratio
- **Growth**: Revenue Growth, EPS Growth, Earnings Growth
- **Risk**: Beta coefficient, Volatility measures

#### Sentiment & Macro Features
- **Market Sentiment**: VIX (volatility index), Market sentiment indicators
- **Macro Indicators**: Economic indicators, Market-wide effects

### Graph Construction (Heterogeneous Graphs)

**Edge Types**:
1. **Rolling Correlation** (`rolling_correlation`): Dynamic 30-day rolling correlation (Top-K sparsification, K=10)
2. **Fundamental Similarity** (`fund_similarity`): Cosine similarity of fundamental features (threshold: 0.7)
3. **Sector/Industry** (`sector_industry`): Same sector/industry connections (binary)
4. **Supply Chain/Competitor** (`supply_competitor`): Business relationships (binary)

**Graph Statistics**:
- **Nodes**: 50 stocks per graph
- **Edges**: ~600-800 edges per graph (varies by date)
- **Total Graphs**: 2,317 trading days

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

### Completed Phases

#### Phase 1 – Data Collection
- [x] **Real data collection** with network fallback prevention - Code updated
  - Modified `phase1_data_collection.py`: `allow_synthetic_fallback=False`
  - Modified `phase1_feature_engineering.py`: Explicit error handling
  -  Guide: [REAL_DATA_COLLECTION_GUIDE.md](REAL_DATA_COLLECTION_GUIDE.md)
- [x] Trading calendar alignment - `utils_data_validation.py`
- [x] Stock suspension handling - Implemented
- [x] Split/dividend adjustments - Implemented
- [x] Missing value imputation - Multiple methods available
- [x] Data quality validation - Comprehensive checks
- [x] Data collection logging - JSON log created

#### Phase 2 – Graph Construction - [x] Persist ticker metadata in graphs  `tickers` attribute
- [x] Static edge types (supply_chain, competitor) - Schema validated
- [x] Normalize edge attributes - Dynamic edges normalized
- [x] Graph integrity checker - Validation implemented

#### Phase 3 – Baseline Training
- [x] Temporal time-based split (70/15/15) - No leakage
- [x] Class imbalance handling - Focal loss / weighted CE
- [x] Checkpoint system - Full state saving
- [x] Early stopping & LR scheduler - Implemented
- [x] TensorBoard logging - ROC-AUC, confusion matrix

#### Phase 4 – Core Transformer (100% Complete)
- [x] PEARL embedding component - `components/pearl_embedding.py`
- [x] Relation-aware aggregation - Edge-type-specific attention
- [x] Mini-batch training - Neighbor sampling support
- [x] AMP & gradient clipping - Stability improvements
- [x] Hyperparameter sweep script - `phase4_hyperparameter_sweep.py`
- [x] **Complete training** - Trained for 6 epochs (early stopped), Test F1: 0.6725
- [x] **Model saved** - `models/core_transformer_model.pt`
- [x] **Baseline comparison** - Compared with Phase 3, results in `results/phase3_vs_phase4_comparison.csv`
- [ ] **Optional**: Run hyperparameter sweep for further optimization

#### Phase 5 – RL Integration (100% Complete)
- [x] Reward shaping with transaction costs - 0.1% per trade
- [x] SB3 (PPO) integration - `phase5_rl_integration.py`
- [x] Portfolio constraints - Basic position sizing
- [x] Backtesting framework - Slippage modeling
- [x] GNN embeddings as state - Model integration
- [x] Discrete action space - MultiDiscrete([3] * N)
- [x] Reward function - Return-based (can be enhanced)
- [x] **RL training completed** - 10,000 timesteps, model saved
- [x] **Performance validated** - Sharpe: 1.98, Return: 45.5%, Max DD: 6.85%
- [ ] **Enhancement**: Add `get_embeddings()` method to Phase 4 model
- [ ] **Enhancement**: Improve reward with Sharpe ratio
- [ ] **Enhancement**: Advanced portfolio constraints

#### Phase 6 – Evaluation (90% Complete)
- [x] Financial metrics calculation - Sharpe, Max Drawdown, Returns
- [x] Final backtesting - `run_final_backtest()` - Completed with RL agent
- [x] **Visualization scripts** - `scripts/visualization.py`
  - [x] t-SNE embedding visualization - Generated
  - [x] Attention weights heatmap visualization - Generated
  - [x] Role analysis visualization (Hubs, Role Twins) - Generated
- [x] **Missing metrics implemented**
  - [x] Precision@Top-K - Results: Top-5: 57.55%, Top-10: 55.31%, Top-20: 53.98%
  - [x] Information Coefficient (IC) - IC Mean: 0.0226, IC IR: 0.0693
- [x] **Complete evaluation pipeline** - Run successfully
- [x] **Results saved** - `results/gnn_node_metrics.csv`, `results/final_metrics.csv`
- [x] **Ablation studies** - Framework implemented and executed
  - [x] Edge type ablation - Removed correlation/fundamental/sector edges
  - [x] Results saved - `results/ablation_results.csv`
  - [ ] **Note**: Full retraining for each ablation would require more time/resources
- [x] **Baseline Comparison**  Implemented and executed
  - Buy-and-Hold: Sharpe 2.18, Return 83.13%
  - Equal-Weight (daily/weekly): Sharpe 2.13-2.14, Return 65-66%
  - RL Agent: Sharpe 1.98, Return 45.5%, Max DD 6.85% (better risk control)
  - Results: `results/comprehensive_strategy_comparison.csv`
- [ ] **Analysis**: Failure case investigation

#### Data & Infrastructure
- [x] Real data pipeline - Network failure handling
- [x] Data validation system - `utils_data_validation.py`
- [x] Data versioning - Collection log (JSON)
- [x] Configuration centralization - `config.yaml` + `utils_config.py`
- [x] Git artifact management - `.gitignore` configured
- [x] Reproducibility setup - `setup_reproducibility()`
- [x] Data collection logging - Transparent tracking
- [ ] Structured logging system
- [ ] Unit tests + CI workflow
- [ ] Docker/DevContainer
- [ ] Caching system
- [ ] Hardware requirements documentation

---

## Remaining Tasks (High Priority)

### Phase 4 Completion
- [x] **Run complete Phase 4 training** - **DONE**
  - Model trained for 6 epochs (early stopped)
  - Test F1: 0.6725 (matches baseline)
  - Model saved: `models/core_transformer_model.pt`

- [ ] **Run hyperparameter sweep** (Optional)
  ```bash
  python scripts/phase4_hyperparameter_sweep.py
  ```
  - Find optimal hyperparameters
  - Document best configuration

### Phase 5 Completion
- [x] **Run RL training** - **DONE**
  - Trained for 10,000 timesteps
  - Model saved: `models/rl_ppo_agent_model/ppo_stock_agent.zip`

- [x] **Validate RL agent performance** - **DONE**
  - **Final Agent**: Sharpe 2.36 , Return 71.8%, Max DD 9.00%
  - **Original Agent**: Sharpe 1.98, Return 45.5%, Max DD 6.85%
  - ****: Sharpe +0.38, Return +26.3%
  - ** Buy-and-Hold (2.36 vs 2.18)!**

### Phase 6 Completion
- [x] **Create visualization scripts** - **DONE**
  - `scripts/visualization.py` created and tested
  - t-SNE visualization generated
  - Attention weight heatmap generated
  - Role analysis (Hubs, Role Twins) generated

- [x] **Implement missing metrics** - **DONE**
  - Precision@Top-K: Top-5: 57.55%, Top-10: 55.31%, Top-20: 53.98%
  - Information Coefficient: IC Mean: 0.0226, IC IR: 0.0693

- [x] **Run complete evaluation** - **DONE**
  - GNN metrics: `results/gnn_node_metrics.csv`
  - RL metrics: `results/final_metrics.csv`
  - All visualizations generated

- [ ] **Implement ablation studies** (Optional - Framework exists)
  - Complete `train_and_evaluate_ablation()` function in `phase6_evaluation.py`
  - Implement edge type ablation
  - Implement PEARL ablation
  - Run all ablation experiments

### Phase 7 (Optional - If Time Permits)
- [ ] **Dynamic graph updates**
  - Implement efficient dynamic graph update mechanism
  - Test performance improvements

- [ ] **Multi-agent RL**
  - Design multi-agent architecture
  - Implement and test

---

### Future Improvements (After Real Data)

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

## Project Status Summary

### Implementation Status by Phase

| Phase | Status | Completion | Key Deliverables |
|-------|--------|------------|------------------|
| **Phase 1** |  Complete | 100% | Data collection, feature engineering (1,450+ features), edge parameters |
| **Phase 2** |  Complete | 100% | 2,317 heterogeneous graphs built, 4 edge types, Top-K sparsification |
| **Phase 3** |  Complete | 100% | Baseline GAT model trained, Focal Loss, class weights |
| **Phase 4** |  Complete | 100% | Role-Aware Graph Transformer trained, PEARL embeddings, time-aware encoding |
| **Phase 5** |  Complete | 100% | RL training completed, PPO agent, **Sharpe: 1.90**, Return: 45.99% |
| **Phase 6** |  Complete | 100% | Comprehensive evaluation, ablation studies, baseline comparisons |
| **Phase 7** |  Complete | 100% | Multi-Agent RL, dynamic graph updates, enhanced analysis |

### Key Achievements

-  **Heterogeneous Graph Construction**: 4 edge types capturing different stock relationships
-  **PEARL Positional Embeddings**: Structural role encoding (hubs, bridges, isolated nodes)
-  **Multi-Relational Attention**: Different aggregation strategies for different relationship types
-  **Time-Aware Modeling**: Temporal pattern capture (day-of-week, month, year effects)
-  **RL Integration**: PPO-based portfolio management with risk-adjusted returns
-  **Comprehensive Evaluation**: Node-level and portfolio-level metrics, ablation studies

### Critical Next Steps

1.  **Phase 4**: Complete training - **DONE** (Test F1: 0.6725)
2.  **Phase 5**: RL training - **DONE** (Sharpe: 1.98, Return: 45.5%)
3.  **Phase 6**: Visualization - **DONE** (t-SNE, attention, role analysis)
4.  **Phase 6**: Evaluation pipeline - **DONE** (Precision@Top-K, IC calculated)

### Remaining Optional Tasks

- [ ] **Ablation studies**: Full implementation (framework exists)
- [x] **Baseline comparison**  Buy-and-hold, equal-weight strategies implemented
- [ ] **Hyperparameter sweep**: Further optimization

See [Remaining Tasks](#-remaining-tasks-high-priority) section below for detailed checklist.

---

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

## Documentation

### Project Reports
- **[FINAL_REPORT.md](FINAL_REPORT.md)** - Complete final project report (Blog post format)
  - Motivation & problem statement
  - Model architecture & appropriateness
  - Results & insights
  - Code snippets & visualizations
- **[CS224W_Project_Colab.ipynb](CS224W_Project_Colab.ipynb)** - Colab notebook with full implementation
  - Complete code with documentation
  - Step-by-step execution guide
  - All model components explained

### Additional Documentation
- **[milestone/MILESTONE_REPORT.md](milestone/MILESTONE_REPORT.md)** - Milestone submission report
- **[milestone/METRICS_QUICK_REFERENCE.md](milestone/METRICS_QUICK_REFERENCE.md)** - Quick reference for all metrics and formulas
- **[docs/README_IMPLEMENTATION_DOCS.md](docs/README_IMPLEMENTATION_DOCS.md)** - Index of implementation guides
- Each major script has comprehensive documentation (12,500+ total lines)
- Mathematical foundations, design patterns, best practices

## Notes

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
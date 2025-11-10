# CS224W Project Milestone Report 
**Project Title**: Graph Neural Networks for Stock Market Prediction and Portfolio Management  
**Date**: November 9, 2025  

---

## 1. Milestone Significance & Strategy
- **Deliverable Focus**:  
  1. Demonstrate the core code and data artifacts for Phases 1–4, showing that the pipeline runs end-to-end.  
  2. Report the current training status and metrics (with the caveat that this run uses synthetic data).  
  3. Provide a draft report covering problem motivation, data processing, model design, and debugging notes.  
- **High-Level Technical Roadmap**:  
  - **Phases 1–2**: Build a multi-modal financial knowledge graph by generating node features from technical indicators, fundamental, sentiment, and macro data; construct heterogeneous edges using rolling correlation, fundamental similarity, and static relations.  
  - **Phase 3**: Start with a “flattened” GNN baseline. We deliberately use a GAT + GCN mix instead of jumping straight to Transformers/LSTM to first validate that the graph inputs and labels are sane; if even a simple graph baseline fails, the issue is likely in the data rather than model capacity.  
  - **Phase 4**: Adopt a **Role-Aware Graph Transformer** (RelationAware GATv2 + PEARL positional embedding). Stocks are connected by multiple relations (industry, supply chain, competition, correlation); treating them uniformly would dilute semantics. The transformer learns a dedicated projection and attention weight per relation, enabling us to capture which relation matters most at any given time and to meet explainability needs.  
  - **Why not other approaches?**  
    - Pure time-series models (LSTM/Transformer) can only see one ticker at a time and cannot directly model cross-equity interactions.  
    - Table-based models (XGBoost/MLP) require heavy manual feature engineering for relationships and do not scale to dynamic graphs.  
    - Simple homogeneous GNNs ignore relation semantics and, in our preliminary tests, suffered from “edge-weight dilution.”  
    - Hence the two-stage strategy: a lightweight graph baseline to sanity-check the pipeline, followed by the relation-aware transformer for fine-grained modeling.  
  - **Goal**: Validate the value of heterogeneous relations in financial prediction and prepare for Phase 5 reinforcement learning. We intentionally separate relation modeling from RL so that downstream policy learning is not burdened by feature/data issues.

---

## 2. Problem Description & Motivation

Financial markets exhibit rich cross-equity relationships (industry hierarchies, supply chains, competitive dynamics, price co-movements). Traditional time-series models struggle to capture these structured interactions. Graph neural networks can perform message passing in heterogeneous networks, helping us reason about systemic risk and interdependencies.

Project goals:
1. Use GNNs to exploit relational signals across equities.  
2. Improve recall of “down” events for risk management.  
3. Apply CS224W graph techniques to a temporal financial graph, evaluating strengths and limitations.

---

## 3. Dataset Description & Processing (Phase 1 & Phase 2)

> **Note**: Because the current environment lacks internet access, we generated synthetic OHLCV data for 50 stocks (2018–2024). In a connected environment we will rerun Phase 1 with real data to obtain meaningful metrics.

### 3.1 Data Collection & Feature Engineering
- **Price (OHLCV)**  
  - Primary source: `yfinance.download()` for SPY Top-50 constituents (broad sector coverage, liquid, long history).  
  - Fallback (used in this run): synthetic prices via `generate_synthetic_prices()`. Purpose: validate code and training stability.  
    1. Log returns follow geometric Brownian motion with drift μ = 0.03/252 and volatility σ = 0.2/√252.  
    2. Cumulative sum of log returns → log-price → exponentiate back to prices.  
    3. Add ±0.2% noise to create open/high/low envelopes.  
    4. Save as `data/raw/stock_prices_ohlcv_raw.csv` with `Date` index.  
  - Alignment: `phase1_feature_engineering.py` sets `Date` as index and forward-fills missing business days.

- **Technical Indicators (10 indicators × 50 stocks = 500 columns)**  
  - Preprocessing: compute log returns, 30-day rolling σ, drop all-null rows, backfill initial windows.  
  - Indicators & formulas (examples):  
    1. `LogRet_{1,5,20}` – short/medium-term momentum.  
    2. `Vol_30d` – 30-day annualized volatility.  
    3. `RSI_14` – Relative Strength Index.  
    4. `MACD`, `MACD_Signal`, `MACD_Hist`.  
    5. `BB_Width` – Bollinger band width `(Upper-Lower)/Middle`.  
    6. `Price_to_SMA50` – deviation from the 50-day moving average.  
    7. `ATR_14` – Average True Range.  
    8. `OBV` (On-Balance Volume).  
    9. `MFI` (Money Flow Index).  
    10. Additional oscillators (`Stochastic %K/%D`, `CCI`) can be added as needed.  
  - Naming: `Feature_TICKER`, e.g. `RSI_14_AAPL`.  
  - Rationale: momentum features capture trend reversals, volatility gauges risk, oscillators signal overbought/oversold, etc. These serve as node features for the GNN.

- **Fundamental / Sentiment / Macro Features (157 columns)**  
  - Fundamentals (PE, PB, ROE, Debt/Equity, RevenueGrowth):  
    - Online mode: pull current data via `yfinance.Ticker().info`, backfill history with sector averages + perturbation.  
    - Offline mode: sample base values, apply AR(1) noise, interpolate to daily frequency.  
  - Sentiment/Macro (VIX, macro_sentiment, risk_free_rate):  
    - Online mode: download actual indicators.  
    - Offline mode: Ornstein–Uhlenbeck process + white noise, aligned to trading days.  
  - Cleaning: `log1p` for skewed metrics (PE, MarketCap), fill remaining gaps with 0, apply `StandardScaler`.  

- **Feature Consolidation & Validation**  
  - Join technical and normalized fundamental/sentiment features by date, drop rows with any NaNs.  
  - Output `data/processed/node_features_X_t_final.csv` (1,778 days × 707 features) with `Date` index.  
  - Validation: inspect mean/variance of each column, spot-check indicator ranges (e.g. RSI ∈ [0,100]).

- **Static Relations (Industry / Supply Chain / Competitors)**  
  - Industry: real mode uses public classifications; offline mode randomly assigns sectors/industries ensuring ≥2 stocks per sector for connectivity.  
  - Supply chain: randomly select 1–3 suppliers per customer, direct edges (`customer → supplier`) with weight 1.  
  - Competitors: match 1–2 similarly sized firms, undirected edges stored as bidirectional pairs.  
  - Saved as `static_sector_industry.csv` and `static_supply_competitor_edges.csv`, later converted to graph edges.

### 3.2 Dynamic Edge Computation
- **Rolling Correlation**: compute 30-day rolling correlation of log returns, retain entries with |corr| ≥ 0.6, map weights to [0,1].  
- **Fundamental Similarity**: cosine similarity on Z-scored fundamental vectors, keep values ≥ 0.8; log and skip days lacking valid vectors.

### 3.3 Heterogeneous Graph Construction (Phase 2)
- **High-Level Steps**:  
  1. Align node feature matrices, dynamic edges, and static edges, filter out incomplete trading days.  
  2. Map tickers to node indices, populate `data['stock'].x`, `tickers`, and `node_ids`.  
  3. Generate dynamic edges (correlation & similarity) with thresholds + Top-K pruning.  
  4. Generate static edges (industry, supply chain, competitors).  
  5. Apply sparsification and normalization, optionally insert self-loops for isolated nodes.  
  6. Build daily `HeteroData` objects (`graph_t_YYYYMMDD.pt`).  
  7. Validate via re-loading, degree stats, weight ranges, and log performance metrics.

- **Detailed Pipeline**:  
  - **Input Alignment**: load `node_features_X_t_final.csv` and dynamic/static edge files; skip days missing any component.  
  - **Nodes & Metadata**: single node type `stock` (50 nodes, 707 features); store `tickers` and `node_ids`.  
  - **Dynamic Edges**:  
    1. **Rolling Correlation**: filter below 0.6, add both directions, normalize weights. Motivation: 0.6 is a common “significant correlation” threshold; keeping all edges would result in ~40% density and severe over-smoothing, so we retain top 5 neighbors per node (~9 average neighbors).  
    2. **Fundamental Similarity**: keep cos-sim ≥0.8 (≈36° angle), add both directions, normalize; Top-K prevents noisy diffusion and surfaces truly similar peers (expected 4–6 neighbors with real data; fewer with synthetic data).  
    3. Map tickers to indices (`ticker2idx`), ensure `edge_index` is `torch.long`, `edge_weight` is `torch.float32`; log unknown tickers.  
  - **Static Edges**: industry (undirected), supply chain (directed), competitor (undirected). After generation, ensure each node has at least one static edge; approximate counts: 120 industry edges, 70 supply chain edges, 80 competitor edges (doubled when stored bidirectionally).  
  - **Sparsity & Stability**: Dynamic edges keep top 5 in- and out-neighbors per node; apply `torch.nan_to_num`; store both normalized and raw weights; self-loops optional (disabled here).  
  - **Serialization**: create `HeteroData`, fill edge indices/weights/attrs per relation, save with `torch.save(..., weights_only=False)` (after registering safe globals).  
  - **Quality Checks**: randomly reload 10 graphs; confirm average degree 13.8 (max 21, min 6); verify normalized weights ∈ [0,1]; ensure 1,778 graphs match trading days; log performance (~0.12s per day, total ~3.5 minutes).

---

## 4. Model Design & Evaluation Workflow (Phase 3 & Phase 4)

### 4.1 Phase 3 – Baseline GNN (Flattened Graph)
- **Input Preparation**  
  - Each trading day produces a single homogeneous `Data` graph by merging all relation types.  
  - Node features: 707-dim vectors from Phase 1 (technical + fundamental + sentiment).  
  - Edge list: concatenation of every relation’s `edge_index`; edge weights are min-max normalized and stored as attributes.  
- **Architecture & Implementation**  
  1. `GATConv(707 → 128, heads=4, dropout=0.3)` – adaptive weighting of neighbors even in the flattened setting.  
  2. `GCNConv(512 → 128)` – stabilizes aggregation and reduces noise introduced by Step 1.  
  3. Classifier head: `Linear(128 → 64) → ReLU → Dropout(0.2) → Linear(64 → 2)` with log-softmax at inference.  
  4. Optional residual connection between Layer 1 & classifier (disabled in current run but implemented for future tests).  
- **Training Configuration**  
  - Loss: Focal Loss (α=0.5, γ=2.0) to emphasize minority cost (“down” class).  
  - Optimizer: Adam (LR 5e-4, weight decay 1e-4).  
  - Scheduler: ReduceLROnPlateau (factor 0.5, patience 3, min LR 1e-6).  
  - Early Stopping: patience 5, min delta 1e-4.  
  - Batch strategy: one graph per batch (sequence over days). 70/15/15 chronological split.  
- **Evaluation & Logging**  
  - Metrics: Accuracy, Macro-F1, ROC-AUC, confusion matrix, class-wise precision/recall; stored per epoch.  
  - TensorBoard: loss, F1, LR, gradient norms.  
  - Checkpoints: best F1 + every 5 epochs (`models/checkpoints/`).  
  - Ablation hooks (implemented but not yet run): remove GAT layer, adjust focal γ, compare weighted cross-entropy.  
- **Why This Baseline**  
  - Confirms that features/labels behave sensibly before adding relation-specific complexity.  
  - Provides a benchmark for future “single-edge vs multi-edge” comparisons on real data.  
  - Lightweight and fast to train (~3.8s/epoch on CPU).

### 4.2 Phase 4 – Role-Aware Graph Transformer (Heterogeneous Graph)
- **Motivation**  
  - Financial graphs contain distinct semantics per relation (industry, supply chain, competition, correlation). We need relation-specific parameters to avoid averaging them away.  
  - Regulators and business stakeholders demand explainability; attention scores per relation provide auditability.  
  - PEARL positional embeddings encode structural roles so that nodes with similar connectivity patterns share representational priors.
- **Architecture**  
  - Input: daily `HeteroData` with separate edge types; PEARL embedding (32-dim) concatenated with 707-dim features.  
  - Transformer Stack (repeat twice):  
    - `HeteroConv` with 5 × `RelationAwareGATv2Conv(edge_type=r, in_dim, out_dim=128, heads=4, dropout=0.3)`; each relation has its own linear projection and attention parameters.  
    - Relation aggregator: learned attention vector (128×5) → softmax → weighted sum of relation outputs.  
    - LayerNorm + residual fusion with previous activations.  
  - Output head: `Linear(128 → 64) → GELU → Dropout(0.2) → Linear(64 → 2)`.  
  - Gradient clipping (max norm 1.0); AMP on GPU (disabled in this CPU run).  
- **Training & Evaluation**  
  - Loss/optimizer/scheduler identical to Phase 3 for apples-to-apples comparison.  
  - Logging: in addition to baseline metrics, we record per-relation attention weights and store them alongside checkpoints.  
  - Current synthetic run: 1 epoch, Test Accuracy ≈0.4897, F1 ≈0.6504 (expected to be near random due to synthetic data).  
- **Planned Analyses**  
  - Compare relation-aware vs flattened baselines once real data is available.  
  - Analyze attention weights during market events (e.g., sector crashes).  
  - Extend with temporal encoders (GRU/LSTM) to capture sequential dependencies.  
  - Run `scripts/phase4_hyperparameter_sweep.py` to tune hidden dims, number of layers, dropout, and learning rate.

### 4.3 Evaluation Pipeline Summary
1. Load or generate daily graphs for train/val/test splits.  
2. Train Phase 3 baseline → log metrics → save best checkpoint.  
3. Train Phase 4 transformer from scratch on same splits.  
4. Export metrics, confusion matrices, ROC curves, and (Phase 4) relation attention statistics.  
5. Compare runs on a unified dashboard (TensorBoard + CSV summaries).  
6. Once real data is available, extend evaluation with profitability metrics and calibration checks.

---

## 5. Code Deliverables & Program Structure

### 5.1 Dataset Processing (Processing the Dataset)
| Script | Purpose | Key Inputs | Outputs / Artifacts | Notes |
| --- | --- | --- | --- | --- |
| `scripts/phase1_data_collection.py` | Fetch raw OHLCV, macro, sentiment data; synthesize fallback series when offline | `config/data_sources.yaml` (optional), ticker list | `data/raw/stock_prices_ohlcv_raw.csv`, `data/raw/sentiment_macro_raw.csv` | Supports resume, logging to `data_collection.log` |
| `scripts/phase1_static_data_collection.py` | Generate static relations (industry, supply chain, competitors) | Ticker list, optional real mappings | `data/raw/static_sector_industry.csv`, `data/raw/static_supply_competitor_edges.csv` | SPY Top-50 fallback ensures connectivity |
| `scripts/phase1_feature_engineering.py` | Compute technical indicators, normalize fundamentals/sentiment, produce node feature matrix and edge params | Files above | `data/processed/node_features_X_t_final.csv`, `data/edges/edges_dynamic_corr_params.pkl`, etc. | Contains detailed logging & sanity checks |
| `scripts/phase2_graph_construction.py` | Build daily heterogeneous graphs with sparsification and normalization | Processed node features + edge params | `data/graphs/graph_t_YYYYMMDD.pt` | Top-K controls, metadata embedding, performance logs |

Usage example:
```bash
python scripts/phase1_data_collection.py --tickers config/tickers_spy50.txt
python scripts/phase1_feature_engineering.py --mode offline
python scripts/phase2_graph_construction.py --output data/graphs
```
Dependencies: `pandas`, `numpy`, `ta-lib`, `torch`, `torch-geometric`, `tqdm`.

### 5.2 Model Training / Evaluation (Training/Evaluating the Model)
| Script | Role | Key CLI Options | Outputs |
| --- | --- | --- | --- |
| `scripts/phase3_baseline_training.py` | Train/evaluate flattened GNN baseline | `--epochs`, `--loss-type`, `--checkpoint-dir`, `--device` | `models/checkpoints/`, TensorBoard runs, `metrics_phase3.csv` |
| `scripts/phase4_core_training.py` | Train/evaluate Role-Aware Graph Transformer | `--epochs`, `--hidden-dim`, `--num-layers`, `--amp`, `--grad-clip` | `models/core_transformer_model.pt`, attention weights logs, `metrics_phase4.csv` |
| `milestone/METRICS_QUICK_REFERENCE.md` | Reference for formulas, interpretations, debugging tips | n/a | Quick reference for review sessions |

Typical workflow:
```bash
python scripts/phase3_baseline_training.py --epochs 40 --loss-type focal
python scripts/phase4_core_training.py --epochs 20 --hidden-dim 256 --num-layers 2 --amp
```
Both scripts automatically resume from checkpoints if `--resume` is provided.

**Runtime & Artifacts**: CPU-only runs take ~4 seconds/epoch (Phase 3) and ~28 seconds/epoch (Phase 4); GPU usage is recommended for longer schedules. Each script writes per-epoch metrics to CSV (`metrics_phase3.csv`, `metrics_phase4.csv`) and logs confusion matrices/ROC curves to `runs/`. Additional evaluation utilities—such as `analysis/plot_metrics.ipynb` for plotting learning curves and `analysis/export_attention.py` for inspecting relation weights—are included to support TA review and future comparisons.

### 5.3 Additional Utilities (Any Other Programs Required)
- `scripts/phase4_hyperparameter_sweep.py`: orchestrates grid search; specify search space via JSON/YAML, stores per-run configs, metrics, and checkpoints under `models/sweeps/`.  
- `scripts/utils_data.py`: shared helpers for loading graphs, batching, and sanity checks.  
- `scripts/rl_environment.py` (draft): Phase 5 reinforcement-learning environment scaffold (state definition, action space, reward structure).  
- `runs/` & `models/`: standardized directories for TensorBoard summaries, checkpoints, and logs; accompanied by a `README_runs.md` on how to interpret them.

---

## 6. Current Metrics & Assessment

- Synthetic data results only validate the pipeline (metrics near random):  
  - Phase 3: Val F1 ≈ 0.67.  
  - Phase 4: Test F1 ≈ 0.65.  
- Phase 3 (epoch 8) snapshot: train/val samples = 63,850/13,700; labels roughly balanced (up 51%, down 49%); confusion matrix shows recall ~0.65 for both classes; ROC-AUC ≈0.51 (as expected for random series).  
- Synthetic data nonetheless helps verify feature scaling, convergence, and logging quality.

### 6.1 Why These Metrics?
- **Accuracy**: quick sanity check in balanced settings; not emphasized for risk management.  
- **F1 Score (macro/micro)**: balances precision/recall, especially important for the “down” class.  
- **ROC-AUC**: threshold-independent ranking ability; monitors whether the model separates classes at all.  
- **Confusion Matrix**: reveals FP/FN patterns that drive subsequent strategy decisions (e.g., increasing down-class recall).  
- **TensorBoard Loss/F1 Curves**: diagnose training stability.  
- **Planned**: cumulative return, max drawdown, Sharpe Ratio, Downside Deviation, Brier Score, and calibration curves once real data is available.

### 6.2 Current Performance & Path Forward
- **Synthetic Data Summary**:  
  - Phase 3: Accuracy ≈0.51, Macro F1 ≈0.67, ROC-AUC ≈0.51.  
  - Phase 4: Accuracy ≈0.49, F1 ≈0.65, ROC-AUC ≈0.50.  
  - These numbers confirm the pipeline runs; they do not imply predictive power.  
- **Why Limited Performance?**  
  - Synthetic prices lack real-world co-movement and structural signals.  
  - Focal Loss primarily stabilizes training; it cannot create signal where none exists.  
- **Real-Data Plan**:  
  1. Rerun Phase 1 with real OHLCV, fundamental, macro, and sentiment data.  
  2. Align trading calendars, handle suspensions/splits/missing values.  
  3. Extend metrics with profitability and calibration measures.  
  4. Examine precision/recall trade-offs for the down class, adjust thresholds or loss weights.  
  5. Tune Top-K or adopt MST-based sparsification if graphs become too dense or sparse.

### 6.3 Risks & Next Steps
1. **Data Quality**: real data may contain anomalies (e.g., stock splits) that can distort correlations; plan to clean and backfill carefully.  
2. **Graph Sparsity vs. Over-Smoothing**: maintain Top-K around 5 for correlations; dynamically adjust with real density statistics.  
3. **Model Architecture**: evaluate alternative convolutions (SAGEConv, edge dropout) for efficiency; analyze transformer attention weights for interpretability; consider temporal encoders.  
4. **Training Practices**: Focal Loss with large γ may require lower LR; consider Warmup + ReduceLROnPlateau; adopt NeighborSampler on GPU.  
5. **Future Metrics**: add return- and risk-based metrics (cumulative return, max drawdown, Sharpe, downside deviation) and calibration checks (Brier, calibration curves) once real labels are available.

---


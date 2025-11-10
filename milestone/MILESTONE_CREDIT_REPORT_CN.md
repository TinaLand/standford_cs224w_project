# CS224W Project Milestone Report (English Version)

**Course**: CS224W – Machine Learning with Graphs  
**Project Title**: Graph Neural Networks for Stock Market Prediction and Portfolio Management  
**Date**: November 9, 2025  
**Milestone Weight**: 2% of total course grade (project component 6.67%, Credit / No Credit)

---

## 1. Milestone Significance & Strategy

- **Course Requirement**: The milestone is graded Credit / No Credit. Its purpose is to ensure the team builds an end-to-end pipeline early and to give the TAs substantive material for feedback.  
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

## 4. Model Design & Evaluation Metrics (Phase 3 & Phase 4)

### 4.1 Phase 3 – Baseline GNN
- **Design Rationale**:  
  - Start with a “minimal viable” graph model by flattening heterogeneous edges into a single `Data` object, verifying labels, losses, and training loops before adding complexity.  
  - Combine `GATConv(707 → 128, heads=4, dropout=0.3)` with `GCNConv(512 → 128)` to benefit from adaptive neighbor weighting (GAT) and smoothing (GCN).  
  - Light classifier head (`Linear(128 → 64) → ReLU → Dropout(0.2) → Linear(64 → 2)`) keeps parameters small to avoid overfitting.  
- **Loss & Imbalance Handling**: Focal Loss (α=0.5, γ=2.0) focuses training on difficult samples, appropriate when “down” events carry higher cost.  
- **Optimization & Scheduling**: Adam (LR 5e-4, weight decay 1e-4); ReduceLROnPlateau (factor 0.5, patience 3, min LR 1e-6); Early Stopping (patience 5, δ=1e-4).  
- **Batching & Splits**: full-graph training (batch size = 1 day); splits 70/15/15 (train/val/test). Training loss decreases smoothly, validation F1 stays ~0.66–0.68.  
- **Monitoring**: TensorBoard for loss/F1/LR curves; confusion matrix & ROC-AUC each epoch to watch for collapse.  
- **Checkpointing**: best F1 + snapshots every 5 epochs, stored in `models/checkpoints/`.  
- **Insights**: flattening heterogeneous edges highlights whether the data is healthy; once verified, we can compare “single-edge baseline” vs “multi-edge baseline” on real data; consider NeighborLoader for larger graphs.

### 4.2 Phase 4 – Role-Aware Graph Transformer
- **Design Rationale**:  
  - Different relations (industry, supply chain, competition, correlation) require relation-specific parameters to avoid semantic dilution.  
  - PEARL positional embeddings inject structural context to reduce isomorphism issues.  
  - Relation attention weights offer interpretability for financial stakeholders.  
- **Architecture**:  
  - Each layer: `HeteroConv` with 5 × `RelationAwareGATv2Conv(edge_type=r, in_dim, out_dim=128, heads=4, dropout=0.3)`.  
  - Relation aggregator: learnable attention vector (128×5) softmaxed across relations.  
  - PEARL positional embedding (32 dims) concatenated with original features.  
- **Training Setup**: two transformer blocks, hidden dim 256; gradient clipping (max norm 1.0); AMP enabled on GPU (disabled in CPU run).  
- **Current Results**: 1 epoch → Test Accuracy ≈0.4897, F1 ≈0.6504 (synthetic data).  
- **Next Steps**: compare relation-aware vs flattened baselines on real data; export attention weights for interpretability; add temporal encoders (GRU/LSTM) to capture dynamics; run `phase4_hyperparameter_sweep.py` for tuning.

---

## 5. Code Deliverables & Program Structure

### 5.1 Dataset Processing (Processing the Dataset)
- `scripts/phase1_data_collection.py`: raw price, macro, sentiment downloads & fallback synthesis.  
- `scripts/phase1_static_data_collection.py`: industry, supply chain, competitor relation generation with SPY Top-50 fallback.  
- `scripts/phase1_feature_engineering.py`: technical indicator computation, normalization of fundamentals/sentiment, dynamic/static edge parameter generation, node feature export.  
- `scripts/phase2_graph_construction.py`: daily heterogeneous graph generation (Top-K sparsification, edge weight normalization, metadata injection).

### 5.2 Model Training / Evaluation (Training/Evaluating the Model)
- `scripts/phase3_baseline_training.py`: baseline GNN training (GAT + GCN), Focal Loss, ReduceLROnPlateau, Early Stopping, TensorBoard, confusion matrix & ROC-AUC.  
- `scripts/phase4_core_training.py`: Role-Aware Graph Transformer training with PEARL embeddings, relation attention, gradient clipping, AMP support.  
- `milestone/METRICS_QUICK_REFERENCE.md`: metric formulas, interpretations, debugging checklist.

### 5.3 Additional Utilities (Any Other Programs Required)
- `scripts/phase4_hyperparameter_sweep.py`: grid search over hidden dims, layers, epochs, learning rates; logs metrics and checkpoints per run.  
- `scripts/utils_data.py`, `scripts/rl_environment.py` (draft): data loading utilities and Phase 5 RL environment sketch.  
- `runs/`, `models/`: TensorBoard logs and model checkpoints for reproducibility.

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


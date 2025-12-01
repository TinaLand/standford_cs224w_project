# Phase 6: Evaluation & Visualization

**Objective:** Quantitatively and qualitatively assess the effectiveness of the model and trading strategy.

## 6.1 Quantitative Evaluation

* **Node-Level Metrics**: Accuracy, F1, IC, **Precision@Top-K**.
* **Portfolio-Level Metrics (Backtested)**: **Cumulative return**, **Sharpe ratio**, **Max drawdown**.

## 6.2 Ablation Studies

Systematically remove or replace components to validate the contribution of key design choices.

* **Edge Type Ablation:**
 * Compare performance using only one edge type (e.g., Sector vs. Correlation vs. Fundamental).
 * Full model (Static + Dynamic edges) vs. Static Backbone Only.
* **Positional Embedding Ablation:**
 * **PEARL** embeddings vs. Fixed Laplacian embeddings vs. No positional embeddings.
* **Threshold Sensitivity:**
 * Test the impact of correlation threshold $\rho$ sensitivity (e.g., $\rho \in [0.4, 0.8]$) on graph sparsity and performance.

## 6.3 Qualitative Visualization

* **Embedding Interpretation:** Use **t-SNE/UMAP** to visualize the learned stock embeddings.
* **Role Analysis:** Interpret the structural roles encoded by PEARL, specifically identifying and analyzing clusters corresponding to:
 * **Hubs:** Highly connected stocks (high influence).
 * **Bridges:** Stocks connecting different sectors or clusters (high cross-market conductivity).
 * **Role Twins:** Stocks with similar structural profiles and comparable market behavior.

## 6.4 Research Experiments (Proposal-Aligned)

Additional experiments addressing proposal requirements for comprehensive analysis:

### 6.4.1 Lookahead Horizon Analysis

Tests different prediction horizons to find optimal lookahead window:

```bash
python scripts/experiment_lookahead_horizons.py
```

* **Tests**: 1, 3, 5, 7, 10 day lookahead horizons
* **Output**: `results/lookahead_horizon_results.csv`
* **Result**: 10 days optimal (Test F1: 0.3576)
* **Runtime**: ~1-2 hours

### 6.4.2 Graph Sparsification Experiments

Evaluates different Top-K thresholds and correlation cutoffs:

```bash
python scripts/experiment_graph_sparsification.py
```

* **Tests**: Various Top-K values and correlation thresholds
* **Output**: `results/graph_sparsification_results.csv`
* **Purpose**: Find optimal graph density for performance
* **Runtime**: ~1-2 hours

### 6.4.3 Robustness Checks

Tests model performance under different transaction cost and slippage assumptions:

```bash
python scripts/experiment_robustness_checks.py
```

* **Tests**: Transaction costs (0% to 0.5%), Slippage (0% to 0.1%)
* **Output**: `results/robustness_checks_results.csv`
* **Purpose**: Validate model robustness to realistic trading conditions
* **Runtime**: ~30 minutes - 1 hour

### 6.4.4 Statistical Significance Testing

Integrated into Phase 6 evaluation pipeline:

```bash
python -m src.evaluation.evaluation
```

* **Tests**: Block bootstrap for Sharpe ratio, t-tests for accuracy
* **Output**: `results/statistical_tests.csv`
* **Purpose**: Validate statistical significance of performance improvements
* **Automatic**: Runs as part of Phase 6 evaluation
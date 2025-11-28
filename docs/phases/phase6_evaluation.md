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
# ✨ Phase 4: Graph Transformer + PEARL Training (MVP 2)

**Objective:** Train the core, role-aware, multi-modal GNN model for robust and interpretable node-level prediction.

## 4.1 Model Architecture

* **Core GNN:** **Graph Transformer** (Ying et al., 2021).
    * **Configuration:** 2 layers, 4 heads, hidden size 256.
    * **Mechanism:** Uses edge-aware global attention to aggregate heterogeneous, multi-relational information from both static and dynamic edges.
* **Structural Role Encoding:** **PEARL Positional Embeddings** (You et al., 2023).
    * **Integration:** Encodes structural roles (hubs, bridges, role twins) and is **concatenated** with the node features.
    * **Rationale:** Provides explainable structural roles and robustness.

## 4.2 Training and Task

1.  **Input:** The full graph with combined static + dynamic edges.
2.  **Dynamic Graph Updates:** Implement the logic to **periodically update** the dynamic edges (correlation, fundamental, news) during training, while maintaining the static backbone.
3.  **Task:** **5-day-ahead stock return sign prediction**.
    * Target $y_{i,t+5} = 1$ if $\frac{p_{i,t+5}-p_{i,t}}{p_{i,t}}>0$, else $0$.
    * Model output $\hat{y}_{i,t+5}=P(y_{i,t+5}=1)$.
4.  **Loss Function:** Cross-entropy loss for the binary classification task.

## 4.3 Node-Level Evaluation Metrics

* Accuracy, F1.
* **Information Coefficient (IC)**.
* **Precision@Top-K**.
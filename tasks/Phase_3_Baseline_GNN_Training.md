# 📈 Phase 3: Baseline GNN Training (MVP 1)

**Objective:** Establish a simple baseline using a standard GNN on a static graph to validate the PyG pipeline and data handling.

## 3.1 Model and Task

* **Baseline Model:** Select from **GCNConv, GATConv, or SAGEConv**.
* **Graph Input:** Initially use the **static graph** (e.g., Sector/Industry edges) and technical features only.
* **Prediction Task:** Node-level prediction: **Next-day stock return sign prediction**.
* **Output Target:** $\hat{y}_{i,t+1}$ (return sign for the next day).

## 3.2 Training Procedure

1.  **Data Preparation:** Load the static `HeteroData` subgraph and prepare it for the chosen GNN layer (demonstrates PyG layers and `HeteroData` handling).
2.  **Model Instantiation:** Initialize the chosen GNN layer.
3.  **Training Loop:** Train the model using a cross-entropy loss (for the binary classification task).
4.  **Result Logging:** Record and save the baseline model's performance metrics.

## 3.3 Evaluation Metrics

* **Node-level Metrics:** Accuracy and F1 Score.
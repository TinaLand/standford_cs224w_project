# Phase 2: Graph Construction

**Objective:** Build a multi-relational PyG `HeteroData` graph with combined static and dynamic edges.

## 2.1 Static Edges (Stable Backbone)

Static edges form the stable structure of the graph.

| Edge Type | Description | Weighting Scheme | Rationale |
| :--- | :--- | :--- | :--- |
| **Sector/Industry** | Stocks in the same sector or industry. | 1.0 (same industry), 0.5 (same sector). | Ensures tight intra-industry clustering while maintaining broader sector-level relations. |
| **Supply Chain** | Customer-supplier relationships. | Binary (1/0). | Reflects revenue dependence; binary simplifies modeling of directional dependence. |
| **Competitor** | Competitor-competitor relationships. | Binary (1/0). | Ensures clear market adjacency and competitive links are modeled. |

## 2.2 Dynamic Edges (Updated Periodically)

Dynamic edges capture evolving market co-movements.

| Edge Type | Description | Weighting Scheme | Connection Condition | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **Rolling Correlation** | Pearson correlation of returns over a 30-day window. | $\rho_{ij,t}$. | $|\rho_{ij,t}| > 0.6$. | Captures meaningful co-movements, avoiding noise/sparsity. |
| **Fundamental Similarity** | Cosine similarity of normalized financial metrics. | $s_{ij}$. | $s_{ij} > 0.8$. | Links stocks with highly similar financial profiles ("role twins"). |
| **News Sentiment** (Optional) | Stock news relationships. | Sentiment score. | - | Adds dynamic market perception. |

## 2.3 PyG Graph Structure

* Use PyTorch Geometric's **`HeteroData`** object.
* Node Set ($V$): Stocks.
* Edge Set ($E_{t}$): Contains all static and dynamic edge types.
* Node Features ($X_{t}$): Stored on the stock nodes.
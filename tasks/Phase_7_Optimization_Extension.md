# 🌟 Phase 7: Optimization & Extension (Optional / MVP 4 & 5)

**Objective:** Implement advanced features and ensure the robustness and scalability of the final framework.

## 7.1 Dynamic Graph Updates

* **Implementation:** Finalize the efficient implementation of the combined graph strategy: **Static edges** maintain stable clusters, while **Dynamic edges** capture evolving market co-movements and cross-sector bridges.
* **Testing:** Evaluate the framework's performance gains when using the dynamic updates, especially during volatile market periods.

## 7.2 Multi-Agent RL Extension (Optional)

* **Architecture:** Explore the use of **Multi-Agent RL** where subsets of the portfolio (e.g., different sectors or risk profiles) are managed by separate, potentially interacting, RL agents.
* **Evaluation:** Compare the performance of the single-agent vs. multi-agent RL strategy.

## 7.3 Expected Contributions

1.  **Dataset:** A dynamic multi-relational market graph dataset (2015-2025) in PyG format.
2.  **Model:** A Role-aware Graph Transformer + RL agent framework.
3.  **Insights:** Interpretable stock roles, cross-sector dependencies, and insights into evolving market structure.
4.  **Performance:** Improved predictive accuracy, financial interpretability, and trading results over traditional time-series or baseline GNNs.
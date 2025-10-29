# 🤖 Phase 5: RL Integration & Portfolio Optimization (MVP 3)

**Objective:** Integrate the GNN embeddings into a Reinforcement Learning agent to optimize sequential portfolio decisions.

## 5.1 RL Key Components Definition

| Element | Definition | Composition / Algorithm |
| :--- | :--- | :--- |
| **State** | The input for the agent's decision. | **Portfolio holdings** + **GNN node embeddings** + **Node features**. |
| **Action** | The operation performed by the agent at each trading day. | **Buy / Sell / Hold** for each stock. |
| **Reward** | The agent's objective function. | **Portfolio return** / **Risk-adjusted return** (e.g., Sharpe Ratio). |
| **Algorithm** | The learning strategy. | **Q-learning** or **Policy Gradient**. |

## 5.2 Portfolio Optimization Pipeline

1.  **Embedding Generation:** Use the trained Graph Transformer (from Phase 4) to generate up-to-date, role-aware stock embeddings at each trading step.
2.  **State Construction:** Concatenate embeddings, current portfolio holdings, and node features to form the complete RL state.
3.  **Agent Decision:** The RL agent receives the state and outputs a sequential action (Buy/Sell/Hold).
4.  **Environment Interaction (Backtesting):** Simulate the trade execution, update the portfolio, and calculate the reward.
5.  **Policy Update:** Use the reward signal to update the RL agent's policy network or Q-function.

## 5.3 Portfolio-Level Evaluation Metrics (Backtesting)

* **Cumulative Return**.
* **Sharpe Ratio**.
* **Max Drawdown**.
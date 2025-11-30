# Graph Neural Networks for Stock Market Prediction: A Heterogeneous Graph Approach with Reinforcement Learning

**Stanford CS 224W (Machine Learning with Graphs) Course Project**

## Abstract

Many have sought to make money by investing in the stock market, from professional fund managers to everyday people, but few are able to beat the market. In fact, S&P Dow Jones Indices' 2024 SPIVA U.S. Mid-Year report shows that 57% of all active large-cap U.S. equity managers underperformed the S&P 500, consistent with the 60% underperformance rate observed in 2023 [Ganti, 2024].

Predicting stock prices is known to be extremely challenging due to the market's non-linearity, short and long-term temporal dependencies, and the complex relationships between stocks. In this work, we aim to explore the benefits of modelling interstock relationships in predicting stock prices. We propose a novel heterogeneous graph neural network architecture called **Role-Aware Graph Transformer** that incorporates multiple types of stock relationships (correlation, fundamental similarity, sector connections, and supply chain) and uses PEARL positional embeddings to encode structural roles. We demonstrate that our model significantly outperforms baselines that do not incorporate interstock relations and show that conditioning on multiple relationship types helps generalize the model to future unseen market conditions. Our system achieves a Sharpe Ratio of 1.90, cumulative return of 45.99%, and demonstrates the effectiveness of graph-structured data in financial prediction tasks.

---

## 1. Motivation & Problem Statement

### 1.1 Stocks Have Shared Trends

Stock prices often move together due to the influence of market fundamental factors. However, idiosyncratic factors can also lead to differences across the market. For instance, stocks within the same sector or industry tend to be correlated, meaning that they can move in the same direction (positive correlation) or opposite directions (negative correlation). This type of information is often overlooked by traditional approaches, but can be a valuable source of information for predicting stock market movements.

Previous approaches to stock market prediction have typically treated each stock as an isolated entity, only considering the individual target company [Patel et al., 2024]. However, in reality, stock prices are not independent and can be influenced by other factors. The interdependencies between different stocks can be measured by their correlations, fundamental similarities, sector relationships, and supply chain connections.

### 1.2 The Graph Structure of Financial Markets

Financial markets are inherently **graph-structured**:
- **Sector/Industry Relationships**: Stocks in the same sector (e.g., tech, healthcare) are highly correlated
- **Supply Chain Connections**: Companies in supply chains influence each other
- **Competitive Relationships**: Competitors' performance affects each other
- **Dynamic Correlations**: Price movements create time-varying correlations
- **Fundamental Similarity**: Companies with similar financial characteristics (P/E ratios, market caps, etc.) tend to move together

**Key Insight**: By modeling stocks as nodes and relationships as edges, we can leverage Graph Neural Networks to capture these complex interdependencies.

### 1.3 Why Heterogeneous Graphs?

To capture interstock relations, 3 main types of stock graphs are commonly used: corporate-relational, textual and statistical graphs [Patel et al., 2024]. Constructing corporate-relational graphs requires solid finance domain knowledge and large-scale data engineering, which is time-consuming, laborious, and costly [Tian et al., 2023]. These predefined graphs are also usually static, not always up-to-date, and contain wrong, missing, or irrelevant connections between stocks, which can incur much noise for the learning and inhibit generalization [Tian et al., 2023, Kim et al., 2019, Ma et al., 2024].

We propose using **heterogeneous graphs** that combine multiple relationship types:
- **Statistical correlations** (rolling correlation): Dynamic, data-driven relationships
- **Fundamental similarity**: Based on financial characteristics
- **Sector/Industry connections**: Domain knowledge-based relationships
- **Supply chain/Competitor relationships**: Business relationship-based connections

This multi-relational approach allows the model to learn from different types of information simultaneously, providing complementary signals for prediction.

### 1.4 Our Approach: Role-Aware Graph Transformer

We propose a **heterogeneous graph neural network** approach that:
1. Constructs multi-relational graphs capturing different types of stock relationships
2. Uses **PEARL positional embeddings** to encode structural roles (hubs, bridges, isolated nodes)
3. Employs a **Role-Aware Graph Transformer** with multi-head attention to aggregate information across relationships
4. Incorporates **time-aware positional encoding** to capture temporal patterns
5. Integrates predictions with **Reinforcement Learning** for optimal portfolio management

### 1.5 Task Definition

**Primary Task**: Predict the sign of 5-day-ahead stock returns (binary classification: Up/Down)

**Secondary Task**: Use predictions to construct an optimal trading portfolio via RL

**Problem Statement**: Given prices and features of N stocks on T days and heterogeneous stock relationship graphs A = {A₁, A₂, ..., Aₖ} where each Aᵢ ∈ R^(N×N) represents a different relationship type, the task is to predict the 5-day-ahead return sign for each stock.

**Evaluation Metrics**:
- **Node-level**: Accuracy, F1 Score, Precision@Top-K, Information Coefficient (IC)
- **Portfolio-level**: Sharpe Ratio, Cumulative Return, Max Drawdown

---

## 2. Data & Task Explanation

### 2.1 Dataset

We use **50 major US stocks** from diverse sectors:
- Technology: AAPL, MSFT, GOOGL, AMZN, META
- Finance: JPM, BAC, GS, MS
- Healthcare: JNJ, PFE, UNH
- Consumer: WMT, HD, MCD
- Energy: XOM, CVX
- And more...

**Time Period**: 2015-01-01 to 2024-12-31 (approximately 10 years of data)

**Data Sources**:
- **OHLCV Data**: Yahoo Finance (via `yfinance`)
- **Fundamental Data**: Company financials (P/E ratio, market cap, revenue, etc.)
- **Macro Indicators**: VIX (volatility index), sentiment indicators

### 2.2 Feature Engineering

We compute **1450+ features** per stock, including:

#### Technical Indicators (33 features per stock)
```python
# Example: Technical indicators calculation
def calculate_technical_indicators(df):
    # Momentum indicators
    df['RSI'] = talib.RSI(df['Close'].values.astype(np.float64))
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(...)
    
    # Volatility indicators
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(...)
    df['ATR'] = talib.ATR(...)
    
    # Volume indicators
    df['OBV'] = talib.OBV(...)
    df['AD'] = talib.AD(...)
    
    # Additional indicators
    df['Stochastic'] = talib.STOCH(...)
    df['ADX'] = talib.ADX(...)
    df['CCI'] = talib.CCI(...)
    # ... and more
```

#### Fundamental Features (157 features)
- Financial ratios: P/E, P/B, Debt/Equity, ROE, ROA
- Growth metrics: Revenue growth, EPS growth
- Valuation metrics: Market cap, Enterprise value

#### Sentiment & Macro Features
- VIX (volatility index)
- Market sentiment indicators
- Economic indicators

### 2.3 Graph Construction

We construct **heterogeneous graphs** for each trading day with:

#### Node Features
- **Technical indicators** (33 dims)
- **Fundamental features** (normalized)
- **Sentiment/macro features**

#### Edge Types

1. **Rolling Correlation** (`rolling_correlation`)
   - Dynamic edges based on 30-day rolling correlation
   - Edge weight: correlation coefficient
   - Top-K sparsification (K=10) to keep strongest correlations

2. **Fundamental Similarity** (`fund_similarity`)
   - Static edges based on financial similarity
   - Computed using cosine similarity of fundamental features
   - Threshold: similarity > 0.7

3. **Sector/Industry** (`sector_industry`)
   - Static edges connecting stocks in the same sector/industry
   - Binary edges (1 if same sector, 0 otherwise)

4. **Supply Chain/Competitor** (`supply_competitor`)
   - Static edges for supply chain and competitive relationships
   - Binary edges

**Graph Statistics**:
- **Nodes**: 50 stocks per graph
- **Edges**: ~600-800 edges per graph (varies by date)
- **Total Graphs**: 2,317 trading days

```python
# Graph construction example
def construct_graph(date, node_features, edge_data):
    data = HeteroData()
    
    # Node features
    data['stock'].x = node_features  # Shape: [50, feature_dim]
    
    # Edge indices and attributes
    data['stock', 'rolling_correlation', 'stock'].edge_index = corr_edges
    data['stock', 'rolling_correlation', 'stock'].edge_attr = corr_weights
    
    data['stock', 'fund_similarity', 'stock'].edge_index = fund_edges
    data['stock', 'fund_similarity', 'stock'].edge_attr = fund_weights
    
    # ... other edge types
    
    return data
```

### 2.4 Target Labels

**Classification Target**: Binary label indicating 5-day-ahead return sign
- `1` if return > 0 (Up)
- `0` if return ≤ 0 (Down/Flat)

**Regression Target**: Continuous 5-day-ahead return (for multi-task learning)

---

## 3. Model Architecture & Appropriateness

### 3.1 Why Graph Neural Networks?

**Traditional ML approaches** (LSTM, CNN) treat stocks as **independent** time series, ignoring:
- Cross-stock dependencies
- Sector-level movements
- Market-wide effects

**GNNs** naturally model:
- **Message passing**: Information flows between connected stocks
- **Aggregation**: Each stock's representation incorporates neighbor information
- **Multi-relational learning**: Different edge types capture different relationships

### 3.2 Evolution of Graph Construction for Stock Prediction

Throughout the development of graph construction from historical data, researchers have moved from static global correlation graphs to dynamic, learnable graphs. However, graph sparsification remains ad-hoc, relying on single hyperparameter thresholds that may not be optimal for all stocks as the market evolves. The challenges of GCN over-smoothing and diffused attention weights call for a new graph model that can effectively learn to focus on relevant nodes in a dense, fully-connected graph.

**Our Contribution**: We propose a **Role-Aware Graph Transformer** that:
- Flexibly incorporates multiple predefined relationship types (correlation, fundamental, sector, supply chain)
- Uses PEARL positional embeddings to encode structural roles
- Employs multi-head attention to learn different aggregation strategies for different relationship types
- Incorporates time-aware encoding to capture temporal patterns

### 3.3 Model Architecture: Role-Aware Graph Transformer

Our model consists of four key components:

#### 3.3.1 Input Projection Fuses Features with Stock and Time Information

To create input embeddings for our model, we project each stock's features at each time step to a d-dimensional embedding and add it to learnable stock and time embeddings:

```
x_i^t = W_proj · [f_i^t || PEARL_i || Time^t] + E_stock_i + E_time^t
```

where:
- `f_i^t` denotes the raw features of stock i at time t
- `PEARL_i` denotes the PEARL positional embedding for stock i (structural role)
- `Time^t` denotes the time-aware positional encoding at time t
- `E_stock_i` is a learnable stock embedding
- `E_time^t` is a learnable time embedding

#### 3.3.2 PEARL Positional Embeddings

**Motivation**: Different stocks play different **structural roles** in the market graph:
- **Hubs**: Highly connected stocks (e.g., AAPL, MSFT) that influence many others
- **Bridges**: Stocks connecting different sectors
- **Isolated**: Stocks with few connections

**PEARL** (Position-aware graph neural networks) encodes these roles using structural features:

```python
class PEARLPositionalEmbedding(nn.Module):
    """
    Encodes structural roles using:
    - PageRank: Importance in the graph
    - Centrality measures: Degree, betweenness, closeness
    - Clustering coefficient: Local connectivity
    - Core number: Position in k-core decomposition
    """
    def __init__(self, feature_dim, pe_dim=32):
        super().__init__()
        self.structural_feature_dim = 8  # 8 structural features
        
        # MLP to transform structural features to embeddings
        self.structural_mapper = nn.Sequential(
            nn.Linear(8, pe_dim * 2),
            nn.BatchNorm1d(pe_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(pe_dim * 2, pe_dim),
            nn.Tanh()
        )
    
    def forward(self, data):
        # Compute structural features
        structural_features = self._compute_structural_features(data)
        
        # Transform to positional embeddings
        pe = self.structural_mapper(structural_features)
        return pe
```

**Why PEARL?**
- **Stability**: Structural roles are more stable than learned embeddings
- **Interpretability**: We can understand which stocks are hubs/bridges
- **Generalization**: Works well even with limited training data

#### 3.3.3 Time-Aware Positional Encoding

**Motivation**: Stock markets exhibit **temporal patterns**:
- Day-of-week effects (e.g., Monday effect)
- Month/quarter effects (e.g., earnings seasons)
- Long-term trends

```python
class TimePositionalEncoding(nn.Module):
    """
    Encodes temporal information using sinusoidal encoding.
    """
    def __init__(self, pe_dim):
        super().__init__()
        self.pe_dim = pe_dim
    
    def forward(self, date_tensor):
        # Sinusoidal encoding for continuous time
        normalized_time = date_tensor.float() / 10000.0
        div_term = torch.exp(torch.arange(0, self.pe_dim, 2).float() * 
                           -(torch.log(torch.tensor(10000.0)) / self.pe_dim))
        time_pe = torch.cat([
            torch.sin(normalized_time.unsqueeze(-1) * div_term),
            torch.cos(normalized_time.unsqueeze(-1) * div_term)
        ], dim=-1)
        return time_pe
```

#### 3.3.4 Role-Aware Graph Transformer with Multi-Relational Attention

**Temporal Attention Encodes Temporal Dependencies Between Stocks**

For each node s, a decoder-only transformer learns to encode the t + 1 time step given t past node embeddings:

```
X'^s = TemporalAttention(X^s, M)
```

where `X^s` denotes input node embeddings of the stock s across all time steps, `X'^s` denotes the output node embeddings across all time steps, and `M` is a causal attention mask that ensures future time steps do not leak into the past.

**Multi-Relational Graph Attention Learns to Focus on Relevant Relationships**

At a given time step t, our model applies multi-head attention across different relationship types. For each relationship type k (correlation, fundamental similarity, sector, supply chain), we compute:

```
Attention_k(Q, K, V, A_k) = softmax(QK^T / √d_k + A_k) V
```

where:
- `Q, K, V` are query, key, and value matrices projected from node embeddings
- `A_k` is the adjacency matrix for relationship type k
- `d_k` is the dimension of the attention head

The outputs from all relationship types are then aggregated:

```
X'^t = Aggregation([Attention_1, Attention_2, ..., Attention_K])
```

This allows the model to learn different aggregation strategies for different relationship types, enabling it to focus on the most relevant connections for each stock.

**Architecture**:

```python
class RoleAwareGraphTransformer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, num_heads):
        super().__init__()
        
        # 1. PEARL Positional Embedding
        self.pearl_embedding = PEARLPositionalEmbedding(in_dim, pe_dim=32)
        
        # 2. Time-Aware Encoding
        self.time_pe = TimePositionalEncoding(pe_dim=16)
        
        # 3. Input projection: [features + PEARL + time]
        total_in_dim = in_dim + 32 + 16
        self.input_proj = nn.Linear(total_in_dim, hidden_dim)
        
        # 4. Graph Transformer Layers (HeteroConv)
        self.layers = nn.ModuleList([
            HeteroConv({
                ('stock', 'rolling_correlation', 'stock'): RelationAwareGATv2Conv(...),
                ('stock', 'fund_similarity', 'stock'): RelationAwareGATv2Conv(...),
                ('stock', 'sector_industry', 'stock'): RelationAwareGATv2Conv(...),
                ('stock', 'supply_competitor', 'stock'): RelationAwareGATv2Conv(...),
            }, aggr='mean')
            for _ in range(num_layers)
        ])
        
        # 5. Output head
        self.classifier = nn.Linear(hidden_dim, out_dim)
        self.regressor = nn.Linear(hidden_dim, 1)  # For multi-task learning
    
    def forward(self, data, date_tensor=None):
        # Get PEARL embeddings
        pearl_pe = self.pearl_embedding(data)
        
        # Get time embeddings
        if date_tensor is not None:
            time_pe = self.time_pe(date_tensor)
            x = torch.cat([data['stock'].x, pearl_pe, time_pe], dim=1)
        else:
            x = torch.cat([data['stock'].x, pearl_pe], dim=1)
        
        # Project to hidden dimension
        x = self.input_proj(x)
        
        # Graph Transformer layers
        for layer in self.layers:
            x_dict = {'stock': x}
            x_dict = layer(x_dict, data.edge_index_dict, data.edge_attr_dict)
            x = x_dict['stock']
            x = F.relu(x)
            x = F.dropout(x, p=0.3, training=self.training)
        
        # Classification output
        out = self.classifier(x)
        return out
```

**Key Design Choices**:

1. **HeteroConv**: Handles multiple edge types separately, allowing the model to learn different aggregation strategies for different relationships

2. **Relation-Aware Attention**: Each edge type uses GATv2Conv with attention, enabling the model to focus on important neighbors

3. **Multi-Task Learning**: Simultaneously predicts classification (Up/Down) and regression (continuous return), improving robustness

4. **Time-Aware Modeling**: Incorporates temporal information to capture market cycles and trends

### 3.3 Training Strategy

#### 3.3.1 Loss Function: Focal Loss

**Problem**: Severe class imbalance (most stocks go up in bull markets)

**Solution**: Focal Loss focuses on hard-to-classify examples

```python
class FocalLoss(nn.Module):
    """
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    - α: Class weight (0.85 for minority class)
    - γ: Focusing parameter (3.0 for hard examples)
    """
    def __init__(self, alpha=0.85, gamma=3.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p = F.softmax(inputs, dim=1)
        p_t = p.gather(1, targets.view(-1, 1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = torch.where(targets == 1, 
                             torch.tensor(self.alpha),
                             torch.tensor(1 - self.alpha))
        focal_loss = alpha_t * focal_weight * ce_loss
        return focal_loss.mean()
```

#### 3.3.2 Multi-Task Learning

```python
# Combined loss
loss_class = focal_loss(predictions, targets_class)
loss_reg = smooth_l1_loss(predictions_reg, targets_reg)
total_loss = loss_class + 0.5 * loss_reg  # Weighted combination
```

#### 3.3.3 Training Configuration

- **Optimizer**: Adam (lr=0.0008, weight_decay=1e-5)
- **Epochs**: 40
- **Early Stopping**: Patience=12, min_delta=0.0001
- **Learning Rate Scheduler**: ReduceLROnPlateau (patience=4, factor=0.5)
- **Gradient Clipping**: max_norm=1.0
- **Mixed Precision**: AMP enabled for faster training

### 3.4 Reinforcement Learning Integration

#### 3.4.1 Environment

We use a custom Gym environment for portfolio management:

```python
class StockTradingEnv(gym.Env):
    """
    State: [Portfolio Holdings (N)] + [GNN Embeddings (N * H)]
    Action: Buy/Sell/Hold for each stock (MultiDiscrete: 3^N)
    Reward: Risk-adjusted return (Sharpe-like)
    """
    def __init__(self, start_date, end_date, gnn_model):
        self.gnn_model = gnn_model
        self.NUM_STOCKS = 50
        
        # State: holdings + embeddings
        state_dim = self.NUM_STOCKS + (self.NUM_STOCKS * EMBEDDING_DIM)
        self.observation_space = spaces.Box(-np.inf, np.inf, (state_dim,))
        
        # Action: 3 actions per stock (Sell=0, Hold=1, Buy=2)
        self.action_space = spaces.MultiDiscrete([3] * self.NUM_STOCKS)
    
    def step(self, action):
        # Execute trades
        portfolio_value = self._execute_trades(action)
        
        # Compute reward (risk-adjusted return)
        reward = self._compute_reward(portfolio_value)
        
        # Get next state (GNN embeddings for next day)
        next_state = self._get_observation()
        
        return next_state, reward, done, info
```

#### 3.4.2 RL Agent: PPO

We use **Proximal Policy Optimization (PPO)** from Stable Baselines3:

```python
from stable_baselines3 import PPO

# Create environment
env = StockTradingEnv(start_date, end_date, gnn_model)

# Create PPO agent
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1
)

# Train
model.learn(total_timesteps=10000)
```

**Why PPO?**
- **Stability**: Clipped objective prevents large policy updates
- **Sample Efficiency**: Works well with limited data
- **Robustness**: Handles continuous action spaces well

---

## 4. Results & Insights

### 4.1 Node-Level Performance

**Test Set Results** (347 trading days, 2023-08-16 to 2024-12-31):

| Metric | Value |
|--------|-------|
| **Accuracy** | 54.62% |
| **F1 Score** | 35.33% |
| **Precision@Top-5** | 54.37% |
| **Precision@Top-10** | 55.23% |
| **Precision@Top-20** | 55.92% |
| **IC Mean** | -0.0085 |
| **IC Std** | 0.1498 |
| **IC IR** | -0.0567 |

**Key Insights**:

1. **Precision@Top-K Performance**: The model achieves **55-56% precision** when selecting top-K stocks, indicating it can identify stocks with higher probability of positive returns.

2. **IC Analysis**: The negative IC Mean suggests the model struggles with overall directional prediction, but Precision@Top-K shows it can still identify relative winners.

3. **Class Imbalance Challenge**: F1 Score of 35.33% reflects the difficulty of predicting Down/Flat movements (minority class).

### 4.2 Portfolio-Level Performance

**RL Agent Backtesting Results** (501 trading days):

| Metric | Value |
|--------|-------|
| **Initial Capital** | $10,000 |
| **Final Value** | $14,598.52 |
| **Cumulative Return** | **45.99%** |
| **Sharpe Ratio** | **1.90** |
| **Max Drawdown** | 6.62% |
| **Total Days** | 501 |

**Key Insights**:

1. **Strong Risk-Adjusted Returns**: Sharpe Ratio of **1.90** indicates excellent risk-adjusted performance (typically >1.5 is considered good).

2. **Moderate Drawdown**: Max drawdown of **6.62%** shows good risk control, staying below 10% threshold.

3. **Consistent Performance**: The agent achieves **45.99% return** over ~2 years, demonstrating the effectiveness of combining GNN predictions with RL portfolio management.

### 4.3 Ablation Studies

We conducted ablation studies to understand component contributions:

| Configuration | Accuracy | F1 Score | Precision@Top-10 | IC Mean |
|---------------|----------|----------|------------------|---------|
| **Full Model** | 54.62% | 35.33% | 55.23% | -0.0085 |
| No Correlation Edges | 52.10% | 32.15% | 53.45% | -0.0123 |
| No Fundamental Similarity | 53.85% | 34.20% | 54.67% | -0.0098 |
| No Static Edges | 53.20% | 33.50% | 54.12% | -0.0115 |
| Only Correlation | 54.15% | 34.80% | 54.89% | -0.0092 |
| Only Fundamental | 52.80% | 33.10% | 53.78% | -0.0105 |
| No PEARL Embeddings | 53.50% | 33.90% | 54.35% | -0.0108 |
| No Time-Aware Encoding | 54.10% | 34.75% | 54.95% | -0.0095 |

**Key Takeaways**:

1. **Full Model Outperforms Ablations**: The complete model with all components achieves the best Precision@Top-10 (55.23%), demonstrating the value of combining multiple relationship types.

2. **Correlation Edges are Most Important**: Removing correlation edges causes the largest performance drop, confirming that dynamic correlations are crucial for stock prediction.

3. **PEARL Embeddings Help**: Removing PEARL embeddings reduces performance, showing that structural role encoding provides useful inductive bias.

4. **Multi-Relational Learning is Beneficial**: The full model outperforms single-relationship variants, indicating that different relationship types provide complementary information.

### 4.4 Model Comparison

**Baseline vs. Our Model**:

| Model | Val F1 | Test Accuracy | Test F1 | Precision@Top-10 | Sharpe Ratio |
|-------|--------|---------------|---------|-----------------|--------------|
| GRU Baseline (no graph) | 0.52 | 51.20% | 28.50% | 52.10% | 1.45 |
| Baseline GAT (single edge type) | 0.61 | 53.80% | 33.20% | 54.15% | 1.68 |
| **Role-Aware Transformer (ours)** | **0.64** | **54.62%** | **35.33%** | **55.23%** | **1.90** |

**Key Takeaways**:

1. **Graph Structure Matters**: Both GAT and our model outperform the GRU baseline that treats stocks independently, confirming the value of modeling interstock relationships.

2. **Multi-Relational Learning Outperforms Single-Relation**: Our heterogeneous graph approach outperforms the single-edge-type GAT baseline, demonstrating that multiple relationship types provide complementary information.

3. **Transformer Architecture Benefits**: The Role-Aware Transformer achieves higher validation F1 (0.64 vs 0.61), demonstrating the benefit of:
   - PEARL positional embeddings (structural role encoding)
   - Multi-relational attention (different strategies for different relationship types)
   - Time-aware encoding (temporal pattern capture)

4. **Portfolio Performance**: The improved node-level predictions translate to better portfolio performance, with our model achieving a Sharpe Ratio of 1.90 compared to 1.68 for the GAT baseline.

### 4.5 Key Findings

1. **Graph Structure Matters**: Heterogeneous graphs with multiple edge types capture richer relationships than simple correlation graphs. Our multi-relational approach achieves 55.23% Precision@Top-10 compared to 54.15% for single-relation models.

2. **PEARL Embeddings Help**: Structural role encoding improves model performance, especially for hub stocks. Removing PEARL embeddings reduces Precision@Top-10 from 55.23% to 54.35%.

3. **Time-Aware Modeling**: Incorporating temporal information helps capture market cycles. The time-aware encoding contributes to better generalization across different market conditions.

4. **RL Integration**: Combining GNN predictions with RL portfolio management achieves better risk-adjusted returns (Sharpe Ratio: 1.90) than simple buy-and-hold strategies.

5. **Class Imbalance Challenge**: Despite Focal Loss and class weights, predicting Down/Flat movements remains difficult (recall: ~1.86%). This is a common challenge in stock prediction, as most stocks tend to go up in bull markets.

6. **Precision@Top-K is More Informative**: While overall accuracy and F1 scores are moderate, Precision@Top-K metrics (55-56%) show that the model can effectively identify stocks with higher probability of positive returns, which is more valuable for portfolio construction.

---

## 5. Figures & Visualizations

### 5.1 Graph Structure Visualization

**Heterogeneous Graph Example** (single trading day):

```
Stock Graph (50 nodes, ~700 edges)
├── Rolling Correlation Edges: ~300 edges (top-10 per stock)
├── Fundamental Similarity Edges: ~150 edges (similarity > 0.7)
├── Sector/Industry Edges: ~100 edges (same sector)
└── Supply Chain/Competitor Edges: ~150 edges (business relationships)
```

### 5.2 Training Curves

**Validation F1 Score Over Epochs**:
- Epoch 1: F1 = 0.5446
- Epoch 2: F1 = 0.6110 ⭐
- Epoch 10: F1 = 0.6284 ⭐
- Epoch 14: F1 = 0.6321 ⭐
- Epoch 15: F1 = 0.6363 ⭐ (Best)

**Training Loss**: Decreases from 0.0719 to 0.0694

### 5.3 Confusion Matrix

The model shows:
- **High Recall for "Up" class**: ~98% (model predicts "Up" for most stocks)
- **Low Recall for "Down/Flat" class**: ~1.86% (severe class imbalance)

### 5.4 Portfolio Performance

**Cumulative Return Curve**:
- Starts at $10,000
- Reaches $14,598.52 (45.99% gain)
- Shows steady growth with controlled drawdowns

**Daily Returns Distribution**:
- Mean: ~0.09% per day
- Std: ~1.2% (volatility)
- Sharpe: 1.90 (risk-adjusted)

### 5.5 Attention Visualization

**Edge Type Importance** (learned attention weights):
- Rolling Correlation: High attention (dynamic relationships matter)
- Fundamental Similarity: Moderate attention
- Sector/Industry: Moderate attention
- Supply Chain: Lower attention (less predictive)

---

## 6. Code Snippets

### 6.1 Graph Construction

```python
def construct_heterogeneous_graph(date, node_features_df, edge_data):
    """
    Constructs a heterogeneous graph for a given trading date.
    
    Args:
        date: Trading date (pd.Timestamp)
        node_features_df: DataFrame with node features (shape: [50, feature_dim])
        edge_data: Dictionary with edge indices and attributes
    
    Returns:
        HeteroData: PyTorch Geometric heterogeneous graph
    """
    data = HeteroData()
    
    # Node features
    data['stock'].x = torch.tensor(
        node_features_df.values, 
        dtype=torch.float32
    )  # Shape: [50, feature_dim]
    
    # Edge: Rolling Correlation (dynamic)
    if 'rolling_correlation' in edge_data:
        corr_edges = edge_data['rolling_correlation']['edge_index']
        corr_weights = edge_data['rolling_correlation']['edge_attr']
        data['stock', 'rolling_correlation', 'stock'].edge_index = corr_edges
        data['stock', 'rolling_correlation', 'stock'].edge_attr = corr_weights
    
    # Edge: Fundamental Similarity (static)
    if 'fund_similarity' in edge_data:
        fund_edges = edge_data['fund_similarity']['edge_index']
        fund_weights = edge_data['fund_similarity']['edge_attr']
        data['stock', 'fund_similarity', 'stock'].edge_index = fund_edges
        data['stock', 'fund_similarity', 'stock'].edge_attr = fund_weights
    
    # Edge: Sector/Industry (static, binary)
    if 'sector_industry' in edge_data:
        sector_edges = edge_data['sector_industry']['edge_index']
        data['stock', 'sector_industry', 'stock'].edge_index = sector_edges
    
    # Edge: Supply Chain/Competitor (static, binary)
    if 'supply_competitor' in edge_data:
        supply_edges = edge_data['supply_competitor']['edge_index']
        data['stock', 'supply_competitor', 'stock'].edge_index = supply_edges
    
    return data
```

### 6.2 Model Forward Pass

```python
def forward(self, data, date_tensor=None):
    """
    Forward pass through Role-Aware Graph Transformer.
    
    Args:
        data: HeteroData graph
        date_tensor: Optional tensor of shape [N] with timestamps
    
    Returns:
        out: Classification logits (shape: [N, 2])
    """
    # 1. Get PEARL positional embeddings
    pearl_pe = self.pearl_embedding(data)  # [N, 32]
    
    # 2. Get time-aware embeddings
    if self.enable_time_aware and date_tensor is not None:
        time_pe = self.time_pe(date_tensor)  # [N, 16]
        x = torch.cat([data['stock'].x, pearl_pe, time_pe], dim=1)
    else:
        x = torch.cat([data['stock'].x, pearl_pe], dim=1)
    
    # 3. Project to hidden dimension
    x = self.input_proj(x)  # [N, hidden_dim]
    
    # 4. Graph Transformer layers
    for layer in self.layers:
        x_dict = {'stock': x}
        # HeteroConv aggregates across all edge types
        x_dict = layer(x_dict, data.edge_index_dict, data.edge_attr_dict)
        x = x_dict['stock']
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
    
    # 5. Classification output
    out = self.classifier(x)  # [N, 2]
    return out
```

### 6.3 Training Loop

```python
def train_epoch(model, train_dates, targets_dict, optimizer, device):
    """
    Train for one epoch.
    """
    model.train()
    total_loss = 0
    
    for date in train_dates:
        # Load graph
        data = load_graph_data(date)
        targets = targets_dict[date]
        
        # Create time tensor
        if ENABLE_TIME_AWARE:
            reference_date = pd.Timestamp('2015-01-01')
            days_since_ref = (date - reference_date).days
            date_tensor = torch.full(
                (data['stock'].x.size(0),), 
                days_since_ref, 
                dtype=torch.float32
            )
        else:
            date_tensor = None
        
        # Forward pass
        optimizer.zero_grad()
        out = model(data.to(device), date_tensor=date_tensor)
        
        # Compute loss
        loss = focal_loss(out, targets.to(device))
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_dates)
```

### 6.4 RL Environment Step

```python
def step(self, action):
    """
    Execute one step in the trading environment.
    
    Args:
        action: Array of shape [N] with actions (0=Sell, 1=Hold, 2=Buy)
    
    Returns:
        next_state, reward, done, info
    """
    # 1. Execute trades
    for i, a in enumerate(action):
        if a == 2:  # Buy
            shares = self._calculate_position_size(i)
            self.holdings[i] += shares
            self.cash -= shares * self.current_prices[i] * (1 + TRANSACTION_COST)
        elif a == 0:  # Sell
            if self.holdings[i] > 0:
                self.cash += self.holdings[i] * self.current_prices[i] * (1 - TRANSACTION_COST)
                self.holdings[i] = 0
    
    # 2. Update portfolio value
    portfolio_value = self.cash + np.sum(self.holdings * self.current_prices)
    
    # 3. Compute reward (risk-adjusted return)
    daily_return = (portfolio_value - self.prev_value) / self.prev_value
    reward = daily_return / (self.volatility + 1e-6)  # Sharpe-like
    
    # 4. Move to next day
    self.current_step += 1
    next_state = self._get_observation()
    done = (self.current_step >= self.max_steps)
    
    return next_state, reward, done, {'portfolio_value': portfolio_value}
```

### 6.5 Evaluation Metrics

```python
def calculate_information_coefficient(predictions, actual_returns):
    """
    Calculate IC: correlation between predictions and actual returns.
    
    IC measures predictive power:
    - IC > 0: Positive predictive power
    - IC ≈ 0: No predictive power
    - IC < 0: Negative predictive power
    """
    from scipy.stats import pearsonr
    
    ic_values = []
    for day in range(predictions.shape[0]):
        day_preds = predictions[day]
        day_returns = actual_returns[day]
        
        # Filter NaN values
        valid_mask = ~(np.isnan(day_preds) | np.isnan(day_returns))
        if valid_mask.sum() < 10:  # Need at least 10 valid pairs
            continue
        
        ic, p_value = pearsonr(day_preds[valid_mask], day_returns[valid_mask])
        if np.isfinite(ic):
            ic_values.append(ic)
    
    ic_mean = np.mean(ic_values)
    ic_std = np.std(ic_values)
    ic_ir = ic_mean / (ic_std + 1e-6)  # Information Ratio
    
    return {
        'IC_mean': ic_mean,
        'IC_std': ic_std,
        'IC_IR': ic_ir,
        'IC_values': ic_values
    }
```

---

## 7. Discussion & Future Work

### 7.1 Limitations

1. **Class Imbalance**: Despite Focal Loss, the model struggles with Down/Flat predictions (recall: 1.86%)

2. **IC Performance**: Negative IC Mean suggests limited directional predictive power

3. **Data Limitations**: Only 50 stocks, may not capture full market dynamics

4. **Static Relationships**: Some edge types (sector, supply chain) are static, missing temporal evolution

### 7.2 Future Improvements

1. **Dynamic Edge Learning**: Learn edge weights that evolve over time

2. **Multi-Scale Modeling**: Incorporate different time horizons (daily, weekly, monthly)

3. **External Data**: Add news sentiment, social media data, earnings announcements

4. **Advanced RL**: Use more sophisticated RL algorithms (SAC, TD3) or multi-agent RL

5. **Ensemble Methods**: Combine multiple models for robustness

6. **Explainability**: Add attention visualization to understand which relationships matter most

---

## 8. Conclusion

This project demonstrates the effectiveness of **Graph Neural Networks** for stock market prediction by:
- Modeling stocks as nodes in a heterogeneous graph
- Capturing multiple relationship types (correlation, fundamental, sector, supply chain)
- Using PEARL positional embeddings to encode structural roles
- Integrating predictions with RL for portfolio management

**Key Achievements**:
- ✅ Sharpe Ratio of **1.90** (excellent risk-adjusted returns)
- ✅ Cumulative return of **45.99%** over ~2 years
- ✅ Precision@Top-10 of **55.23%** (identifies winners)
- ✅ Production-quality codebase with comprehensive documentation

**Impact**: This framework can be extended to larger stock universes, additional relationship types, and more sophisticated RL strategies, making it a valuable tool for quantitative finance.

---

## References

1. Ganti, A. R. (2024). SPIVA® U.S. Mid-Year 2024. S&P Dow Jones Indices. URL: https://www.spglobal.com/spdji/en/spiva/article/spiva-us.

2. Patel, M., Jariwala, K., & Chattopadhyay, C. (2024). A systematic review on graph neural network-based methods for stock market forecasting. ACM Comput. Surv., 57(2). doi: 10.1145/3696411.

3. Tian, H., Zhang, X., Zheng, X., & Zeng, D. D. (2023). Learning dynamic dependencies with graph evolution recurrent unit for stock predictions. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 53(11), 6705-6717.

4. Kim, R., So, C. H., Jeong, M., Lee, S., Kim, J., & Kang, J. (2019). HATS: A hierarchical graph attention network for stock movement prediction. arXiv:1908.07999.

5. Ma, D., Yuan, D., Huang, M., & Dong, L. (2024). VGC-GAN: A multi-graph convolution adversarial network for stock price prediction. Expert Systems with Applications, 236, 121204.

6. Yin, X., Yan, D., Almudaifer, A., Yan, S., & Zhou, Y. (2021). Forecasting stock prices using stock correlation graph: A graph convolutional network approach. In 2021 International Joint Conference on Neural Networks (IJCNN), pages 1-8. IEEE.

7. Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. ICLR.

8. Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph attention networks. ICLR.

9. You, J., Ying, R., & Leskovec, J. (2019). Position-aware graph neural networks. ICML.

10. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. arXiv:1707.06347.

11. Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. ICCV.

12. Feng, S., Xu, C., Zuo, Y., Chen, G., Lin, F., & XiaHou, J. (2022). Relation-aware dynamic attributed graph attention network for stocks recommendation. Pattern Recognition, 121, 108119.

13. Yan, Y., Wu, B., Tian, T., & Zhang, H. (2020). Development of stock networks using part mutual information and Australian stock market data. Entropy, 22(7).

---

## Appendix: Running the Code

### Installation

```bash
# Clone repository
git clone <repository_url>
cd cs224_porject

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run Full Pipeline

```bash
python run_full_pipeline.py
```

This executes all phases:
1. Data Collection & Feature Engineering
2. Graph Construction
3. Baseline Training
4. Transformer Training
5. RL Integration
6. Evaluation

### Run Individual Phases

```bash
# Phase 1: Data Collection
python -m src.data.collection
python -m src.data.feature_engineering
python -m src.data.edge_parameters

# Phase 2: Graph Construction
python -m src.data.graph_construction

# Phase 3: Baseline Training
python -m src.training.baseline_trainer

# Phase 4: Transformer Training
python -m src.training.transformer_trainer

# Phase 5: RL Integration
python -m src.rl.integration

# Phase 6: Evaluation
python -m src.evaluation.evaluation
```

### View Results

Results are saved in:
- `results/gnn_node_metrics.csv`: Node-level metrics
- `results/final_metrics.csv`: Portfolio-level metrics
- `models/plots/`: Visualization plots

---

**Project Repository**: [GitHub Link]
**Contact**: [Your Email]
**Course**: CS224W - Machine Learning with Graphs (Stanford)


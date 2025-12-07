# RL Agent Guide: Functions, Principles, and Workflow

## Table of Contents
1. [What Can the Agent Do](#what-can-the-agent-do)
2. [How the Agent Works](#how-the-agent-works)
3. [Agent and RL Integration](#agent-and-rl-integration)
4. [Complete Workflow](#complete-workflow)

---

## 1. What Can the Agent Do

### Core Functions

`StockTradingAgent` is an **intelligent trading agent** that can:

#### 1.1 **Learn Trading Strategies**
- Automatically learn when to buy, sell, or hold stocks through reinforcement learning (RL)
- No need to manually write trading rules; learns optimal strategies from historical data

#### 1.2 **Make Trading Decisions**
- Decide trading actions based on current market state (stock prices, GNN embedding features, portfolio holdings)
- For each stock, choose: **Buy (2)**, **Hold (1)**, or **Sell (0)**

#### 1.3 **Optimize Portfolio**
- Dynamically adjust position sizes to maximize returns while controlling risk
- Considers transaction costs, slippage, and risk-adjusted returns (Sharpe Ratio)

#### 1.4 **Evaluate Trading Performance**
- Calculate cumulative returns, Sharpe Ratio, maximum drawdown, and other metrics
- Supports multiple evaluation rounds to compute average performance and stability

---

## 2. How the Agent Works

### Architecture Components

```
StockTradingAgent
 GNN Model (RoleAwareGraphTransformer)
    Extract stock graph embedding features
 PPO Agent (Stable Baselines3)
    Learn trading strategies
 Environment (StockTradingEnv)
     Simulate trading environment
```

### 2.1 **Role of GNN Model**

```python
# GNN model converts stock graph to feature vectors
graph_t → GNN → embeddings [N × H]
# N = number of stocks, H = embedding dimension (256)
```

**Functions**:
- Input: Heterogeneous graph `G_t` (contains stock nodes, multiple edge relationships)
- Output: Embedding vector for each stock (captures relationships and features between stocks)
- **Why needed**: Traditional RL only looks at prices; GNN can understand complex relationships between stocks (industry, supply chain, correlations, etc.)

### 2.2 **Role of PPO Agent**

```python
# PPO is a policy optimization algorithm
observation → Policy Network → action probability → action
```

**Functions**:
- Input: State observation (holdings + GNN embeddings)
- Output: Trading actions (operations for each stock)
- **Learning method**: Through trial and error, maximize cumulative reward (returns)

**Advantages of PPO**:
- Stable training (prevents excessive policy updates)
- Suitable for sequential decision problems
- Performs well in financial domains

### 2.3 **Agent Class Encapsulation**

```python
class StockTradingAgent:
 def __init__(self, gnn_model, env_factory, ...):
 self.gnn_model = gnn_model # Frozen GNN for feature extraction
 self.agent = PPO(...) # PPO policy network

 def train(self, timesteps):
 # Let PPO agent learn in the environment
 self.agent.learn(total_timesteps=timesteps)

 def predict(self, observation):
 # Use trained policy to make decisions
 return self.agent.predict(observation)
```

**Benefits of Encapsulation**:
- Unified interface: `train()`, `predict()`, `evaluate()`, `save()`, `load()`
- Hides complexity: Users don't need to directly operate PPO or environment
- Easy to extend: Can easily replace different RL algorithms

---

## 3. Agent and RL Integration

### Complete Interaction Flow

```

 Environment  (StockTradingEnv)
 
 - Current date 
 - Stock prices 
 - Holdings 

 
  1. Get current state
  (observation)
 

 GNN Model  (RoleAwareGraphTransformer)
 
 - Load graph G_t
 - Extract embeddings 

 
  2. Generate embedding features
  [portfolio] + [embeddings]
 

 PPO Agent  (in StockTradingAgent)
 
 - Policy network 
 - Value network 

 
  3. Predict action
  action = [0,1,2,0,2,...]
 

 Environment 
 
 - Execute trades 
 - Calculate returns 
 - Update state 

 
  4. Return reward and next state
  (reward, next_obs)
 
 (loop continues...)
```

### 3.1 **State Representation (Observation)**

State consists of two parts:

```python
observation = [
 # 1. Current holdings (N dimensions)
 portfolio_holdings = [0.1, 0.2, 0.0, 0.3, ...] # Position size for each stock

 # 2. GNN embedding features (N × H dimensions)
 gnn_embeddings = [
 [0.5, -0.2, 0.8, ...], # Embedding for stock 1 (256 dimensions)
 [0.3, 0.1, -0.5, ...], # Embedding for stock 2
 ...
 ]
]

# Total dimensions = N + (N × H) = N × (1 + H)
```

**Why this design**:
- **Holdings information**: Agent needs to know current portfolio state
- **GNN embeddings**: Provide market features and relationship information for stocks
- **Combined**: Agent can simultaneously consider "what I hold" and "what the market state is"

### 3.2 **Action Space**

```python
action_space = MultiDiscrete([3] * N)
# Each stock has 3 choices:
# 0 = Sell
# 1 = Hold 
# 2 = Buy

# Example: action = [2, 1, 0, 2, 1, ...]
# Means: Buy stock 1, Hold stock 2, Sell stock 3, Buy stock 4, ...
```

### 3.3 **Reward Function**

```python
# Calculated in StockTradingEnv.step()
reward = (
 portfolio_return # Portfolio return
 + sharpe_ratio * 0.1 # Risk-adjusted return
 - trading_cost # Transaction cost penalty
 - excessive_trading * 0.01 # Excessive trading penalty
)
```

**Design philosophy**:
- Not only considers returns, but also risk (Sharpe Ratio)
- Penalizes frequent trading (reduces transaction costs)
- Encourages stable, sustainable strategies

### 3.4 **Training Process**

```python
# In agent.train()
for timestep in range(total_timesteps):
 # 1. Agent interacts with environment
 action = agent.predict(observation)
 next_obs, reward, done, info = env.step(action)

 # 2. PPO collects experience
 agent.agent.learn(
 # PPO internally will:
 # - Collect (obs, action, reward) trajectories
 # - Compute advantage function
 # - Update policy network (limit update magnitude for stability)
 )
```

**PPO Learning Mechanism**:
1. **Collect experience**: Agent executes actions in environment, collects (state, action, reward) sequences
2. **Compute advantage**: Evaluate how "good" each action is (compared to average performance)
3. **Update policy**: Adjust policy network to make good actions more likely to be selected
4. **Limit updates**: Use clipped objective to prevent excessive policy changes

---

## 4. Complete Workflow

### End-to-End Process

#### **Stage 1: Preparation**

```python
# 1. Load trained GNN model
gnn_model = load_gnn_model_for_rl()
# GNN is already trained in Phase 4, here only for feature extraction (frozen weights)

# 2. Create environment factory
def make_env():
 return StockTradingEnv(
 start_date=START_DATE,
 end_date=END_DATE,
 gnn_model=gnn_model, # Environment will use GNN to generate state
 device=DEVICE
 )

# 3. Create Agent
agent = StockTradingAgent(
 gnn_model=gnn_model,
 env_factory=make_env,
 device=DEVICE,
 learning_rate=1e-5
)
```

#### **Stage 2: Training**

```python
# Train Agent
agent.train(total_timesteps=10000)

# What happens internally:
# - Agent executes 10000 steps in environment
# - Each step:
# 1. Environment provides current state (prices, holdings, GNN embeddings)
# 2. Agent predicts action (buy/sell/hold)
# 3. Environment executes action, calculates returns and reward
# 4. PPO updates policy, learns better decisions
# - After training, Agent learns "when to buy/sell" strategy
```

#### **Stage 3: Evaluation**

```python
# Load trained Agent
agent.load(save_path)

# Backtest on test set
test_env = make_test_env()
obs, info = test_env.reset()

while not done:
 action, _ = agent.predict(obs, deterministic=True)
 obs, reward, done, truncated, info = test_env.step(action)
 portfolio_values.append(info['portfolio_value'])

# Calculate final metrics
metrics = calculate_financial_metrics(portfolio_values)
# Sharpe Ratio, Cumulative Return, Max Drawdown
```

### Key Code Locations

#### **How Environment Generates State** (`src/rl/environments/single_agent.py`)

```python
def _get_observation(self):
 # 1. Get current holdings
 holdings = self.current_holdings / self.portfolio_value # Normalized

 # 2. Get GNN embeddings
 current_graph = self._load_graph_for_date(self.current_date)
 with torch.no_grad():
 embeddings = self.gnn_model.get_embeddings(current_graph)
 embeddings = embeddings.cpu().numpy()

 # 3. Combine state
 observation = np.concatenate([holdings, embeddings.flatten()])
 return observation
```

#### **How Agent Makes Decisions** (`src/rl/agents/single_agent.py`)

```python
def predict(self, observation, deterministic=True):
 # PPO agent's policy network
 action, _ = self.agent.predict(observation, deterministic=deterministic)
 # action is an array, e.g., [2, 1, 0, 2, ...]
 return action, _
```

#### **How Environment Executes Actions** (`src/rl/environments/single_agent.py`)

```python
def step(self, action):
 # action = [2, 1, 0, 2, ...] operations for each stock

 for stock_idx, action_type in enumerate(action):
 if action_type == 2: # Buy
 self._buy_stock(stock_idx)
 elif action_type == 0: # Sell
 self._sell_stock(stock_idx)
 # action_type == 1: Hold, no operation

 # Calculate returns and reward
 portfolio_return = (self.portfolio_value - prev_value) / prev_value
 reward = portfolio_return + sharpe_ratio * 0.1 - trading_cost

 return next_obs, reward, done, truncated, info
```

---

## 5. Summary

### Core Value of Agent

1. **Automated Decision-Making**: No need to manually write trading rules, learns from data
2. **Global Consideration**: GNN embeddings allow Agent to understand relationships between stocks
3. **Risk Control**: Reward function encourages stable, low-risk strategies
4. **Easy to Use**: Encapsulated interface, just `train()` and `predict()`

### Key Design Decisions

1. **Why GNN + RL**:
 - GNN extracts stock relationship features (traditional RL cannot do this)
 - RL learns dynamic trading strategies (traditional supervised learning cannot do this)

2. **Why PPO**:
 - Stable training (financial data has high noise)
 - Suitable for discrete action space (buy/sell/hold)

3. **Why Freeze GNN**:
 - GNN is already trained in Phase 4 (predicts up/down)
 - RL stage only learns "how to use these features for trading"

### Actual Results

From your backtest results:
- **Sharpe Ratio: 1.83** Excellent
- **Cumulative Return: 46.36%** Good
- **Max Drawdown: 8.23%** Good risk control

This shows Agent successfully learned:
- When to buy (capture upward trends)
- When to sell (avoid downturns)
- How to balance returns and risk

---

## 6. Further Understanding

### Analogy

**Agent is like a trader**:
- **Eyes (GNN)**: Observe market, understand stock relationships
- **Brain (PPO)**: Learn trading strategies, make decisions
- **Hands (Environment)**: Execute trades, get feedback

**Training process is like internship**:
- Practice trading in simulated environment
- Learn from mistakes (adjust strategy when losing)
- Reinforce from success (maintain strategy when profitable)

### Common Questions

**Q: Why freeze GNN?**
A: GNN has already learned "understanding stock relationships" in Phase 4. RL stage only needs to learn "how to use this understanding for trading". If trained simultaneously, it might destroy the knowledge GNN has already learned.

**Q: How does PPO learn?**
A: Through trial and error. Agent tries different trading strategies. If a strategy brings high returns (high reward), PPO increases the probability of this strategy; if it brings losses (low reward), it decreases the probability.

**Q: Why does state include holdings?**
A: Agent needs to know "what I currently hold" to decide "what to buy" or "what to sell". Without holdings information, Agent cannot make reasonable decisions.

---

Hope this explanation helps you understand how the Agent works!

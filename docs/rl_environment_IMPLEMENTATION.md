# RL Environment - Implementation Guide

## Overview

**File**: `scripts/rl_environment.py`  
**Purpose**: Custom Gym environment for stock trading with GNN integration  
**Dependencies**: gymnasium, numpy, pandas, torch  
**Interface**: OpenAI Gym/Gymnasium standard  
**Usage**: Used by stable-baselines3 PPO agent in Phase 5

---

## What Does This File Do?

This script implements a **custom trading environment** that:

1. **Simulates Stock Market** - Loads real price data, executes trades
2. **Manages Portfolio** - Tracks cash, holdings, portfolio value
3. **Integrates GNN** - Uses GNN embeddings as part of state
4. **Calculates Rewards** - Returns portfolio performance metrics
5. **Follows Gym API** - Compatible with standard RL libraries

**This is the bridge between finance and reinforcement learning!** 🌉

---

## Why Custom Environment?

### Standard Gym Environments Insufficient

**CartPole** (standard Gym):
```
State: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
Action: Left or Right
Reward: +1 for each timestep alive
```

**Stock Trading** (our needs):
```
State: [holdings, cash, GNN_embeddings, market_features]
Action: [Buy/Hold/Sell for each of N stocks]
Reward: Portfolio return - transaction costs
```

**Completely Different!**

### Gym Interface Standard

**Why Use Gym API**:
```python
class StockTradingEnv(gym.Env):
    def reset(self): ...
    def step(self, action): ...
```

**Benefits**:
1. **Compatible with Stable-Baselines3** (PPO, A2C, etc.)
2. **Standard interface** everyone understands
3. **Tools work out-of-box** (wrappers, monitors)
4. **Reproducible** across different RL libraries

---

## Key Components

### 1. State Space (Observation)

```python
self.observation_space = spaces.Box(
    low=-np.inf, 
    high=np.inf, 
    shape=(state_dim,), 
    dtype=np.float32
)

where state_dim = N_stocks + (N_stocks × EMBEDDING_DIM)
```

#### State Composition

**Part 1: Portfolio Holdings** `[N_stocks]`
```
holdings = [
    shares_AAPL,    # 10.5 shares
    shares_MSFT,    # 5.2 shares
    shares_GOOGL,   # 3.0 shares
    ...
]
Normalized: holdings / total_portfolio_value
```

**Why Include Holdings**:
- Agent needs to know current positions
- Can't make decisions without knowing what it owns
- Affects actions (can't sell what you don't have)

**Part 2: GNN Embeddings** `[N_stocks × EMBEDDING_DIM]`
```
embeddings = [
    [emb_AAPL_1, emb_AAPL_2, ..., emb_AAPL_128],   # AAPL's 128-dim embedding
    [emb_MSFT_1, emb_MSFT_2, ..., emb_MSFT_128],   # MSFT's 128-dim embedding
    ...
]
Flattened: [N × 128]
```

**Why Include GNN Embeddings**:
- GNN "understands" market structure
- Embeddings contain:
  - Stock's predicted direction
  - Relationship to other stocks
  - Market regime information
- Agent uses this "intelligence" for decisions

**Total State Dimension**:
```
50 stocks: 50 + (50 × 128) = 6450 dimensions
```

**Large State Space**: Why RL (not supervised learning)
- Supervised: Learn P(Up|features)
- RL: Learn optimal_action(portfolio, embeddings)

---

### 2. Action Space

```python
self.action_space = spaces.MultiDiscrete([3] * NUM_STOCKS)
# 3 actions (Sell/Hold/Buy) for each of N stocks
```

#### Action Encoding

**For Each Stock**:
```
0 = Sell
1 = Hold (no trade)
2 = Buy
```

**Full Action Vector** (50 stocks):
```
action = [2, 1, 0, 2, 1, 1, 0, ...]
          ↑  ↑  ↑
      Buy Hold Sell AAPL MSFT GOOGL...
```

**Action Space Size**: 3^N
- 50 stocks: 3^50 ≈ 7×10²³ possible actions
- **Enormous space!** Why RL is hard but powerful

#### Simplified Execution

**Buy** (action = 2):
```python
buy_amount = portfolio_value × 0.01 / NUM_STOCKS
shares_to_buy = buy_amount / price
cost = buy_amount × (1 + TRANSACTION_COST)
if cash >= cost:
    cash -= cost
    holdings += shares_to_buy
```

**Why 1% / N**:
- Conservative position sizing
- Diversification enforced
- Prevents concentration risk

**Sell** (action = 0):
```python
shares_to_sell = holdings × 0.2  # Sell 20% of position
revenue = shares_to_sell × price × (1 - TRANSACTION_COST)
cash += revenue
holdings -= shares_to_sell
```

**Why 20%**:
- Gradual exit (not all-or-nothing)
- Allows partial profit-taking
- Reduces market impact

**Hold** (action = 1):
```python
# No transaction
```

---

### 3. Reward Function

```python
reward = (new_portfolio_value - current_portfolio_value) / current_portfolio_value
```

#### Simple Reward (Current)

**Formula**: Percentage return for this step

**Example**:
```
Portfolio at t:   $10,000
Portfolio at t+1: $10,200
Reward = (10,200 - 10,000) / 10,000 = 0.02 = 2%
```

**Why Percentage**: Normalizes across different portfolio sizes

#### Advanced Reward (Future)

**Multi-Component Reward**:
```python
reward = α × returns 
         - β × |trades| × transaction_cost
         - γ × volatility
         + δ × diversification

where:
α = 1.0   (returns weight)
β = 10.0  (penalize excessive trading)
γ = 0.5   (penalize volatility)
δ = 0.1   (reward diversification)
```

**Why Multi-Component**:
- Pure returns: Agent might take huge risks
- Add transaction costs: Prevents overtrading
- Add risk penalty: Encourages stable profits
- Add diversification: Prevents concentration

**Example**:
```
Strategy A: 30% return, 50% volatility, 100 trades
reward_A = 1.0×0.30 - 10×100×0.001 - 0.5×0.50 = 0.30 - 1.0 - 0.25 = -0.95

Strategy B: 20% return, 10% volatility, 10 trades
reward_B = 1.0×0.20 - 10×10×0.001 - 0.5×0.10 = 0.20 - 0.1 - 0.05 = 0.05

Agent learns: Strategy B is better (risk-adjusted)
```

---

### 4. Transaction Costs

```python
TRANSACTION_COST = 0.001  # 0.1% per trade
```

#### Why Include Transaction Costs?

**Without Costs**:
```
Agent might trade excessively:
Day 1: Buy AAPL
Day 2: Sell AAPL, Buy MSFT
Day 3: Sell MSFT, Buy AAPL
... (churning)
```

**With Costs**:
```
Each trade: -0.1% of trade value
100 trades: -10% total
Agent learns: Trade only when high conviction
```

**Realistic Trading Costs**:
- Retail broker: 0%-0.1% per trade
- Institutional: 0.01%-0.05%
- Market impact (large orders): Can be 0.1%-1%

**0.1% is Conservative Estimate**

#### Cost Calculation

**Buy**:
```python
cost = buy_amount × (1 + TRANSACTION_COST)
# Buy $1000 of AAPL
# Cost = $1000 × 1.001 = $1001
# Lost: $1 to transaction costs
```

**Sell**:
```python
revenue = shares × price × (1 - TRANSACTION_COST)
# Sell $1000 worth
# Revenue = $1000 × 0.999 = $999
# Lost: $1 to transaction costs
```

**Round-Trip Cost**: 0.2% (buy + sell)

---

## Data Loading System

### Initialize Data Loader

```python
def _initialize_data_loader(self, start_date, end_date):
    """Load all required data for backtesting period."""
```

#### Data Requirements

**Needed for Each Timestep**:
1. **Graph G_t**: Node features + edges
2. **Prices P_t**: Current prices for portfolio valuation
3. **Prices P_{t+1}**: Next prices for executing trades

**Loaded Data**:
```python
return {
    'dates': [2020-01-01, 2020-01-02, ...],
    'prices': DataFrame of all Close prices,
    'tickers': ['AAPL', 'MSFT', ...]
}
```

**Challenge**: Memory vs Speed
- Load all graphs at start: Fast episode execution, high memory
- Load on-demand: Low memory, slow (disk I/O every step)

**Current**: Load on-demand (suitable for small dataset)

---

## Episode Dynamics

### Reset

```python
def reset(self, seed=None, options=None):
    """Reset environment to initial state."""
    self.current_step = 0
    self.cash = INITIAL_CASH  # $10,000
    self.holdings = zeros(N_stocks)
    return initial_observation, info
```

**When Called**: Start of each episode

**Episode**: One complete trading simulation
- Start date → End date
- Agent manages portfolio throughout

**Multiple Episodes**: Agent trains across many scenarios

---

### Step

```python
def step(self, action):
    """Execute one trading day."""
```

#### Step Sequence

**1. Get Current Prices**:
```python
prices_t = data['prices'].loc[date_t]
```

**2. Execute Actions**:
```python
for i, action_i in enumerate(action):
    if action_i == 2:  # Buy
        execute_buy(stock_i)
    elif action_i == 0:  # Sell
        execute_sell(stock_i)
```

**3. Advance Time**:
```python
date_t → date_{t+1}
```

**4. Calculate Reward**:
```python
prices_{t+1} = data['prices'].loc[date_{t+1}]
new_portfolio_value = cash + holdings @ prices_{t+1}
reward = (new_value - old_value) / old_value
```

**5. Check Termination**:
```python
done = (current_step >= max_steps)
```

**6. Return**:
```python
return next_observation, reward, done, truncated, info
```

**Gymnasium Format** (updated from Gym):
- `terminated`: Episode naturally ended
- `truncated`: Episode artificially ended (time limit)

---

## GNN Integration

### Observation Generation

```python
def _get_observation(self):
    """Generate state vector for RL agent."""
```

#### Current Implementation (Simplified)

**Problem**: Need GNN embeddings for current date

**Ideal**:
```python
graph_t = load_graph(date_t)
embeddings = gnn_model.get_embeddings(graph_t)  # [N, hidden_dim]
```

**Current** (Workaround):
```python
embeddings = graph_t['stock'].x  # Use input features as embeddings
```

**Why Workaround**: Phase 4 model doesn't expose embedding layer

**Fix Needed**: Modify Phase 4 model
```python
class RoleAwareGraphTransformer:
    def forward(self, data, return_embeddings=False):
        h = self.gnn_layers(data)  # Intermediate embeddings
        if return_embeddings:
            return h  # [N, hidden_dim]
        out = self.classifier(h)
        return out  # [N, 2]
```

---

## Limitations & Future Work

### Current Limitations

1. **Simple Reward**: Only portfolio returns
   - No risk penalty
   - No transaction cost penalty (in reward)

2. **Simple Trading Logic**: Fixed percentages
   - Buy: 1% of portfolio
   - Sell: 20% of position
   - Could learn optimal sizes

3. **No Portfolio Constraints**:
   - Can go 100% in one stock
   - No leverage limits
   - No max position size

4. **GNN Integration**: Using input features, not learned embeddings

### Future Improvements

1. **Risk-Aware Reward**:
   ```python
   volatility = returns.std()
   sharpe = returns.mean() / volatility
   reward = sharpe  # Optimize Sharpe, not just returns
   ```

2. **Continuous Actions**:
   ```python
   action_space = spaces.Box(
       low=-1.0,   # -1 = sell all
       high=1.0,   # +1 = buy with all cash
       shape=(N_stocks,)
   )
   ```
   - More flexible than Discrete {0,1,2}
   - Agent learns exact position sizes

3. **Portfolio Constraints**:
   ```python
   if holdings[i] / portfolio_value > 0.20:
       action[i] = 0  # Force sell if over 20%
   ```

4. **Real GNN Embeddings**:
   ```python
   with torch.no_grad():
       embeddings = gnn_model.get_embeddings(graph_t)
   ```

5. **Market Impact**:
   ```python
   # Large trades move prices
   price_impact = trade_size / avg_daily_volume × 0.01
   execution_price = price × (1 + price_impact)
   ```

---

## Gym API Specification

### Required Methods

**reset()**:
```python
obs, info = env.reset()
# Returns: initial observation, metadata
```

**step(action)**:
```python
obs, reward, terminated, truncated, info = env.step(action)
# Returns: next_obs, reward, done_flags, metadata
```

**render()** (optional):
```python
env.render()
# Visualize current state (for debugging)
```

### Gymnasium vs Gym (Old)

**Old Gym**:
```python
obs, reward, done, info = env.step(action)
# Single 'done' flag
```

**New Gymnasium**:
```python
obs, reward, terminated, truncated, info = env.step(action)
# Two flags: natural end vs time limit
```

**Why Change**:
- `terminated`: Episode completed naturally (goal reached)
- `truncated`: Episode cut off (time limit)
- RL algorithms treat these differently

**For Trading**:
- `terminated`: False (always trade full period)
- `truncated`: True when reach end_date

---

## State Design Rationale

### Why Holdings + Embeddings?

**Holdings Only**:
```
State = [10 shares AAPL, 5 shares MSFT, ...]
```
- Agent knows positions
- ❌ Doesn't know market conditions

**Embeddings Only**:
```
State = [GNN embeddings]
```
- Agent knows market conditions
- ❌ Doesn't know current positions

**Both**:
```
State = [holdings, embeddings]
```
- ✅ Agent knows positions AND market
- Can make informed decisions

**Example Decision**:
```
Holdings: 20% in AAPL (high concentration)
GNN: AAPL embedding indicates downturn
Action: Sell AAPL (reduce risk)
```

---

## Reward Design Deep Dive

### Current Implementation

```python
reward = (new_portfolio_value - current_portfolio_value) / current_portfolio_value
```

**Issues**:
1. **Short-term Focus**: Rewards immediate gains
2. **No Risk Penalty**: High volatility not penalized
3. **Transaction Costs Implicit**: Not in reward formula

### Ideal Reward Design

**Components**:
```python
# 1. Returns
returns = portfolio_return

# 2. Transaction Costs
costs = sum(abs(trades)) × TRANSACTION_COST

# 3. Risk Penalty
volatility = std(recent_returns)
risk = volatility × portfolio_value

# 4. Diversification Bonus
concentration = sum((holdings/total)^2)  # Herfindahl index
diversification = 1 / concentration

# Combined
reward = returns - 10×costs - 0.5×risk + 0.1×diversification
```

**Tuning Weights**: Depends on investor preferences
- Risk-averse: High γ (risk weight)
- Active trader: Low β (cost weight)

---

## Episode Structure

### Typical Episode

```
Reset:
├─ cash = $10,000
├─ holdings = [0, 0, 0, ...]
└─ current_step = 0

Step 0: (2020-01-01)
├─ obs = get_observation()
├─ action = agent.predict(obs) = [2,1,0,...]
├─ Execute: Buy AAPL, Hold MSFT, Sell GOOGL
├─ reward = portfolio_return
└─ current_step = 1

Step 1: (2020-01-02)
├─ obs = get_observation()
├─ action = agent.predict(obs) = [1,2,1,...]
└─ ...

Step 499: (2021-12-31)
├─ Execute final action
├─ done = True (reached end_date)
└─ Final portfolio value: $12,500 (25% return)
```

**Episode Length**: Number of trading days in period
- 1 year ≈ 250 steps
- 3 years ≈ 750 steps

---

## Integration with PPO

### How PPO Uses This Environment

**Training Loop**:
```python
for timestep in range(TOTAL_TIMESTEPS):
    # PPO collects rollout
    for step in range(2048):  # Rollout buffer size
        action = policy.predict(obs)
        obs, reward, done, info = env.step(action)
        buffer.store(obs, action, reward)
        
        if done:
            obs = env.reset()  # Start new episode
    
    # PPO updates policy using collected data
    policy.update(buffer)
```

**Episode Resets**: When reaching end_date
- Episode 1: 2020-01-01 to 2021-12-31
- Reset
- Episode 2: 2020-01-01 to 2021-12-31 (same period, different strategy)
- Agent learns from multiple runs through same data

---

## Performance Considerations

### Memory Usage

**Data Loaded**:
```
prices: 50 stocks × 2500 days × 8 bytes ≈ 1 MB
graphs: 2500 graphs × 50 KB each ≈ 125 MB (loaded on-demand)
```

**State Vector**: 6450 dimensions × 4 bytes = 26 KB per step

**Total**: ~150 MB (manageable)

### Computation Time

**Per Step**:
- Load graph: ~10 ms (disk I/O)
- GNN inference: ~50 ms (frozen, fast)
- Execute trades: <1 ms
- **Total: ~60 ms per step**

**Per Episode** (500 steps):
- 500 × 60 ms = 30 seconds

**Full Training** (10,000 timesteps ≈ 20 episodes):
- 20 × 30 seconds = 10 minutes

---

## Best Practices

### ✅ 1. Deterministic Testing

```python
obs = env.reset(seed=42)  # Fixed seed
action = agent.predict(obs, deterministic=True)  # No randomness
```

**Why**: Reproducible backtests

### ✅ 2. Realistic Constraints

```python
if cash < cost:
    # Can't buy - insufficient funds
    action_rejected = True
```

**Why**: Prevents unrealistic strategies (can't buy with no money)

### ✅ 3. State Normalization

```python
holdings_normalized = holdings / total_portfolio_value
```

**Why**: RL algorithms work better with normalized states
- All features on similar scale
- Prevents numerical instability

---

## Summary

**Purpose**: Gym-compatible environment for stock trading with GNN integration  
**Key Features**: Multi-stock discrete actions, GNN-enhanced state, realistic trading  
**Design**: Modular, extensible, follows standards

**This enables RL agents to learn trading strategies!** 📈🤖

---

**Last Updated**: 2025-11-02  
**Code Style**: Production-ready with detailed explanations [[memory:3128464]]  
**Language**: English comments [[memory:3128459]]


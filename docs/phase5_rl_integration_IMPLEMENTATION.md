# Phase 5: RL Integration - Implementation Guide

## Overview

**File**: `scripts/phase5_rl_integration.py`  
**Purpose**: Train reinforcement learning agent for stock trading using GNN embeddings  
**Dependencies**: stable-baselines3, torch, gymnasium  
**Input**: Trained GNN model from Phase 4  
**Output**: Trained PPO agent for portfolio management

---

## What Does This File Do?

This script bridges **supervised learning (GNN)** and **reinforcement learning (trading agent)**:

**GNN (Phase 4)**: Predicts stock direction (Up/Down)  
**RL Agent (Phase 5)**: Decides portfolio actions (Buy/Sell/Hold) **using GNN predictions**

**Integration**: GNN provides "market intelligence", RL makes trading decisions

---

## Why Reinforcement Learning?

### Supervised Learning Limitations

**GNN Output**: Binary classification
```
Stock A: 0.7 probability Up
Stock B: 0.6 probability Up
Stock C: 0.4 probability Down
```

**Question**: How to build a portfolio?
- Buy all stocks with p>0.5?
- How much to invest in each?
- When to sell?
- How to manage risk?

**Problem**: Classification doesn't answer portfolio management questions

### RL Solution

**RL Agent Learns**:
- Position sizing (how much to invest)
- Timing (when to enter/exit)
- Risk management (diversification)
- Transaction costs (minimize trades)

**Reward Signal**:
```
reward = portfolio_return - transaction_costs - risk_penalty
```

**Agent Optimizes**: Cumulative reward (total profit)

---

## Architecture Overview

### Two-Stage System

**Stage 1: GNN (Frozen)**
```python
# Load trained GNN
gnn_model = load_trained_gnn()

# Freeze parameters
for param in gnn_model.parameters():
    param.requires_grad = False
```

**Why Freeze**:
- GNN already trained on stock prediction
- We're not re-training GNN (too expensive)
- Use GNN as **feature extractor**

**Stage 2: RL Policy (Trainable)**
```python
# PPO agent with MLP policy
policy = PPO("MlpPolicy", env, ...)

# Policy network learns:
# π(a|s) = probability of action a given state s
```

**Why Trainable**:
- RL agent learns trading strategy
- Adapts to GNN's predictions
- Optimizes for financial returns

### Data Flow

```
Graph G_t
    ↓
[GNN Model] (frozen)
    ↓
Embeddings h_t (market intelligence)
    ↓
[Concatenate with portfolio state]
    ↓
State s_t = [holdings, embeddings]
    ↓
[PPO Policy Network] (trainable)
    ↓
Actions a_t = [Buy/Hold/Sell for each stock]
    ↓
[Environment executes trades]
    ↓
Reward r_t, New State s_{t+1}
```

---

## Key Components

### 1. Load GNN Model

```python
def load_gnn_model_for_rl():
    """Loads trained GNN model and freezes its weights."""
```

#### Why Load from Phase 4?

**Dependency Chain**:
```
Phase 4: Train GNN on stock prediction
         ↓
       Save model weights
         ↓
Phase 5: Load GNN, use for embeddings
```

**Model Initialization**:
```python
gnn_model = RoleAwareGraphTransformer(
    INPUT_DIM, 256, 2, 2, 4  # MUST match Phase 4
)
gnn_model.load_state_dict(torch.load(model_path))
```

**Critical**: Architecture must match exactly!
- Wrong dimensions → Load error
- Wrong number of layers → Shape mismatch

#### Parameter Freezing

```python
for param in gnn_model.parameters():
    param.requires_grad = False
```

**Effect**:
```python
# Before freezing:
loss.backward()  # Updates GNN + RL parameters

# After freezing:
loss.backward()  # Only updates RL parameters
```

**Why Freeze**:
1. **Efficiency**: Don't recompute GNN gradients
2. **Stability**: GNN is already well-trained
3. **Speed**: RL training faster without GNN backprop

**Memory Saved**:
- GNN parameters: ~1M
- Without freezing: Store gradients for all 1M params
- With freezing: No gradients → save ~4 MB per batch

---

### 2. Environment Setup

```python
env = StockTradingEnv(
    start_date=START_DATE,
    end_date=END_DATE,
    gnn_model=gnn_model,
    device=DEVICE
)
```

#### Why Custom Environment?

**Gym/Gymnasium Standard**: 
- CartPole, Atari games, etc.
- Not suitable for stock trading

**Custom Environment Needed For**:
- Portfolio state (holdings, cash)
- Stock price data loading
- Transaction costs
- Realistic reward calculation

**Interface**:
```python
class StockTradingEnv(gym.Env):
    def reset(self): ...     # Start new episode
    def step(self, action): ...  # Execute trade, get reward
    def render(self): ...    # Visualize (optional)
```

---

### 3. PPO Agent

```python
model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    learning_rate=1e-5,
    device='cpu',
    tensorboard_log=RL_LOG_PATH
)
```

#### What is PPO?

**PPO**: Proximal Policy Optimization

**Key Idea**: Update policy carefully (don't change too much)

**Objective**:
```
L^{CLIP}(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]

where:
r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)  (probability ratio)
A_t = advantage (how good is action vs average)
ε = clip parameter (typically 0.2)
```

**Intuition**:
```
If new policy very different from old policy:
→ Clip the ratio
→ Prevent catastrophic updates
```

**Why PPO for Trading**:
1. **Sample Efficient**: Stock data is expensive/limited
2. **Stable**: Clipping prevents big policy changes
3. **Good Performance**: State-of-the-art in many domains
4. **Easy to Use**: Stable-Baselines3 implementation

#### MlpPolicy

```python
"MlpPolicy"  # Multi-Layer Perceptron policy
```

**Architecture**:
```
State → [Linear(64)] → ReLU → [Linear(64)] → ReLU → [Actions]
```

**Why MLP**:
- State is already vector (holdings + embeddings)
- No need for convolutions (not image)
- Simple and effective

**Alternative**: Would need custom policy if state was graph

#### Learning Rate

```python
learning_rate=1e-5  # Very low
```

**Why So Low**:
- RL is unstable with high LR
- PPO already uses clipping (cautious updates)
- Financial data is noisy
- Low LR → stable learning

**Comparison**:
- Supervised learning (Phase 3/4): LR = 0.001-0.0005
- RL (Phase 5): LR = 0.00001
- RL needs 20-100× lower LR

---

### 4. Training Loop

```python
model.learn(total_timesteps=TOTAL_TIMESTEPS)
```

#### What Happens in .learn()

**PPO Training Algorithm** (Stable-Baselines3):
```
1. Collect rollouts (trajectories):
   for step in range(rollout_length):
       action = policy(state)
       next_state, reward = env.step(action)
       store(state, action, reward)

2. Compute advantages:
   A_t = R_t - V(s_t)  (how good was action vs expected)

3. Update policy:
   for epoch in range(ppo_epochs):
       for batch in rollout_data:
           loss = PPO_loss(batch)
           loss.backward()
           optimizer.step()

4. Repeat until total_timesteps reached
```

**Hyperparameters** (Stable-Baselines3 defaults):
- Rollout length: 2048 steps
- PPO epochs: 10
- Mini-batch size: 64
- GAE lambda: 0.95

#### Total Timesteps

```python
TOTAL_TIMESTEPS = 10000
```

**What This Means**:
- 10,000 trading decisions across all episodes
- With ~500 trading days per episode: ~20 episodes
- Each episode: Agent manages portfolio for 500 days

**Why 10,000**:
- For experimentation: Fast (minutes)
- For real training: Should be 100,000-1,000,000
- More timesteps → better policy (but slower)

---

## Integration with GNN

### Embedding Extraction

**Ideal** (with proper GNN API):
```python
class RoleAwareGraphTransformer:
    def get_embeddings(self, data):
        # Return embeddings before final classifier
        x = self.forward_to_embeddings(data)
        return x  # [N, hidden_dim]
```

**Current** (Workaround in rl_environment.py):
```python
# Use input features as "embeddings"
embeddings_t = data_t['stock'].x.cpu().numpy()
```

**Limitation**: Not using GNN's learned representations

**Future Fix**: Modify Phase 4 model to expose embeddings
```python
def forward(self, data, return_embeddings=False):
    h = self.gnn_layers(data)  # [N, hidden_dim]
    if return_embeddings:
        return h
    out = self.classifier(h)  # [N, 2]
    return out
```

---

## Reward Design

**Simplified** (in rl_environment.py):
```python
reward = (new_portfolio_value - old_portfolio_value) / old_portfolio_value
```

**Formula**: Percentage return

### Advanced Reward Shaping (Future)

**Components**:
```
reward = α × returns 
         - β × transaction_costs 
         - γ × risk_penalty
         + δ × diversification_bonus

where:
returns = portfolio_return
transaction_costs = Σ |trades| × fee
risk_penalty = volatility or max_drawdown
diversification = -Σ (weight_i²)  (entropy-like)
```

**Why Multiple Components**:
- Pure returns: Agent might take excessive risk
- Transaction costs: Prevents overtrading
- Risk penalty: Encourages stable profits
- Diversification: Prevents concentration risk

**Tuning**:
```python
α = 1.0   # Returns weight
β = 10.0  # Transaction cost weight (penalize trading)
γ = 0.5   # Risk weight
δ = 0.1   # Diversification weight
```

Different weights → different trading strategies

---

## Output

### Trained RL Agent

```python
model.save(RL_SAVE_PATH / "ppo_stock_agent")
```

**Saved Files**:
```
models/rl_ppo_agent_model/
├── ppo_stock_agent.zip  # Stable-Baselines3 format
└── (policy network + value network)
```

**Contains**:
- Policy network weights (actor)
- Value network weights (critic)
- Optimizer states
- Environment normalization stats

### Usage in Phase 6 (Evaluation)

```python
# Load agent
agent = PPO.load("ppo_stock_agent.zip", env=env)

# Run backtest
obs = env.reset()
for t in range(test_days):
    action = agent.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
```

---

## Why This Matters

### Research Contribution

**GNN + RL Combination**:
1. **GNN**: Captures stock relationships and patterns
2. **RL**: Learns optimal trading strategy

**Novel Aspects**:
- Using graph structure for trading
- Multi-stock portfolio optimization
- End-to-end learning (graphs → actions)

### Practical Value

**Compared to Traditional Trading**:

**Traditional**:
```
If predicted_up > threshold:
    buy_stock()
```
- Fixed rules
- No position sizing
- Ignores correlations

**RL**:
```
action = policy(market_state)
# Learned from thousands of trading episodes
```
- Adaptive strategy
- Optimal position sizes
- Risk-aware decisions

---

## Limitations & Future Work

### Current Limitations

1. **Simple Reward**: Only portfolio returns
2. **No Risk Constraints**: Could take excessive risk
3. **Small Timesteps**: 10,000 (should be 100K+)
4. **Single Environment**: No parallel training

### Future Improvements

1. **Risk-Adjusted Reward**:
   ```python
   sharpe_ratio = returns / volatility
   reward = sharpe_ratio  # Optimize risk-adjusted returns
   ```

2. **Portfolio Constraints**:
   ```python
   # Max position size: 20% per stock
   # Max leverage: 1.0 (no borrowing)
   # Min diversification: Hold 10+ stocks
   ```

3. **Parallel Environments**:
   ```python
   vec_env = make_vec_env(make_env, n_envs=8)
   # 8× faster data collection
   ```

4. **Curriculum Learning**:
   ```python
   # Start with easy periods (stable markets)
   # Gradually increase difficulty (volatile markets)
   ```

---

## Summary

**Purpose**: Train RL agent for portfolio management using GNN intelligence  
**Algorithm**: PPO (Proximal Policy Optimization)  
**Innovation**: GNN-guided trading decisions  
**Output**: Trained agent ready for backtesting

**This enables actionable trading strategies!** 💰

---

**Last Updated**: 2025-11-02  
**Code Style**: Research prototype [[memory:3128464]]  
**Language**: English comments [[memory:3128459]]


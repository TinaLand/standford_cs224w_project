# RL Agent Usage Guide

## StockTradingAgent Class

`StockTradingAgent` is a class that encapsulates a PPO agent and GNN model, providing a unified interface to train, evaluate, and use the RL agent.

## Basic Usage

### 1. Create Agent

```python
from src.rl.agents.single_agent import StockTradingAgent
from src.rl.environments.single_agent import StockTradingEnv

# Define environment factory function
def make_env():
 return StockTradingEnv(
 start_date=START_DATE,
 end_date=END_DATE,
 gnn_model=gnn_model,
 device=device
 )

# Create agent
agent = StockTradingAgent(
 gnn_model=gnn_model,
 env_factory=make_env,
 device=device,
 learning_rate=1e-5,
 tensorboard_log=log_path, # Optional
 policy="MlpPolicy",
 verbose=1
)
```

### 2. Train Agent

```python
# Train agent
training_stats = agent.train(
 total_timesteps=10000,
 progress_bar=True
)

# Save agent
agent.save(save_path)
```

### 3. Load Trained Agent

```python
# Create new agent instance
agent = StockTradingAgent(...)

# Load trained weights
agent.load(load_path)
```

### 4. Use Agent for Prediction

```python
# Use in environment
obs, info = env.reset()
action, _ = agent.predict(obs, deterministic=True)
obs, reward, done, truncated, info = env.step(action)
```

### 5. Evaluate Agent

```python
# Evaluate agent performance
metrics = agent.evaluate(
 env=test_env,
 n_episodes=10,
 deterministic=True
)

print(f"Mean return: {metrics['mean_return']}")
print(f"Std return: {metrics['std_return']}")
```

## Main Methods

- `train(total_timesteps, callback, progress_bar)`: Train agent
- `predict(observation, deterministic)`: Predict action
- `evaluate(env, n_episodes, deterministic)`: Evaluate performance
- `save(path)`: Save agent
- `load(path)`: Load agent

## Advantages

1. **Modular**: Encapsulates RL logic in an independent class
2. **Easy to Use**: Unified interface, reduces complexity
3. **Extensible**: Can easily add new agent types or features
4. **Maintainable**: Clear code structure, easy to debug and optimize

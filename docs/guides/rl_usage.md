# RL Agent 使用指南

## StockTradingAgent 类

`StockTradingAgent` 是一个封装了 PPO agent 和 GNN 模型的类，提供了统一的接口来训练、评估和使用 RL agent。

## 基本用法

### 1. 创建 Agent

```python
from rl_agent import StockTradingAgent
from src.rl.environment import StockTradingEnv

# 定义环境工厂函数
def make_env():
 return StockTradingEnv(
 start_date=START_DATE,
 end_date=END_DATE,
 gnn_model=gnn_model,
 device=device
 )

# 创建 agent
agent = StockTradingAgent(
 gnn_model=gnn_model,
 env_factory=make_env,
 device=device,
 learning_rate=1e-5,
 tensorboard_log=log_path, # 可选
 policy="MlpPolicy",
 verbose=1
)
```

### 2. 训练 Agent

```python
# 训练 agent
training_stats = agent.train(
 total_timesteps=10000,
 progress_bar=True
)

# 保存 agent
agent.save(save_path)
```

### 3. 加载已训练的 Agent

```python
# 创建新的 agent 实例
agent = StockTradingAgent(...)

# 加载已训练的权重
agent.load(load_path)
```

### 4. 使用 Agent 进行预测

```python
# 在环境中使用
obs, info = env.reset()
action, _ = agent.predict(obs, deterministic=True)
obs, reward, done, truncated, info = env.step(action)
```

### 5. 评估 Agent

```python
# 评估 agent 性能
metrics = agent.evaluate(
 env=test_env,
 n_episodes=10,
 deterministic=True
)

print(f"平均收益: {metrics['mean_return']}")
print(f"标准差: {metrics['std_return']}")
```

## 主要方法

- `train(total_timesteps, callback, progress_bar)`: 训练 agent
- `predict(observation, deterministic)`: 预测动作
- `evaluate(env, n_episodes, deterministic)`: 评估性能
- `save(path)`: 保存 agent
- `load(path)`: 加载 agent

## 优势

1. **模块化**: 将 RL 逻辑封装在独立的类中
2. **易用性**: 统一的接口，降低使用复杂度
3. **可扩展**: 可以轻松添加新的 agent 类型或功能
4. **可维护**: 清晰的代码结构，便于调试和优化

# Multi-Agent RL Implementation Guide

## Implementation Approach Selection

Based on your project characteristics (stock trading, GNN, RL), we chose:

### **Cooperative Multi-Agent RL (Cooperative MARL) - CTDE Architecture**

**Why this approach?**
- Most intuitive: Group by sector, each agent manages one sector
- Fits financial scenarios: Different sectors have different trading patterns
- Easy to implement: Extends existing single-agent system
- Effective: Can learn sector-specific strategies

---

## Architecture Design

### Overall Architecture

```

 Multi-Agent RL System (CTDE) 
 
    
  Agent 1:   Agent 2:   Agent N:  
  Technology   Healthcare   Finance  
  (10 stocks)   (10 stocks)   (10 stocks)
    
    
  
  
  
  Mixing Network  
  (QMIX-style)  
  
  
  
  Global Reward  
  (Portfolio)  
  

```

### Core Components

1. **SectorAgent**: Agent for each sector
2. **MultiAgentCoordinator**: Coordinator (CTDE)
3. **MixingNetwork**: QMIX-style mixing network
4. **MultiAgentTradingEnv**: Multi-agent environment

---

## Implementation Details

### 1. Sector Grouping

**File**: `scripts/multi_agent_rl_coordinator.py`

```python
class SectorGrouping:
 @staticmethod
 def load_sector_mapping() -> Dict[str, List[str]]:
 """Load sector grouping from static_sector_industry.csv"""

 # Returns:
 # {
 # 'Technology': ['AAPL', 'MSFT', 'GOOGL', ...],
 # 'Healthcare': ['JNJ', 'PFE', 'UNH', ...],
 # 'Financials': ['JPM', 'BAC', 'WFC', ...],
 # ...
 # }
```

### 2. Individual Agent (SectorAgent)

```python
class SectorAgent:
 """Manages stocks in a specific sector"""

 def __init__(self, sector_name, tickers, gnn_model, ...):
 # Each agent has its own PPO policy network
 self.agent = PPO(policy="MlpPolicy", ...)

 def predict(self, observation):
 """Predict actions for stocks in this sector"""
 return self.agent.predict(observation)
```

**Features**:
- Each agent independently manages stocks in one sector
- Uses the same GNN model for embeddings
- Has its own PPO policy network

### 3. Mixing Network (MixingNetwork)

```python
class MixingNetwork(nn.Module):
 """QMIX-style mixing network"""

 def forward(self, q_values, global_state):
 """
 Mix individual Q-values into global Q-value

 Args:
 q_values: [batch, num_agents] - Q-values from each agent
 global_state: [batch, state_dim] - Global state (portfolio state)

 Returns:
 global_q: [batch, 1] - Mixed global Q-value
 """
 # Use hypernetwork to generate mixing weights
 # Ensure monotonicity (satisfies IGM condition)
```

**QMIX Principles**:
- **Individual Q-values**: Each agent outputs its own Q-value
- **Mixing Network**: Mixes individual Q-values into global Q-value
- **Monotonicity**: Ensures global Q-value increases monotonically with individual Q-values
- **Centralized Training**: Uses global information during training
- **Decentralized Execution**: Each agent makes independent decisions during execution

### 4. Coordinator (MultiAgentCoordinator)

```python
class MultiAgentCoordinator:
 """Multi-agent coordinator (CTDE)"""

 def get_agent_actions(self, observations):
 """Get actions from all agents (decentralized execution)"""
 actions = {}
 for sector, agent in self.agents.items():
 actions[sector] = agent.predict(observations[sector])
 return actions

 def merge_actions(self, actions_dict):
 """Merge all agent actions into global action"""
 # Merge sector actions into all-stock actions
 return combined_actions

 def compute_global_q_value(self, q_values_dict, global_state):
 """Compute global Q-value (centralized training)"""
 return self.mixing_network(q_values, global_state)
```

---

## Training Process (CTDE)

### Centralized Training

```
1. Collect experiences:
 - Each agent independently interacts with environment
 - Collect (obs, action, reward, next_obs)
 - Compute individual Q-values

2. Mix Q-values:
 - Use Mixing Network to mix individual Q-values
 - Get global Q-value

3. Update policies:
 - Update Mixing Network using global reward
 - Update individual agent policies using individual rewards
```

### Decentralized Execution

```
1. Get observations:
 - Each agent gets observations for its sector

2. Independent decisions:
 - Each agent independently predicts actions
 - No communication needed

3. Merge actions:
 - Coordinator merges all actions
 - Execute global action
```

---

## Implementation Steps

### Step 1: Create Multi-Agent System

```python
# Load GNN model
gnn_model = load_gnn_model_for_rl()

# Create coordinator
coordinator = MultiAgentCoordinator(
 gnn_model=gnn_model,
 sector_groups=SectorGrouping.load_sector_mapping(),
 all_tickers=all_tickers,
 device=device
)
```

### Step 2: Train Individual Agents

```python
# Create environment for each sector
for sector_name, agent in coordinator.agents.items():
 # Create environment for this sector (only stocks in this sector)
 env = create_sector_env(sector_name, ...)

 # Train this agent
 agent.agent.learn(total_timesteps=timesteps_per_agent)
```

### Step 3: Train Mixing Network

```python
# Collect experiences
experiences = collect_experiences(coordinator, env, n_steps)

# Train mixing network
for experience in experiences:
 # Compute individual Q-values
 q_values = {sector: agent.compute_q(obs) for sector, agent in coordinator.agents.items()}

 # Compute global Q-value
 global_q = coordinator.compute_global_q_value(q_values, global_state)

 # Compute target Q-value
 target_q = reward + gamma * next_global_q

 # Update mixing network
 coordinator.train_mixing_network(q_values, global_state, target_q)
```

### Step 4: Evaluation

```python
# Evaluate multi-agent system
results = evaluate_multi_agent_system(
 coordinator=coordinator,
 start_date=START_DATE,
 end_date=END_DATE
)
```

---

## Three Implementation Approaches Comparison

### Approach 1: Cooperative (Recommended) 

**Architecture**: CTDE with QMIX

**Advantages**:
- Most intuitive, easy to implement
- Each agent specializes (by sector)
- Can learn sector-specific strategies
- Global optimization (shared reward)

**Implementation Files**:
- `scripts/multi_agent_rl_coordinator.py` 
- `src/multi_agent_training.py` 

**Status**: Framework implemented, needs training loop completion

---

### Approach 2: Adversarial (Optional)

**Architecture**: Predictor vs Critic (GAN-style)

**Implementation Idea**:
```python
class PredictorAgent:
 """Predicts stock movements"""
 def __init__(self, gnn_model):
 self.gnn = gnn_model
 self.policy = PPO(...)

class CriticAgent:
 """Evaluates prediction quality"""
 def __init__(self):
 self.discriminator = MLP(...)

# Adversarial training
for epoch in range(epochs):
 # Train Predictor
 predictions = predictor.predict(obs)
 critic_score = critic.evaluate(predictions, true_labels)
 predictor_loss = -critic_score # Maximize critic score
 predictor.update(predictor_loss)

 # Train Critic
 critic_loss = critic.train(predictions, true_labels)
 critic.update(critic_loss)
```

**Advantages**:
- May improve prediction robustness
- Similar to GAN, may discover new patterns

**Disadvantages**:
- Training instability
- Requires more tuning

**Status**: Not implemented (optional)

---

### Approach 3: Hierarchical (Optional)

**Architecture**: High-level + Low-level Agents

**Implementation Idea**:
```python
class HighLevelAgent:
 """High-level agent: macro decisions"""
 def predict(self, macro_graph):
 # Observe macro graph (country/industry relationships)
 # Output: Long-term trend (overall position ratio for next month)
 return long_term_allocation

class LowLevelAgent:
 """Low-level agent: micro decisions"""
 def predict(self, stock_graph, high_level_decision):
 # Observe stock detail graph
 # Under high-level guidance, output short-term buy/sell signals
 return short_term_actions
```

**Advantages**:
- Handles multiple time scales
- Combines macro + micro

**Disadvantages**:
- High implementation complexity
- Requires hierarchical structure design

**Status**: Not implemented (optional)

---

## How to Use

### Quick Start

```bash
# 1. Ensure Phase 4 model is trained
python src/training/core_training.py

# 2. Run multi-agent training
python src/multi_agent_training.py
```

### Configuration Options

Modify in `src/rl/training/multi_agent_training.py`:

```python
# Training configuration
TOTAL_TIMESTEPS = 10000 # Total training steps
LEARNING_RATE = 1e-5 # Learning rate
START_DATE = pd.to_datetime('2023-01-01')
END_DATE = pd.to_datetime('2024-12-31')
```

---

## Expected Effects

### Advantages

1. **Specialization**
 - Each agent focuses on specific sector
 - Can learn sector-specific trading patterns

2. **Risk Diversification**
 - Different sector agents make independent decisions
 - Better risk control

3. **Interpretability**
 - Can analyze each agent's contribution
 - Easier to understand decision process

### Potential Issues

1. **May not be better**
 - Single agent already performs well (Sharpe 2.36)
 - Multi-agent may not bring significant improvement

2. **Training complexity**
 - Need to train multiple agents
 - Need to train mixing network
 - Training time significantly increases

3. **Coordination difficulties**
 - How to allocate capital?
 - How to handle conflicts?

---

## Implementation Status

### Completed 

- `multi_agent_rl_coordinator.py`: Core architecture
 - SectorAgent class
 - MultiAgentCoordinator class
 - MixingNetwork class
 - SectorGrouping utility

- `src/rl/training/multi_agent_training.py`: Training script framework
 - Training loop structure
 - Evaluation framework

### Needs Completion 

1. **Environment Splitting**
 - Current: All agents use full environment
 - Need: Create independent environment for each sector (only stocks in that sector)

2. **Observation Space Splitting**
 - Current: All agents see full observation
 - Need: Each agent only sees observations for its sector

3. **Complete Training Loop**
 - Current: Simplified training loop
 - Need: Complete CTDE training process
 - Need: Q-value collection and mixing network training

4. **Action Space Handling**
 - Current: Simple action merging
 - Need: Handle capital allocation, conflict resolution

---

## Implementation Recommendations

### Priority 1: Complete Cooperative Multi-Agent (Recommended)

**Why**:
- Most intuitive, best fits project characteristics
- Framework already implemented, just need to complete details

**What to do**:
1. Implement sector environment splitting
2. Complete observation space splitting
3. Implement complete CTDE training loop
4. Add capital allocation mechanism

**Time estimate**: 1-2 weeks

### Priority 2: Adversarial Multi-Agent (Optional)

**Why**:
- May improve prediction robustness
- But training is unstable, requires more tuning

**Time estimate**: 2-3 weeks

### Priority 3: Hierarchical Multi-Agent (Optional)

**Why**:
- Handles multiple time scales, high research value
- But high implementation complexity

**Time estimate**: 3-4 weeks

---

## Code Structure

```
scripts/
 multi_agent_rl_coordinator.py # Core architecture 
  SectorGrouping # Sector grouping
  SectorAgent # Individual agent
  MixingNetwork # QMIX mixing network
  MultiAgentCoordinator # Coordinator

 src/rl/training/multi_agent_training.py # Training script 
  train_multi_agent_system() # Training function
  evaluate_multi_agent_system() # Evaluation function

 src/rl/multi_agent_env.py # Multi-agent environment (to be implemented)
  SectorTradingEnv # Sector-specific environment
```

---

## Summary

### Current Status

- **Architecture implemented**: Core components all created
- **Needs completion**: Training loop and environment splitting
- **Documentation complete**: Implementation guide provided

### Next Steps

1. **Complete environment splitting** (High priority)
2. **Implement complete training loop** (High priority)
3. **Add capital allocation mechanism** (Medium priority)
4. **Evaluation and comparison** (Medium priority)

### Recommendations

**For current project**:
- Multi-agent RL is an optional extension
- Single agent already performs well
- Can skip if time is limited

**If implementing**:
- Recommend completing cooperative multi-agent first
- This is the most intuitive and practical approach
- Can verify if multi-agent is really better

---

**Implementation Files**:
- `scripts/multi_agent_rl_coordinator.py` 
- `src/multi_agent_training.py` 
- `docs/MULTI_AGENT_RL_IMPLEMENTATION.md` 

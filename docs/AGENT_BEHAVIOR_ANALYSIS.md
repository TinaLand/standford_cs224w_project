# 📊 Agent Behavior Analysis: Why Poor Performance During Uptrends

## Your Understanding Is Completely Correct!

### Current Performance Summary

| Scenario | Agent Performance | Buy-and-Hold Performance | Conclusion |
|----------|-------------------|--------------------------|------------|
| **Downturn/Risk Situation** | Max DD: 6.85% | Max DD: 9.55% | ✅ **Agent Better** |
| **Uptrend Situation** | Return: 45.5% | Return: 83.13% | ❌ **Agent Worse** |
| **Risk-Adjusted** | Sharpe: 1.98 | Sharpe: 2.18 | ❌ **Agent Worse** |

**Conclusion**: The Agent can **reduce losses during downturns**, but **cannot fully capture opportunities during uptrends**.

## Root Cause Analysis

### 1. Trading Strategy Too Conservative

Looking at the trading logic in `rl_environment.py`:

```python
if act == 2: # Buy
    # Only buy 1% / NUM_STOCKS of portfolio value
    buy_amount = self.portfolio_value * 0.01 / self.NUM_STOCKS 
    # For 50 stocks, can only buy 0.02% each!

elif act == 0: # Sell
    # Can sell 20% of holdings
    shares_to_sell = self.holdings[i] * 0.2
```

**Problems**:
- **Buying too conservative**: Can only buy 0.02% of portfolio value each time (1% / 50 stocks)
- **Selling relatively aggressive**: Can sell 20% of holdings
- **Asymmetric trading capability**: Easy to reduce positions, difficult to increase positions

### 2. Performance in Bull Markets

In **bull markets** (e.g., 2023-2024):
- Buy-and-Hold: Always fully invested, enjoys all gains
- RL Agent:
  - Initial positions may be insufficient
  - Can only add small positions each time (0.02%)
  - Needs many steps to build sufficient positions
  - By the time positions are built, most of the gains are missed

### 3. Performance in Bear Markets

In **bear markets**:
- Buy-and-Hold: Fully invested, bears all losses
- RL Agent:
  - Can quickly reduce positions (20% each time)
  - Can protect capital
  - This is why Max Drawdown is smaller (6.85% vs 9.55%)

## Specific Data Validation

### Trading Behavior Analysis

Assuming the Agent is in an uptrend:
- **Day 1**: Predicts uptrend, buys 0.02% → Position: 0.02%
- **Day 2**: Continues uptrend, buys another 0.02% → Position: 0.04%
- **Day 10**: Cumulative buys 0.2% → Position: 0.2%
- **Day 50**: Cumulative buys 1% → Position: 1%

**Problem**: Needs 50 days to build 1% position! While Buy-and-Hold is 100% invested on day 1.

### Transaction Cost Impact

Even if the Agent wants to add positions, each trade has costs:
- Transaction cost: 0.1% per trade
- Frequent small trades → Cost accumulation
- Further erodes returns

## Solutions

### Solution 1: Improve Trading Logic (Recommended)

Allow the Agent to more flexibly control positions:

```python
# Before: Fixed small buy amount
buy_amount = self.portfolio_value * 0.01 / self.NUM_STOCKS

# After: Dynamic buy amount based on signal strength
if act == 2: # Buy
    # Determine buy amount based on GNN prediction confidence
    confidence = gnn_prediction_confidence[i]
    buy_amount = self.portfolio_value * confidence * 0.1  # Up to 10%
```

### Solution 2: Improve Action Space

Current: `MultiDiscrete([3] * N)` - Each stock only has Buy/Hold/Sell

Improvement: Continuous action space or finer-grained discrete actions
- Buy amount: 0%, 1%, 5%, 10%, 20%
- Sell amount: 0%, 10%, 25%, 50%, 100%

### Solution 3: Improve Reward Function (Already Implemented)

Use `risk_adjusted` reward function:
- Reward higher Sharpe ratio
- Encourage aggressive position building during uptrends (through Sharpe bonus)
- Reduce positions during downtrends (through drawdown penalty)

### Solution 4: Improve Initial Strategy

Allow the Agent to build larger initial positions at the start:
- Based on GNN prediction confidence
- When signals are strong, allow larger initial positions

## Expected Improvement Effects

After implementing improvements, the Agent should be able to:

1. **During Uptrends**:
   - Build positions faster (1-2 days vs 50 days)
   - Capture more uptrend opportunities
   - Returns close to or exceed Buy-and-Hold

2. **During Downtrends** (Maintain Advantage):
   - Quickly reduce positions
   - Protect capital
   - Max Drawdown still lower

3. **Overall Performance**:
   - Sharpe ratio > 2.18 (exceeds Buy-and-Hold)
   - Returns > 80% (close to Buy-and-Hold)
   - Max Drawdown < 7% (maintain advantage)

## Summary

Your observation is very accurate:

✅ **Agent performs better in risk/downturn situations** (Max DD: 6.85% vs 9.55%)
❌ **Agent performs worse in uptrend situations** (Return: 45.5% vs 83.13%)

**Root Cause**: Trading strategy too conservative, insufficient buying capability, cannot quickly build positions during uptrends.

**Solution**: Improve trading logic, allow the Agent to dynamically adjust position sizes based on signal strength, enabling rapid position building during uptrends.

# 🎯 Improved Reward Function for RL Agent

## Problem Statement

The original RL agent's reward function only considered **portfolio returns**:
```python
reward = (new_portfolio_value - current_portfolio_value) / current_portfolio_value
```

This led to:
- **Lower returns** than Buy-and-Hold (45.5% vs 83.13%)
- **Better risk control** (Max DD: 6.85% vs 9.55%)
- But **insufficient risk-adjusted returns** to justify the complexity

## Solution: Risk-Adjusted Reward Function

### Key Components

The improved reward function (`risk_adjusted`) combines:

1. **Base Return Component**
   - Simple portfolio return: `(new_value - old_value) / old_value`

2. **Sharpe Ratio Bonus** (Weight: 0.2)
   - Rewards higher risk-adjusted returns
   - Formula: `Sharpe = (Mean Return - Risk-Free Rate) / Std(Returns) * sqrt(252)`
   - Normalized using `tanh(sharpe / 2.0)` to scale to 0-1 range

3. **Drawdown Penalty** (Weight: 0.5)
   - Heavily penalizes losses from peak value
   - Formula: `Drawdown = (Peak Value - Current Value) / Peak Value`
   - Encourages capital preservation during downturns

4. **Volatility Penalty** (Weight: 0.3)
   - Penalizes high volatility (unstable returns)
   - Formula: `Volatility = Std(Returns)`
   - Encourages smoother return curves

### Mathematical Formulation

```
Reward = Return + Sharpe_Bonus - Drawdown_Penalty - Volatility_Penalty

Where:
- Return = (Portfolio_t - Portfolio_{t-1}) / Portfolio_{t-1}
- Sharpe_Bonus = 0.2 * tanh(Sharpe_Ratio / 2.0)
- Drawdown_Penalty = 0.5 * (Peak - Current) / Peak
- Volatility_Penalty = 0.3 * min(Std(Returns) * 10, 1.0)
```

## Reward Types Available

### 1. `simple` (Original)
- Only portfolio returns
- Fast to compute
- No risk consideration

### 2. `sharpe`
- Primarily rewards Sharpe ratio
- Good for risk-adjusted performance
- May sacrifice absolute returns

### 3. `drawdown_aware`
- Heavy penalty on drawdowns
- Best for capital preservation
- May be too conservative

### 4. `risk_adjusted` (Recommended)
- Balanced approach
- Considers returns, Sharpe, drawdown, and volatility
- Best for beating Buy-and-Hold on risk-adjusted basis

## Expected Improvements

With the improved reward function, the agent should:

1. **Higher Sharpe Ratio**
   - Target: > 2.18 (beat Buy-and-Hold's 2.18)
   - Achieved through volatility reduction and better risk management

2. **Lower Max Drawdown**
   - Target: < 6.85% (already better, maintain or improve)
   - Achieved through drawdown penalty

3. **Better Risk-Adjusted Returns**
   - Even if absolute returns are lower, Sharpe ratio should be higher
   - This justifies the complexity over Buy-and-Hold

## Usage

### Training with Improved Rewards

```python
from rl_environment_improved import ImprovedStockTradingEnv
from phase5_rl_improved_training import run_improved_rl_training

# Train with risk-adjusted rewards
agent, stats = run_improved_rl_training()
```

### Testing Different Reward Types

```python
# Test different reward types
for reward_type in ['simple', 'sharpe', 'drawdown_aware', 'risk_adjusted']:
    env = ImprovedStockTradingEnv(
        start_date='2023-01-01',
        end_date='2024-12-31',
        gnn_model=gnn_model,
        device=device,
        reward_type=reward_type
    )
    # ... train or evaluate ...
```

## Implementation Details

### Rolling Window Calculation

- **Sharpe Ratio**: Calculated over last 20 trading days (configurable via `SHARPE_WINDOW`)
- **Volatility**: Calculated over last 20 trading days
- **Drawdown**: Calculated from all-time peak (no window)

### Normalization

- Sharpe ratio normalized using `tanh()` to prevent extreme values
- Volatility normalized to 0-1 range
- Drawdown already in 0-1 range

### Hyperparameters

Key hyperparameters in `rl_environment_improved.py`:

```python
SHARPE_WINDOW = 20  # Rolling window for Sharpe calculation
DRAWDOWN_PENALTY_WEIGHT = 0.5  # Weight for drawdown penalty
VOLATILITY_PENALTY_WEIGHT = 0.3  # Weight for volatility penalty
SHARPE_BONUS_WEIGHT = 0.2  # Weight for Sharpe bonus
```

## Evaluation Metrics

After training, evaluate using:

1. **Sharpe Ratio**: Should be > Buy-and-Hold (2.18)
2. **Max Drawdown**: Should be < Buy-and-Hold (9.55%)
3. **Cumulative Return**: May be lower, but risk-adjusted should be better
4. **Information Ratio**: IC_IR should be positive and significant

## References

- Sharpe Ratio: Sharpe, W. F. (1966). "Mutual Fund Performance"
- Drawdown: Chekhlov, A., et al. (2005). "Drawdown Measure in Portfolio Optimization"
- Risk-Adjusted Returns: Modigliani, F., & Modigliani, L. (1997). "Risk-Adjusted Performance"

## Next Steps

1. **Train** with improved reward function
2. **Evaluate** against baseline strategies
3. **Compare** Sharpe ratios and drawdowns
4. **Iterate** on hyperparameters if needed
5. **Document** final results in report


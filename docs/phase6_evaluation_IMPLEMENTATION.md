# Phase 6: Evaluation & Ablation Studies - Implementation Guide

## Overview

**File**: `scripts/phase6_evaluation.py`  
**Purpose**: Final evaluation and ablation studies for research paper  
**Dependencies**: stable-baselines3, pandas, numpy  
**Input**: Trained GNN + RL agent  
**Output**: Performance metrics, ablation results, comparison tables

---

## What Does This File Do?

This script performs **comprehensive evaluation** for the research paper:

1. **Final Backtesting** - Test RL agent on held-out test set
2. **Financial Metrics** - Calculate Sharpe ratio, max drawdown, returns
3. **Ablation Studies** - Prove each component's contribution
4. **Baseline Comparisons** - Compare against buy-and-hold, etc.

**This generates the results for your paper!** 📊

---

## Why This Phase?

### Research Requirements

**Academic Paper Needs**:
1. **Main Results**: Final test performance
2. **Ablations**: Which components matter?
3. **Comparisons**: Better than baselines?
4. **Statistical Significance**: Is improvement real?

**This File Provides**: All of the above

---

## Key Components

### 1. Financial Metrics Calculation

```python
def calculate_financial_metrics(portfolio_values, trading_days):
    """Calculate key financial metrics."""
```

#### Cumulative Return

**Formula**:
```
R_cumulative = (V_final / V_initial) - 1
```

**Example**:
```
Initial: $10,000
Final: $12,500
Return = (12,500 / 10,000) - 1 = 0.25 = 25%
```

**Why Important**: Primary performance metric

#### Sharpe Ratio

**Formula**:
```
Sharpe = (μ_r - R_f) / σ_r × √252

where:
μ_r = mean daily return
σ_r = std of daily returns
R_f = risk-free rate (≈0 for simplicity)
252 = trading days per year
```

**Interpretation**:
```
Sharpe < 1.0: Poor risk-adjusted return
Sharpe = 1.0: Acceptable
Sharpe = 2.0: Good
Sharpe > 3.0: Excellent
```

**Why √252**:
- Daily returns → annual returns scaling
- `σ_annual = σ_daily × √252`
- Standardizes comparison across strategies

**Mathematical Derivation**:
```
Annual return variance:
Var(R_year) = Var(Σ_{i=1}^{252} r_i)

If returns independent:
= Σ Var(r_i) = 252 × Var(r_daily)

Therefore:
σ_year = √(252 × σ²_daily) = √252 × σ_daily
```

**Why Sharpe Matters**:
- 30% return with 40% volatility: Sharpe = 0.75 (risky)
- 20% return with 10% volatility: Sharpe = 2.0 (better!)

**Risk-Adjusted Performance** more meaningful than raw returns

#### Maximum Drawdown

**Formula**:
```
MDD = max_t ((Peak_t - V_t) / Peak_t)

where Peak_t = max(V_1, V_2, ..., V_t)
```

**Example**:
```
Day 1: $10,000
Day 50: $15,000  ← Peak
Day 75: $12,000  ← Drawdown = (15,000-12,000)/15,000 = 20%
Day 100: $16,000 ← New peak, but MDD still 20%
```

**Why Important**:
- Measures **worst-case loss**
- Investors care about downside risk
- "How much can I lose?"

**Interpretation**:
```
MDD < 10%: Low risk
MDD = 20%: Moderate risk
MDD > 30%: High risk
MDD > 50%: Extreme risk
```

**For Trading Strategies**:
- Hedge fund standard: MDD < 20%
- Retail acceptable: MDD < 30%

---

### 2. Final Backtesting

```python
def run_final_backtest(gnn_model, rl_agent_path):
    """Runs final backtest on held-out test set."""
```

#### Walk-Forward Testing

**Standard Backtest** (naive):
```
Train on: 2015-2020
Test on:  2020-2024 (all at once)
```

**Walk-Forward** (realistic):
```
Train on: 2015-2020
Test on:  2020-01-01 to 2020-01-31
Retrain on: 2015-2020-01
Test on:  2020-02-01 to 2020-02-28
...
```

**Current Implementation**: Standard (test on fixed period)
**Future**: Walk-forward for robustness

#### Episode Execution

```python
obs = env.reset()
done = False

while not done:
    action = agent.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    portfolio_values.append(info['portfolio_value'])
```

**Deterministic = True**: No exploration during testing
- Training: Explore (try random actions)
- Testing: Exploit (use best known actions)

---

### 3. Ablation Studies

```python
def train_and_evaluate_ablation(ablation_name, config_modifier):
    """Re-trains model with modified configuration."""
```

#### What is an Ablation Study?

**Purpose**: Prove each component contributes to performance

**Method**: Remove component, measure performance drop

**Example**:
```
Full Model:          F1 = 0.70
Remove PEARL:        F1 = 0.65  (-0.05, PEARL helps!)
Remove Sector Edges: F1 = 0.66  (-0.04, sector edges help!)
Remove Focal Loss:   F1 = 0.63  (-0.07, focal loss critical!)
```

**Conclusion**: All components necessary

#### Planned Ablations

```python
ablations = [
    'Full_Model',           # Baseline
    'Abl_NoPEARL',         # Remove PEARL positional encoding
    'Abl_OnlyStatic',      # Remove dynamic edges
    'Abl_NoFundSim',       # Remove fundamental similarity
    'Abl_FixedLaplacian',  # Use fixed PE instead of PEARL
]
```

**Research Questions**:
1. Does PEARL help? (NoPEARL vs Full)
2. Are dynamic edges important? (OnlyStatic vs Full)
3. Do fundamentals matter? (NoFundSim vs Full)
4. Is PEARL better than fixed PE? (FixedLaplacian vs Full)

#### Implementation Note

**Current**: Placeholder
```python
# Would need to:
# 1. Modify model architecture
# 2. Retrain from scratch
# 3. Evaluate on test set
# 4. Compare metrics
```

**Why Placeholder**: Each ablation = full training run
- 1 ablation = 60-90 minutes
- 5 ablations = 5-8 hours
- Beyond scope of quick demonstration

**For Real Research**:
- Create automated ablation script
- Run overnight
- Collect all results in table

---

## Output Format

### Final Metrics CSV

```csv
strategy,         final_value, Sharpe_Ratio, Cumulative_Return, Max_Drawdown
Core_GNN_RL,      12450.00,    1.85,          0.245,             0.12
Baseline_GNN_RL,  11800.00,    1.62,          0.180,             0.15
Buy_And_Hold,     11200.00,    1.20,          0.120,             0.22
```

**Usage**: Copy to paper as Table 1

### Ablation Results CSV

```csv
strategy,              Sharpe_Ratio, Cumulative_Return, ΔF1_from_Full
Full_Model,            1.85,         0.245,             0.00
Abl_NoPEARL,          1.72,         0.220,             -0.05
Abl_OnlyStatic,       1.68,         0.215,             -0.04
Abl_NoFundSim,        1.74,         0.225,             -0.03
```

**Usage**: Copy to paper as Table 2 (Ablation Study)

---

## Baseline Comparisons

### Comparison Strategies

**1. Buy-and-Hold**:
```python
# Buy all stocks equally at start
# Hold until end
```
- Simplest strategy
- Market return baseline

**2. Equal-Weight Rebalance**:
```python
# Every month: rebalance to equal weights
```
- Captures diversification benefit

**3. Momentum**:
```python
# Buy top 10 performers last month
```
- Traditional quant strategy

**4. Mean Reversion**:
```python
# Buy stocks that fell, sell stocks that rose
```
- Contrarian strategy

**Our Model Should Beat All**: Otherwise, why use GNN+RL?

---

## Statistical Significance Testing

### Why Needed?

**Problem**: Markets are noisy

**Example**:
```
Our model: 24.5% return
Baseline:  22.0% return

Is 2.5% difference real or luck?
```

**Solution**: Statistical testing

### Methods (Future Implementation)

**1. Bootstrap Resampling**:
```python
for i in range(1000):
    sample_returns = np.random.choice(returns, size=len(returns))
    sample_sharpe = calculate_sharpe(sample_returns)
    bootstrap_sharpes.append(sample_sharpe)

p_value = (bootstrap_sharpes > our_sharpe).mean()
```

**2. Multiple Time Periods**:
```
Test on: 2020-2021, 2021-2022, 2022-2023, 2023-2024
If model wins in 4/4 periods → significant
```

**3. Cross-Validation** (time series):
```
Rolling window validation
Train: [2015-2019], Test: [2020]
Train: [2016-2020], Test: [2021]
...
```

---

## Integration with Paper

### Figures to Generate

**Figure 1**: Equity Curve
```python
plt.plot(dates, portfolio_values)
plt.title("Portfolio Value Over Time")
```

**Figure 2**: Drawdown Chart
```python
drawdown = (cummax - portfolio_values) / cummax
plt.fill_between(dates, drawdown)
```

**Figure 3**: Ablation Bar Chart
```python
plt.bar(ablations, f1_scores)
plt.title("Ablation Study Results")
```

**Figure 4**: Comparison Table
- Our model vs baselines
- Multiple metrics

---

## Summary

**Purpose**: Generate all results for research paper  
**Key Outputs**: Performance metrics, ablation results, comparisons  
**Design**: Comprehensive evaluation framework  
**Status**: Scaffolded (needs full ablation implementation)

**This validates your research contribution!** 🎓

---

**Last Updated**: 2025-11-02  
**Code Style**: Research-focused [[memory:3128464]]  
**Language**: English comments [[memory:3128459]]


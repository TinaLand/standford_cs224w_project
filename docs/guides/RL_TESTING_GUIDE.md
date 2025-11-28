# RL Agent Testing Guide

## Quick Start

### Get Latest Test Results

**Option 1: Quick Test (Fast, ~1-2 minutes)**
```bash
python scripts/quick_rl_test.py
```
- Runs a short backtest (500 steps)
- Shows key metrics (Sharpe, Return, Drawdown)
- Saves plot to `results/quick_rl_test_results.png`

**Option 2: Full Evaluation (Complete, ~10-30 minutes)**
```bash
python scripts/run_full_rl_evaluation.py
```
- Trains agent if needed (or uses existing)
- Runs full backtest on test set
- Generates comprehensive report and plots
- Saves to `results/rl_evaluation_report_*.csv` and `results/rl_performance_plots_*.png`

**Option 3: View Latest Results**
```bash
python scripts/view_rl_results.py
```
- Shows latest test results
- Displays report summary
- Lists available plots

## Test Results Location

All results are saved in `results/` directory:

- **Quick Test**: `results/quick_rl_test_results.png`
- **Full Reports**: `results/rl_evaluation_report_YYYYMMDD_HHMMSS.csv`
- **Full Plots**: `results/rl_performance_plots_YYYYMMDD_HHMMSS.png`

## Understanding Results

### Key Metrics

1. **Total Return**: Percentage gain/loss from initial investment
 - Good: >20% over test period
 - Excellent: >40%

2. **Sharpe Ratio**: Risk-adjusted return
 - Good: >1.0
 - Excellent: >1.5
 - Outstanding: >2.0

3. **Max Drawdown**: Maximum peak-to-trough decline
 - Good: <15%
 - Excellent: <10%

### Latest Test Results (from quick test)

Based on the latest run:
- **Initial Value**: $10,000.00
- **Final Value**: $14,489.80
- **Total Return**: 44.90%
- **Sharpe Ratio**: 1.96 (Excellent!)
- **Max Drawdown**: 8.63% (Excellent!)
- **Steps**: 500

## Dynamic Visualization

### Real-time Training Monitor

To monitor training in real-time:

1. **TensorBoard** (if available):
 ```bash
 tensorboard --logdir logs/rl_logs
 ```
 Then open http://localhost:6006

2. **Progress Updates**:
 - Quick test shows progress every 50 steps
 - Full evaluation shows detailed progress

### Plot Types

The visualization includes:

1. **Portfolio Value Over Time**: Shows how portfolio value changes
2. **Cumulative Returns**: Percentage returns over time
3. **Reward Distribution**: Histogram of rewards received
4. **Drawdown Analysis**: Shows risk periods

## Troubleshooting

### Agent Observation Space Mismatch

If you see: `Observation spaces do not match`

**Solution**: The agent needs to be retrained with the correct observation space. The script will automatically retrain if needed.

### Missing Dependencies

If you see: `ImportError: You must install tqdm and rich`

**Solution**: 
```bash
pip install tqdm rich
# Or
pip install stable-baselines3[extra]
```

### No Results Found

If `view_rl_results.py` shows no results:

1. Run `quick_rl_test.py` first to generate initial results
2. Check that `results/` directory exists
3. Verify you have write permissions

## Next Steps

After viewing results:

1. **If results are good**: Consider running full evaluation for comprehensive analysis
2. **If results need improvement**: 
 - Increase training timesteps in `src/rl/integration.py`
 - Adjust reward function in `rl_environment.py`
 - Tune hyperparameters

3. **For production use**:
 - Run full evaluation on longer test period
 - Compare with baseline strategies
 - Perform ablation studies

## Command Summary

```bash
# Quick test (recommended first)
python scripts/quick_rl_test.py

# View latest results
python scripts/view_rl_results.py

# Full evaluation (when ready)
python scripts/run_full_rl_evaluation.py

# Verify connection (one-time)
python scripts/test_rl_agent_connection.py
```

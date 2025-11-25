# Dynamic Visualization Guide

## Overview

Two dynamic visualization tools have been created to show real-time performance during RL training and testing.

## Tool 1: Live Backtest Visualization

### Purpose
Shows real-time performance during backtesting with live updates.

### Features
- **4 Real-time Panels**:
  1. Portfolio Value Over Time
  2. Cumulative Returns (%)
  3. Reward Per Step
  4. Drawdown Analysis

- **Live Metrics Display**:
  - Current step
  - Portfolio value
  - Total return percentage
  - Average reward
  - Maximum drawdown

### Usage
```bash
python scripts/dynamic_rl_visualization.py
```

### What Happens
1. Loads GNN model and agent
2. Opens visualization window
3. Runs backtest with live updates
4. Updates plots every 100ms
5. Shows final metrics when complete

### Controls
- **Close window** to stop backtest early
- Plots update automatically
- No interaction needed during run

## Tool 2: Live Training Monitor

### Purpose
Monitors RL agent training in real-time.

### Features
- **2 Real-time Panels**:
  1. Episode Rewards (individual + 10-episode average)
  2. Reward Over Timesteps

- **Training Statistics**:
  - Episode count
  - Latest reward
  - Average reward
  - Total timesteps

### Usage
```bash
python scripts/live_training_monitor.py
```

### What Happens
1. Loads GNN model
2. Opens visualization window
3. Starts training with live monitoring
4. Updates plots every 500ms
5. Shows training progress

### Controls
- **Close window** to stop training
- Watch rewards improve over time
- See training convergence

## Comparison

| Feature | Live Backtest | Training Monitor |
|---------|--------------|------------------|
| **Purpose** | Test trained agent | Monitor training |
| **Updates** | Every 100ms | Every 500ms |
| **Panels** | 4 panels | 2 panels |
| **Metrics** | Performance | Training progress |
| **Use Case** | After training | During training |

## Quick Start Examples

### Example 1: Watch Backtest Live
```bash
# Run live backtest visualization
python scripts/dynamic_rl_visualization.py

# You'll see:
# - Portfolio value updating in real-time
# - Returns percentage changing
# - Rewards per step
# - Drawdown analysis
```

### Example 2: Monitor Training
```bash
# Run training with live monitor
python scripts/live_training_monitor.py

# You'll see:
# - Episode rewards improving
# - Average reward trending up
# - Training statistics
```

## Technical Details

### Update Frequency
- **Backtest**: 100ms intervals (smooth updates)
- **Training**: 500ms intervals (less frequent, more stable)

### Data Collection
- Uses `deque` for efficient data storage
- Limits to last 1000 points for performance
- Real-time calculation of metrics

### Visualization Library
- **matplotlib** with `FuncAnimation`
- Non-blocking display
- Responsive to window close events

## Tips

1. **For Best Results**:
   - Use live backtest after training is complete
   - Use training monitor during initial training
   - Close window if you want to stop early

2. **Performance**:
   - Updates are lightweight
   - Can handle 1000+ steps smoothly
   - Memory efficient with deque

3. **Troubleshooting**:
   - If window doesn't appear: Check matplotlib backend
   - If updates are slow: Reduce max_steps parameter
   - If plots are empty: Wait for first data point

## Integration with Other Tools

These tools work alongside:
- `quick_rl_test.py` - Quick static test
- `run_full_rl_evaluation.py` - Full evaluation
- `view_rl_results.py` - View saved results

## Next Steps

1. **First Time**: Try live backtest to see current performance
2. **Training**: Use training monitor to watch agent learn
3. **Analysis**: Use static tools for detailed analysis

## Command Summary

```bash
# Dynamic visualization
python scripts/dynamic_rl_visualization.py    # Live backtest
python scripts/live_training_monitor.py        # Training monitor

# Static analysis
python scripts/quick_rl_test.py               # Quick test
python scripts/view_rl_results.py               # View results
python scripts/run_full_rl_evaluation.py       # Full evaluation
```


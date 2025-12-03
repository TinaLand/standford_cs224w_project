# src/evaluation Module Guide

## Overview

The `src/evaluation/` module contains **core evaluation and analysis functionality** for the project. It provides comprehensive metrics calculation, ablation studies, visualizations, and advanced analysis tools.

## Module Structure

```
src/evaluation/
├── evaluation.py              # Core evaluation metrics
├── ablation.py                # Ablation studies
├── visualization.py           # Visualization utilities
├── enhanced_evaluation.py     # Enhanced evaluation with RL
├── deep_analysis.py           # Deep analysis tools
└── enhancements/              # Advanced analysis modules
    ├── failure_analysis.py
    ├── sensitivity_analysis.py
    ├── edge_importance.py
    ├── multi_agent_analysis.py
    ├── cross_period_validation.py
    └── enhanced_ablation.py
```

## Core Modules

### 1. `evaluation.py` - Core Evaluation Metrics

**Purpose**: Main evaluation module for calculating performance metrics

**Key Functions**:
- `calculate_precision_at_topk()` - Precision@Top-K metric
- `calculate_information_coefficient()` - Information Coefficient (IC)
- `evaluate_model()` - Comprehensive model evaluation
- `calculate_financial_metrics()` - Portfolio-level metrics (Sharpe, Drawdown, etc.)

**Usage**:
```python
from src.evaluation.evaluation import evaluate_model, calculate_precision_at_topk

# Evaluate model on test set
metrics = evaluate_model(model, test_data)
precision_top10 = calculate_precision_at_topk(predictions, targets, k=10)
```

**Metrics Calculated**:
- Node-level: Accuracy, F1, Precision, Recall, Precision@Top-K
- Portfolio-level: Sharpe Ratio, Max Drawdown, Cumulative Return
- Ranking: Information Coefficient (IC)

### 2. `ablation.py` - Ablation Studies

**Purpose**: Run ablation studies to understand component contributions

**Key Functions**:
- `run_ablation_study()` - Main ablation study runner
- `evaluate_ablation_config()` - Evaluate specific configuration
- `compare_ablation_results()` - Compare different configurations

**Usage**:
```python
from src.evaluation.ablation import run_ablation_study

# Run ablation study
results = run_ablation_study(
    configs=['full', 'no_pearl', 'single_edge', 'no_time_aware'],
    train_dates=train_dates,
    val_dates=val_dates,
    test_dates=test_dates
)
```

**Ablation Configurations**:
- Full model (baseline)
- No PEARL embeddings
- Single edge type
- No time-aware encoding
- GAT baseline

### 3. `visualization.py` - Visualization Utilities

**Purpose**: Generate visualizations for analysis and reports

**Key Functions**:
- `plot_training_curves()` - Training/validation loss curves
- `plot_attention_heatmap()` - Attention weight visualizations
- `plot_portfolio_performance()` - Portfolio value over time
- `plot_ablation_results()` - Ablation study visualizations

**Usage**:
```python
from src.evaluation.visualization import plot_training_curves, plot_attention_heatmap

# Plot training curves
plot_training_curves(train_losses, val_losses, save_path='plots/training.png')

# Plot attention heatmap
plot_attention_heatmap(attention_weights, save_path='plots/attention.png')
```

### 4. `enhanced_evaluation.py` - Enhanced Evaluation with RL

**Purpose**: Evaluation that integrates RL agent performance

**Key Functions**:
- `evaluate_rl_agent()` - Evaluate RL agent performance
- `compare_strategies()` - Compare different trading strategies
- `calculate_portfolio_metrics()` - Portfolio-level metrics

**Usage**:
```python
from src.evaluation.enhanced_evaluation import evaluate_rl_agent

# Evaluate RL agent
metrics = evaluate_rl_agent(agent, test_env, n_episodes=10)
```

### 5. `deep_analysis.py` - Deep Analysis Tools

**Purpose**: Advanced analysis and insights

**Key Functions**:
- `analyze_prediction_errors()` - Error pattern analysis
- `analyze_feature_importance()` - Feature importance analysis
- `analyze_temporal_patterns()` - Temporal pattern analysis

## Enhancement Modules (`enhancements/`)

### `failure_analysis.py`
- Analyze worst-performing periods
- Error pattern identification
- Drawdown analysis

### `sensitivity_analysis.py`
- Transaction cost sensitivity
- Slippage sensitivity
- Parameter sensitivity

### `edge_importance.py`
- Edge type importance analysis
- Sector subgraph analysis
- Correlation vs fundamental comparison

### `multi_agent_analysis.py`
- Multi-agent decision analysis
- Sector performance analysis
- Mixing network weight analysis

### `cross_period_validation.py`
- Cross-period validation
- Temporal stability analysis

### `enhanced_ablation.py`
- Enhanced ablation studies
- Component interaction analysis

## Integration with Main Pipeline

### Phase 6: Evaluation

The main pipeline (`run_full_pipeline.py`) uses `src/evaluation` in Phase 6:

```python
# Phase 6: Evaluation
from src.evaluation.evaluation import evaluate_model
from src.evaluation.ablation import run_ablation_study

# Evaluate model
metrics = evaluate_model(model, test_data)

# Run ablation study
ablation_results = run_ablation_study(...)
```

## Key Differences from `scripts/`

| Aspect | `src/evaluation/` | `scripts/` |
|--------|------------------|-----------|
| **Purpose** | Core evaluation modules, reusable | Standalone utility scripts |
| **Structure** | Organized Python modules | Flat script files |
| **Import** | Imported as modules | Run directly |
| **Reusability** | High - used across project | Low - one-off scripts |
| **Main Pipeline** | Part of main pipeline | Called separately |

## Usage Examples

### Example 1: Basic Model Evaluation

```python
from src.evaluation.evaluation import evaluate_model
from src.utils.graph_loader import load_graph_data

# Load test data
test_graph = load_graph_data(test_date)

# Evaluate model
metrics = evaluate_model(model, test_graph)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1']:.4f}")
```

### Example 2: Ablation Study

```python
from src.evaluation.ablation import run_ablation_study

# Run ablation study
results = run_ablation_study(
    configs=['full', 'no_pearl', 'single_edge'],
    train_dates=train_dates,
    test_dates=test_dates
)

# Compare results
for config, metrics in results.items():
    print(f"{config}: Accuracy={metrics['accuracy']:.4f}")
```

### Example 3: Visualization

```python
from src.evaluation.visualization import plot_training_curves

# Plot training curves
plot_training_curves(
    train_losses=history['train_loss'],
    val_losses=history['val_loss'],
    save_path='results/plots/training_curves.png'
)
```

## Summary

The `src/evaluation/` module is the **core evaluation engine** of the project:

1. **Metrics Calculation** - Comprehensive performance metrics
2. **Ablation Studies** - Component contribution analysis
3. **Visualization** - Analysis plots and charts
4. **Advanced Analysis** - Deep insights and patterns
5. **RL Integration** - RL agent evaluation

It's a **reusable, modular** component that's integrated into the main pipeline, unlike `scripts/` which contains standalone utility scripts.

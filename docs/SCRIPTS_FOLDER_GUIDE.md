# Scripts Folder Guide

## Overview

The `scripts/` folder contains **standalone utility scripts** for:
1. **Figure Generation** - Creating visualizations for reports
2. **Experiments** - Running research experiments and analysis
3. **Evaluation** - Running baseline comparisons and ablation studies

## Script Categories

### 1. Figure Generation Scripts

#### `generate_report_figures.py` (84KB)
- **Purpose**: Generate all main figures for the final report
- **Output**: Creates figures in `figures/` directory
- **Usage**: `python scripts/generate_report_figures.py`
- **Figures Generated**:
  - Figure 1: System Architecture
  - Figure 2: Training Curves
  - Figure 3: Model Comparison
  - Figure 4: Portfolio Performance
  - Figure 5: Ablation Study
  - Figure 6: Attention Heatmap
  - Figure 7a-7e: Graph Structure Visualizations
  - Figure 8: Regime Performance
  - Figure 9: GNN Architecture

#### `create_additional_figures.py` (13KB)
- **Purpose**: Generate additional analysis figures
- **Output**: IC analysis, PEARL visualization, Precision@Top-K, MARL decision flow
- **Usage**: `python scripts/create_additional_figures.py`

#### `check_and_generate_all_figures.py` (5.5KB)
- **Purpose**: Verify and generate all missing figures
- **Usage**: `python scripts/check_and_generate_all_figures.py`

### 2. Experiment Scripts

#### `experiment_lookahead_horizons.py` (11KB)
- **Purpose**: Test different prediction horizons (1, 3, 5, 7, 10 days)
- **Usage**: `python scripts/experiment_lookahead_horizons.py`
- **Output**: Results in `results/experiment_lookahead_horizons.csv`

#### `experiment_graph_sparsification.py` (9.4KB)
- **Purpose**: Evaluate Top-K thresholds and correlation cutoffs
- **Usage**: `python scripts/experiment_graph_sparsification.py`
- **Output**: Results in `results/experiment_graph_sparsification.csv`

#### `experiment_robustness_checks.py` (10KB)
- **Purpose**: Test transaction cost and slippage sensitivity
- **Usage**: `python scripts/experiment_robustness_checks.py`
- **Output**: Results in `results/experiment_robustness_checks.csv`

### 3. Evaluation Scripts

#### `run_baseline_comparison.py` (1.1KB)
- **Purpose**: Compare GCN, GAT, GraphSAGE, HGT, Logistic Regression, MLP, LSTM
- **Usage**: `python scripts/run_baseline_comparison.py`
- **Output**: Results in `results/baseline_model_comparison.csv`

#### `run_improved_ablation.py` (1.1KB)
- **Purpose**: Run improved ablation studies with full retraining
- **Usage**: `python scripts/run_improved_ablation.py`
- **Output**: Results in `results/ablation_results.csv`

#### `run_marl_ablation.py` (3.8KB)
- **Purpose**: Run Multi-Agent RL ablation studies
- **Usage**: `python scripts/run_marl_ablation.py`
- **Output**: Results in `results/marl_ablation_results.csv`

#### `analyze_ic_deep.py` (7.0KB)
- **Purpose**: Deep analysis of Information Coefficient (IC)
- **Usage**: `python scripts/analyze_ic_deep.py`
- **Output**: IC analysis plots and statistics

## Key Differences from `src/` Folder

| Aspect | `scripts/` | `src/` |
|--------|------------|--------|
| **Purpose** | Standalone utilities, experiments, visualizations | Core modules, reusable components |
| **Structure** | Flat structure, independent scripts | Organized modules (`src/training/`, `src/evaluation/`, etc.) |
| **Dependencies** | May import from `src/` | Self-contained modules |
| **Usage** | Run directly: `python scripts/script_name.py` | Import as modules: `from src.training import ...` |
| **Main Pipeline** | Not part of main pipeline | Part of main pipeline (`run_full_pipeline.py`) |

## When to Use Scripts

### Use `scripts/` for:
-  Generating figures for reports
-  Running one-off experiments
-  Quick analysis and visualization
-  Testing specific configurations

### Use `src/` modules for:
-  Core functionality (training, evaluation)
-  Reusable components
-  Main pipeline execution
-  Production code

## Integration with Main Pipeline

The main pipeline (`run_full_pipeline.py`) uses `src/` modules, not `scripts/`:
- Phase 1-6: All use `src/` modules
- Scripts are called separately for:
  - Figure generation (after pipeline completes)
  - Additional experiments (optional)
  - Deep analysis (optional)

## Summary

The `scripts/` folder is a **utility workspace** for:
1. **Visualization** - Creating report figures
2. **Experimentation** - Running research experiments
3. **Analysis** - Deep dives into specific metrics

It complements the main `src/` codebase by providing standalone tools for analysis and visualization.

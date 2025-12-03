# Scripts Folder - Usage Guide

## Overview

The `scripts/` folder contains **standalone utility scripts** that can be run individually or in batch.

## Running Scripts

### Option 1: Run All Scripts at Once (Recommended)

Use the unified runner to execute all scripts:

```bash
python scripts/run_all_scripts.py
```

This will give you options to:
- Run all scripts
- Run figure generation scripts only
- Run experiment scripts only
- Run evaluation scripts only
- Select individual scripts

### Option 2: Run Individual Scripts

Each script can be run directly:

#### Figure Generation
```bash
# Generate all main report figures
python scripts/generate_report_figures.py

# Generate additional analysis figures
python scripts/create_additional_figures.py

# Check and generate missing figures
python scripts/check_and_generate_all_figures.py
```

#### Experiments
```bash
# Test different prediction horizons
python scripts/experiment_lookahead_horizons.py

# Evaluate graph sparsification strategies
python scripts/experiment_graph_sparsification.py

# Test transaction cost and slippage sensitivity
python scripts/experiment_robustness_checks.py
```

#### Evaluation
```bash
# Compare baseline models (GCN, GAT, GraphSAGE, etc.)
python scripts/run_baseline_comparison.py

# Run improved ablation studies
python scripts/run_improved_ablation.py

# Run Multi-Agent RL ablation studies
python scripts/run_marl_ablation.py

# Deep IC analysis
python scripts/analyze_ic_deep.py
```

## Running Evaluation Modules

### Option 1: Run All Evaluation Modules at Once

```bash
python scripts/run_all_evaluation.py
```

This will give you options to:
- Run all modules
- Run core evaluation modules only
- Run enhancement modules only
- Select individual modules

### Option 2: Run Individual Modules

Evaluation modules in `src/evaluation/` can be run as Python modules:

```bash
# Main evaluation pipeline (Phase 6)
python -m src.evaluation.evaluation

# Ablation studies
python -m src.evaluation.ablation

# Enhanced evaluation
python -m src.evaluation.enhanced_evaluation

# Enhanced ablation (with retraining)
python -m src.evaluation.enhanced_ablation

# Deep analysis
python -m src.evaluation.deep_analysis

# Statistical tests
python -m src.evaluation.statistical_tests

# Visualizations
python -m src.evaluation.visualization
```

#### Enhancement Modules
```bash
# Edge importance analysis
python -m src.evaluation.enhancements.edge_importance

# Cross-period validation
python -m src.evaluation.enhancements.cross_period_validation

# Multi-agent RL analysis
python -m src.evaluation.enhancements.multi_agent_analysis

# Sensitivity analysis
python -m src.evaluation.enhancements.sensitivity_analysis

# Failure analysis
python -m src.evaluation.enhancements.failure_analysis
```

## Script Categories

### 1. Figure Generation Scripts
- **Purpose**: Create visualizations for reports
- **Output**: Figures in `figures/` directory
- **Runtime**: ~5-30 minutes depending on script

### 2. Experiment Scripts
- **Purpose**: Run research experiments and analysis
- **Output**: Results in `results/` directory (CSV files)
- **Runtime**: ~1-2 hours per script

### 3. Evaluation Scripts
- **Purpose**: Run baseline comparisons and ablation studies
- **Output**: Results in `results/` directory
- **Runtime**: ~1-4 hours per script (some require retraining)

## Quick Reference

| Script | Purpose | Runtime | Output |
|--------|---------|---------|--------|
| `run_all_scripts.py` | Run all scripts in batch | Varies | All outputs |
| `generate_report_figures.py` | Generate main figures | ~30 min | `figures/` |
| `run_baseline_comparison.py` | Compare baselines | ~1-2 hours | `results/baseline_model_comparison.csv` |
| `run_improved_ablation.py` | Ablation studies | ~2-3 hours | `results/ablation_results.csv` |
| `analyze_ic_deep.py` | IC analysis | ~10 min | IC plots and stats |

## Notes

- **Individual scripts**: Each script can be run independently
- **Batch runner**: Use `run_all_scripts.py` for convenience
- **Evaluation modules**: Use `run_all_evaluation.py` or run as modules
- **Dependencies**: Most scripts require trained models from previous phases
- **Order matters**: Some scripts depend on outputs from others

## Integration with Main Pipeline

The main pipeline (`run_full_pipeline.py` in project root) automatically calls some scripts after Phase 6:
- `scripts/generate_report_figures.py`
- `scripts/analyze_ic_deep.py`
- `scripts/create_additional_figures.py`

Other scripts are optional and can be run separately for additional analysis.


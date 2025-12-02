# Figures Checklist for A+ Grade

This document lists all required figures for the A+ grade and their current status.

## Core Figures (5 Required)

| # | Figure Name | File | Status | Description |
|---|-------------|------|--------|-------------|
| 1 | System Architecture | `figure1_system_architecture.png` | ✅ | High-level system architecture showing data flow from raw data → GNN → MARL → Portfolio |
| 4 | Portfolio Performance | `figure4_portfolio_performance.png` | ✅ | Cumulative portfolio value with baseline comparisons (MARL vs Single-Agent vs Equal-Weight) |
| 5 | Ablation Study | `figure5_ablation_study.png` | ✅ | Dual metrics (Precision@Top-10 and Sharpe Ratio) showing progressive component contributions |
| 9 | GNN Architecture | `figure9_gnn_architecture.png` | ✅ | GNN message passing mechanism with PEARL embeddings and multi-relational attention |
| 10 | Multi-Task Loss | `figure10_multitask_loss.png` | ✅ | Multi-task learning loss structure (Focal Loss + MSE Loss) |

## Supporting Figures (5 Recommended)

| # | Figure Name | File | Status | Description |
|---|-------------|------|--------|-------------|
| 6 | Attention Heatmap | `figure6_attention_heatmap.png` | ✅ | GNN attention weights by edge type and stock (explainability) |
| 8 | Regime Performance | `figure8_regime_performance.png` | ✅ | Model performance across different market regimes (bull, volatile, bear) |
| 11 | IC Analysis | `figure_ic_analysis.png` | ✅ | IC time series and distribution analysis (explaining negative IC) |
| 12 | PEARL Visualization | `figure_pearl_embedding_visualization.png` | ✅ | t-SNE visualization of PEARL embeddings showing structural roles |
| 13 | Precision@Top-K | `figure_precision_topk_curve.png` | ✅ | Precision@Top-K curve from K=1 to K=20 |
| 14 | MARL Decision Flow | `figure_marl_decision_flow.png` | ✅ | Multi-Agent RL decision flow diagram (CTDE + QMIX) |

## Additional Figures

| # | Figure Name | File | Status | Description |
|---|-------------|------|--------|-------------|
| 2 | Training Curves | `figure2_training_curves.png` | ✅ | Training and validation curves |
| 3 | Model Comparison | `figure3_model_comparison.png` | ✅ | Comprehensive model comparison (GCN, GAT, GraphSAGE, HGT, etc.) |
| 7a-7e | Graph Structure | `figure7a-7e_*.png` | ✅ | Detailed graph structure visualizations (5 sub-figures) |

## Figure Generation Commands

### Generate All Main Figures
```bash
python scripts/generate_report_figures.py
```

### Generate Additional Figures (IC, PEARL, Precision@Top-K, MARL)
```bash
python scripts/create_additional_figures.py
```

### Generate IC Analysis
```bash
python scripts/analyze_ic_deep.py
```

### Check All Figures
```bash
python scripts/check_and_generate_all_figures.py
```

## Figure Requirements for A+ Grade

### ✅ All Core Figures (5/5)
- [x] System Architecture Diagram
- [x] Portfolio Performance with Baselines
- [x] Ablation Study (Dual Metrics)
- [x] GNN Architecture & Message Passing
- [x] Multi-Task Loss Structure

### ✅ All Supporting Figures (6/6)
- [x] Attention Heatmap
- [x] Regime Performance
- [x] IC Analysis
- [x] PEARL Embedding Visualization
- [x] Precision@Top-K Curve
- [x] MARL Decision Flow

### Key Improvements Made

1. **Ablation Study (Figure 5)**: 
   - Now shows dual metrics (Precision@Top-10 and Sharpe Ratio)
   - Demonstrates progressive component contributions
   - Clear differentiation between configurations

2. **Portfolio Performance (Figure 4)**:
   - Includes multiple baseline comparisons (MARL vs Single-Agent vs Equal-Weight)
   - Shows key metrics (Sharpe Ratio, Max Drawdown, Cumulative Return)
   - Highlights Max Drawdown period

3. **All Additional Figures**:
   - IC Analysis: Explains negative IC with time series and distribution
   - PEARL Visualization: t-SNE showing structural roles
   - Precision@Top-K Curve: Demonstrates ranking capability
   - MARL Decision Flow: Illustrates CTDE + QMIX architecture

## Status: ✅ All Figures Complete

All 14 required figures for A+ grade are generated and referenced in `FINAL_REPORT.md`.


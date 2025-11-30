# Documentation Index

Welcome to the CS224W Stock RL GNN Project documentation. This directory contains comprehensive documentation for all aspects of the project.

## Quick Links

- **[Main README](../README.md)** - Project overview and quick start guide
- **[Final Report](../FINAL_REPORT.md)** - Complete project report (Blog post format)
- **[Colab Notebook](../CS224W_Project_Colab.ipynb)** - Full implementation in Colab format

---

## Phase Documentation

Detailed documentation for each phase of the project:

| Phase | Document | Description |
|-------|----------|-------------|
| **Phase 1** | [Data Collection & Feature Engineering](phases/phase1_data_collection.md) | Raw data acquisition, feature computation, edge parameter calculation |
| **Phase 2** | [Graph Construction](phases/phase2_graph_construction.md) | Heterogeneous graph construction with 4 edge types |
| **Phase 3** | [Baseline Training](phases/phase3_baseline_training.md) | Baseline GAT model training with Focal Loss |
| **Phase 4** | [Transformer Training](phases/phase4_transformer_training.md) | Role-Aware Graph Transformer with PEARL embeddings |
| **Phase 5** | [RL Integration](phases/phase5_rl_integration.md) | PPO-based reinforcement learning for portfolio management |
| **Phase 6** | [Evaluation](phases/phase6_evaluation.md) | Comprehensive evaluation metrics and analysis |
| **Phase 7** | [Optimization](phases/phase7_optimization.md) | Multi-Agent RL and dynamic graph updates |

---

## Guides

How-to guides and tutorials:

| Guide | Description |
|-------|-------------|
| [RL Agent Guide](guides/rl_agent_guide.md) | Understanding the RL agent architecture and workflow |
| [RL Usage Guide](guides/rl_usage.md) | How to use the RL agent for trading |
| [RL Testing Guide](guides/RL_TESTING_GUIDE.md) | Testing and debugging RL components |
| [Testing Guide](guides/TESTING_GUIDE.md) | General testing and validation procedures |

---

## Analysis Reports

Performance and analysis reports:

| Report | Description |
|--------|-------------|
| [Performance Evaluation](analysis/PERFORMANCE_EVALUATION.md) | Detailed performance analysis and metrics |
| [GNN Prediction Analysis](analysis/GNN_PREDICTION_ANALYSIS.md) | Analysis of GNN model predictions |
| [Project Score Evaluation](analysis/PROJECT_SCORE_EVALUATION.md) | Overall project grade evaluation |
| [Complexity Evaluation](analysis/PROJECT_COMPLEXITY_EVALUATION.md) | Project complexity and technical depth analysis |

---

## Implementation Details

Technical implementation documentation:

| Document | Description |
|----------|-------------|
| [PEARL Implementation](implementation/PEARL_IMPLEMENTATION_STATUS.md) | PEARL positional embedding implementation details |
| [Multi-Agent RL Explanation](implementation/MULTI_AGENT_RL_EXPLANATION.md) | Multi-Agent RL architecture and design |
| [Multi-Agent RL Implementation](implementation/MULTI_AGENT_RL_IMPLEMENTATION.md) | Implementation details for Multi-Agent RL |
| [Phase 6 & 7 Implementation](implementation/PHASE6_PHASE7_IMPLEMENTATION.md) | Implementation status for Phase 6 and 7 |
| [Proposal Implementation Checklist](implementation/PROPOSAL_IMPLEMENTATION_CHECKLIST.md) | Checklist of proposal requirements vs. implementation |

---

## Documentation Structure

```
docs/
 README.md                    # This file - documentation index

 phases/                      # Phase-by-phase documentation
    phase1_data_collection.md
    phase2_graph_construction.md
    phase3_baseline_training.md
    phase4_transformer_training.md
    phase5_rl_integration.md
    phase6_evaluation.md
    phase7_optimization.md

 guides/                      # How-to guides
    rl_agent_guide.md
    rl_usage.md
    RL_TESTING_GUIDE.md
    TESTING_GUIDE.md

 analysis/                    # Analysis reports
    PERFORMANCE_EVALUATION.md
    GNN_PREDICTION_ANALYSIS.md
    PROJECT_SCORE_EVALUATION.md
    PROJECT_COMPLEXITY_EVALUATION.md

 implementation/              # Technical implementation details
     PEARL_IMPLEMENTATION_STATUS.md
     MULTI_AGENT_RL_EXPLANATION.md
     MULTI_AGENT_RL_IMPLEMENTATION.md
     PHASE6_PHASE7_IMPLEMENTATION.md
     PROPOSAL_IMPLEMENTATION_CHECKLIST.md
```

---

## Key Project Components

### Model Architecture
- **Role-Aware Graph Transformer**: Multi-relational graph transformer with PEARL embeddings
- **PEARL Positional Embeddings**: Structural role encoding (hubs, bridges, isolated nodes)
- **Time-Aware Encoding**: Temporal pattern capture
- **Multi-Relational Attention**: Different aggregation strategies for different edge types

### Reinforcement Learning Agents
- **Single-Agent RL**: PPO-based portfolio optimization using GNN predictions
  - See [RL Agent Guide](guides/rl_agent_guide.md) for detailed architecture
  - See [RL Usage Guide](guides/rl_usage.md) for usage instructions
- **Multi-Agent RL**: Sector-specialized agents with CTDE architecture and QMIX-style mixing network
  - See [Multi-Agent RL Explanation](implementation/MULTI_AGENT_RL_EXPLANATION.md) for architecture details
  - See [Multi-Agent RL Implementation](implementation/MULTI_AGENT_RL_IMPLEMENTATION.md) for implementation guide

### Data & Graphs
- **50 Major US Stocks**: Diverse sectors (Technology, Finance, Healthcare, etc.)
- **10 Years of Data**: 2015-2024 (~2,500 trading days)
- **Heterogeneous Graphs**: 4 edge types (correlation, fundamental, sector, supply chain)
- **1,450+ Features**: Technical indicators, fundamentals, sentiment, macro

### Results
- **Node-Level**: Accuracy 54.62%, F1 35.33%, Precision@Top-10 55.23%
- **Portfolio-Level**: Sharpe Ratio 1.90, Cumulative Return 45.99%, Max Drawdown 6.62%

---

## Documentation Standards

All documentation follows these standards:
- **Language**: English (except where explicitly noted)
- **Format**: Markdown (.md)
- **Structure**: Clear headings, code blocks, tables
- **Code Examples**: Complete, runnable code snippets
- **Mathematical Notation**: LaTeX-style formulas where appropriate

---

## Finding What You Need

### For New Users
1. Start with [Main README](../README.md) for project overview
2. Read [Phase 1 Documentation](phases/phase1_data_collection.md) to understand data collection
3. Follow phase-by-phase guides in `phases/` directory

### For Developers
1. Check [Implementation Details](implementation/) for technical deep-dives
2. Review [Guides](guides/) for specific how-to instructions
3. See [Analysis Reports](analysis/) for performance insights

### For Researchers
1. Read [Final Report](../FINAL_REPORT.md) for complete methodology
2. Review [Analysis Reports](analysis/) for detailed results
3. Check [Proposal Implementation Checklist](implementation/PROPOSAL_IMPLEMENTATION_CHECKLIST.md) for requirements

---

## Contributing to Documentation

When adding or updating documentation:
1. Follow the existing structure and format
2. Use clear, concise language
3. Include code examples where relevant
4. Update this README.md if adding new documents
5. Ensure all links are working

---

## Related Resources

- **Course**: CS224W - Machine Learning with Graphs (Stanford)
- **Project Repository**: See main [README](../README.md) for repository information
- **External References**: See [FINAL_REPORT.md](../FINAL_REPORT.md) for complete reference list

---

*Last Updated: 2024-12-29*

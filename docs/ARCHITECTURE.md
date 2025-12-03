# Project Architecture Documentation

## Overview

This document provides a high-level overview of the project architecture, including module structure, data flow, and key components.

## Module Structure

### Core Modules

```
src/
├── utils/                    # Centralized utilities
│   ├── paths.py             # Path configuration
│   ├── graph_loader.py      # Graph loading utilities
│   └── constants.py         # Configuration constants
│
├── data/                     # Data processing
│   ├── collection.py        # Data collection
│   ├── feature_engineering.py  # Feature computation
│   ├── edge_parameters.py   # Edge parameter calculation
│   └── graph_construction.py  # Graph construction
│
├── models/                    # Model definitions
│   └── components/          # Model components
│       ├── pearl_embedding.py
│       └── transformer_layer.py
│
├── training/                  # Training scripts
│   ├── baseline_trainer.py
│   ├── transformer_trainer.py
│   └── baseline_comparison.py
│
├── rl/                       # Reinforcement Learning
│   ├── agents/              # RL agents
│   │   ├── base.py
│   │   ├── single_agent.py
│   │   └── sector_agent.py
│   ├── environments/        # Trading environments
│   │   ├── base.py
│   │   ├── single_agent.py
│   │   └── multi_agent.py
│   ├── coordination/        # Multi-agent coordination
│   │   ├── coordinator.py
│   │   └── mixing.py
│   └── training/            # RL training
│       ├── single_agent.py
│       └── multi_agent.py
│
└── evaluation/               # Evaluation and analysis
    ├── evaluation.py
    ├── ablation.py
    ├── visualization.py
    └── enhancements/       # Advanced analysis
```

## Data Flow

### Phase 1: Data Collection & Feature Engineering
```
Raw Data (yfinance) 
  → OHLCV Data 
  → Technical Indicators 
  → Feature Matrix
```

### Phase 2: Graph Construction
```
Feature Matrix + Edge Parameters 
  → Heterogeneous Graph (4 edge types)
  → Graph Snapshots (per trading day)
```

### Phase 3-4: Model Training
```
Graph Snapshots 
  → GNN Model (PEARL + Transformer)
  → Predictions (Classification + Regression)
```

### Phase 5: RL Integration
```
GNN Predictions 
  → RL Environment 
  → PPO Agent 
  → Portfolio Actions
```

### Phase 6: Evaluation
```
Model Predictions + Portfolio Actions 
  → Metrics (Accuracy, F1, Sharpe, etc.)
  → Visualizations
```

## Key Components

### 1. Path Configuration (`src/utils/paths.py`)
Centralized path management for all project directories and files.

**Usage**:
```python
from src.utils.paths import PROJECT_ROOT, DATA_GRAPHS_DIR, MODELS_DIR
```

### 2. Graph Loading (`src/utils/graph_loader.py`)
Unified graph data loading utilities.

**Usage**:
```python
from src.utils.graph_loader import load_graph_data
data = load_graph_data(date)
```

### 3. Configuration Constants (`src/utils/constants.py`)
Centralized hyperparameters and configuration values.

**Usage**:
```python
from src.utils.constants import DEVICE, HIDDEN_CHANNELS, LEARNING_RATE
```

### 4. Model Architecture
- **Role-Aware Graph Transformer**: Multi-relational graph transformer
- **PEARL Embeddings**: Structural role encoding
- **Time-Aware Encoding**: Temporal pattern capture

### 5. RL Components
- **Single-Agent RL**: PPO-based portfolio optimization
- **Multi-Agent RL**: Sector-specialized agents with CTDE architecture

## Import Patterns

### Standard Imports
```python
# Paths
from src.utils.paths import PROJECT_ROOT, DATA_GRAPHS_DIR

# Graph Loading
from src.utils.graph_loader import load_graph_data

# Constants
from src.utils.constants import DEVICE, HIDDEN_CHANNELS

# Models
from src.training.transformer_trainer import RoleAwareGraphTransformer

# RL
from src.rl.environments.single_agent import StockTradingEnv
from src.rl.agents.single_agent import StockTradingAgent
```

## File Organization

### Data Files
- `data/raw/` - Raw OHLCV data
- `data/processed/` - Processed features
- `data/edges/` - Edge parameters
- `data/graphs/` - Graph snapshots

### Model Files
- `models/` - Trained model weights
- `models/checkpoints/` - Training checkpoints

### Results
- `results/` - Evaluation results and metrics
- `results/plots/` - Generated plots

### Figures
- `figures/` - All report figures

## Best Practices

1. **Always use centralized paths**: Import from `src.utils.paths`
2. **Use graph loader**: Import from `src.utils.graph_loader`
3. **Use constants**: Import from `src.utils.constants`
4. **Follow import patterns**: Use proper module paths
5. **Document changes**: Update relevant documentation

## Migration Notes

### Old Patterns (Deprecated)
```python
# ❌ Old
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
from phase4_core_training import load_graph_data
from rl_environment import StockTradingEnv
```

### New Patterns (Current)
```python
# ✅ New
from src.utils.paths import PROJECT_ROOT
from src.utils.graph_loader import load_graph_data
from src.rl.environments.single_agent import StockTradingEnv
```

---

*For detailed implementation guides, see [Implementation Documentation](implementation/)*


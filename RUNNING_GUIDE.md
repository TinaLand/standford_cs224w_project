# Complete Running Guide

Complete guide to run the entire CS224W Stock RL GNN project from scratch.

---

## Prerequisites

1. **Python 3.8+** installed
2. **Virtual environment** (recommended)
3. **Dependencies** installed

### Install Dependencies

```bash
cd cs224_porject
pip install -r requirements.txt
```

### Verify Data Files

Ensure you have:
- `data/raw/stock_prices_ohlcv_raw.csv` (or run Phase 1 to generate)
- `data/raw/static_sector_industry.csv`

---

## Running the Full Pipeline

### Option 1: Run All Phases at Once (Recommended)

```bash
python run_full_pipeline.py
```

This will execute all phases in sequence:
1. Phase 1: Data Collection & Feature Engineering
2. Phase 2: Graph Construction
3. Phase 3: Baseline Training
4. Phase 4: Transformer Training
5. Phase 5: RL Integration
6. Phase 6: Evaluation

**Expected Time**: ~2-4 hours (depending on hardware)

---

### Option 2: Run Phases Individually

#### Phase 1: Data Collection & Feature Engineering

```bash
# Step 1: Download raw stock data
python -m src.data.collection

# Step 2: Calculate technical indicators and normalize features
python -m src.data.feature_engineering

# Step 3: Calculate edge parameters (correlations, similarities)
python -m src.data.edge_parameters
```

**Outputs**:
- `data/raw/stock_prices_ohlcv_raw.csv`
- `data/processed/node_features_X_t_final.csv`
- `data/edges/edges_dynamic_corr_params.pkl`
- `data/edges/edges_dynamic_fund_sim_params.csv`

---

#### Phase 2: Graph Construction

```bash
python -m src.data.graph_construction
```

**Outputs**:
- `data/graphs/graph_t_YYYYMMDD.pt` (one per trading day)
- Graph objects: PyG `HeteroData` with:
  - Node type: `'stock'`
  - Edge types: `'rolling_correlation'`, `'fund_similarity'`, `'sector_industry'`

---

#### Phase 3: Baseline GNN Training

```bash
python -m src.training.baseline_trainer
```

**Outputs**:
- `models/baseline_gcn_model.pt`
- `models/checkpoints/checkpoint_*.pt`
- `models/plots/confusion_matrix_*.png`
- Training logs in `runs/` (TensorBoard)

**Configuration**:
- Loss type: `'focal'` (handles class imbalance)
- Epochs: 40
- Learning rate: 0.0005

---

#### Phase 4: Transformer Training

```bash
python -m src.training.transformer_trainer
```

**Outputs**:
- `models/core_transformer_model.pt`
- `models/checkpoints/checkpoint_*.pt`
- Training logs in `runs/`

**Features**:
- Role-Aware Graph Transformer
- PEARL positional embeddings
- Multi-head attention

---

#### Phase 5: RL Integration

```bash
python -m src.rl.training.single_agent
```

**Outputs**:
- `models/rl_ppo_agent_model/ppo_stock_agent.zip`
- Training logs in `runs/`

**Features**:
- PPO algorithm (Stable Baselines3)
- GNN-based observation space
- Risk-adjusted reward function

---

#### Phase 6: Evaluation

```bash
python -m src.evaluation.evaluation
```

**Outputs**:
- `results/gnn_node_metrics.csv`
- `results/comprehensive_strategy_comparison.csv`
- `results/ablation_results.csv`
- Visualizations in `models/plots/`

**Metrics**:
- Sharpe Ratio
- Max Drawdown
- Cumulative Return
- Precision@Top-K
- Information Coefficient (IC)

---

#### Optional: Baseline Model Comparison (Grading Requirement)

Compare multiple model architectures as required by the grading rubric:

```bash
python scripts/run_baseline_comparison.py
```

**Outputs**:
- `results/baseline_model_comparison.csv` - Comparison table
- `results/baseline_model_comparison.json` - Detailed metrics

**Models Compared**:
- **GNN Baselines**: GCN, GAT, GraphSAGE, HGT
- **Non-Graph Baselines**: Logistic Regression, MLP, LSTM

**Purpose**: Addresses grading rubric requirement "Comparison between multiple model architectures" for "Insights + results (10 points)"

**Runtime**: ~1-2 hours

---

## A+ Enhancements

Run all enhancement analysis scripts:

```bash
python -m src.scripts.run_all_enhancements
```

This includes:
1. **Multi-Agent Decision Analysis** - Analyzes agent disagreements and sector performance
2. **Failure Analysis** - Identifies worst-performing periods and error patterns
3. **Edge Importance Analysis** - Ranks edge types and analyzes sector subgraphs
4. **Cross-Period Validation** - Tests performance across market regimes
5. **Sensitivity Analysis** - Tests robustness to transaction costs and parameters

**Outputs**:
- `results/enhancements/` - All enhancement analysis results
- Visualizations and detailed reports

---

## Expected Outputs Summary

### Data Files
```
data/
 raw/
    stock_prices_ohlcv_raw.csv
    static_sector_industry.csv
 processed/
    node_features_X_t_final.csv
 edges/
    edges_dynamic_corr_params.pkl
    edges_dynamic_fund_sim_params.csv
 graphs/
     graph_t_YYYYMMDD.pt (one per trading day)
```

### Models
```
models/
 baseline_gcn_model.pt
 core_transformer_model.pt
 rl_ppo_agent_model/
    ppo_stock_agent.zip
 checkpoints/
     checkpoint_best.pt
     checkpoint_latest.pt
```

### Results
```
results/
 gnn_node_metrics.csv
 comprehensive_strategy_comparison.csv
 ablation_results.csv
 enhancements/
     multi_agent_analysis/
     failure_analysis/
     edge_importance/
     cross_period_validation/
     sensitivity_analysis/
```

---

## Important Notes

### 1. **Import Paths**

After reorganization, all imports use the new structure:
```python
# Old (won't work)
from phase4_core_training import ...

# New (correct)
from src.training.transformer_trainer import RoleAwareGraphTransformer
from src.rl.integration import load_gnn_model_for_rl
```

### 2. **Running from Project Root**

Always run scripts from the project root:
```bash
cd cs224_porject
python run_full_pipeline.py
```

### 3. **Module Execution**

Use `-m` flag for module execution:
```bash
python -m src.data.collection
```

### 4. **PyTorch >= 2.6 Compatibility**

All scripts handle PyTorch 2.6+ serialization:
- Uses `torch.serialization.add_safe_globals()`
- Uses `torch.load(..., weights_only=False)`

---

## Troubleshooting

### Import Errors

**Problem**: `ModuleNotFoundError` or import errors

**Solution**:
1. Ensure you're in the project root directory
2. Check that `src/` directory exists
3. Verify all `__init__.py` files are present
4. Try: `export PYTHONPATH="${PYTHONPATH}:$(pwd)"`

### Path Errors

**Problem**: File not found errors

**Solution**:
1. Check `PROJECT_ROOT` calculation in scripts
2. Ensure paths are relative to project root
3. Verify data files exist in expected locations

### Missing Dependencies

**Problem**: `ImportError` for packages

**Solution**:
```bash
pip install -r requirements.txt
```

### CUDA/GPU Issues

**Problem**: CUDA errors or GPU not detected

**Solution**:
- Scripts automatically fall back to CPU if CUDA unavailable
- Check CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`

---

## Quick Reference

All essential commands in one place:

```bash
# Full pipeline (recommended)
python run_full_pipeline.py

# Individual phases
python -m src.data.collection
python -m src.data.feature_engineering
python -m src.data.edge_parameters
python -m src.data.graph_construction
python -m src.training.baseline_trainer
python -m src.training.transformer_trainer
python -m src.rl.integration
python -m src.evaluation.evaluation

# A+ Enhancements
python -m src.scripts.run_all_enhancements
```

---

## Next Steps

After running the pipeline:

1. **Check Results**: Review `results/` directory
2. **View Visualizations**: Check `models/plots/`
3. **Review Documentation**: See `docs/` for detailed guides
4. **Analyze Performance**: Review `results/comprehensive_strategy_comparison.csv`

---

**For detailed documentation, see [docs/README.md](docs/README.md)**


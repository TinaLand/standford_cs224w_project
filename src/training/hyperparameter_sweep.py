#!/usr/bin/env python3
"""
phase4_hyperparameter_sweep.py
--------------------------------

Utility script to perform a lightweight hyperparameter sweep for the
Phase 4 core graph transformer model. The sweep runs multiple training
jobs sequentially (grid search) and records validation/test metrics for
each configuration.

Usage:
    python scripts/phase4_hyperparameter_sweep.py

Outputs:
    - Checkpoints for each run saved under models/sweeps/<timestamp>/
    - JSON summary file capturing config + metrics for every run
"""

from pathlib import Path
from datetime import datetime
import itertools
import json
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import the Phase 4 training pipeline from transformer_trainer
from src.training.transformer_trainer import (
    RoleAwareGraphTransformer,
    run_training_pipeline,
    load_graph_data,
    create_target_labels,
    HIDDEN_CHANNELS,
    LEARNING_RATE,
    NUM_LAYERS,
    NUM_HEADS,
    NUM_EPOCHS,
    DEVICE,
    MODELS_DIR as DEFAULT_MODELS_DIR
)

# Create a mock module-like object for compatibility
class CoreModule:
    """Wrapper to make transformer_trainer compatible with sweep script"""
    HIDDEN_CHANNELS = HIDDEN_CHANNELS
    LEARNING_RATE = LEARNING_RATE
    NUM_LAYERS = NUM_LAYERS
    NUM_HEADS = NUM_HEADS
    NUM_EPOCHS = NUM_EPOCHS
    MODELS_DIR = DEFAULT_MODELS_DIR
    MODEL_SAVE_NAME = "core_transformer_model.pt"
    
    @staticmethod
    def run_training_pipeline():
        """Wrapper for run_training_pipeline"""
        return run_training_pipeline()

core = CoreModule()
SWEEP_ROOT = PROJECT_ROOT / "models" / "sweeps"


def build_search_grid():
    """Return a list of hyperparameter dictionaries to evaluate."""
    hidden_channels = [192, 256]
    learning_rates = [5e-4, 1e-4]
    num_layers = [2, 3]
    num_heads = [4]
    num_epochs = [5, 8]

    grid = []
    for hc, lr, layers, heads, epochs in itertools.product(
        hidden_channels,
        learning_rates,
        num_layers,
        num_heads,
        num_epochs,
    ):
        grid.append(
            {
                "hidden_channels": hc,
                "learning_rate": lr,
                "num_layers": layers,
                "num_heads": heads,
                "num_epochs": epochs,
            }
        )
    return grid


def configure_training_params(config, run_dir, run_name):
    """Configure training parameters for this sweep run."""
    # Store original values
    originals = {
        "HIDDEN_CHANNELS": HIDDEN_CHANNELS,
        "LEARNING_RATE": LEARNING_RATE,
        "NUM_LAYERS": NUM_LAYERS,
        "NUM_HEADS": NUM_HEADS,
        "NUM_EPOCHS": NUM_EPOCHS,
    }
    
    # Temporarily modify the transformer_trainer module's constants
    import src.training.transformer_trainer as trainer_module
    trainer_module.HIDDEN_CHANNELS = config["hidden_channels"]
    trainer_module.LEARNING_RATE = config["learning_rate"]
    trainer_module.NUM_LAYERS = config["num_layers"]
    trainer_module.NUM_HEADS = config["num_heads"]
    trainer_module.NUM_EPOCHS = config["num_epochs"]
    trainer_module.MODELS_DIR = run_dir
    trainer_module.MODEL_SAVE_NAME = f"{run_name}.pt"
    
    run_dir.mkdir(parents=True, exist_ok=True)

    return originals, trainer_module


def restore_training_params(originals, trainer_module):
    """Restore original training parameters."""
    trainer_module.HIDDEN_CHANNELS = originals["HIDDEN_CHANNELS"]
    trainer_module.LEARNING_RATE = originals["LEARNING_RATE"]
    trainer_module.NUM_LAYERS = originals["NUM_LAYERS"]
    trainer_module.NUM_HEADS = originals["NUM_HEADS"]
    trainer_module.NUM_EPOCHS = originals["NUM_EPOCHS"]
    trainer_module.MODELS_DIR = DEFAULT_MODELS_DIR
    trainer_module.MODEL_SAVE_NAME = "core_transformer_model.pt"


def run_single_experiment(config, run_dir, run_name):
    """Execute one training run with the provided configuration."""
    originals, trainer_module = configure_training_params(config, run_dir, run_name)
    try:
        print("\n" + "=" * 70)
        print(f"🎯 Starting sweep run: {run_name}")
        print(f"    hidden_channels={config['hidden_channels']}, "
              f"num_layers={config['num_layers']}, "
              f"num_heads={config['num_heads']}, "
              f"learning_rate={config['learning_rate']:.2e}, "
              f"num_epochs={config['num_epochs']}")
        print("=" * 70)

        results = trainer_module.run_training_pipeline()
        if results is None:
            results = {}

        run_summary = {
            "run_name": run_name,
            "config": config,
            "results": results,
        }
        return run_summary
    finally:
        restore_training_params(originals, trainer_module)


def write_summary(summary_records, summary_path):
    """Persist sweep results to JSON."""
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "num_runs": len(summary_records),
        "runs": summary_records,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = SWEEP_ROOT / timestamp
    sweep_dir.mkdir(parents=True, exist_ok=True)

    search_grid = build_search_grid()
    summary_records = []
    best_record = None

    for idx, config in enumerate(search_grid, start=1):
        run_name = (
            f"run{idx:02d}_hc{config['hidden_channels']}"
            f"_lr{config['learning_rate']:.0e}"
            f"_layers{config['num_layers']}"
            f"_epochs{config['num_epochs']}"
        )
        run_dir = sweep_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        record = run_single_experiment(config, run_dir, run_name)
        summary_records.append(record)

        val_f1 = record.get("results", {}).get("best_val_f1", float("-inf"))
        if best_record is None or val_f1 > best_record.get("results", {}).get("best_val_f1", float("-inf")):
            best_record = record

    summary_path = sweep_dir / "sweep_summary.json"
    write_summary(summary_records, summary_path)

    print("\n" + "=" * 70)
    print(f"✅ Sweep completed. Results saved to: {summary_path}")
    if best_record:
        best_metrics = best_record.get("results", {})
        print("🏆 Best configuration:")
        print(f"   Run: {best_record['run_name']}")
        print(f"   Config: {best_record['config']}")
        print(f"   Best Val F1: {best_metrics.get('best_val_f1', 'N/A')}")
        print(f"   Test Accuracy: {best_metrics.get('test_accuracy', 'N/A')}")
        print(f"   Test F1: {best_metrics.get('test_f1', 'N/A')}")
        print(f"   Checkpoint: {best_metrics.get('model_checkpoint', 'N/A')}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()


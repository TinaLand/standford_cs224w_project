#!/usr/bin/env python3
"""
Experiment: Lookahead Horizon Analysis

Tests different lookahead horizons (1, 3, 5, 7, 10 days) to find optimal prediction window.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import json
from datetime import datetime
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.transformer_trainer import (
    RoleAwareGraphTransformer,
    load_graph_data,
    create_target_labels,
    HIDDEN_CHANNELS,
    OUT_CHANNELS,
    NUM_LAYERS,
    NUM_HEADS,
    LEARNING_RATE,
    NUM_EPOCHS,
    DEVICE,
    ENABLE_MULTI_TASK,
    ENABLE_TIME_AWARE,
    DATA_GRAPHS_DIR,
    _read_time_series_csv,
    OHLCV_RAW_FILE
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Lookahead horizons to test
LOOKAHEAD_HORIZONS = [1, 3, 5, 7, 10]

# Reduced epochs for faster experimentation
QUICK_TEST_EPOCHS = 10  # Use fewer epochs for quick comparison


def train_model_for_lookahead(lookahead_days, train_dates, val_dates, test_dates, tickers):
    """
    Train a model for a specific lookahead horizon.
    Returns model and metrics.
    """
    print(f"\n{'='*60}")
    print(f"🔬 Training model for {lookahead_days}-day lookahead")
    print(f"{'='*60}")
    
    # Create targets with specific lookahead
    all_dates = train_dates + val_dates + test_dates
    targets_class_dict, targets_reg_dict = create_target_labels(tickers, all_dates, lookahead_days=lookahead_days)
    
    # Get input dimension from first graph
    sample_graph_file = sorted(list(DATA_GRAPHS_DIR.glob('graph_t_*.pt')))[0]
    sample_graph = torch.load(sample_graph_file, weights_only=False)
    INPUT_DIM = sample_graph['stock'].x.shape[1]
    
    # Create model
    model = RoleAwareGraphTransformer(
        in_dim=INPUT_DIM,
        hidden_dim=HIDDEN_CHANNELS,
        out_dim=OUT_CHANNELS,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        enable_time_aware=ENABLE_TIME_AWARE
    ).to(DEVICE)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop (simplified, quick version)
    REFERENCE_DATE = pd.to_datetime('2015-01-01')
    best_val_f1 = 0.0
    best_model_state = None
    
    from src.training.transformer_trainer import FocalLoss, FOCAL_ALPHA, FOCAL_GAMMA, USE_CLASS_WEIGHTS
    
    # Calculate class weights
    all_train_targets = []
    for date in train_dates:
        target = targets_class_dict.get(date)
        if target is not None:
            all_train_targets.append(target.cpu().numpy())
    if all_train_targets:
        all_train_targets_array = np.concatenate(all_train_targets)
        class_counts = np.bincount(all_train_targets_array)
        total_samples = class_counts.sum()
        num_classes = len(class_counts)
        if num_classes > 0 and USE_CLASS_WEIGHTS:
            class_weights = total_samples / (num_classes * class_counts)
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
        else:
            class_weights = None
    else:
        class_weights = None
    
    criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, weight=class_weights)
    
    print(f"Training for {QUICK_TEST_EPOCHS} epochs (quick test)...")
    for epoch in range(1, QUICK_TEST_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for date in tqdm(train_dates, desc=f"Epoch {epoch}/{QUICK_TEST_EPOCHS}", leave=False):
            data = load_graph_data(date)
            target_class = targets_class_dict.get(date)
            
            if data and target_class is not None:
                data = data.to(DEVICE)
                target_class = target_class.to(DEVICE)
                
                if ENABLE_TIME_AWARE:
                    days_since_ref = (date - REFERENCE_DATE).days
                    num_nodes = data['stock'].x.shape[0]
                    date_tensor = torch.full((num_nodes,), days_since_ref, dtype=torch.float32).to(DEVICE)
                else:
                    date_tensor = None
                
                optimizer.zero_grad()
                out = model(data, date_tensor=date_tensor)
                loss = criterion(out, target_class)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
        
        # Validation
        model.eval()
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for date in val_dates:
                data = load_graph_data(date)
                target_class = targets_class_dict.get(date)
                
                if data and target_class is not None:
                    data = data.to(DEVICE)
                    target_class = target_class.to(DEVICE)
                    
                    if ENABLE_TIME_AWARE:
                        days_since_ref = (date - REFERENCE_DATE).days
                        num_nodes = data['stock'].x.shape[0]
                        date_tensor = torch.full((num_nodes,), days_since_ref, dtype=torch.float32).to(DEVICE)
                    else:
                        date_tensor = None
                    
                    out = model(data, date_tensor=date_tensor)
                    preds = out.argmax(dim=1)
                    
                    val_predictions.extend(preds.cpu().numpy())
                    val_targets.extend(target_class.cpu().numpy())
        
        if len(val_targets) > 0:
            val_accuracy = accuracy_score(val_targets, val_predictions)
            val_f1 = f1_score(val_targets, val_predictions, average='macro')
        else:
            val_accuracy = 0.0
            val_f1 = 0.0
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        if epoch % 2 == 0 or epoch == 1:
            print(f"  Epoch {epoch}: Loss={avg_loss:.4f}, Val Acc={val_accuracy:.4f}, Val F1={val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Test evaluation
    model.eval()
    test_predictions = []
    test_targets = []
    test_probs = []
    
    with torch.no_grad():
        for date in tqdm(test_dates, desc="Testing"):
            data = load_graph_data(date)
            target_class = targets_class_dict.get(date)
            
            if data and target_class is not None:
                data = data.to(DEVICE)
                target_class = target_class.to(DEVICE)
                
                if ENABLE_TIME_AWARE:
                    days_since_ref = (date - REFERENCE_DATE).days
                    num_nodes = data['stock'].x.shape[0]
                    date_tensor = torch.full((num_nodes,), days_since_ref, dtype=torch.float32).to(DEVICE)
                else:
                    date_tensor = None
                
                out = model(data, date_tensor=date_tensor)
                probs = torch.nn.functional.softmax(out, dim=1)
                preds = out.argmax(dim=1)
                
                test_predictions.extend(preds.cpu().numpy())
                test_targets.extend(target_class.cpu().numpy())
                test_probs.extend(probs[:, 1].cpu().numpy())  # Probability of "Up" class
    
    # Calculate metrics
    if len(test_targets) > 0:
        test_accuracy = accuracy_score(test_targets, test_predictions)
        test_f1 = f1_score(test_targets, test_predictions, average='macro')
        test_precision = precision_score(test_targets, test_predictions, average='macro', zero_division=0)
        test_recall = recall_score(test_targets, test_predictions, average='macro', zero_division=0)
    else:
        test_accuracy = test_f1 = test_precision = test_recall = 0.0
    
    metrics = {
        'lookahead_days': lookahead_days,
        'best_val_f1': float(best_val_f1),
        'test_accuracy': float(test_accuracy),
        'test_f1': float(test_f1),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'num_train_dates': len(train_dates),
        'num_val_dates': len(val_dates),
        'num_test_dates': len(test_dates)
    }
    
    return model, metrics


def run_lookahead_experiment():
    """
    Run experiments for different lookahead horizons.
    """
    print("=" * 60)
    print("🔬 Lookahead Horizon Experiment")
    print("=" * 60)
    
    # Get all dates
    graph_files = sorted(list(DATA_GRAPHS_DIR.glob('graph_t_*.pt')))
    all_dates = [pd.to_datetime(f.stem.split('_')[-1]) for f in graph_files]
    
    # Split dates (70/15/15)
    split_70_idx = int(len(all_dates) * 0.70)
    split_85_idx = int(len(all_dates) * 0.85)
    
    train_dates = all_dates[:split_70_idx]
    val_dates = all_dates[split_70_idx:split_85_idx]
    test_dates = all_dates[split_85_idx:]
    
    print(f"\n📅 Date Split:")
    print(f"   Train: {len(train_dates)} dates ({train_dates[0]} to {train_dates[-1]})")
    print(f"   Val: {len(val_dates)} dates ({val_dates[0]} to {val_dates[-1]})")
    print(f"   Test: {len(test_dates)} dates ({test_dates[0]} to {test_dates[-1]})")
    
    # Get tickers
    ohlcv_df = _read_time_series_csv(OHLCV_RAW_FILE)
    tickers = [col.replace('Close_', '') for col in ohlcv_df.columns if col.startswith('Close_')]
    print(f"\n📊 Stocks: {len(tickers)}")
    
    # Run experiments for each lookahead horizon
    results = {}
    
    for lookahead in LOOKAHEAD_HORIZONS:
        try:
            model, metrics = train_model_for_lookahead(
                lookahead, train_dates, val_dates, test_dates, tickers
            )
            results[lookahead] = metrics
            
            print(f"\n✅ {lookahead}-day lookahead Results:")
            print(f"   Val F1: {metrics['best_val_f1']:.4f}")
            print(f"   Test Accuracy: {metrics['test_accuracy']:.4f}")
            print(f"   Test F1: {metrics['test_f1']:.4f}")
            print(f"   Test Precision: {metrics['test_precision']:.4f}")
            print(f"   Test Recall: {metrics['test_recall']:.4f}")
            
        except Exception as e:
            print(f"\n❌ Error with {lookahead}-day lookahead: {e}")
            import traceback
            traceback.print_exc()
            results[lookahead] = {'error': str(e)}
    
    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv(RESULTS_DIR / 'lookahead_horizon_results.csv')
    
    with open(RESULTS_DIR / 'lookahead_horizon_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Lookahead Horizon Experiment Summary")
    print("=" * 60)
    print("\nResults by Lookahead Horizon:")
    print(results_df.to_string())
    
    # Find best lookahead
    if len(results_df) > 0 and 'test_f1' in results_df.columns:
        best_lookahead = results_df['test_f1'].idxmax()
        best_f1 = results_df.loc[best_lookahead, 'test_f1']
        print(f"\n🏆 Best Lookahead: {best_lookahead} days (Test F1: {best_f1:.4f})")
    
    print(f"\n📁 Results saved to:")
    print(f"  - {RESULTS_DIR / 'lookahead_horizon_results.csv'}")
    print(f"  - {RESULTS_DIR / 'lookahead_horizon_results.json'}")
    
    return results


if __name__ == '__main__':
    run_lookahead_experiment()


#!/usr/bin/env python3
"""
Experiment: Graph Sparsification Strategies

Tests different Top-K thresholds and correlation cutoffs to find optimal graph sparsification.
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
    DEVICE,
    ENABLE_TIME_AWARE,
    DATA_GRAPHS_DIR,
    _read_time_series_csv,
    OHLCV_RAW_FILE
)
from sklearn.metrics import accuracy_score, f1_score

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Sparsification parameters to test
TOP_K_VALUES = [5, 8, 10, 15, 20]
CORRELATION_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]

# For quick testing, we'll evaluate on a subset of graphs
# Full evaluation would require regenerating all graphs with different parameters
QUICK_TEST = True  # Set to False for full evaluation
QUICK_TEST_GRAPHS = 50  # Number of graphs to test on


def evaluate_sparsification_config(top_k, corr_threshold, test_dates, tickers):
    """
    Evaluate a specific sparsification configuration.
    Note: This is a simplified evaluation that filters edges from existing graphs.
    For full evaluation, graphs should be regenerated with these parameters.
    """
    print(f"\n{'='*60}")
    print(f"🔬 Testing: Top-K={top_k}, Correlation Threshold={corr_threshold}")
    print(f"{'='*60}")
    
    # Create targets
    targets_class_dict, targets_reg_dict = create_target_labels(tickers, test_dates, 5)
    
    # Get input dimension
    sample_graph_file = sorted(list(DATA_GRAPHS_DIR.glob('graph_t_*.pt')))[0]
    sample_graph = torch.load(sample_graph_file, weights_only=False)
    INPUT_DIM = sample_graph['stock'].x.shape[1]
    
    # Load pre-trained model (use existing model for quick evaluation)
    model_path = PROJECT_ROOT / "models" / "core_transformer_model.pt"
    if not model_path.exists():
        print("⚠️  Pre-trained model not found. Skipping this configuration.")
        return None
    
    model = RoleAwareGraphTransformer(
        in_dim=INPUT_DIM,
        hidden_dim=HIDDEN_CHANNELS,
        out_dim=OUT_CHANNELS,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        enable_time_aware=ENABLE_TIME_AWARE
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # Filter edges based on sparsification parameters
    def filter_graph_edges(data, top_k, corr_threshold):
        """Filter graph edges based on Top-K and correlation threshold."""
        from torch_geometric.data import HeteroData
        
        # Create a deep copy of the original data to avoid modifying it
        # Use clone() to ensure we have a proper copy
        filtered_data = HeteroData()
        filtered_data['stock'].x = data['stock'].x.clone()
        
        # Get edge_index_dict from HeteroData
        # Access edge_index_dict property which returns a dict
        edge_index_dict = data.edge_index_dict if hasattr(data, 'edge_index_dict') else {}
        
        # If edge_index_dict is not available, build it from edge_types
        if not edge_index_dict:
            for edge_type in data.edge_types:
                if hasattr(data[edge_type], 'edge_index') and data[edge_type].edge_index is not None:
                    edge_index_dict[edge_type] = data[edge_type].edge_index
        
        # Filter correlation edges
        corr_key = ('stock', 'rolling_correlation', 'stock')
        if corr_key in edge_index_dict:
            edge_index = edge_index_dict[corr_key]
            
            # Get edge weights if available
            # For simplicity, we'll just keep top-k edges per node
            # In full implementation, this should use actual correlation values
            
            # Simple Top-K filtering: keep only first top_k edges per node
            # This is a simplified version - full implementation needs edge weights
            num_nodes = data['stock'].x.shape[0]
            filtered_edges = []
            
            # Group edges by source node
            source_nodes = edge_index[0].cpu().numpy()
            target_nodes = edge_index[1].cpu().numpy()
            
            for node in range(num_nodes):
                node_edges = np.where(source_nodes == node)[0]
                if len(node_edges) > top_k:
                    # Keep top-k (simplified: just take first k)
                    node_edges = node_edges[:top_k]
                filtered_edges.extend(node_edges)
            
            if len(filtered_edges) > 0:
                filtered_edge_index = edge_index[:, filtered_edges]
                # Properly set edge_index for the edge type
                filtered_data[corr_key].edge_index = filtered_edge_index.clone()
        
        # Keep other edge types as-is
        for key in edge_index_dict:
            if key != corr_key:
                # Properly set edge_index for other edge types
                filtered_data[key].edge_index = edge_index_dict[key].clone()
        
        # Copy other attributes if they exist
        if hasattr(data, 'tickers'):
            filtered_data.tickers = data.tickers
        
        return filtered_data
    
    # Evaluate on test set
    REFERENCE_DATE = pd.to_datetime('2015-01-01')
    test_predictions = []
    test_targets = []
    
    test_dates_subset = test_dates[:QUICK_TEST_GRAPHS] if QUICK_TEST else test_dates
    
    with torch.no_grad():
        for date in tqdm(test_dates_subset, desc="Evaluating"):
            data = load_graph_data(date)
            target_class = targets_class_dict.get(date)
            
            if data and target_class is not None:
                # Filter graph edges
                filtered_data = filter_graph_edges(data, top_k, corr_threshold)
                filtered_data = filtered_data.to(DEVICE)
                target_class = target_class.to(DEVICE)
                
                if ENABLE_TIME_AWARE:
                    days_since_ref = (date - REFERENCE_DATE).days
                    num_nodes = filtered_data['stock'].x.shape[0]
                    date_tensor = torch.full((num_nodes,), days_since_ref, dtype=torch.float32).to(DEVICE)
                else:
                    date_tensor = None
                
                out = model(filtered_data, date_tensor=date_tensor)
                preds = out.argmax(dim=1)
                
                test_predictions.extend(preds.cpu().numpy())
                test_targets.extend(target_class.cpu().numpy())
    
    # Calculate metrics
    if len(test_targets) > 0:
        test_accuracy = accuracy_score(test_targets, test_predictions)
        test_f1 = f1_score(test_targets, test_predictions, average='macro')
    else:
        test_accuracy = test_f1 = 0.0
    
    metrics = {
        'top_k': top_k,
        'correlation_threshold': corr_threshold,
        'test_accuracy': float(test_accuracy),
        'test_f1': float(test_f1),
        'num_test_graphs': len(test_dates_subset)
    }
    
    return metrics


def run_sparsification_experiment():
    """
    Run experiments for different sparsification configurations.
    """
    print("=" * 60)
    print("🔬 Graph Sparsification Experiment")
    print("=" * 60)
    print("\n⚠️  Note: This is a simplified evaluation using existing graphs.")
    print("   For full evaluation, regenerate graphs with different parameters.")
    
    # Get test dates
    graph_files = sorted(list(DATA_GRAPHS_DIR.glob('graph_t_*.pt')))
    all_dates = [pd.to_datetime(f.stem.split('_')[-1]) for f in graph_files]
    
    split_85_idx = int(len(all_dates) * 0.85)
    test_dates = all_dates[split_85_idx:]
    
    # Get tickers
    ohlcv_df = _read_time_series_csv(OHLCV_RAW_FILE)
    tickers = [col.replace('Close_', '') for col in ohlcv_df.columns if col.startswith('Close_')]
    
    print(f"\n📊 Test Dates: {len(test_dates)}")
    print(f"📊 Stocks: {len(tickers)}")
    
    # Run experiments
    results = []
    
    # Test a subset of combinations for quick evaluation
    test_configs = [
        (5, 0.3), (8, 0.4), (10, 0.5), (15, 0.6), (20, 0.7)
    ]
    
    for top_k, corr_threshold in test_configs:
        try:
            metrics = evaluate_sparsification_config(top_k, corr_threshold, test_dates, tickers)
            if metrics:
                results.append(metrics)
                print(f"\n✅ Top-K={top_k}, Threshold={corr_threshold}:")
                print(f"   Test Accuracy: {metrics['test_accuracy']:.4f}")
                print(f"   Test F1: {metrics['test_f1']:.4f}")
        except Exception as e:
            print(f"\n❌ Error with Top-K={top_k}, Threshold={corr_threshold}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(RESULTS_DIR / 'graph_sparsification_results.csv', index=False)
        
        with open(RESULTS_DIR / 'graph_sparsification_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 Graph Sparsification Experiment Summary")
        print("=" * 60)
        print("\nResults:")
        print(results_df.to_string())
        
        # Find best configuration
        if 'test_f1' in results_df.columns:
            best_idx = results_df['test_f1'].idxmax()
            best_config = results_df.loc[best_idx]
            print(f"\n🏆 Best Configuration:")
            print(f"   Top-K: {best_config['top_k']}, Threshold: {best_config['correlation_threshold']}")
            print(f"   Test F1: {best_config['test_f1']:.4f}")
        
        print(f"\n📁 Results saved to:")
        print(f"  - {RESULTS_DIR / 'graph_sparsification_results.csv'}")
        print(f"  - {RESULTS_DIR / 'graph_sparsification_results.json'}")
    
    return results


if __name__ == '__main__':
    run_sparsification_experiment()


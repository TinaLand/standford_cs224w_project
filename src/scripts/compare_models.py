# scripts/compare_phase3_phase4.py
"""
Compare Phase 3 Baseline GAT model with Phase 4 Role-Aware Graph Transformer.

This script evaluates both models on the same test set and generates a comparison report.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / 'scripts'))

# Import models
from phase3_baseline_training import BaselineGNN
from src.training.transformer_trainer import RoleAwareGraphTransformer, create_target_labels, _read_time_series_csv
from src.utils.graph_loader import load_graph_data
from src.utils.paths import OHLCV_RAW_FILE
from phase6_evaluation import calculate_precision_at_topk, calculate_information_coefficient

# Configuration
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model paths
PHASE3_MODEL_PATH = MODELS_DIR / "baseline_gcn_model.pt"
PHASE4_MODEL_PATH = MODELS_DIR / "core_transformer_model.pt"


def load_phase3_model():
    """Load Phase 3 baseline GAT model."""
    print("\n--- Loading Phase 3 Baseline Model ---")
    
    # Model parameters (from phase3_baseline_training.py)
    IN_CHANNELS = 15  # Node feature dimension
    HIDDEN_CHANNELS = 128
    OUT_CHANNELS = 2
    
    model = BaselineGNN(IN_CHANNELS, HIDDEN_CHANNELS, OUT_CHANNELS)
    
    if PHASE3_MODEL_PATH.exists():
        model.load_state_dict(torch.load(PHASE3_MODEL_PATH, map_location=DEVICE, weights_only=False))
        print(f"✅ Phase 3 model loaded from: {PHASE3_MODEL_PATH}")
    else:
        print(f"⚠️  Phase 3 model not found at {PHASE3_MODEL_PATH}")
        print("   Using untrained model for comparison")
    
    model.to(DEVICE)
    model.eval()
    return model


def load_phase4_model():
    """Load Phase 4 Role-Aware Graph Transformer model."""
    print("\n--- Loading Phase 4 Transformer Model ---")
    
    # Model parameters (from phase4_core_training.py)
    # These must match the saved model exactly
    IN_DIM = 15
    HIDDEN_DIM = 256  # Must match saved model
    OUT_DIM = 2
    NUM_LAYERS = 2
    NUM_HEADS = 4  # Must match saved model (4 heads, 64 per head = 256 total)
    
    model = RoleAwareGraphTransformer(IN_DIM, HIDDEN_DIM, OUT_DIM, NUM_LAYERS, NUM_HEADS)
    
    if PHASE4_MODEL_PATH.exists():
        model.load_state_dict(torch.load(PHASE4_MODEL_PATH, map_location=DEVICE, weights_only=False))
        print(f"✅ Phase 4 model loaded from: {PHASE4_MODEL_PATH}")
    else:
        print(f"⚠️  Phase 4 model not found at {PHASE4_MODEL_PATH}")
        print("   Using untrained model for comparison")
    
    model.to(DEVICE)
    model.eval()
    return model


def evaluate_model_phase3(model, test_dates, targets_dict, tickers):
    """Evaluate Phase 3 baseline model (uses simple edge_index)."""
    print("\n--- Evaluating Phase 3 Baseline Model ---")
    
    all_predictions = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for date in tqdm(test_dates, desc="Phase 3 Evaluation"):
            data = load_graph_data(date)
            target = targets_dict.get(date)
            
            if data and target is not None:
                # Phase 3 uses simple edge_index (combine all edges)
                # Get the first edge type as a simple baseline
                edge_types = list(data.edge_index_dict.keys())
                if edge_types:
                    # Use first edge type (or combine all)
                    edge_index = data[edge_types[0]].edge_index
                else:
                    continue
                
                x = data['stock'].x.to(DEVICE)
                
                # Get predictions
                out = model(x, edge_index)
                probs = F.softmax(out, dim=1)
                preds = out.argmax(dim=1).cpu().numpy()
                
                all_predictions.append(preds)
                all_targets.append(target.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
    
    if len(all_predictions) == 0:
        return {}
    
    # Calculate metrics
    predictions_array = np.array(all_predictions)
    targets_array = np.array(all_targets)
    probs_array = np.array(all_probs)
    
    flat_predictions = predictions_array.flatten()
    flat_targets = targets_array.flatten()
    
    accuracy = accuracy_score(flat_targets, flat_predictions)
    f1 = f1_score(flat_targets, flat_predictions, average='macro')
    
    # Precision@Top-K
    precision_k5 = calculate_precision_at_topk(probs_array, targets_array, k=5)
    precision_k10 = calculate_precision_at_topk(probs_array, targets_array, k=10)
    precision_k20 = calculate_precision_at_topk(probs_array, targets_array, k=20)
    
    # IC (if we have returns)
    ic_results = {}
    try:
        ohlcv_df = _read_time_series_csv(OHLCV_RAW_FILE)
        close_cols = [col for col in ohlcv_df.columns if col.startswith('Close_')]
        close_prices = ohlcv_df[close_cols].copy()
        forward_returns = (close_prices.shift(-5) - close_prices) / close_prices
        
        actual_returns_list = []
        for date in test_dates:
            date_str = date.strftime('%Y-%m-%d')
            if date_str in forward_returns.index:
                returns_vector = []
                for ticker in tickers:
                    col_name = f'Close_{ticker}'
                    if col_name in forward_returns.columns:
                        ret = forward_returns.loc[date_str, col_name]
                        returns_vector.append(ret if not pd.isna(ret) else 0.0)
                    else:
                        returns_vector.append(0.0)
                actual_returns_list.append(returns_vector)
            else:
                actual_returns_list.append([0.0] * len(tickers))
        
        actual_returns_array = np.array(actual_returns_list)
        pred_probs_up = probs_array[:, :, 1]
        ic_results = calculate_information_coefficient(pred_probs_up, actual_returns_array)
    except Exception as e:
        print(f"⚠️  Could not calculate IC: {e}")
        ic_results = {'IC_mean': 0.0, 'IC_std': 0.0, 'IC_IR': 0.0}
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision_at_top5': precision_k5,
        'precision_at_top10': precision_k10,
        'precision_at_top20': precision_k20,
        **ic_results
    }


def evaluate_model_phase4(model, test_dates, targets_dict, tickers):
    """Evaluate Phase 4 transformer model (uses HeteroData)."""
    print("\n--- Evaluating Phase 4 Transformer Model ---")
    
    all_predictions = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for date in tqdm(test_dates, desc="Phase 4 Evaluation"):
            data = load_graph_data(date)
            target = targets_dict.get(date)
            
            if data and target is not None:
                # Phase 4 uses HeteroData directly
                out = model(data.to(DEVICE))
                probs = F.softmax(out, dim=1)
                preds = out.argmax(dim=1).cpu().numpy()
                
                all_predictions.append(preds)
                all_targets.append(target.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
    
    if len(all_predictions) == 0:
        return {}
    
    # Calculate metrics
    predictions_array = np.array(all_predictions)
    targets_array = np.array(all_targets)
    probs_array = np.array(all_probs)
    
    flat_predictions = predictions_array.flatten()
    flat_targets = targets_array.flatten()
    
    accuracy = accuracy_score(flat_targets, flat_predictions)
    f1 = f1_score(flat_targets, flat_predictions, average='macro')
    
    # Precision@Top-K
    precision_k5 = calculate_precision_at_topk(probs_array, targets_array, k=5)
    precision_k10 = calculate_precision_at_topk(probs_array, targets_array, k=10)
    precision_k20 = calculate_precision_at_topk(probs_array, targets_array, k=20)
    
    # IC
    ic_results = {}
    try:
        ohlcv_df = _read_time_series_csv(OHLCV_RAW_FILE)
        close_cols = [col for col in ohlcv_df.columns if col.startswith('Close_')]
        close_prices = ohlcv_df[close_cols].copy()
        forward_returns = (close_prices.shift(-5) - close_prices) / close_prices
        
        actual_returns_list = []
        for date in test_dates:
            date_str = date.strftime('%Y-%m-%d')
            if date_str in forward_returns.index:
                returns_vector = []
                for ticker in tickers:
                    col_name = f'Close_{ticker}'
                    if col_name in forward_returns.columns:
                        ret = forward_returns.loc[date_str, col_name]
                        returns_vector.append(ret if not pd.isna(ret) else 0.0)
                    else:
                        returns_vector.append(0.0)
                actual_returns_list.append(returns_vector)
            else:
                actual_returns_list.append([0.0] * len(tickers))
        
        actual_returns_array = np.array(actual_returns_list)
        pred_probs_up = probs_array[:, :, 1]
        ic_results = calculate_information_coefficient(pred_probs_up, actual_returns_array)
    except Exception as e:
        print(f"⚠️  Could not calculate IC: {e}")
        ic_results = {'IC_mean': 0.0, 'IC_std': 0.0, 'IC_IR': 0.0}
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision_at_top5': precision_k5,
        'precision_at_top10': precision_k10,
        'precision_at_top20': precision_k20,
        **ic_results
    }


def main():
    """Main comparison function."""
    print("=" * 60)
    print("🔬 Phase 3 vs Phase 4 Model Comparison")
    print("=" * 60)
    
    # Load models
    phase3_model = load_phase3_model()
    phase4_model = load_phase4_model()
    
    # Get test dates and tickers
    graph_files = sorted(list((PROJECT_ROOT / "data" / "graphs").glob('graph_t_*.pt')))
    all_dates = [pd.to_datetime(f.stem.split('_')[-1]) for f in graph_files]
    
    # Use same split as Phase 4 (70/15/15)
    split_85_idx = int(len(all_dates) * 0.85)
    test_dates = all_dates[split_85_idx:]
    
    # Get tickers
    sample_graph = torch.load(graph_files[0], weights_only=False)
    if 'tickers' in sample_graph:
        tickers = sample_graph['tickers']
    else:
        ohlcv_df = _read_time_series_csv(OHLCV_RAW_FILE)
        tickers = [col.replace('Close_', '') for col in ohlcv_df.columns if col.startswith('Close_')]
    
    # Create targets
    targets_dict = create_target_labels(tickers, all_dates, lookahead_days=5)
    
    print(f"\n📊 Test Set: {len(test_dates)} days, {len(tickers)} stocks")
    
    # Evaluate both models
    phase3_metrics = evaluate_model_phase3(phase3_model, test_dates, targets_dict, tickers)
    phase4_metrics = evaluate_model_phase4(phase4_model, test_dates, targets_dict, tickers)
    
    # Create comparison DataFrame
    comparison_data = {
        'Model': ['Phase 3 Baseline (GAT)', 'Phase 4 Transformer'],
        'Accuracy': [phase3_metrics.get('accuracy', 0), phase4_metrics.get('accuracy', 0)],
        'F1_Score': [phase3_metrics.get('f1_score', 0), phase4_metrics.get('f1_score', 0)],
        'Precision@Top-5': [phase3_metrics.get('precision_at_top5', 0), phase4_metrics.get('precision_at_top5', 0)],
        'Precision@Top-10': [phase3_metrics.get('precision_at_top10', 0), phase4_metrics.get('precision_at_top10', 0)],
        'Precision@Top-20': [phase3_metrics.get('precision_at_top20', 0), phase4_metrics.get('precision_at_top20', 0)],
        'IC_Mean': [phase3_metrics.get('IC_mean', 0), phase4_metrics.get('IC_mean', 0)],
        'IC_IR': [phase3_metrics.get('IC_IR', 0), phase4_metrics.get('IC_IR', 0)],
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Calculate improvements
    print("\n" + "=" * 60)
    print("📊 COMPARISON RESULTS")
    print("=" * 60)
    print(comparison_df.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("📈 IMPROVEMENTS (Phase 4 vs Phase 3)")
    print("=" * 60)
    
    improvements = {}
    for metric in ['Accuracy', 'F1_Score', 'Precision@Top-5', 'Precision@Top-10', 'Precision@Top-20', 'IC_Mean', 'IC_IR']:
        phase3_val = comparison_df.loc[0, metric]
        phase4_val = comparison_df.loc[1, metric]
        if phase3_val > 0:
            improvement = ((phase4_val - phase3_val) / phase3_val) * 100
            improvements[metric] = improvement
            print(f"  {metric:20s}: {phase4_val:.4f} vs {phase3_val:.4f} ({improvement:+.2f}%)")
        else:
            print(f"  {metric:20s}: {phase4_val:.4f} vs {phase3_val:.4f} (N/A)")
    
    # Save results
    comparison_df.to_csv(RESULTS_DIR / 'phase3_vs_phase4_comparison.csv', index=False)
    print(f"\n✅ Comparison results saved to: {RESULTS_DIR / 'phase3_vs_phase4_comparison.csv'}")
    
    return comparison_df


if __name__ == '__main__':
    main()


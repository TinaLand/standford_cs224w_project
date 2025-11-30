"""
Enhanced Ablation Studies for A+ Grade
Actually retrains models for each configuration to show real differences
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import json

from src.training.transformer_trainer import (
    RoleAwareGraphTransformer,
    load_graph_data,
    create_target_labels,
    run_training_pipeline,
    HIDDEN_CHANNELS,
    NUM_LAYERS,
    NUM_HEADS,
    OUT_CHANNELS,
    LEARNING_RATE,
    NUM_EPOCHS,
    DEVICE,
    ENABLE_MULTI_TASK,
    ENABLE_TIME_AWARE
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_GRAPHS_DIR = PROJECT_ROOT / "data" / "graphs"

# Ablation configurations
ABLATION_CONFIGS = {
    'Full_Model': {
        'use_correlation': True,
        'use_fund_similarity': True,
        'use_static_edges': True,
        'description': 'Full model with all edge types'
    },
    'Abl_NoCorrelation': {
        'use_correlation': False,
        'use_fund_similarity': True,
        'use_static_edges': True,
        'description': 'Without correlation edges'
    },
    'Abl_NoFundSim': {
        'use_correlation': True,
        'use_fund_similarity': False,
        'use_static_edges': True,
        'description': 'Without fundamental similarity edges'
    },
    'Abl_NoStatic': {
        'use_correlation': True,
        'use_fund_similarity': True,
        'use_static_edges': False,
        'description': 'Without static edges (sector/industry)'
    },
    'Abl_OnlyCorrelation': {
        'use_correlation': True,
        'use_fund_similarity': False,
        'use_static_edges': False,
        'description': 'Only correlation edges'
    },
    'Abl_OnlyFundSim': {
        'use_correlation': False,
        'use_fund_similarity': True,
        'use_static_edges': False,
        'description': 'Only fundamental similarity edges'
    }
}


def filter_graph_edges(data, config: Dict) -> torch.Tensor:
    """
    Filter graph edges based on ablation configuration.
    """
    edge_index_dict = {}
    
    if config['use_correlation'] and 'rolling_correlation' in data.edge_index_dict:
        edge_index_dict['rolling_correlation'] = data.edge_index_dict['rolling_correlation']
    
    if config['use_fund_similarity'] and 'fund_similarity' in data.edge_index_dict:
        edge_index_dict['fund_similarity'] = data.edge_index_dict['fund_similarity']
    
    if config['use_static_edges']:
        if 'sector_industry' in data.edge_index_dict:
            edge_index_dict['sector_industry'] = data.edge_index_dict['sector_industry']
        if 'supply_competitor' in data.edge_index_dict:
            edge_index_dict['supply_competitor'] = data.edge_index_dict['supply_competitor']
    
    # Create new HeteroData with filtered edges
    from torch_geometric.data import HeteroData
    filtered_data = HeteroData()
    filtered_data['stock'].x = data['stock'].x
    filtered_data.edge_index_dict = edge_index_dict
    
    return filtered_data


def train_ablation_model(
    config_name: str,
    config: Dict,
    train_dates: List[pd.Timestamp],
    val_dates: List[pd.Timestamp],
    targets_class_dict: Dict,
    targets_reg_dict: Dict,
    input_dim: int
) -> Tuple[RoleAwareGraphTransformer, Dict]:
    """
    Train a model for a specific ablation configuration.
    """
    print(f"\n{'='*60}")
    print(f"🔬 Training Ablation Model: {config_name}")
    print(f"   {config['description']}")
    print(f"{'='*60}")
    
    # Create model
    model = RoleAwareGraphTransformer(
        in_dim=input_dim,
        hidden_dim=HIDDEN_CHANNELS,
        out_dim=OUT_CHANNELS,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        enable_time_aware=ENABLE_TIME_AWARE
    ).to(DEVICE)
    
    # Custom data loader that filters edges
    def load_filtered_graph_data(date):
        data = load_graph_data(date)
        if data:
            return filter_graph_edges(data, config)
        return None
    
    # Train model
    best_model, metrics = run_training_pipeline(
        model=model,
        train_dates=train_dates,
        val_dates=val_dates,
        targets_class_dict=targets_class_dict,
        targets_reg_dict=targets_reg_dict,
        load_graph_data_fn=load_filtered_graph_data,
        save_path=MODELS_DIR / f"ablation_{config_name.lower()}.pt"
    )
    
    return best_model, metrics


def evaluate_ablation_model(
    model: RoleAwareGraphTransformer,
    config: Dict,
    test_dates: List[pd.Timestamp],
    targets_class_dict: Dict,
    tickers: List[str]
) -> Dict:
    """
    Evaluate an ablation model on test set.
    """
    from sklearn.metrics import accuracy_score, f1_score
    
    all_predictions = []
    all_targets = []
    all_probs = []
    
    REFERENCE_DATE = pd.to_datetime('2015-01-01')
    
    model.eval()
    for date in tqdm(test_dates, desc=f"Evaluating {config.get('description', 'model')}"):
        data = load_graph_data(date)
        target = targets_class_dict.get(date)
        
        if data and target is not None:
            # Filter edges
            filtered_data = filter_graph_edges(data, config)
            
            with torch.no_grad():
                days_since_ref = (date - REFERENCE_DATE).days
                num_nodes = filtered_data['stock'].x.shape[0]
                date_tensor = torch.full((num_nodes,), days_since_ref, dtype=torch.float32).to(DEVICE) if ENABLE_TIME_AWARE else None
                
                out = model(filtered_data.to(DEVICE), date_tensor=date_tensor)
                probs = torch.nn.functional.softmax(out, dim=1)
                preds = out.argmax(dim=1).cpu().numpy()
            
            all_predictions.append(preds)
            all_targets.append(target.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    if len(all_predictions) == 0:
        return {'accuracy': 0.0, 'f1_score': 0.0, 'precision_at_top10': 0.0}
    
    predictions_array = np.array(all_predictions)
    targets_array = np.array(all_targets)
    probs_array = np.array(all_probs)
    
    # Calculate metrics
    flat_predictions = predictions_array.flatten()
    flat_targets = targets_array.flatten()
    
    accuracy = accuracy_score(flat_targets, flat_predictions)
    f1 = f1_score(flat_targets, flat_predictions, average='macro')
    
    # Precision@Top-10
    from src.evaluation.evaluation import calculate_precision_at_topk
    precision_at_10 = calculate_precision_at_topk(probs_array, targets_array, k=10)
    
    return {
        'accuracy': float(accuracy),
        'f1_score': float(f1),
        'precision_at_top10': float(precision_at_10)
    }


def run_enhanced_ablation_studies():
    """
    Run enhanced ablation studies with actual retraining.
    """
    print("=" * 60)
    print("🔬 Enhanced Ablation Studies for A+ Grade")
    print("=" * 60)
    
    # Get all dates
    graph_files = sorted(list(DATA_GRAPHS_DIR.glob('graph_t_*.pt')))
    all_dates = [pd.to_datetime(f.stem.split('_')[-1]) for f in graph_files]
    
    # Split dates
    split_70_idx = int(len(all_dates) * 0.70)
    split_85_idx = int(len(all_dates) * 0.85)
    
    train_dates = all_dates[:split_70_idx]
    val_dates = all_dates[split_70_idx:split_85_idx]
    test_dates = all_dates[split_85_idx:]
    
    print(f"\n📅 Date Split:")
    print(f"   Train: {len(train_dates)} dates ({train_dates[0]} to {train_dates[-1]})")
    print(f"   Val: {len(val_dates)} dates ({val_dates[0]} to {val_dates[-1]})")
    print(f"   Test: {len(test_dates)} dates ({test_dates[0]} to {test_dates[-1]})")
    
    # Get tickers and input dimension
    sample_graph = torch.load(graph_files[0], weights_only=False)
    input_dim = sample_graph['stock'].x.shape[1]
    
    from src.training.transformer_trainer import _read_time_series_csv, OHLCV_RAW_FILE
    ohlcv_df = _read_time_series_csv(OHLCV_RAW_FILE)
    tickers = [col.replace('Close_', '') for col in ohlcv_df.columns if col.startswith('Close_')]
    
    # Create targets
    targets_class_dict, targets_reg_dict = create_target_labels(tickers, all_dates, lookahead_days=5)
    
    # Run ablation studies
    ablation_results = {}
    
    for config_name, config in ABLATION_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"🔬 Processing: {config_name}")
        print(f"{'='*60}")
        
        try:
            # Train model
            model, train_metrics = train_ablation_model(
                config_name=config_name,
                config=config,
                train_dates=train_dates,
                val_dates=val_dates,
                targets_class_dict=targets_class_dict,
                targets_reg_dict=targets_reg_dict,
                input_dim=input_dim
            )
            
            # Evaluate on test set
            test_metrics = evaluate_ablation_model(
                model=model,
                config=config,
                test_dates=test_dates,
                targets_class_dict=targets_class_dict,
                tickers=tickers
            )
            
            ablation_results[config_name] = {
                **test_metrics,
                'train_val_f1': train_metrics.get('best_val_f1', 0),
                'description': config['description']
            }
            
            print(f"\n✅ {config_name} Results:")
            print(f"   Test Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"   Test F1: {test_metrics['f1_score']:.4f}")
            print(f"   Precision@Top-10: {test_metrics['precision_at_top10']:.4f}")
            
        except Exception as e:
            print(f"❌ Error training {config_name}: {e}")
            import traceback
            traceback.print_exc()
            ablation_results[config_name] = {
                'accuracy': 0.0,
                'f1_score': 0.0,
                'precision_at_top10': 0.0,
                'error': str(e)
            }
    
    # Save results
    results_df = pd.DataFrame(ablation_results).T
    results_df.to_csv(RESULTS_DIR / 'enhanced_ablation_results.csv')
    
    with open(RESULTS_DIR / 'enhanced_ablation_results.json', 'w') as f:
        json.dump(ablation_results, f, indent=2, default=str)
    
    print("\n" + "=" * 60)
    print("✅ Enhanced Ablation Studies Complete!")
    print("=" * 60)
    print(f"\n📁 Results saved to:")
    print(f"  - {RESULTS_DIR / 'enhanced_ablation_results.csv'}")
    print(f"  - {RESULTS_DIR / 'enhanced_ablation_results.json'}")
    
    return ablation_results


if __name__ == '__main__':
    run_enhanced_ablation_studies()


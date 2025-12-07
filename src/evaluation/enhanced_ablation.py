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


def filter_graph_edges(data, config: Dict):
    """
    Filter graph edges based on ablation configuration.
    Returns a new HeteroData with only the specified edge types.
    """
    from torch_geometric.data import HeteroData
    
    # Create new HeteroData
    filtered_data = HeteroData()
    
    # Copy node features
    filtered_data['stock'].x = data['stock'].x.clone()
    
    # Initialize edge_index_dict
    filtered_data.edge_index_dict = {}
    
    # Build edge_index_dict from data if it doesn't exist
    if hasattr(data, 'edge_index_dict') and isinstance(data.edge_index_dict, dict) and len(data.edge_index_dict) > 0:
        edge_index_dict = data.edge_index_dict
    else:
        # Build edge_index_dict from edge_types
        edge_index_dict = {}
        for edge_type in data.edge_types:
            try:
                if hasattr(data[edge_type], 'edge_index') and data[edge_type].edge_index is not None:
                    edge_index_dict[edge_type] = data[edge_type].edge_index
            except (KeyError, AttributeError):
                continue
    
    # Filter edges based on configuration
    if config['use_correlation']:
        edge_key = ('stock', 'rolling_correlation', 'stock')
        if edge_key in edge_index_dict:
            filtered_data.edge_index_dict[edge_key] = edge_index_dict[edge_key].clone()
        elif edge_key in data.edge_types:
            try:
                filtered_data.edge_index_dict[edge_key] = data[edge_key].edge_index.clone()
            except (KeyError, AttributeError):
                pass  # Edge type doesn't exist, skip
    
    if config['use_fund_similarity']:
        edge_key = ('stock', 'fund_similarity', 'stock')
        if edge_key in edge_index_dict:
            filtered_data.edge_index_dict[edge_key] = edge_index_dict[edge_key].clone()
        elif edge_key in data.edge_types:
            try:
                filtered_data.edge_index_dict[edge_key] = data[edge_key].edge_index.clone()
            except (KeyError, AttributeError):
                pass
    
    if config['use_static_edges']:
        edge_key = ('stock', 'sector_industry', 'stock')
        if edge_key in edge_index_dict:
            filtered_data.edge_index_dict[edge_key] = edge_index_dict[edge_key].clone()
        elif edge_key in data.edge_types:
            try:
                filtered_data.edge_index_dict[edge_key] = data[edge_key].edge_index.clone()
            except (KeyError, AttributeError):
                pass
        
        edge_key = ('stock', 'supply_competitor', 'stock')
        if edge_key in edge_index_dict:
            filtered_data.edge_index_dict[edge_key] = edge_index_dict[edge_key].clone()
        elif edge_key in data.edge_types:
            try:
                filtered_data.edge_index_dict[edge_key] = data[edge_key].edge_index.clone()
            except (KeyError, AttributeError):
                pass
    
    # Ensure at least one edge type exists (required by model)
    if len(filtered_data.edge_index_dict) == 0:
        # If no edges match, create a minimal self-loop graph to prevent errors
        num_nodes = filtered_data['stock'].x.shape[0]
        # Create self-loops as fallback
        import torch
        self_loops = torch.arange(num_nodes, dtype=torch.long, device=filtered_data['stock'].x.device).repeat(2, 1)
        fallback_key = ('stock', 'rolling_correlation', 'stock')
        filtered_data.edge_index_dict[fallback_key] = self_loops
        print(f"  Warning: No edges found for config, using self-loops as fallback")
    
    # Copy tickers if they exist
    if hasattr(data, 'tickers'):
        filtered_data.tickers = data.tickers
    
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
    print(f" Training Ablation Model: {config_name}")
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
    
    # Train model using simplified training loop with filtered graphs
    from src.training.transformer_trainer import (
        FocalLoss, FOCAL_ALPHA, FOCAL_GAMMA, USE_CLASS_WEIGHTS
    )
    import torch.optim as optim
    from sklearn.metrics import f1_score, accuracy_score
    from tqdm import tqdm
    
    # Calculate class weights if needed
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
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train the model with filtered graphs
    best_val_f1 = 0.0
    best_model_state = None
    REFERENCE_DATE = pd.to_datetime('2015-01-01')
    
    print(f"Training {config_name} for {NUM_EPOCHS} epochs...")
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for date in tqdm(train_dates, desc=f"Epoch {epoch}/{NUM_EPOCHS}"):
            data = load_filtered_graph_data(date)
            target_class = targets_class_dict.get(date)
            target_reg = targets_reg_dict.get(date) if ENABLE_MULTI_TASK else None
            
            if data and target_class is not None:
                data = data.to(DEVICE)
                target_class = target_class.to(DEVICE)
                
                # Create date tensor if time-aware
                if ENABLE_TIME_AWARE:
                    days_since_ref = (date - REFERENCE_DATE).days
                    num_nodes = data['stock'].x.shape[0]
                    date_tensor = torch.full((num_nodes,), days_since_ref, dtype=torch.float32).to(DEVICE)
                else:
                    date_tensor = None
                
                optimizer.zero_grad()
                out = model(data, date_tensor=date_tensor)
                
                # Calculate loss
                criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
                loss = criterion(out, target_class)
                
                if ENABLE_MULTI_TASK and target_reg is not None:
                    target_reg = target_reg.to(DEVICE)
                    reg_loss = torch.nn.functional.mse_loss(out[:, 1], target_reg)
                    loss = loss + 0.1 * reg_loss
                
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
                data = load_filtered_graph_data(date)
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
        print(f"  Epoch {epoch}/{NUM_EPOCHS}: Loss={avg_loss:.4f}, Val Acc={val_accuracy:.4f}, Val F1={val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Save model
    torch.save(model.state_dict(), MODELS_DIR / f"ablation_{config_name.lower()}.pt")
    
    metrics = {
        'best_val_f1': best_val_f1,
        'best_val_accuracy': val_accuracy
    }
    
    best_model = model
    
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
    print(" Enhanced Ablation Studies for A+ Grade")
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
    
    print(f"\n Date Split:")
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
        print(f" Processing: {config_name}")
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
            
            print(f"\n {config_name} Results:")
            print(f"   Test Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"   Test F1: {test_metrics['f1_score']:.4f}")
            print(f"   Precision@Top-10: {test_metrics['precision_at_top10']:.4f}")
            
        except Exception as e:
            print(f" Error training {config_name}: {e}")
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
    print(" Enhanced Ablation Studies Complete!")
    print("=" * 60)
    print(f"\n Results saved to:")
    print(f"  - {RESULTS_DIR / 'enhanced_ablation_results.csv'}")
    print(f"  - {RESULTS_DIR / 'enhanced_ablation_results.json'}")
    
    return ablation_results


if __name__ == '__main__':
    run_enhanced_ablation_studies()


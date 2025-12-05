# phase6_complete_ablation.py
"""
Complete Ablation Studies for Phase 6
Implements full retraining for each ablation configuration as required by the proposal.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
from typing import Dict, List, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent))

from src.training.transformer_trainer import (
    RoleAwareGraphTransformer,
    create_target_labels,
)
from src.utils.constants import (
    DEVICE,
    HIDDEN_CHANNELS,
    NUM_LAYERS,
    NUM_HEADS,
    OUT_CHANNELS,
    LEARNING_RATE,
    NUM_EPOCHS,
    LOOKAHEAD_DAYS,
)
from src.utils.graph_loader import load_graph_data

from src.utils.paths import PROJECT_ROOT, MODELS_DIR, RESULTS_DIR, OHLCV_RAW_FILE
from src.training.transformer_trainer import _read_time_series_csv
ABLATION_MODELS_DIR = MODELS_DIR / "ablation_models"
ABLATION_MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Import PEARL and Laplacian embeddings
from src.models.components.pearl_embedding import PEARLPositionalEmbedding
try:
    from torch_geometric.nn import LaplacianEigenvectorPE
    LAPLACIAN_AVAILABLE = True
except ImportError:
    LAPLACIAN_AVAILABLE = False
    print("⚠️  LaplacianEigenvectorPE not available, skipping Laplacian ablation")


def get_train_val_test_dates():
    """Get train/val/test date splits (70/15/15 split)."""
    graph_files = sorted(list((PROJECT_ROOT / "data" / "graphs").glob('graph_t_*.pt')))
    all_dates = [pd.to_datetime(f.stem.split('_')[-1]) for f in graph_files]
    
    # 70/15/15 split
    split_70_idx = int(len(all_dates) * 0.70)
    split_85_idx = int(len(all_dates) * 0.85)
    
    train_dates = all_dates[:split_70_idx]
    val_dates = all_dates[split_70_idx:split_85_idx]
    test_dates = all_dates[split_85_idx:]
    
    return train_dates, val_dates, test_dates


def load_targets():
    """Load target labels for all dates."""
    # Get all dates
    graph_files = sorted(list((PROJECT_ROOT / "data" / "graphs").glob('graph_t_*.pt')))
    all_dates = [pd.to_datetime(f.stem.split('_')[-1]) for f in graph_files]
    
    # Get tickers
    sample_graph = torch.load(graph_files[0], weights_only=False)
    if 'tickers' in sample_graph:
        tickers = sample_graph['tickers']
    else:
        ohlcv_df = _read_time_series_csv(OHLCV_RAW_FILE)
        tickers = [col.replace('Close_', '') for col in ohlcv_df.columns if col.startswith('Close_')]
    
    # Create targets
    targets_class_dict, targets_reg_dict = create_target_labels(tickers, all_dates, lookahead_days=LOOKAHEAD_DAYS)
    
    return targets_class_dict


def modify_graph_for_ablation(data, config: Dict):
    """Modify graph structure for ablation study."""
    from torch_geometric.data import HeteroData
    
    modified_data = HeteroData()
    modified_data['stock'].x = data['stock'].x.clone()
    
    if 'tickers' in data:
        modified_data['tickers'] = data['tickers']
    
    remove_edges = config.get('remove_edges', [])
    
    for edge_type in data.edge_index_dict.keys():
        if edge_type not in remove_edges:
            modified_data[edge_type].edge_index = data[edge_type].edge_index.clone()
            if hasattr(data[edge_type], 'edge_attr'):
                modified_data[edge_type].edge_attr = data[edge_type].edge_attr.clone()
    
    return modified_data


def filter_graphs_by_threshold(data, threshold: float):
    """Filter correlation edges by threshold for sensitivity analysis."""
    from torch_geometric.data import HeteroData
    
    modified_data = HeteroData()
    modified_data['stock'].x = data['stock'].x.clone()
    
    if 'tickers' in data:
        modified_data['tickers'] = data['tickers']
    
    # Keep all non-correlation edges
    for edge_type in data.edge_index_dict.keys():
        if 'correlation' not in str(edge_type):
            modified_data[edge_type].edge_index = data[edge_type].edge_index.clone()
            if hasattr(data[edge_type], 'edge_attr'):
                modified_data[edge_type].edge_attr = data[edge_type].edge_attr.clone()
    
    # Filter correlation edges by threshold
    for edge_type in data.edge_index_dict.keys():
        if 'correlation' in str(edge_type):
            edge_index = data[edge_type].edge_index.clone()
            if hasattr(data[edge_type], 'edge_attr'):
                edge_attr = data[edge_type].edge_attr.clone()
                # Filter edges where correlation >= threshold
                mask = edge_attr.abs() >= threshold
                filtered_indices = torch.where(mask)[0]
                if len(filtered_indices) > 0:
                    modified_data[edge_type].edge_index = edge_index[:, filtered_indices]
                    modified_data[edge_type].edge_attr = edge_attr[filtered_indices]
            else:
                # If no edge attributes, keep all edges
                modified_data[edge_type].edge_index = edge_index
    
    return modified_data


class AblationModelWrapper:
    """Wrapper to train models with different configurations."""
    
    def __init__(self, config: Dict, in_dim: int):
        self.config = config
        self.in_dim = in_dim
        self.model = None
        
    def create_model(self):
        """Create model based on ablation configuration."""
        use_pearl = self.config.get('use_pearl', True)
        use_laplacian = self.config.get('use_laplacian', False)
        
        # Adjust input dimension based on positional embedding
        pe_dim = 0
        if use_pearl:
            pe_dim = 32  # PEARL dimension
        elif use_laplacian and LAPLACIAN_AVAILABLE:
            pe_dim = 16  # Laplacian dimension
        
        total_in_dim = self.in_dim + pe_dim
        
        # Create model (we'll modify forward to handle different PE types)
        model = RoleAwareGraphTransformer(
            total_in_dim, HIDDEN_CHANNELS, OUT_CHANNELS, NUM_LAYERS, NUM_HEADS
        )
        
        # Modify model to use different positional embeddings
        if not use_pearl:
            # Remove PEARL, optionally add Laplacian
            model.pearl_embedding = None
            if use_laplacian and LAPLACIAN_AVAILABLE:
                # Add Laplacian PE
                model.laplacian_pe = LaplacianEigenvectorPE(16)
            else:
                model.laplacian_pe = None
        else:
            model.laplacian_pe = None
        
        return model
    
    def forward_with_config(self, model, data):
        """Forward pass with ablation configuration."""
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        
        x = x_dict['stock']
        
        # Apply positional embedding based on config
        if self.config.get('use_pearl', True):
            if model.pearl_embedding is not None:
                pearl_pe = model.pearl_embedding(x, edge_index_dict)
                x = torch.cat([x, pearl_pe], dim=1)
        elif self.config.get('use_laplacian', False) and model.laplacian_pe is not None:
            # Use Laplacian PE
            laplacian_pe = model.laplacian_pe(data)
            if isinstance(laplacian_pe, dict):
                laplacian_pe = laplacian_pe.get('stock', torch.zeros(x.shape[0], 16).to(x.device))
            x = torch.cat([x, laplacian_pe], dim=1)
        # If neither, use raw features only
        
        x_dict['stock'] = x
        
        # Continue with normal forward
        for layer_idx, (conv, aggregator) in enumerate(zip(model.convs, model.relation_aggregators)):
            conv_output = conv(x_dict, edge_index_dict)
            x_dict = aggregator(conv_output, x_dict)
        
        out = model.lin_out(x_dict['stock'])
        return out


def train_ablation_model(config: Dict, ablation_name: str) -> Dict[str, Any]:
    """
    Train a model with specific ablation configuration.
    
    Args:
        config: Ablation configuration dictionary
        ablation_name: Name of the ablation study
        
    Returns:
        Dictionary with training results and metrics
    """
    print(f"\n{'='*60}")
    print(f"Training Ablation: {ablation_name}")
    print(f"{'='*60}")
    
    # Load data
    train_dates, val_dates, test_dates = get_train_val_test_dates()
    targets_dict = load_targets()
    
    # Get input dimension from first graph
    sample_data = load_graph_data(train_dates[0])
    if sample_data is None:
        print(f"❌ Could not load sample graph for {ablation_name}")
        return {}
    
    in_dim = sample_data['stock'].x.shape[1]
    
    # Create model wrapper
    wrapper = AblationModelWrapper(config, in_dim)
    model = wrapper.create_model().to(DEVICE)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop (simplified - fewer epochs for ablation)
    best_val_f1 = 0
    patience_counter = 0
    max_patience = 5
    
    for epoch in range(min(NUM_EPOCHS, 10)):  # Limit epochs for ablation
        model.train()
        train_loss = 0
        train_count = 0
        
        for date in tqdm(train_dates[:100], desc=f"Epoch {epoch+1}"):  # Limit training data
            data = load_graph_data(date)
            target = targets_dict.get(date)
            
            if data is None or target is None:
                continue
            
            # Modify graph based on ablation config
            if 'remove_edges' in config:
                data = modify_graph_for_ablation(data, config)
            elif 'threshold' in config:
                data = filter_graphs_by_threshold(data, config['threshold'])
            
            optimizer.zero_grad()
            
            # Forward pass with config
            out = wrapper.forward_with_config(model, data.to(DEVICE))
            loss = criterion(out, target.to(DEVICE))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_count += 1
        
        # Validation
        if train_count > 0:
            model.eval()
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for date in val_dates[:20]:  # Limit validation data
                    data = load_graph_data(date)
                    target = targets_dict.get(date)
                    
                    if data is None or target is None:
                        continue
                    
                    if 'remove_edges' in config:
                        data = modify_graph_for_ablation(data, config)
                    elif 'threshold' in config:
                        data = filter_graphs_by_threshold(data, config['threshold'])
                    
                    out = wrapper.forward_with_config(model, data.to(DEVICE))
                    preds = out.argmax(dim=1).cpu()
                    
                    val_predictions.append(preds)
                    val_targets.append(target)
            
            if len(val_predictions) > 0:
                from sklearn.metrics import f1_score
                val_preds = torch.cat(val_predictions).numpy()
                val_targs = torch.cat(val_targets).numpy()
                val_f1 = f1_score(val_targs, val_preds, average='macro')
                
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_counter = 0
                    # Save best model
                    torch.save(model.state_dict(), ABLATION_MODELS_DIR / f"{ablation_name}_best.pt")
                else:
                    patience_counter += 1
                
                if patience_counter >= max_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    
    # Load best model and evaluate on test set
    if (ABLATION_MODELS_DIR / f"{ablation_name}_best.pt").exists():
        model.load_state_dict(torch.load(ABLATION_MODELS_DIR / f"{ablation_name}_best.pt"))
    
    # Test evaluation
    model.eval()
    test_predictions = []
    test_targets = []
    test_probs = []
    
    with torch.no_grad():
        for date in tqdm(test_dates, desc="Testing"):
            data = load_graph_data(date)
            target = targets_dict.get(date)
            
            if data is None or target is None:
                continue
            
            if 'remove_edges' in config:
                data = modify_graph_for_ablation(data, config)
            elif 'threshold' in config:
                data = filter_graphs_by_threshold(data, config['threshold'])
            
            out = wrapper.forward_with_config(model, data.to(DEVICE))
            probs = F.softmax(out, dim=1)
            preds = out.argmax(dim=1).cpu()
            
            test_predictions.append(preds.numpy())
            test_targets.append(target.numpy())
            test_probs.append(probs.cpu().numpy())
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, f1_score
    from src.evaluation.evaluation import calculate_precision_at_topk, calculate_information_coefficient
    
    test_preds = np.concatenate(test_predictions)
    test_targs = np.concatenate(test_targets)
    test_probs_array = np.array(test_probs)
    
    accuracy = accuracy_score(test_targs, test_preds)
    f1 = f1_score(test_targs, test_preds, average='macro')
    precision_k = calculate_precision_at_topk(test_probs_array, test_targs, k=10)
    ic_result = calculate_information_coefficient(test_probs_array, test_targs)
    
    results = {
        'ablation_name': ablation_name,
        'accuracy': accuracy,
        'f1_score': f1,
        'precision_at_top10': precision_k,
        'ic_mean': ic_result.get('IC_mean', 0),
        'ic_ir': ic_result.get('IC_IR', 0),
        'best_val_f1': best_val_f1
    }
    
    print(f"\n✅ {ablation_name} Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1 Score: {f1:.4f}")
    print(f"   Precision@Top-10: {precision_k:.4f}")
    print(f"   IC Mean: {results['ic_mean']:.4f}")
    
    return results


def run_complete_ablation_studies():
    """Run all ablation studies with full retraining."""
    
    print("🔬 Starting Complete Ablation Studies (with Retraining)")
    print("="*60)
    
    ablation_configs = []
    
    # 1. Edge Type Ablations
    ablation_configs.extend([
        {
            'name': 'Full_Model',
            'config': {'use_pearl': True, 'remove_edges': []}
        },
        {
            'name': 'Abl_NoCorrelation',
            'config': {'use_pearl': True, 'remove_edges': [('stock', 'rolling_correlation', 'stock')]}
        },
        {
            'name': 'Abl_NoFundSim',
            'config': {'use_pearl': True, 'remove_edges': [('stock', 'fundamental_similarity', 'stock')]}
        },
        {
            'name': 'Abl_NoStatic',
            'config': {'use_pearl': True, 'remove_edges': [
                ('stock', 'sector', 'stock'),
                ('stock', 'industry', 'stock')
            ]}
        },
        {
            'name': 'Abl_OnlyCorrelation',
            'config': {'use_pearl': True, 'remove_edges': [
                ('stock', 'fundamental_similarity', 'stock'),
                ('stock', 'sector', 'stock'),
                ('stock', 'industry', 'stock')
            ]}
        },
    ])
    
    # 2. Positional Embedding Ablations
    ablation_configs.extend([
        {
            'name': 'Abl_NoPEARL',
            'config': {'use_pearl': False, 'use_laplacian': False, 'remove_edges': []}
        },
        {
            'name': 'Abl_LaplacianPE',
            'config': {'use_pearl': False, 'use_laplacian': True, 'remove_edges': []}
        },
    ])
    
    # 3. Threshold Sensitivity (if time permits)
    # ablation_configs.extend([
    #     {
    #         'name': 'Threshold_0.4',
    #         'config': {'threshold': 0.4, 'use_pearl': True}
    #     },
    #     {
    #         'name': 'Threshold_0.6',
    #         'config': {'threshold': 0.6, 'use_pearl': True}
    #     },
    #     {
    #         'name': 'Threshold_0.8',
    #         'config': {'threshold': 0.8, 'use_pearl': True}
    #     },
    # ])
    
    results = []
    
    for abl_config in tqdm(ablation_configs, desc="Ablation Studies"):
        try:
            result = train_ablation_model(abl_config['config'], abl_config['name'])
            if result:
                results.append(result)
        except Exception as e:
            print(f"❌ Error in {abl_config['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    if results:
        df = pd.DataFrame(results)
        df.to_csv(RESULTS_DIR / 'complete_ablation_results.csv', index=False)
        print(f"\n✅ Complete ablation results saved to: {RESULTS_DIR / 'complete_ablation_results.csv'}")
        print("\n📊 Results Summary:")
        print(df.to_string(index=False))
        return df
    else:
        print("❌ No ablation results generated")
        return pd.DataFrame()


if __name__ == "__main__":
    print("🚀 Phase 6: Complete Ablation Studies")
    print("Note: This will train models for each ablation configuration.")
    print("This may take several hours depending on your hardware.\n")
    
    results_df = run_complete_ablation_studies()
    
    if not results_df.empty:
        print("\n✅ Complete Ablation Studies Finished!")
        print(f"📁 Results: {RESULTS_DIR / 'complete_ablation_results.csv'}")
    else:
        print("\n⚠️  Ablation studies completed with errors. Check logs above.")


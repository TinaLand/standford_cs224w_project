# scripts/visualization.py
"""
Visualization scripts for Phase 6 evaluation.
Implements:
1. t-SNE/UMAP embedding visualization
2. Attention weights heatmap visualization
3. Role analysis visualization (Hubs, Bridges, Role Twins)
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import sys

# Try to import UMAP (optional)
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("  UMAP not available. Install with: pip install umap-learn")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / 'scripts'))
sys.path.append(str(PROJECT_ROOT / 'scripts' / 'components'))

from src.training.transformer_trainer import RoleAwareGraphTransformer
from src.utils.graph_loader import load_graph_data
from utils_data import load_data_file

# Configuration
MODELS_DIR = PROJECT_ROOT / "models"
DATA_GRAPHS_DIR = PROJECT_ROOT / "data" / "graphs"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = MODELS_DIR / "plots"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ensure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Fix PyTorch serialization
import torch.serialization
from torch_geometric.data.storage import BaseStorage, NodeStorage, EdgeStorage, GlobalStorage
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([BaseStorage, NodeStorage, EdgeStorage, GlobalStorage])


def load_trained_model():
    """Load the trained Core Transformer model."""
    model_path = MODELS_DIR / 'core_transformer_model.pt'
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load sample graph to get input dimension
    sample_graph = list(DATA_GRAPHS_DIR.glob('graph_t_*.pt'))[0]
    temp_data = torch.load(sample_graph, weights_only=False)
    INPUT_DIM = temp_data['stock'].x.shape[1]
    
    # Initialize and load model
    model = RoleAwareGraphTransformer(INPUT_DIM, 256, 2, 2, 4).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=False))
    model.eval()
    
    return model, INPUT_DIM


def extract_embeddings(model, dates, tickers):
    """
    Extract node embeddings from the trained model for multiple dates.
    Returns: embeddings_dict[date] = np.array of shape (num_stocks, embedding_dim)
    """
    print(f"\n Extracting embeddings for {len(dates)} dates...")
    embeddings_dict = {}
    
    with torch.no_grad():
        for date in dates:
            data = load_graph_data(date)
            if data is None:
                continue
            
            # Get embeddings (before final classification layer)
            # The model's forward method returns embeddings
            data = data.to(DEVICE)
            embeddings = model(data)  # Shape: (num_stocks, embedding_dim)
            embeddings_dict[date] = embeddings.cpu().numpy()
    
    print(f" Extracted embeddings for {len(embeddings_dict)} dates")
    return embeddings_dict


def visualize_embeddings_tsne(embeddings_dict, tickers, output_path=None):
    """
    Visualize embeddings using t-SNE.
    """
    print("\n Creating t-SNE visualization...")
    
    # Use embeddings from the most recent date
    if not embeddings_dict:
        print(" No embeddings available")
        return
    
    latest_date = max(embeddings_dict.keys())
    embeddings = embeddings_dict[latest_date]
    
    # Apply t-SNE
    print("   Applying t-SNE (this may take a moment)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(tickers)-1))
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         s=100, alpha=0.6, c=range(len(tickers)), cmap='viridis')
    
    # Add ticker labels
    for i, ticker in enumerate(tickers):
        plt.annotate(ticker, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                    fontsize=8, alpha=0.7)
    
    plt.colorbar(scatter, label='Stock Index')
    plt.title(f't-SNE Visualization of Stock Embeddings\n(Date: {latest_date.date()})', fontsize=14)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.tight_layout()
    
    if output_path is None:
        output_path = PLOTS_DIR / 'embeddings_tsne.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f" Saved t-SNE plot to: {output_path}")
    plt.close()


def visualize_embeddings_umap(embeddings_dict, tickers, output_path=None):
    """
    Visualize embeddings using UMAP (if available).
    """
    if not UMAP_AVAILABLE:
        print("  UMAP not available, skipping UMAP visualization")
        return
    
    print("\n Creating UMAP visualization...")
    
    if not embeddings_dict:
        print(" No embeddings available")
        return
    
    latest_date = max(embeddings_dict.keys())
    embeddings = embeddings_dict[latest_date]
    
    # Apply UMAP
    print("   Applying UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(tickers)-1))
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         s=100, alpha=0.6, c=range(len(tickers)), cmap='viridis')
    
    # Add ticker labels
    for i, ticker in enumerate(tickers):
        plt.annotate(ticker, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                    fontsize=8, alpha=0.7)
    
    plt.colorbar(scatter, label='Stock Index')
    plt.title(f'UMAP Visualization of Stock Embeddings\n(Date: {latest_date.date()})', fontsize=14)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.tight_layout()
    
    if output_path is None:
        output_path = PLOTS_DIR / 'embeddings_umap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f" Saved UMAP plot to: {output_path}")
    plt.close()


def visualize_attention_weights(model, data, tickers, output_path=None):
    """
    Visualize attention weights as a heatmap.
    Note: This requires modifying the model to return attention weights.
    For now, we'll create a placeholder visualization.
    """
    print("\n Creating attention weights heatmap...")
    
    # TODO: Modify model to return attention weights
    # For now, create a placeholder based on graph structure
    data = data.to(DEVICE)
    
    # Get edge indices to create adjacency-like matrix
    num_stocks = len(tickers)
    attention_matrix = np.zeros((num_stocks, num_stocks))
    
    # Sum attention from all edge types
    for edge_type in data.edge_index_dict.keys():
        edge_index = data[edge_type].edge_index.cpu().numpy()
        if len(edge_index) > 0:
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[0, i], edge_index[1, i]
                if src < num_stocks and dst < num_stocks:
                    attention_matrix[src, dst] += 1.0
    
    # Normalize
    attention_matrix = attention_matrix / (attention_matrix.sum(axis=1, keepdims=True) + 1e-8)
    
    # Create heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(attention_matrix, 
                xticklabels=tickers, 
                yticklabels=tickers,
                cmap='YlOrRd', 
                cbar_kws={'label': 'Attention Weight'},
                square=True,
                linewidths=0.5)
    plt.title('Attention Weights Heatmap (Graph Structure Based)', fontsize=14)
    plt.xlabel('Target Stock')
    plt.ylabel('Source Stock')
    plt.tight_layout()
    
    if output_path is None:
        output_path = PLOTS_DIR / 'attention_weights_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f" Saved attention heatmap to: {output_path}")
    plt.close()


def analyze_roles(embeddings_dict, tickers, output_path=None):
    """
    Analyze and visualize stock roles (Hubs, Bridges, Role Twins).
    """
    print("\n Analyzing stock roles...")
    
    if not embeddings_dict:
        print(" No embeddings available")
        return
    
    # Use embeddings from multiple dates to get average
    all_embeddings = []
    for date, emb in embeddings_dict.items():
        all_embeddings.append(emb)
    avg_embeddings = np.mean(all_embeddings, axis=0)
    
    # Calculate pairwise distances
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(avg_embeddings))
    
    # Identify Hubs: stocks with high average connection strength
    # (In a real implementation, this would use graph centrality measures)
    hub_scores = distances.mean(axis=1)
    hub_indices = np.argsort(hub_scores)[-10:]  # Top 10 hubs
    
    # Identify Role Twins: stocks with similar embeddings
    # Find pairs with smallest distances
    role_twins = []
    for i in range(len(tickers)):
        for j in range(i+1, len(tickers)):
            if distances[i, j] < np.percentile(distances[distances > 0], 10):
                role_twins.append((tickers[i], tickers[j], distances[i, j]))
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Hub scores
    hub_scores_sorted = sorted(enumerate(hub_scores), key=lambda x: x[1], reverse=True)
    top_hubs = [tickers[i] for i, _ in hub_scores_sorted[:10]]
    top_scores = [score for _, score in hub_scores_sorted[:10]]
    
    axes[0].barh(range(len(top_hubs)), top_scores)
    axes[0].set_yticks(range(len(top_hubs)))
    axes[0].set_yticklabels(top_hubs)
    axes[0].set_xlabel('Hub Score (Average Distance)')
    axes[0].set_title('Top 10 Hub Stocks')
    axes[0].invert_yaxis()
    
    # Plot 2: Role Twins (similar stocks)
    if role_twins:
        top_twins = sorted(role_twins, key=lambda x: x[2])[:10]
        twin_pairs = [f"{t1}-{t2}" for t1, t2, _ in top_twins]
        twin_distances = [d for _, _, d in top_twins]
        
        axes[1].barh(range(len(twin_pairs)), twin_distances)
        axes[1].set_yticks(range(len(twin_pairs)))
        axes[1].set_yticklabels(twin_pairs)
        axes[1].set_xlabel('Embedding Distance')
        axes[1].set_title('Top 10 Role Twin Pairs (Most Similar)')
        axes[1].invert_yaxis()
    else:
        axes[1].text(0.5, 0.5, 'No role twins found', 
                    ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Role Twins')
    
    plt.tight_layout()
    
    if output_path is None:
        output_path = PLOTS_DIR / 'role_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f" Saved role analysis to: {output_path}")
    plt.close()
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'ticker': tickers,
        'hub_score': hub_scores
    })
    results_df = results_df.sort_values('hub_score', ascending=False)
    results_df.to_csv(RESULTS_DIR / 'role_analysis.csv', index=False)
    print(f" Saved role analysis results to: {RESULTS_DIR / 'role_analysis.csv'}")


def main():
    """Main function to run all visualizations."""
    print(" Starting Visualization Pipeline")
    print("=" * 50)
    
    # Load model
    try:
        model, INPUT_DIM = load_trained_model()
        print(" Model loaded successfully")
    except Exception as e:
        print(f" Error loading model: {e}")
        return
    
    # Get tickers from a sample graph
    sample_graph = list(DATA_GRAPHS_DIR.glob('graph_t_*.pt'))[0]
    sample_data = torch.load(sample_graph, weights_only=False)
    if 'tickers' in sample_data:
        tickers = sample_data['tickers']
    else:
        # Fallback: use default ticker list
        tickers = [f"STOCK_{i}" for i in range(50)]
        print("  Tickers not found in graph, using default")
    
    # Get dates for embedding extraction (use recent dates)
    all_dates = sorted([pd.to_datetime(f.stem.split('_')[-1]) 
                       for f in DATA_GRAPHS_DIR.glob('graph_t_*.pt')])
    recent_dates = all_dates[-10:]  # Use last 10 dates
    
    # Extract embeddings
    try:
        embeddings_dict = extract_embeddings(model, recent_dates, tickers)
    except Exception as e:
        print(f" Error extracting embeddings: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create visualizations
    try:
        visualize_embeddings_tsne(embeddings_dict, tickers)
        visualize_embeddings_umap(embeddings_dict, tickers)
        
        # Use latest date for attention visualization
        latest_date = max(embeddings_dict.keys())
        latest_data = load_graph_data(latest_date)
        if latest_data:
            visualize_attention_weights(model, latest_data, tickers)
        
        analyze_roles(embeddings_dict, tickers)
        
        print("\n" + "=" * 50)
        print(" All visualizations completed!")
        print(f" Plots saved to: {PLOTS_DIR}")
        print(f" Results saved to: {RESULTS_DIR}")
        
    except Exception as e:
        print(f" Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()


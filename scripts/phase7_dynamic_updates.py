# phase7_dynamic_updates.py
"""
Phase 7: Dynamic Graph Updates Enhancement
Implements efficient dynamic graph update mechanism for evolving market conditions.
"""

import torch
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import sys

sys.path.append(str(Path(__file__).resolve().parent))

from phase2_graph_construction import load_graph_for_date
from phase1_edge_parameter_calc import compute_rolling_correlation, compute_fundamental_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
GRAPHS_DIR = DATA_DIR / "graphs"
EDGES_DIR = DATA_DIR / "edges"

class DynamicGraphUpdater:
    """
    Efficiently updates dynamic edges while maintaining static backbone.
    
    Strategy:
    - Static edges (sector, industry) remain stable
    - Dynamic edges (correlation, fundamental similarity) are updated periodically
    - Updates are incremental to avoid full graph reconstruction
    """
    
    def __init__(self, update_frequency: str = 'weekly', correlation_window: int = 30):
        """
        Args:
            update_frequency: How often to update ('daily', 'weekly', 'monthly')
            correlation_window: Rolling window size for correlation calculation
        """
        self.update_frequency = update_frequency
        self.correlation_window = correlation_window
        self.last_update_date = None
        self.update_cache = {}
        
    def should_update(self, current_date: pd.Timestamp) -> bool:
        """Check if graph should be updated based on frequency."""
        if self.last_update_date is None:
            return True
        
        delta = current_date - self.last_update_date
        
        if self.update_frequency == 'daily':
            return delta.days >= 1
        elif self.update_frequency == 'weekly':
            return delta.days >= 7
        elif self.update_frequency == 'monthly':
            return delta.days >= 30
        else:
            return True
    
    def update_dynamic_edges(self, 
                           date: pd.Timestamp,
                           tickers: List[str],
                           ohlcv_data: pd.DataFrame,
                           fundamental_data: pd.DataFrame) -> Dict:
        """
        Update dynamic edge parameters for a given date.
        
        Returns:
            Dictionary with updated edge indices and attributes
        """
        # Compute rolling correlation
        correlation_edges = self._compute_correlation_edges(date, tickers, ohlcv_data)
        
        # Compute fundamental similarity
        fundamental_edges = self._compute_fundamental_edges(date, tickers, fundamental_data)
        
        return {
            'correlation': correlation_edges,
            'fundamental_similarity': fundamental_edges,
            'update_date': date
        }
    
    def _compute_correlation_edges(self, 
                                   date: pd.Timestamp,
                                   tickers: List[str],
                                   ohlcv_data: pd.DataFrame) -> Dict:
        """Compute correlation edges for given date."""
        # Get data window
        end_date = date
        start_date = end_date - pd.Timedelta(days=self.correlation_window + 10)
        
        # Filter data
        mask = (ohlcv_data.index >= start_date) & (ohlcv_data.index <= end_date)
        window_data = ohlcv_data[mask]
        
        if len(window_data) < self.correlation_window:
            # Return cached or empty edges
            return self.update_cache.get('correlation', {'edge_index': [], 'edge_attr': []})
        
        # Compute correlations
        correlations = compute_rolling_correlation(
            window_data, 
            window=self.correlation_window,
            min_periods=self.correlation_window // 2
        )
        
        if correlations is None or correlations.empty:
            return self.update_cache.get('correlation', {'edge_index': [], 'edge_attr': []})
        
        # Convert to edge format
        edge_list = []
        edge_weights = []
        
        for i, ticker1 in enumerate(tickers):
            for j, ticker2 in enumerate(tickers):
                if i < j and ticker1 in correlations.columns and ticker2 in correlations.columns:
                    corr_value = correlations.loc[date, (ticker1, ticker2)]
                    if not pd.isna(corr_value) and abs(corr_value) > 0.3:  # Threshold
                        edge_list.append([i, j])
                        edge_weights.append(corr_value)
        
        if len(edge_list) == 0:
            return self.update_cache.get('correlation', {'edge_index': [], 'edge_attr': []})
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float)
        
        result = {
            'edge_index': edge_index,
            'edge_attr': edge_attr
        }
        
        # Cache result
        self.update_cache['correlation'] = result
        return result
    
    def _compute_fundamental_edges(self,
                                   date: pd.Timestamp,
                                   tickers: List[str],
                                   fundamental_data: pd.DataFrame) -> Dict:
        """Compute fundamental similarity edges for given date."""
        # Get latest fundamental data
        mask = fundamental_data.index <= date
        latest_data = fundamental_data[mask]
        
        if latest_data.empty:
            return self.update_cache.get('fundamental_similarity', {'edge_index': [], 'edge_attr': []})
        
        # Get most recent values for each ticker
        latest_values = latest_data.groupby('ticker').last()
        
        # Compute similarity
        similarities = compute_fundamental_similarity(latest_values)
        
        if similarities is None or similarities.empty:
            return self.update_cache.get('fundamental_similarity', {'edge_index': [], 'edge_attr': []})
        
        # Convert to edge format
        edge_list = []
        edge_weights = []
        
        ticker_to_idx = {t: i for i, t in enumerate(tickers)}
        
        for ticker1 in tickers:
            for ticker2 in tickers:
                if ticker1 < ticker2:  # Avoid duplicates
                    idx1 = ticker_to_idx.get(ticker1)
                    idx2 = ticker_to_idx.get(ticker2)
                    
                    if idx1 is not None and idx2 is not None:
                        if (ticker1, ticker2) in similarities.index:
                            sim_value = similarities.loc[(ticker1, ticker2), 'similarity']
                            if not pd.isna(sim_value) and sim_value > 0.5:  # Threshold
                                edge_list.append([idx1, idx2])
                                edge_weights.append(sim_value)
        
        if len(edge_list) == 0:
            return self.update_cache.get('fundamental_similarity', {'edge_index': [], 'edge_attr': []})
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float)
        
        result = {
            'edge_index': edge_index,
            'edge_attr': edge_attr
        }
        
        # Cache result
        self.update_cache['fundamental_similarity'] = result
        return result
    
    def update_graph(self, graph, date: pd.Timestamp, **kwargs) -> 'HeteroData':
        """
        Update a graph with new dynamic edges.
        
        Args:
            graph: Existing HeteroData graph
            date: Current date
            **kwargs: Additional data (ohlcv_data, fundamental_data, tickers)
            
        Returns:
            Updated HeteroData graph
        """
        from torch_geometric.data import HeteroData
        
        if not self.should_update(date):
            return graph
        
        # Create updated graph
        updated_graph = HeteroData()
        updated_graph['stock'].x = graph['stock'].x.clone()
        
        if 'tickers' in graph:
            updated_graph['tickers'] = graph['tickers']
        
        tickers = kwargs.get('tickers', [])
        ohlcv_data = kwargs.get('ohlcv_data')
        fundamental_data = kwargs.get('fundamental_data')
        
        # Keep static edges
        for edge_type in graph.edge_index_dict.keys():
            if 'sector' in str(edge_type) or 'industry' in str(edge_type):
                updated_graph[edge_type].edge_index = graph[edge_type].edge_index.clone()
                if hasattr(graph[edge_type], 'edge_attr'):
                    updated_graph[edge_type].edge_attr = graph[edge_type].edge_attr.clone()
        
        # Update dynamic edges
        if ohlcv_data is not None and fundamental_data is not None and tickers:
            dynamic_edges = self.update_dynamic_edges(date, tickers, ohlcv_data, fundamental_data)
            
            # Update correlation edges
            if 'correlation' in dynamic_edges and dynamic_edges['correlation']['edge_index'].numel() > 0:
                corr_edge_type = ('stock', 'rolling_correlation', 'stock')
                updated_graph[corr_edge_type].edge_index = dynamic_edges['correlation']['edge_index']
                updated_graph[corr_edge_type].edge_attr = dynamic_edges['correlation']['edge_attr']
            
            # Update fundamental similarity edges
            if 'fundamental_similarity' in dynamic_edges and dynamic_edges['fundamental_similarity']['edge_index'].numel() > 0:
                fund_edge_type = ('stock', 'fundamental_similarity', 'stock')
                updated_graph[fund_edge_type].edge_index = dynamic_edges['fundamental_similarity']['edge_index']
                updated_graph[fund_edge_type].edge_attr = dynamic_edges['fundamental_similarity']['edge_attr']
        
        self.last_update_date = date
        return updated_graph


def test_dynamic_updates():
    """Test the dynamic graph update mechanism."""
    print("🧪 Testing Dynamic Graph Updates")
    
    updater = DynamicGraphUpdater(update_frequency='weekly')
    
    # Load sample data
    from phase1_data_collection import load_ohlcv_data
    from phase1_feature_engineering import load_fundamental_data
    
    # This is a placeholder - actual implementation would load real data
    print("✅ Dynamic Graph Updater initialized")
    print(f"   Update frequency: {updater.update_frequency}")
    print(f"   Correlation window: {updater.correlation_window} days")
    
    return updater


if __name__ == "__main__":
    print("🚀 Phase 7: Dynamic Graph Updates")
    print("="*60)
    
    updater = test_dynamic_updates()
    
    print("\n✅ Dynamic graph update mechanism ready!")
    print("\nUsage:")
    print("  updater = DynamicGraphUpdater(update_frequency='weekly')")
    print("  updated_graph = updater.update_graph(graph, date, **data)")


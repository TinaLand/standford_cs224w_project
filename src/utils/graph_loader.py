"""
Graph data loading utilities.

This module provides a centralized function for loading graph data,
eliminating duplicate implementations across the codebase.
"""

from pathlib import Path
from typing import Optional
import torch
from torch_geometric.data import HeteroData
import pandas as pd

from .paths import DATA_GRAPHS_DIR


def load_graph_data(date: pd.Timestamp) -> Optional[HeteroData]:
    """
    Loads a single graph snapshot for the given date.
    
    Args:
        date: pandas Timestamp or datetime object
        
    Returns:
        data: PyG HeteroData object or None if file not found
        
    Raises:
        None - Returns None on errors to allow graceful handling
    """
    date_str = date.strftime('%Y%m%d')
    filepath = DATA_GRAPHS_DIR / f'graph_t_{date_str}.pt'
    
    if not filepath.exists():
        return None
    
    try:
        # Load the PyG graph object with weights_only=False for PyTorch 2.6+
        data = torch.load(filepath, weights_only=False)
        return data
    except Exception as e:
        # Silent failure - caller should handle None return
        # Uncomment for debugging:
        # print(f"Error loading graph {filepath.name}: {e}")
        return None


def load_graph_data_with_error(date: pd.Timestamp) -> Optional[HeteroData]:
    """
    Loads a single graph snapshot for the given date, with error logging.
    
    Args:
        date: pandas Timestamp or datetime object
        
    Returns:
        data: PyG HeteroData object or None if file not found
        
    Note:
        This version prints errors, useful for debugging.
    """
    date_str = date.strftime('%Y%m%d')
    filepath = DATA_GRAPHS_DIR / f'graph_t_{date_str}.pt'
    
    if not filepath.exists():
        print(f"Warning: Graph file not found: {filepath.name}")
        return None
    
    try:
        data = torch.load(filepath, weights_only=False)
        return data
    except Exception as e:
        print(f"Error loading graph {filepath.name}: {e}")
        return None


__all__ = ["load_graph_data", "load_graph_data_with_error"]


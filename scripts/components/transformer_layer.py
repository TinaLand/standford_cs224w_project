# scripts/components/transformer_layer.py
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv

class EdgeAwareGATv2Conv(GATv2Conv):
    """
    An edge-aware GATv2 Convolution layer suitable for use within HeteroConv.
    
    The proposal's Graph Transformer uses edge-aware global attention 
    to aggregate heterogeneous relationships. [cite: 85]
    GATv2Conv inherently includes a fixed mechanism that is highly effective 
    for message passing and aggregation.
    """
    def __init__(self, in_channels, out_channels, heads, dropout):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout,
            add_self_loops=False # We handle self-loops via residual connections if needed
        )

    def forward(self, x, edge_index):
        """
        Args:
            x (Tensor): Node features.
            edge_index (Tensor): Edge index for the specific edge type.
            
        Returns:
            Tensor: Aggregated node features.
        """
        # GATv2Conv's forward pass handles the multi-head attention and aggregation.
        return super().forward(x, edge_index)
# scripts/components/transformer_layer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv
from typing import Dict, Tuple, Any
import math

class RelationAwareAttention(nn.Module):
    """
    Edge-type specific attention mechanism that learns different attention patterns
    for different relationship types (sector, competitor, supply chain, etc.).
    """
    def __init__(self, in_dim: int, out_dim: int, num_heads: int, edge_types: list, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.edge_types = edge_types
        self.dropout = dropout
        
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        
        # Edge-type specific parameters
        self.edge_type_embeddings = nn.ModuleDict()
        self.edge_type_transforms = nn.ModuleDict()
        self.edge_type_attention_weights = nn.ModuleDict()
        
        for edge_type in edge_types:
            edge_type_str = '_'.join(edge_type) if isinstance(edge_type, tuple) else str(edge_type)
            
            # Relation embedding for this edge type
            self.edge_type_embeddings[edge_type_str] = nn.Parameter(
                torch.randn(self.head_dim) / math.sqrt(self.head_dim)
            )
            
            # Edge-type specific transformation
            self.edge_type_transforms[edge_type_str] = nn.Linear(in_dim, out_dim)
            
            # Edge-type specific attention weights
            self.edge_type_attention_weights[edge_type_str] = nn.Parameter(
                torch.randn(num_heads, self.head_dim) / math.sqrt(self.head_dim)
            )
        
        # Standard attention components
        self.q_proj = nn.Linear(in_dim, out_dim)
        self.k_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)
        
        self.layer_norm = nn.LayerNorm(out_dim)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: Any) -> torch.Tensor:
        """
        Args:
            x: Node features [N, in_dim]
            edge_index: Edge connections [2, E]
            edge_type: Edge type identifier
        """
        batch_size, seq_len = x.size(0), x.size(0)
        
        # Get edge type string
        edge_type_str = '_'.join(edge_type) if isinstance(edge_type, tuple) else str(edge_type)
        
        # Standard attention projections
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Edge-type specific transformations
        if edge_type_str in self.edge_type_transforms:
            x_edge = self.edge_type_transforms[edge_type_str](x)
            
            # Add relation-specific bias to keys and values
            edge_embedding = self.edge_type_embeddings[edge_type_str]
            edge_attention = self.edge_type_attention_weights[edge_type_str]
            
            # Reshape for broadcasting
            edge_embedding = edge_embedding.view(1, 1, 1, self.head_dim)
            edge_attention = edge_attention.view(1, 1, self.num_heads, self.head_dim)
            
            # Apply edge-type specific modifications
            k = k + edge_embedding
            v = v + edge_attention * x_edge.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch, heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention with edge awareness
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply edge-specific attention mask if needed
        if edge_index.size(1) > 0:
            # Create attention mask based on edge connectivity
            mask = torch.zeros(batch_size, seq_len, seq_len, device=x.device)
            mask[edge_index[0], edge_index[1]] = 1
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.out_dim)
        
        # Final projection and residual connection
        out = self.out_proj(out)
        out = self.layer_norm(out + x if x.size(-1) == self.out_dim else out)
        
        return out

class RelationAwareAggregator(nn.Module):
    """
    Aggregates information from different edge types with learned importance weights.
    """
    def __init__(self, hidden_dim: int, edge_types: list):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_types = edge_types
        
        # Learn importance weights for each relation type
        self.relation_weights = nn.Parameter(torch.ones(len(edge_types)))
        
        # Relation-specific transformation before aggregation
        self.relation_transforms = nn.ModuleDict()
        for i, edge_type in enumerate(edge_types):
            edge_type_str = '_'.join(edge_type) if isinstance(edge_type, tuple) else str(edge_type)
            self.relation_transforms[edge_type_str] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        
        self.final_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, relation_outputs: Dict[Any, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            relation_outputs: Dictionary mapping edge types to their outputs
        Returns:
            Aggregated representation
        """
        transformed_outputs = []
        weights = []
        
        for i, edge_type in enumerate(self.edge_types):
            edge_type_str = '_'.join(edge_type) if isinstance(edge_type, tuple) else str(edge_type)
            
            if edge_type in relation_outputs:
                # Transform the output for this relation
                output = relation_outputs[edge_type]
                if edge_type_str in self.relation_transforms:
                    output = self.relation_transforms[edge_type_str](output)
                
                transformed_outputs.append(output)
                weights.append(self.relation_weights[i])
        
        if not transformed_outputs:
            # Fallback if no relations are present
            return torch.zeros(1, self.hidden_dim, device=next(self.parameters()).device)
        
        # Weighted aggregation
        weights = F.softmax(torch.stack(weights), dim=0)
        aggregated = sum(w * out for w, out in zip(weights, transformed_outputs))
        
        return self.final_transform(aggregated)

class RelationAwareGATv2Conv(GATv2Conv):
    """
    Enhanced GATv2Conv with relation-aware attention.
    """
    def __init__(self, in_channels: int, out_channels: int, heads: int = 1, 
                 concat: bool = True, negative_slope: float = 0.2, dropout: float = 0.0,
                 edge_type: Any = None, **kwargs):
        super().__init__(in_channels, out_channels, heads, concat, negative_slope, dropout, **kwargs)
        
        self.edge_type = edge_type
        
        # Add edge-type specific parameters
        if edge_type is not None:
            final_out_channels = out_channels * heads if concat else out_channels
            self.edge_embedding = nn.Parameter(torch.randn(final_out_channels) / math.sqrt(final_out_channels))
            self.edge_attention_bias = nn.Parameter(torch.randn(heads) / math.sqrt(heads))
    
    def forward(self, x, edge_index, **kwargs):
        # Standard GATv2 forward pass with edge-type modifications
        out = super().forward(x, edge_index, **kwargs)
        
        # Apply edge-type specific transformations
        if hasattr(self, 'edge_embedding'):
            out = out + self.edge_embedding.unsqueeze(0)
            
        return out
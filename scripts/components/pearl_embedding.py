# scripts/components/pearl_embedding.py
import torch
import torch.nn as nn
import torch.nn.init as init

class PEARLPositionalEmbedding(nn.Module):
    """
    Simulates the PEARL Positional Embedding generation.
    
    PEARL encodes structural roles (hubs, bridges, role twins) and is 
    concatenated with node features. 
    
    In a full implementation, this layer would:
    1. Pre-calculate structural properties (e.g., shortest path distances, PageRank).
    2. Learn a transformation of these properties to generate stable embeddings.
    
    Here, we use a simple two-layer MLP to map initial node features to a fixed 
    positional embedding vector based on local structural hints (as a proxy).
    """
    def __init__(self, feature_dim: int, pe_dim: int = 32):
        """
        Args:
            feature_dim: The input dimension (original node feature size).
            pe_dim: The output dimension of the positional embedding.
        """
        super().__init__()
        self.pe_dim = pe_dim
        
        # Simple MLP to learn the positional encoding from input features
        self.mapper = nn.Sequential(
            nn.Linear(feature_dim, pe_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(pe_dim * 2, pe_dim),
            nn.Tanh() # Use tanh to keep embeddings normalized between -1 and 1
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight, gain=1.414)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [N, F].
            
        Returns:
            Positional Embedding [N, PE_DIM].
        """
        # Learn the structural role embedding based on input features (proxy)
        return self.mapper(x)
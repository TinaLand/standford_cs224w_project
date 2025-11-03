# scripts/components/pearl_embedding.py
import torch
import torch.nn as nn
import torch.nn.init as init
import networkx as nx
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class PEARLPositionalEmbedding(nn.Module):
    """
    Full PEARL (Position-aware graph neural networks) Positional Embedding implementation.
    
    PEARL encodes structural roles (hubs, bridges, role twins) by:
    1. Pre-calculating structural properties (PageRank, centrality, clustering, etc.)
    2. Learning transformations of these properties to generate stable embeddings.
    
    This implementation computes real structural features and learns to embed them.
    """
    def __init__(self, feature_dim: int, pe_dim: int = 32, cache_structural_features: bool = True):
        """
        Args:
            feature_dim: The input dimension (original node feature size).
            pe_dim: The output dimension of the positional embedding.
            cache_structural_features: Whether to cache structural computations.
        """
        super().__init__()
        self.pe_dim = pe_dim
        self.feature_dim = feature_dim
        self.cache_structural_features = cache_structural_features
        
        # Number of structural features we compute
        self.structural_feature_dim = 8  # pagerank, degree_centrality, betweenness, closeness, 
                                        # clustering, core_number, avg_neighbor_degree, triangles
        
        # Cache for structural features
        self._structural_cache = {}
        self._graph_hash_cache = None
        
        # MLP to transform structural features into positional embeddings
        self.structural_mapper = nn.Sequential(
            nn.Linear(self.structural_feature_dim, pe_dim * 2),
            nn.BatchNorm1d(pe_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(pe_dim * 2, pe_dim),
            nn.Tanh()
        )
        
        # Optional: Combine with learned features from node attributes
        self.feature_mapper = nn.Sequential(
            nn.Linear(feature_dim, pe_dim),
            nn.ReLU(),
            nn.Linear(pe_dim, pe_dim),
            nn.Tanh()
        )
        
        # Relation-aware attention mechanism
        self.relation_attention = nn.ModuleDict()
        self.relation_weights = nn.Parameter(torch.ones(5))  # 5 edge types
        
        # Create separate attention for each relation type
        self.edge_type_names = ['sector_industry', 'competitor', 'supply_chain', 'rolling_correlation', 'fund_similarity']
        for edge_type in self.edge_type_names:
            self.relation_attention[edge_type] = nn.MultiheadAttention(pe_dim, num_heads=2, batch_first=True)
        
        # Global attention for combining all relations
        self.global_attention = nn.MultiheadAttention(pe_dim, num_heads=4, batch_first=True)
        self.layer_norm = nn.LayerNorm(pe_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight, gain=1.414)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def _compute_graph_hash(self, edge_index_dict: Dict) -> str:
        """Compute a hash of the graph structure for caching."""
        hash_parts = []
        for edge_type, edge_index in sorted(edge_index_dict.items()):
            if isinstance(edge_type, tuple):
                edge_type_str = '_'.join(edge_type)
            else:
                edge_type_str = str(edge_type)
            hash_parts.append(f"{edge_type_str}_{hash(edge_index.cpu().numpy().tobytes())}")
        return '_'.join(hash_parts)

    def _create_networkx_graph(self, edge_index_dict: Dict, num_nodes: int) -> nx.Graph:
        """Convert PyG heterogeneous graph to NetworkX for structural analysis."""
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        
        # Add all edges from different edge types
        for _, edge_index in edge_index_dict.items():
            if edge_index.size(1) > 0:  # Check if there are edges
                edges = edge_index.cpu().numpy().T
                # Filter out self-loops
                edges = [(u, v) for u, v in edges if u != v]
                G.add_edges_from(edges)
        
        # Remove any remaining self-loops
        G.remove_edges_from(nx.selfloop_edges(G))
        
        return G

    def _compute_structural_features(self, G: nx.Graph, num_nodes: int) -> torch.Tensor:
        """Compute comprehensive structural features for each node."""
        
        # Initialize feature matrix
        features = torch.zeros(num_nodes, self.structural_feature_dim)
        
        if G.number_of_edges() == 0:
            return features  # Return zeros for disconnected graph
        
        try:
            # 1. PageRank (measures node importance)
            pagerank = nx.pagerank(G, max_iter=100, tol=1e-4)
            for node, pr in pagerank.items():
                if node < num_nodes:
                    features[node, 0] = pr
            
            # 2. Degree Centrality (normalized degree)
            degree_cent = nx.degree_centrality(G)
            for node, dc in degree_cent.items():
                if node < num_nodes:
                    features[node, 1] = dc
            
            # 3. Betweenness Centrality (bridge importance) - sample for large graphs
            if G.number_of_nodes() > 1000:
                k = min(100, G.number_of_nodes())
                betweenness = nx.betweenness_centrality(G, k=k)
            else:
                betweenness = nx.betweenness_centrality(G)
            for node, bc in betweenness.items():
                if node < num_nodes:
                    features[node, 2] = bc
            
            # 4. Closeness Centrality (proximity to all nodes)
            try:
                closeness = nx.closeness_centrality(G)
                for node, cc in closeness.items():
                    if node < num_nodes:
                        features[node, 3] = cc
            except:
                pass  # Skip if graph is disconnected
            
            # 5. Clustering Coefficient (local connectivity)
            clustering = nx.clustering(G)
            for node, clust in clustering.items():
                if node < num_nodes:
                    features[node, 4] = clust
            
            # 6. Core Number (k-core decomposition)
            core_numbers = nx.core_number(G)
            max_core = max(core_numbers.values()) if core_numbers else 1
            for node, core in core_numbers.items():
                if node < num_nodes:
                    features[node, 5] = core / max_core  # Normalize
            
            # 7. Average Neighbor Degree
            avg_neighbor_deg = nx.average_neighbor_degree(G)
            max_avg_deg = max(avg_neighbor_deg.values()) if avg_neighbor_deg else 1
            for node, avg_deg in avg_neighbor_deg.items():
                if node < num_nodes:
                    features[node, 6] = avg_deg / max_avg_deg  # Normalize
            
            # 8. Triangle Count (local structural richness)
            triangles = nx.triangles(G)
            max_triangles = max(triangles.values()) if triangles else 1
            for node, tri_count in triangles.items():
                if node < num_nodes:
                    features[node, 7] = tri_count / max_triangles  # Normalize
                    
        except Exception as e:
            print(f"Warning: Error computing structural features: {e}")
            # Return normalized random features as fallback
            features = torch.randn(num_nodes, self.structural_feature_dim) * 0.1
            
        return features

    def forward(self, x: torch.Tensor, edge_index_dict: Optional[Dict] = None) -> torch.Tensor:
        """
        Args:
            x: Node features [N, F].
            edge_index_dict: Dictionary of edge indices for different edge types.
            
        Returns:
            Positional Embedding [N, PE_DIM].
        """
        num_nodes = x.size(0)
        device = x.device
        
        # Compute structural features
        if edge_index_dict is not None:
            # Check cache
            graph_hash = self._compute_graph_hash(edge_index_dict)
            
            if self.cache_structural_features and graph_hash == self._graph_hash_cache and hasattr(self, '_cached_structural_features'):
                structural_features = self._cached_structural_features.to(device)
            else:
                # Create NetworkX graph and compute features
                G = self._create_networkx_graph(edge_index_dict, num_nodes)
                structural_features = self._compute_structural_features(G, num_nodes).to(device)
                
                # Cache results
                if self.cache_structural_features:
                    self._cached_structural_features = structural_features.cpu()
                    self._graph_hash_cache = graph_hash
        else:
            # Fallback: use small random structural features
            structural_features = torch.randn(num_nodes, self.structural_feature_dim, device=device) * 0.1
        
        # Transform structural features to embeddings
        structural_pe = self.structural_mapper(structural_features)
        
        # Transform node features to embeddings
        feature_pe = self.feature_mapper(x)
        
        # Relation-aware attention processing
        if edge_index_dict is not None:
            relation_embeddings = []
            active_relations = []
            
            # Process each relation type separately
            for i, edge_type_name in enumerate(self.edge_type_names):
                # Find matching edge type in the dictionary
                matching_edge_type = None
                for edge_type in edge_index_dict.keys():
                    if isinstance(edge_type, tuple) and len(edge_type) == 3:
                        if edge_type[1] == edge_type_name:  # Match middle element (relation name)
                            matching_edge_type = edge_type
                            break
                
                if matching_edge_type is not None and edge_index_dict[matching_edge_type].size(1) > 0:
                    # Apply relation-specific attention
                    structural_pe_reshaped = structural_pe.unsqueeze(0)
                    feature_pe_reshaped = feature_pe.unsqueeze(0)
                    
                    relation_pe, _ = self.relation_attention[edge_type_name](
                        query=structural_pe_reshaped,
                        key=feature_pe_reshaped,
                        value=feature_pe_reshaped
                    )
                    
                    relation_embeddings.append(relation_pe.squeeze(0))
                    active_relations.append(i)
            
            # Aggregate relation-specific embeddings
            if relation_embeddings:
                # Weight each relation by learned importance
                relation_weights = torch.softmax(self.relation_weights[active_relations], dim=0)
                weighted_relations = sum(w * emb for w, emb in zip(relation_weights, relation_embeddings))
                
                # Global attention to combine with structural features
                structural_pe_reshaped = structural_pe.unsqueeze(0)
                weighted_relations_reshaped = weighted_relations.unsqueeze(0)
                
                final_pe, _ = self.global_attention(
                    query=structural_pe_reshaped,
                    key=weighted_relations_reshaped,
                    value=weighted_relations_reshaped
                )
                final_pe = final_pe.squeeze(0)
            else:
                # Fallback to simple combination if no relations are active
                structural_pe_reshaped = structural_pe.unsqueeze(0)
                feature_pe_reshaped = feature_pe.unsqueeze(0)
                
                final_pe, _ = self.global_attention(
                    query=structural_pe_reshaped,
                    key=feature_pe_reshaped,
                    value=feature_pe_reshaped
                )
                final_pe = final_pe.squeeze(0)
        else:
            # Fallback when no edge information is available
            structural_pe_reshaped = structural_pe.unsqueeze(0)
            feature_pe_reshaped = feature_pe.unsqueeze(0)
            
            final_pe, _ = self.global_attention(
                query=structural_pe_reshaped,
                key=feature_pe_reshaped,
                value=feature_pe_reshaped
            )
            final_pe = final_pe.squeeze(0)
        
        # Apply layer norm with residual connection
        final_pe = self.layer_norm(final_pe + structural_pe)
        
        return final_pe
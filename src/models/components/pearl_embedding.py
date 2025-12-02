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
        self.relation_weights = nn.Parameter(torch.ones(4))  # 4 edge types
        
        # Create separate attention for each relation type
        self.edge_type_names = ['sector_industry', 'supply_competitor', 'rolling_correlation', 'fund_similarity']
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

    def _compute_simplified_structural_features(self, edge_index_dict: Dict, num_nodes: int, device: torch.device) -> torch.Tensor:
        """
        Compute simplified structural features for large graphs/mini-batches.
        Uses improved local connectivity analysis for better structural role capture.
        """
        features = torch.zeros(num_nodes, self.structural_feature_dim, device=device)
        
        # Create adjacency information for all relation types
        all_edges = []
        degree_by_relation = torch.zeros(num_nodes, len(edge_index_dict), device=device)
        total_degree = torch.zeros(num_nodes, device=device)
        
        # Relation-specific importance weights
        relation_weights = {
            'sector_industry': 1.0,
            'supply_competitor': 0.8, 
            'rolling_correlation': 0.9,
            'fund_similarity': 0.7
        }
        
        # Process each relation type
        for rel_idx, (edge_type, edge_index) in enumerate(edge_index_dict.items()):
            if edge_index.size(1) == 0:
                continue
                
            # Get relation name from tuple
            if isinstance(edge_type, tuple) and len(edge_type) == 3:
                rel_name = edge_type[1]
                weight = relation_weights.get(rel_name, 0.5)
            else:
                weight = 0.5
            
            # Filter valid edges
            valid_edges = (edge_index[0] != edge_index[1]) & (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
            if not valid_edges.any():
                continue
                
            valid_edge_index = edge_index[:, valid_edges]
            all_edges.append(valid_edge_index)
            
            # Compute relation-specific degrees
            src, dst = valid_edge_index[0], valid_edge_index[1]
            degree_by_relation[src, rel_idx] += 1
            degree_by_relation[dst, rel_idx] += 1
            total_degree[src] += weight
            total_degree[dst] += weight
        
        # Combine all edges for triangle counting
        if all_edges:
            combined_edges = torch.cat(all_edges, dim=1)
            
            # Efficient triangle counting using adjacency matrix
            adj_lists = {}
            for i in range(combined_edges.size(1)):
                src, dst = combined_edges[0, i].item(), combined_edges[1, i].item()
                if src not in adj_lists:
                    adj_lists[src] = set()
                if dst not in adj_lists:
                    adj_lists[dst] = set()
                adj_lists[src].add(dst)
                adj_lists[dst].add(src)
            
            # Count triangles per node
            triangle_count = torch.zeros(num_nodes, device=device)
            for node in range(min(num_nodes, len(adj_lists))):
                if node in adj_lists:
                    neighbors = list(adj_lists[node])
                    for i, n1 in enumerate(neighbors):
                        for n2 in neighbors[i+1:]:
                            if n2 in adj_lists.get(n1, set()):
                                triangle_count[node] += 1
        else:
            triangle_count = torch.zeros(num_nodes, device=device)
        
        # Compute clustering coefficient approximation
        clustering = torch.zeros(num_nodes, device=device)
        for node in range(num_nodes):
            degree = total_degree[node]
            if degree > 1:
                max_triangles = degree * (degree - 1) / 2
                clustering[node] = triangle_count[node] / max_triangles if max_triangles > 0 else 0
        
        # Normalize all metrics
        max_degree = total_degree.max() if total_degree.max() > 0 else 1
        max_triangles = triangle_count.max() if triangle_count.max() > 0 else 1
        
        # Fill improved structural features
        features[:, 0] = total_degree / max_degree  # Improved degree centrality
        features[:, 1] = total_degree / max_degree  # Degree centrality
        features[:, 2] = (total_degree >= total_degree.quantile(0.8)).float()  # High degree nodes (betweenness proxy)
        features[:, 3] = torch.exp(-total_degree / max_degree)  # Improved closeness proxy
        features[:, 4] = clustering  # Actual clustering coefficient
        features[:, 5] = torch.sqrt(total_degree / max_degree)  # Core number proxy
        features[:, 6] = degree_by_relation.sum(dim=1) / (degree_by_relation.sum(dim=1).max() + 1e-8)  # Multi-relation connectivity
        features[:, 7] = triangle_count / (max_triangles + 1e-8)  # Actual triangle count
        
        return features

    def forward(self, x: torch.Tensor, edge_index_dict: Optional[Dict] = None, 
                batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            x: Node features [N, F].
            edge_index_dict: Dictionary of edge indices for different edge types.
            batch_size: For mini-batch training, size of the current batch
            
        Returns:
            Positional Embedding [N, PE_DIM].
        """
        num_nodes = x.size(0)
        device = x.device
        
        # Adaptive structural computation based on graph size and training mode
        use_full_structural = num_nodes <= 500 or not self.training  # Use full computation for smaller graphs or during inference
        
        # Compute structural features
        if edge_index_dict is not None:
            if use_full_structural:
                # Check cache for full graph structural computation
                graph_hash = self._compute_graph_hash(edge_index_dict)
                
                if self.cache_structural_features and graph_hash == self._graph_hash_cache and hasattr(self, '_cached_structural_features'):
                    cached_features = self._cached_structural_features
                    if cached_features.size(0) == num_nodes:  # Ensure size matches
                        structural_features = cached_features.to(device)
                    else:
                        # Cache size mismatch, recompute
                        G = self._create_networkx_graph(edge_index_dict, num_nodes)
                        structural_features = self._compute_structural_features(G, num_nodes).to(device)
                        if self.cache_structural_features:
                            self._cached_structural_features = structural_features.cpu()
                            self._graph_hash_cache = graph_hash
                else:
                    # Create NetworkX graph and compute features
                    G = self._create_networkx_graph(edge_index_dict, num_nodes)
                    structural_features = self._compute_structural_features(G, num_nodes).to(device)
                    
                    # Cache results
                    if self.cache_structural_features:
                        self._cached_structural_features = structural_features.cpu()
                        self._graph_hash_cache = graph_hash
            else:
                # For large graphs/mini-batches, use improved simplified structural features
                structural_features = self._compute_simplified_structural_features(edge_index_dict, num_nodes, device)
        else:
            # Fallback: use small random structural features
            structural_features = torch.randn(num_nodes, self.structural_feature_dim, device=device) * 0.01
        
        # Transform structural features to embeddings
        structural_pe = self.structural_mapper(structural_features)
        
        # Transform node features to embeddings
        feature_pe = self.feature_mapper(x)
        
        # Relation-aware attention processing
        if edge_index_dict is not None:
            relation_embeddings = []
            active_relations = []
            
            # Create edge type mapping for robust matching
            edge_type_mapping = {}
            for edge_type in edge_index_dict.keys():
                if isinstance(edge_type, tuple) and len(edge_type) == 3:
                    relation_name = edge_type[1]
                    edge_type_mapping[relation_name] = edge_type
                else:
                    # Handle string edge types
                    edge_type_mapping[str(edge_type)] = edge_type
            
            # Process each relation type separately
            for i, edge_type_name in enumerate(self.edge_type_names):
                matching_edge_type = edge_type_mapping.get(edge_type_name)
                
                if matching_edge_type is not None and edge_index_dict[matching_edge_type].size(1) > 0:
                    # Apply relation-specific attention with edge information
                    edge_index = edge_index_dict[matching_edge_type]
                    
                    # Create edge-aware attention by incorporating connectivity
                    structural_pe_reshaped = structural_pe.unsqueeze(0)
                    feature_pe_reshaped = feature_pe.unsqueeze(0)
                    
                    # Compute edge-weighted features
                    if edge_index.size(1) > 0:
                        src_nodes = edge_index[0]
                        dst_nodes = edge_index[1]
                        
                        # Create edge-weighted query based on connectivity
                        edge_weights = torch.ones(edge_index.size(1), device=structural_pe.device, dtype=structural_pe.dtype) / edge_index.size(1)
                        if len(src_nodes) > 0:
                            # Weight query by edge connectivity strength
                            connectivity_weight = torch.zeros_like(structural_pe[:, 0])
                            connectivity_weight.scatter_add_(0, src_nodes.long(), edge_weights)
                            connectivity_weight.scatter_add_(0, dst_nodes.long(), edge_weights)
                            connectivity_weight = connectivity_weight.unsqueeze(-1)
                            
                            # Apply connectivity weighting to structural features
                            weighted_structural_pe = structural_pe * (1 + connectivity_weight)
                            weighted_structural_pe_reshaped = weighted_structural_pe.unsqueeze(0)
                        else:
                            weighted_structural_pe_reshaped = structural_pe_reshaped
                    else:
                        weighted_structural_pe_reshaped = structural_pe_reshaped
                    
                    relation_pe, attention_weights = self.relation_attention[edge_type_name](
                        query=weighted_structural_pe_reshaped,
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
        
        # Over-smoothing prevention through adaptive residual weighting
        # Compute structural diversity as a measure of local complexity
        if edge_index_dict is not None:
            structural_diversity = torch.std(structural_pe, dim=1, keepdim=True)
            diversity_weight = torch.sigmoid(structural_diversity * 2.0)  # Adaptive weighting based on local structure
        else:
            diversity_weight = torch.ones_like(structural_pe[:, :1]) * 0.5
        
        # Apply adaptive residual connection to prevent over-smoothing
        # High diversity nodes get stronger residual connections to preserve structural identity
        adaptive_residual = diversity_weight * structural_pe + (1 - diversity_weight) * final_pe
        final_pe = self.layer_norm(final_pe + adaptive_residual)
        
        return final_pe
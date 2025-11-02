"""
Graph structure update simulator with stratified sampling.
Handles node/edge addition and deletion scenarios.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import networkx as nx


class GraphUpdater:
    """
    Creates fair holdout sets and simulates graph structural changes.
    """
    
    def __init__(self, 
                 adj: torch.Tensor,
                 features: torch.Tensor, 
                 labels: torch.Tensor,
                 initial_ratio: float = 0.9,
                 n_buckets: int = 5,
                 seed: int = 42):
        """
        Args:
            adj: Sparse adjacency matrix [N x N]
            features: Node features [N x D]
            labels: Node labels [N]
            initial_ratio: Ratio for initial core graph (default: 0.9)
            n_buckets: Number of buckets to split holdout (default: 5)
            seed: Random seed
        """
        self.adj = adj.coalesce()
        self.features = features
        self.labels = labels
        self.initial_ratio = initial_ratio
        self.n_buckets = n_buckets
        self.seed = seed
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.num_nodes = adj.size(0)
        self.device = adj.device
        
        # Convert to NetworkX for easier manipulation
        self.G = self._sparse_to_networkx(adj)
        
    def _sparse_to_networkx(self, adj: torch.Tensor) -> nx.Graph:
        """Convert sparse adjacency to NetworkX graph."""
        adj = adj.coalesce().cpu()
        indices = adj.indices().numpy()
        
        G = nx.Graph()
        G.add_nodes_from(range(self.num_nodes))
        edges = list(zip(indices[0], indices[1]))
        G.add_edges_from(edges)
        
        return G
    
    def _compute_degree_percentiles(self) -> Tuple[float, float]:
        """Compute 33rd and 67th percentiles of degree distribution."""
        degrees = [self.G.degree(n) for n in self.G.nodes()]
        p33 = np.percentile(degrees, 33)
        p67 = np.percentile(degrees, 67)
        
        return p33, p67
    
    def _stratified_holdout_selection(self, 
                                     holdout_ratio: float) -> List[int]:
        """
        Select holdout nodes using stratified sampling based on degree.
        
        Args:
            holdout_ratio: Ratio of nodes to hold out
            
        Returns:
            List of holdout node indices
        """
        p33, p67 = self._compute_degree_percentiles()
        
        print(f"Degree thresholds: Low ≤ {p33:.1f}, "
              f"Mid: ({p33:.1f}, {p67:.1f}], High > {p67:.1f}")
        
        # Classify nodes into degree bins
        degree_bins = {
            'low': [],
            'mid': [],
            'high': []
        }
        
        for node in self.G.nodes():
            degree = self.G.degree(node)
            if degree <= p33:
                degree_bins['low'].append(node)
            elif degree <= p67:
                degree_bins['mid'].append(node)
            else:
                degree_bins['high'].append(node)
        
        # Sample from each bin proportionally
        target_holdout = int(self.num_nodes * holdout_ratio)
        holdout_nodes = []
        
        for bin_name, nodes in degree_bins.items():
            bin_ratio = len(nodes) / self.num_nodes
            n_sample = int(target_holdout * bin_ratio)
            
            # Random sample from this bin
            if n_sample > 0 and len(nodes) >= n_sample:
                sampled = np.random.choice(nodes, n_sample, replace=False)
                holdout_nodes.extend(sampled.tolist())
        
        # If we didn't get exactly target_holdout, sample remaining
        if len(holdout_nodes) < target_holdout:
            remaining_nodes = list(set(self.G.nodes()) - set(holdout_nodes))
            n_additional = target_holdout - len(holdout_nodes)
            additional = np.random.choice(remaining_nodes, n_additional, replace=False)
            holdout_nodes.extend(additional.tolist())
        
        return holdout_nodes[:target_holdout]
    
    def _check_connectivity(self, 
                           holdout_nodes: List[int], 
                           core_nodes: List[int]) -> bool:
        """
        Check if holdout nodes have at least one edge to core.
        
        Args:
            holdout_nodes: Nodes in holdout set
            core_nodes: Nodes in core graph
            
        Returns:
            True if all holdout nodes are connected to core
        """
        core_set = set(core_nodes)
        
        for node in holdout_nodes:
            neighbors = list(self.G.neighbors(node))
            if not any(n in core_set for n in neighbors):
                return False
        
        return True
    
    def _split_into_buckets(self, 
                           holdout_nodes: List[int]) -> Dict[str, List[int]]:
        """
        Split holdout nodes into n_buckets using round-robin.
        
        Args:
            holdout_nodes: List of holdout node indices
            
        Returns:
            Dictionary mapping bucket names to node lists
        """
        # Sort by degree for fair distribution
        sorted_nodes = sorted(holdout_nodes, 
                            key=lambda n: self.G.degree(n))
        
        # Round-robin assignment
        buckets = {chr(65 + i): [] for i in range(self.n_buckets)}  # A, B, C, D, E
        
        for idx, node in enumerate(sorted_nodes):
            bucket_name = chr(65 + (idx % self.n_buckets))
            buckets[bucket_name].append(node)
        
        return buckets
    
    def _validate_buckets(self, 
                         buckets: Dict[str, List[int]]) -> float:
        """
        Validate that buckets are well-balanced.
        
        Args:
            buckets: Dictionary of bucket name to node lists
            
        Returns:
            Coefficient of variation (CV) of average degrees
        """
        print("\n=== Bucket Validation ===")
        
        stats = {}
        for name, nodes in buckets.items():
            degrees = [self.G.degree(n) for n in nodes]
            stats[name] = {
                'size': len(nodes),
                'avg_degree': np.mean(degrees),
                'std_degree': np.std(degrees)
            }
        
        # Print statistics
        for name, stat in stats.items():
            print(f"Bucket {name}: {stat['size']} nodes, "
                  f"avg_degree={stat['avg_degree']:.2f}±{stat['std_degree']:.2f}")
        
        # Compute coefficient of variation
        avg_degrees = [s['avg_degree'] for s in stats.values()]
        cv = np.std(avg_degrees) / np.mean(avg_degrees)
        print(f"\nCoefficient of Variation: {cv:.3f}")
        
        if cv > 0.15:
            print("⚠️  Warning: Buckets are not well-balanced (CV > 0.15)")
        else:
            print("✅ Buckets are well-balanced")
        
        return cv
    
    def create_incremental_scenario(self) -> Tuple[Dict, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create incremental scenario: 90% core + 2%×5 buckets.
        
        Returns:
            buckets: Dictionary of bucket name to node indices
            core_adj: Sparse adjacency of 90% core graph
            core_features: Features of core nodes
            core_labels: Labels of core nodes
        """
        holdout_ratio = 1.0 - self.initial_ratio
        
        # Step 1: Select holdout nodes with stratification
        holdout_nodes = self._stratified_holdout_selection(holdout_ratio)
        core_nodes = list(set(range(self.num_nodes)) - set(holdout_nodes))
        
        print(f"\nSelected {len(holdout_nodes)} holdout nodes, "
              f"{len(core_nodes)} core nodes")
        
        # Step 2: Check connectivity
        is_connected = self._check_connectivity(holdout_nodes, core_nodes)
        print(f"Connectivity check: {'✅ Passed' if is_connected else '⚠️  Failed'}")
        
        # Step 3: Split into buckets
        buckets = self._split_into_buckets(holdout_nodes)
        
        # Step 4: Validate buckets
        cv = self._validate_buckets(buckets)
        
        # Step 5: Create core graph
        core_nodes_sorted = sorted(core_nodes)
        core_adj, core_features, core_labels = self._extract_subgraph(core_nodes_sorted)
        
        # Convert bucket node indices to new indexing after core extraction
        node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(core_nodes_sorted)}
        
        return buckets, core_adj, core_features, core_labels, node_mapping
    
    def _extract_subgraph(self, 
                         nodes: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract subgraph for given nodes.
        
        Args:
            nodes: List of node indices
            
        Returns:
            sub_adj: Sparse adjacency of subgraph
            sub_features: Features of subgraph nodes
            sub_labels: Labels of subgraph nodes
        """
        nodes = sorted(nodes)
        node_set = set(nodes)
        
        # Extract edges within subgraph
        adj = self.adj.coalesce()
        indices = adj.indices().cpu()
        values = adj.values().cpu()
        
        # Create mapping: old_idx -> new_idx
        node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(nodes)}
        
        # Filter edges
        new_indices = []
        new_values = []
        
        for i in range(indices.size(1)):
            src, dst = indices[0, i].item(), indices[1, i].item()
            if src in node_set and dst in node_set:
                new_src = node_mapping[src]
                new_dst = node_mapping[dst]
                new_indices.append([new_src, new_dst])
                new_values.append(values[i].item())
        
        if len(new_indices) == 0:
            # Empty graph
            sub_adj = torch.sparse_coo_tensor(
                torch.zeros(2, 0, dtype=torch.long),
                torch.zeros(0),
                (len(nodes), len(nodes)),
                device=self.device
            )
        else:
            new_indices = torch.tensor(new_indices, dtype=torch.long).t()
            new_values = torch.tensor(new_values, dtype=torch.float32)
            
            sub_adj = torch.sparse_coo_tensor(
                new_indices,
                new_values,
                (len(nodes), len(nodes)),
                device=self.device
            )
        
        # Extract features and labels
        nodes_tensor = torch.tensor(nodes, dtype=torch.long)
        sub_features = self.features[nodes_tensor]
        sub_labels = self.labels[nodes_tensor]
        
        return sub_adj.coalesce(), sub_features, sub_labels
    
    def add_bucket_to_graph(self,
                           current_adj: torch.Tensor,
                           current_features: torch.Tensor,
                           current_labels: torch.Tensor,
                           bucket_nodes: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Add a bucket of nodes to current graph.
        
        Args:
            current_adj: Current sparse adjacency
            current_features: Current features
            current_labels: Current labels
            bucket_nodes: Nodes to add
            
        Returns:
            new_adj: Updated sparse adjacency
            new_features: Updated features
            new_labels: Updated labels
        """
        # Get features and labels for new nodes
        bucket_nodes_tensor = torch.tensor(bucket_nodes, dtype=torch.long)
        new_features_to_add = self.features[bucket_nodes_tensor]
        new_labels_to_add = self.labels[bucket_nodes_tensor]
        
        # Concatenate features and labels
        new_features = torch.cat([current_features, new_features_to_add], dim=0)
        new_labels = torch.cat([current_labels, new_labels_to_add], dim=0)
        
        # Get edges involving new nodes
        current_num_nodes = current_adj.size(0)
        new_num_nodes = current_num_nodes + len(bucket_nodes)
        
        # Create mapping for bucket nodes
        bucket_node_mapping = {old_idx: current_num_nodes + i 
                              for i, old_idx in enumerate(bucket_nodes)}
        
        # Extract edges from original graph involving bucket nodes
        adj = self.adj.coalesce()
        indices = adj.indices().cpu()
        values = adj.values().cpu()
        
        new_edges = []
        new_edge_values = []
        
        bucket_set = set(bucket_nodes)
        
        for i in range(indices.size(1)):
            src, dst = indices[0, i].item(), indices[1, i].item()
            
            # Both in bucket: internal edge
            if src in bucket_set and dst in bucket_set:
                new_src = bucket_node_mapping[src]
                new_dst = bucket_node_mapping[dst]
                new_edges.append([new_src, new_dst])
                new_edge_values.append(values[i].item())
            
            # One in bucket, one in existing graph
            # (This will be handled by the delta computation)
        
        # Expand current adjacency
        from utils.sparse_ops import SparseOperations
        expanded_adj = SparseOperations.expand_sparse_tensor(
            current_adj, 
            (current_num_nodes, current_num_nodes),
            (new_num_nodes, new_num_nodes)
        )
        
        # Add new edges
        if len(new_edges) > 0:
            new_edges_tensor = torch.tensor(new_edges, dtype=torch.long).t()
            new_values_tensor = torch.tensor(new_edge_values, dtype=torch.float32)
            
            new_edges_sparse = torch.sparse_coo_tensor(
                new_edges_tensor,
                new_values_tensor,
                (new_num_nodes, new_num_nodes),
                device=self.device
            )
            
            new_adj = expanded_adj + new_edges_sparse
            new_adj = new_adj.coalesce()
        else:
            new_adj = expanded_adj
        
        return new_adj, new_features, new_labels


if __name__ == "__main__":
    print("GraphUpdater module - use in experiments")
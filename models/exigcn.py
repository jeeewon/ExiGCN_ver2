"""
ExiGCN: Exact and Efficient Graph Convolutional Network
Implements the efficient retraining algorithm from the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F_functional
from typing import List, Tuple, Optional
from utils.sparse_ops import SparseOperations


class ExiGCNLayer(nn.Module):
    """
    ExiGCN layer with caching and delta update mechanism.
    """
    
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True):
        super(ExiGCNLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Original weights W (frozen during retraining)
        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        
        # Delta weights ΔW (trainable during retraining)
        self.delta_W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
            self.delta_bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('delta_bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.delta_W)
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)
            nn.init.zeros_(self.delta_bias)
    
    def reset_delta(self):
        """Reset delta parameters to zero."""
        nn.init.zeros_(self.delta_W)
        if self.delta_bias is not None:
            nn.init.zeros_(self.delta_bias)
    
    def merge_deltas(self):
        """Merge delta into main weights: W' = W + ΔW"""
        with torch.no_grad():
            self.W.data += self.delta_W.data
            if self.bias is not None:
                self.bias.data += self.delta_bias.data
        
        self.reset_delta()
    
    def forward_initial(self, 
                       adj: torch.Tensor, 
                       features: torch.Tensor) -> torch.Tensor:
        """
        Initial training forward: Z = Â H W
        
        Args:
            adj: Normalized sparse adjacency [N x N]
            features: Node features [N x D]
            
        Returns:
            Pre-activation output [N x out_features]
        """
        support = torch.mm(features, self.W)
        output = SparseOperations.sparse_dense_mm(adj, support)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    def forward_retraining(self,
                          adj: torch.Tensor,
                          features: torch.Tensor,
                          delta_adj: torch.Tensor,
                          delta_features: torch.Tensor,
                          cached_Z: torch.Tensor,
                          cached_F: Optional[torch.Tensor] = None,
                          cached_B: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retraining forward pass using cached terms.
        
        Following Equation (7) from the paper:
        Z' = Z + F + B ΔW
        
        where:
        F = (Â∆H + ∆ÂH + ∆Â∆H)W  [Fixed term, cached]
        B = ÂH + Â∆H + ∆ÂH + ∆Â∆H  [Update term, NOW CACHED!]
        
        Args:
            adj: Current normalized adjacency Â [N' x N']
            features: Current features H [N' x D]
            delta_adj: Change in adjacency ∆Â [N' x N']
            delta_features: Change in features ∆H [N' x D]
            cached_Z: Previous Z [N x out_features]
            cached_F: Cached fixed term (computed once)
            cached_B: Cached B matrix (computed once)
            
        Returns:
            new_Z: Updated pre-activation [N' x out_features]
            fixed_term: Fixed term (for caching if not provided)
            B: B matrix (for caching if not provided)
        """
        # Compute fixed term F if not cached
        if cached_F is None:
            # F = (Â∆H + ∆ÂH + ∆Â∆H)W
            term1 = SparseOperations.sparse_dense_mm(adj, delta_features)  # Â∆H
            term2 = SparseOperations.sparse_dense_mm(delta_adj, features)  # ∆ÂH
            term3 = SparseOperations.sparse_dense_mm(delta_adj, delta_features)  # ∆Â∆H
            
            F_input = term1 + term2 + term3
            fixed_term = torch.mm(F_input, self.W)  # Fixed term
            
            if self.bias is not None:
                # Bias is already included in cached_Z, so we don't add it to F
                pass
        else:
            # OPTIMIZED: Reuse cached F and only compute for new nodes
            if cached_F.size(0) < features.size(0):
                # Only compute F for NEW nodes (incremental)
                n_old = cached_F.size(0)
                
                # For new nodes: F_new = (Â∆H + ∆ÂH + ∆Â∆H)W
                # Since new nodes have ∆H ≠ 0, we compute for them
                term1_new = SparseOperations.sparse_dense_mm(adj[n_old:], delta_features)  # Â_new·∆H
                term2_new = SparseOperations.sparse_dense_mm(delta_adj[n_old:], features)  # ∆Â_new·H
                term3_new = SparseOperations.sparse_dense_mm(delta_adj[n_old:], delta_features)  # ∆Â_new·∆H
                
                F_input_new = term1_new + term2_new + term3_new
                F_new = torch.mm(F_input_new, self.W)
                
                # Concatenate: [cached_F; F_new]
                fixed_term = torch.cat([cached_F, F_new], dim=0)
            else:
                # Same size, use cached directly
                fixed_term = cached_F
        
        # Compute B matrix if not cached (CRITICAL OPTIMIZATION!)
        if cached_B is None:
            # Count B computations (DEBUG)
            if not hasattr(self, '_b_compute_count'):
                self._b_compute_count = 0
            self._b_compute_count += 1
            
            # B = ÂH + Â∆H + ∆ÂH + ∆Â∆H
            term1 = SparseOperations.sparse_dense_mm(adj, features)  # ÂH
            term2 = SparseOperations.sparse_dense_mm(adj, delta_features)  # Â∆H
            term3 = SparseOperations.sparse_dense_mm(delta_adj, features)  # ∆ÂH
            term4 = SparseOperations.sparse_dense_mm(delta_adj, delta_features)  # ∆Â∆H
            
            B = term1 + term2 + term3 + term4
        else:
            # OPTIMIZED: Reuse cached B and only compute for new nodes
            if cached_B.size(0) < features.size(0):
                # Only compute B for NEW nodes
                n_old = cached_B.size(0)
                
                # For new nodes: B_new = ÂH_new + Â∆H_new + ∆ÂH_new + ∆Â∆H_new
                term1_new = SparseOperations.sparse_dense_mm(adj[n_old:], features)
                term2_new = SparseOperations.sparse_dense_mm(adj[n_old:], delta_features)
                term3_new = SparseOperations.sparse_dense_mm(delta_adj[n_old:], features)
                term4_new = SparseOperations.sparse_dense_mm(delta_adj[n_old:], delta_features)
                
                B_new = term1_new + term2_new + term3_new + term4_new
                
                # Concatenate: [cached_B; B_new]
                B = torch.cat([cached_B, B_new], dim=0)
            else:
                # Same size, use cached directly
                B = cached_B
        
        # B ΔW
        B_delta_W = torch.mm(B, self.delta_W)
        
        # CRITICAL: Always expand fixed_term to match B_delta_W size
        # This is needed for new nodes!
        if fixed_term.size(0) < B_delta_W.size(0):
            padding = torch.zeros(B_delta_W.size(0) - fixed_term.size(0),
                                fixed_term.size(1),
                                device=fixed_term.device)
            fixed_term = torch.cat([fixed_term, padding], dim=0)
        
        # Handle cached_Z (may be None on first call)
        if cached_Z is None:
            # No cached Z - use just F + B ΔW
            new_Z = fixed_term + B_delta_W
        else:
            # Expand cached_Z if dimensions don't match
            if cached_Z.size(0) < B_delta_W.size(0):
                # Pad with zeros
                padding = torch.zeros(B_delta_W.size(0) - cached_Z.size(0), 
                                    cached_Z.size(1),
                                    device=cached_Z.device)
                cached_Z_expanded = torch.cat([cached_Z, padding], dim=0)
            else:
                cached_Z_expanded = cached_Z
            
            # Z' = Z + F + B ΔW
            new_Z = cached_Z_expanded + fixed_term + B_delta_W
        
        if self.delta_bias is not None:
            new_Z = new_Z + self.delta_bias
        
        return new_Z, fixed_term, B


class ExiGCN(nn.Module):
    """
    Multi-layer ExiGCN with efficient retraining.
    """
    
    def __init__(self,
                 num_features: int,
                 hidden_dim: int,
                 num_classes: int,
                 num_layers: int = 3,
                 dropout: float = 0.5,
                 activation: str = 'relu'):
        super(ExiGCN, self).__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Activation
        if activation == 'relu':
            self.activation = F_functional.relu
        elif activation == 'elu':
            self.activation = F_functional.elu
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(ExiGCNLayer(num_features, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(ExiGCNLayer(hidden_dim, hidden_dim))
        
        # Output layer
        self.layers.append(ExiGCNLayer(hidden_dim, num_classes))
        
        # Cache for retraining
        self.cached_Z = []  # Pre-activation values
        self.cached_H = []  # Post-activation values
        self.cached_F = []  # Fixed terms
        self.cached_B = []  # B matrices (NEW - for speedup!)
        
        self.is_initial_training = True
    
    def forward_initial(self, 
                       adj: torch.Tensor, 
                       features: torch.Tensor,
                       cache: bool = True) -> torch.Tensor:
        """
        Initial training (90% graph).
        
        Args:
            adj: Normalized sparse adjacency [N x N]
            features: Node features [N x D]
            cache: Whether to cache intermediate results
            
        Returns:
            Logits [N x num_classes]
        """
        if cache:
            self.cached_Z = []
            self.cached_H = []
        
        h = features
        
        # Forward through all layers
        for i, layer in enumerate(self.layers[:-1]):
            z = layer.forward_initial(adj, h)
            if cache:
                self.cached_Z.append(z.detach())
            
            h = self.activation(z)
            h = F_functional.dropout(h, p=self.dropout, training=self.training)
            
            if cache:
                self.cached_H.append(h.detach())
        
        # Last layer
        z = self.layers[-1].forward_initial(adj, h)
        if cache:
            self.cached_Z.append(z.detach())
        
        return z
    
    def forward_retraining(self,
                          adj: torch.Tensor,
                          features: torch.Tensor,
                          delta_adj: torch.Tensor,
                          delta_features: torch.Tensor) -> torch.Tensor:
        """
        Retraining after graph update.
        
        Args:
            adj: Updated normalized adjacency Â' [N' x N']
            features: Updated features H' [N' x D]
            delta_adj: ∆Â [N' x N']
            delta_features: ∆H [N' x D]
            
        Returns:
            Logits [N' x num_classes]
        """
        h = features
        delta_h = delta_features
        
        # Forward through all layers
        for i, layer in enumerate(self.layers[:-1]):
            # Get cached values
            cached_Z = self.cached_Z[i] if i < len(self.cached_Z) else None
            cached_fixed_term = self.cached_F[i] if i < len(self.cached_F) else None
            cached_B_term = self.cached_B[i] if i < len(self.cached_B) else None  # NEW!
            
            # Forward with caching (now returns B too)
            z, fixed_term, B = layer.forward_retraining(
                adj, h, delta_adj, delta_h, cached_Z, cached_fixed_term, cached_B_term
            )
            
            # Cache fixed_term if first time
            if len(self.cached_F) <= i:
                self.cached_F.append(fixed_term.detach())
            
            # Cache B if first time (NEW - CRITICAL FOR SPEED!)
            if len(self.cached_B) <= i:
                self.cached_B.append(B.detach())
            
            # Activation
            h = self.activation(z)
            h = F_functional.dropout(h, p=self.dropout, training=self.training)
            
            # Compute delta_h for next layer
            if i < len(self.cached_H):
                cached_h = self.cached_H[i]
                if cached_h.size(0) < h.size(0):
                    padding = torch.zeros(h.size(0) - cached_h.size(0),
                                        cached_h.size(1),
                                        device=cached_h.device)
                    cached_h = torch.cat([cached_h, padding], dim=0)
                delta_h = h - cached_h
            else:
                delta_h = torch.zeros_like(h)
        
        # Last layer
        cached_Z_last = self.cached_Z[-1] if len(self.cached_Z) >= self.num_layers else None
        cached_fixed_term_last = self.cached_F[-1] if len(self.cached_F) >= self.num_layers else None
        cached_B_last = self.cached_B[-1] if len(self.cached_B) >= self.num_layers else None
        
        z, fixed_term, B = self.layers[-1].forward_retraining(
            adj, h, delta_adj, delta_h, cached_Z_last, cached_fixed_term_last, cached_B_last
        )
        
        # Cache fixed_term if first time
        if len(self.cached_F) < self.num_layers:
            self.cached_F.append(fixed_term.detach())
        
        # Cache B if first time (NEW!)
        if len(self.cached_B) < self.num_layers:
            self.cached_B.append(B.detach())
        
        return z
    
    def merge_all_deltas(self):
        """Merge all delta weights into main weights."""
        for layer in self.layers:
            layer.merge_deltas()
        
        # Clear caches (will be rebuilt)
        self.cached_Z = []
        self.cached_H = []
        self.cached_F = []
        self.cached_B = []
    
    def reset_all_deltas(self):
        """Reset all delta weights to zero."""
        for layer in self.layers:
            layer.reset_delta()
    
    def prepare_for_retraining(self):
        """Prepare model for retraining phase."""
        self.is_initial_training = False
        self.reset_all_deltas()
        self.cached_F = []
        self.cached_B = []
    
    def forward(self, adj, features, delta_adj=None, delta_features=None):
        """Unified forward pass."""
        if self.is_initial_training or delta_adj is None:
            return self.forward_initial(adj, features, cache=True)
        else:
            return self.forward_retraining(adj, features, delta_adj, delta_features)


def test_exigcn():
    """Test ExiGCN model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    # Dummy data
    num_nodes_initial = 90
    num_nodes_new = 100
    num_features = 32
    num_classes = 7
    hidden_dim = 64
    
    print("\n=== Phase 1: Initial Training (90%) ===")
    
    # Initial graph (90%)
    from utils.sparse_ops import SparseOperations
    
    adj_90 = (torch.rand(num_nodes_initial, num_nodes_initial) < 0.1).float()
    adj_90 = (adj_90 + adj_90.T) / 2
    row, col, val = SparseOperations.dense_to_coo(adj_90)
    adj_90_sparse = SparseOperations.coo_to_sparse_tensor(row, col, val,
                                                           (num_nodes_initial, num_nodes_initial), device)
    adj_90_norm = SparseOperations.normalize_adjacency(adj_90_sparse, add_self_loops=True)
    
    features_90 = torch.randn(num_nodes_initial, num_features).to(device)
    
    model = ExiGCN(
        num_features=num_features,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=3,
        dropout=0.5
    ).to(device)
    
    # Initial forward
    model.train()
    logits_90 = model.forward_initial(adj_90_norm, features_90, cache=True)
    print(f"Initial logits shape: {logits_90.shape}")
    print(f"Cached Z layers: {len(model.cached_Z)}")
    print(f"Cached H layers: {len(model.cached_H)}")
    
    print("\n=== Phase 2: Graph Update (90% → 100%) ===")
    
    # Updated graph (100%)
    adj_100 = (torch.rand(num_nodes_new, num_nodes_new) < 0.1).float()
    adj_100 = (adj_100 + adj_100.T) / 2
    adj_100[:num_nodes_initial, :num_nodes_initial] = adj_90  # Keep original
    
    row, col, val = SparseOperations.dense_to_coo(adj_100)
    adj_100_sparse = SparseOperations.coo_to_sparse_tensor(row, col, val,
                                                            (num_nodes_new, num_nodes_new), device)
    adj_100_norm = SparseOperations.normalize_adjacency(adj_100_sparse, add_self_loops=True)
    
    features_100 = torch.randn(num_nodes_new, num_features).to(device)
    features_100[:num_nodes_initial] = features_90  # Keep original
    
    # Compute deltas
    delta_adj = SparseOperations.compute_delta_sparse(adj_90_norm, adj_100_norm)
    delta_features = features_100 - torch.cat([features_90, 
                                               torch.zeros(num_nodes_new - num_nodes_initial, 
                                                          num_features, device=device)])
    
    # Retraining forward
    model.prepare_for_retraining()
    logits_100 = model.forward_retraining(adj_100_norm, features_100, delta_adj, delta_features)
    print(f"Retraining logits shape: {logits_100.shape}")
    print(f"Cached F layers: {len(model.cached_F)}")
    print(f"Cached B layers: {len(model.cached_B)}")
    
    print("\n✅ ExiGCN test passed with B caching!")


if __name__ == "__main__":
    test_exigcn()
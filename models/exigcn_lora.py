"""
ExiGCN with LoRA (Low-Rank Adaptation)
Reduces retraining parameters by 90%+
"""

import torch
import torch.nn as nn
import torch.nn.functional as F_functional
from typing import Optional, Tuple
import sys
sys.path.append('..')
from utils.sparse_ops import SparseOperations


class ExiGCNLayerLoRA(nn.Module):
    """
    ExiGCN Layer with LoRA for efficient retraining.
    
    Instead of learning full ΔW, we learn:
    ΔW = A @ B where A: [in × rank], B: [rank × out]
    
    This reduces parameters from in×out to in×rank + rank×out
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 rank: int = 8, lora_alpha: float = 16.0):
        super(ExiGCNLayerLoRA, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank
        
        # Main weights (frozen during retraining)
        self.W = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # LoRA low-rank matrices (trainable during retraining)
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        # Kaiming initialization for main weights
        nn.init.kaiming_uniform_(self.W, a=5**0.5)
        nn.init.zeros_(self.bias)
        
        # LoRA initialization (as per LoRA paper)
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
    
    def forward_initial(self, adj: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Initial training forward (standard GCN)."""
        h_agg = SparseOperations.sparse_dense_mm(adj, features)
        z = torch.mm(h_agg, self.W)
        
        if self.bias is not None:
            z = z + self.bias
        
        return z
    
    def forward_retraining(self,
                          adj: torch.Tensor,
                          features: torch.Tensor,
                          delta_adj: torch.Tensor,
                          delta_features: torch.Tensor,
                          cached_Z: torch.Tensor,
                          cached_F: Optional[torch.Tensor] = None,
                          cached_B: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retraining forward with LoRA."""
        # Compute F if not cached
        if cached_F is None:
            term1 = SparseOperations.sparse_dense_mm(adj, delta_features)
            term2 = SparseOperations.sparse_dense_mm(delta_adj, features)
            term3 = SparseOperations.sparse_dense_mm(delta_adj, delta_features)
            
            F_input = term1 + term2 + term3
            fixed_term = torch.mm(F_input, self.W)
        else:
            fixed_term = cached_F
        
        # Compute B if not cached
        if cached_B is None:
            term1 = SparseOperations.sparse_dense_mm(adj, features)
            term2 = SparseOperations.sparse_dense_mm(adj, delta_features)
            term3 = SparseOperations.sparse_dense_mm(delta_adj, features)
            term4 = SparseOperations.sparse_dense_mm(delta_adj, delta_features)
            
            B = term1 + term2 + term3 + term4
        else:
            B = cached_B
        
        # LoRA: ΔW = A @ B * scaling
        delta_W = torch.mm(self.lora_A, self.lora_B) * self.scaling
        
        # B @ ΔW
        B_delta_W = torch.mm(B, delta_W)
        
        # Handle cached_Z
        if cached_Z is None:
            new_Z = fixed_term + B_delta_W
        else:
            if cached_Z.size(0) < B_delta_W.size(0):
                padding = torch.zeros(B_delta_W.size(0) - cached_Z.size(0), 
                                    cached_Z.size(1),
                                    device=cached_Z.device)
                cached_Z_expanded = torch.cat([cached_Z, padding], dim=0)
            else:
                cached_Z_expanded = cached_Z
            
            if fixed_term.size(0) < B_delta_W.size(0):
                padding = torch.zeros(B_delta_W.size(0) - fixed_term.size(0),
                                    fixed_term.size(1),
                                    device=fixed_term.device)
                fixed_term_expanded = torch.cat([fixed_term, padding], dim=0)
            else:
                fixed_term_expanded = fixed_term
            
            new_Z = cached_Z_expanded + fixed_term_expanded + B_delta_W
        
        return new_Z, fixed_term, B
    
    def merge_lora(self):
        """Merge LoRA weights into main weights."""
        with torch.no_grad():
            delta_W = torch.mm(self.lora_A, self.lora_B) * self.scaling
            self.W.add_(delta_W)
            self.lora_A.zero_()
            self.lora_B.zero_()
    
    def reset_lora(self):
        """Reset LoRA matrices to zero."""
        with torch.no_grad():
            self.lora_A.zero_()
            self.lora_B.zero_()
    
    def get_num_params(self):
        """Get number of parameters."""
        main_params = self.W.numel() + self.bias.numel()
        lora_params = self.lora_A.numel() + self.lora_B.numel()
        return {
            'main': main_params,
            'lora': lora_params,
            'total': main_params + lora_params,
            'reduction': 1 - (lora_params / main_params) if main_params > 0 else 0
        }


class ExiGCNLoRA(nn.Module):
    """Multi-layer ExiGCN with LoRA."""
    
    def __init__(self,
                 num_features: int,
                 hidden_dim: int,
                 num_classes: int,
                 num_layers: int = 3,
                 dropout: float = 0.5,
                 activation: str = 'relu',
                 lora_rank: int = 8,
                 lora_alpha: float = 16.0):
        super(ExiGCNLoRA, self).__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = dropout
        self.lora_rank = lora_rank
        
        if activation == 'relu':
            self.activation = F_functional.relu
        elif activation == 'elu':
            self.activation = F_functional.elu
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build layers with LoRA
        self.layers = nn.ModuleList()
        self.layers.append(ExiGCNLayerLoRA(num_features, hidden_dim, 
                                           rank=lora_rank, lora_alpha=lora_alpha))
        
        for _ in range(num_layers - 2):
            self.layers.append(ExiGCNLayerLoRA(hidden_dim, hidden_dim,
                                               rank=lora_rank, lora_alpha=lora_alpha))
        
        self.layers.append(ExiGCNLayerLoRA(hidden_dim, num_classes,
                                           rank=lora_rank, lora_alpha=lora_alpha))
        
        # Caches
        self.cached_Z = []
        self.cached_H = []
        self.cached_F = []
        self.cached_B = []
        
        self.is_initial_training = True
    
    def forward_initial(self, adj, features, cache=True):
        """Initial training forward."""
        h = features
        
        for i, layer in enumerate(self.layers[:-1]):
            z = layer.forward_initial(adj, h)
            
            if cache and len(self.cached_Z) <= i:
                self.cached_Z.append(z.detach())
            
            h = self.activation(z)
            h = F_functional.dropout(h, p=self.dropout, training=self.training)
            
            if cache and len(self.cached_H) <= i:
                self.cached_H.append(h.detach())
        
        z = self.layers[-1].forward_initial(adj, h)
        
        if cache and len(self.cached_Z) < self.num_layers:
            self.cached_Z.append(z.detach())
        
        return z
    
    def forward_retraining(self, adj, features, delta_adj, delta_features):
        """Retraining forward with LoRA."""
        h = features
        delta_h = delta_features
        
        for i, layer in enumerate(self.layers[:-1]):
            cached_Z = self.cached_Z[i] if i < len(self.cached_Z) else None
            cached_F = self.cached_F[i] if i < len(self.cached_F) else None
            cached_B = self.cached_B[i] if i < len(self.cached_B) else None
            
            z, F, B = layer.forward_retraining(
                adj, h, delta_adj, delta_h, cached_Z, cached_F, cached_B
            )
            
            if len(self.cached_F) <= i:
                self.cached_F.append(F.detach())
            if len(self.cached_B) <= i:
                self.cached_B.append(B.detach())
            
            h = self.activation(z)
            h = F_functional.dropout(h, p=self.dropout, training=self.training)
            
            if i < len(self.cached_H):
                cached_h = self.cached_H[i]
                if cached_h.size(0) < h.size(0):
                    padding = torch.zeros(h.size(0) - cached_h.size(0),
                                        cached_h.size(1), device=cached_h.device)
                    cached_h = torch.cat([cached_h, padding], dim=0)
                delta_h = h - cached_h
            else:
                delta_h = torch.zeros_like(h)
        
        # Last layer
        cached_Z = self.cached_Z[-1] if len(self.cached_Z) >= self.num_layers else None
        cached_F = self.cached_F[-1] if len(self.cached_F) >= self.num_layers else None
        cached_B = self.cached_B[-1] if len(self.cached_B) >= self.num_layers else None
        
        z, F, B = self.layers[-1].forward_retraining(
            adj, h, delta_adj, delta_h, cached_Z, cached_F, cached_B
        )
        
        if len(self.cached_F) < self.num_layers:
            self.cached_F.append(F.detach())
        if len(self.cached_B) < self.num_layers:
            self.cached_B.append(B.detach())
        
        return z
    
    def merge_all_lora(self):
        """Merge all LoRA weights."""
        for layer in self.layers:
            layer.merge_lora()
        self.cached_Z = []
        self.cached_H = []
        self.cached_F = []
    
    def reset_all_lora(self):
        """Reset all LoRA matrices."""
        for layer in self.layers:
            layer.reset_lora()
    
    def prepare_for_retraining(self):
        """Prepare for retraining."""
        self.is_initial_training = False
        self.reset_all_lora()
        self.cached_F = []
    
    def freeze_main_weights(self):
        """Freeze main weights, only train LoRA."""
        for layer in self.layers:
            layer.W.requires_grad = False
            layer.bias.requires_grad = False
            layer.lora_A.requires_grad = True
            layer.lora_B.requires_grad = True
    
    def get_lora_params(self):
        """Get LoRA parameters for optimizer."""
        params = []
        for layer in self.layers:
            params.extend([layer.lora_A, layer.lora_B])
        return params
    
    def get_num_params(self):
        """Get parameter statistics."""
        stats = {
            'main': 0,
            'lora': 0,
            'total': 0
        }
        
        for layer in self.layers:
            layer_stats = layer.get_num_params()
            stats['main'] += layer_stats['main']
            stats['lora'] += layer_stats['lora']
            stats['total'] += layer_stats['total']
        
        stats['reduction'] = 1 - (stats['lora'] / stats['main']) if stats['main'] > 0 else 0
        return stats
    
    def forward(self, adj, features, delta_adj=None, delta_features=None):
        """Unified forward."""
        if self.is_initial_training or delta_adj is None:
            return self.forward_initial(adj, features, cache=True)
        else:
            return self.forward_retraining(adj, features, delta_adj, delta_features)


if __name__ == "__main__":
    print("✅ ExiGCN with LoRA - Ready to use!")
    print("\nQuick test:")
    model = ExiGCNLoRA(100, 64, 10, lora_rank=8)
    stats = model.get_num_params()
    print(f"  Total params: {stats['total']:,}")
    print(f"  LoRA params: {stats['lora']:,}")
    print(f"  Reduction: {stats['reduction']:.1%}")
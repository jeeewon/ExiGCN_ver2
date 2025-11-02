"""
Base GCN implementation for full retraining.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F_functional
from utils.sparse_ops import SparseOperations


class GCNLayer(nn.Module):
    """
    Single GCN layer: H' = σ(Â H W)
    """
    
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 bias: bool = True):
        super(GCNLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters using Glorot initialization."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, adj: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Â H W
        
        Args:
            adj: Normalized sparse adjacency matrix [N x N]
            features: Node features [N x D]
            
        Returns:
            Output features [N x out_features]
        """
        # H W
        support = torch.mm(features, self.weight)
        
        # Â (H W)
        output = SparseOperations.sparse_dense_mm(adj, support)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


class BaseGCN(nn.Module):
    """
    Multi-layer GCN for node classification.
    Used as baseline for full retraining comparison.
    """
    
    def __init__(self,
                 num_features: int,
                 hidden_dim: int,
                 num_classes: int,
                 num_layers: int = 3,
                 dropout: float = 0.5,
                 activation: str = 'relu'):
        super(BaseGCN, self).__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Activation function
        if activation == 'relu':
            self.activation = F_functional.relu
        elif activation == 'elu':
            self.activation = F_functional.elu
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(GCNLayer(num_features, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GCNLayer(hidden_dim, hidden_dim))
        
        # Output layer
        self.layers.append(GCNLayer(hidden_dim, num_classes))
    
    def forward(self, adj: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all layers.
        
        Args:
            adj: Normalized sparse adjacency [N x N]
            features: Node features [N x D]
            
        Returns:
            Logits [N x num_classes]
        """
        h = features
        
        # Pass through all layers except last
        for i, layer in enumerate(self.layers[:-1]):
            h = layer(adj, h)
            h = self.activation(h)
            h = F_functional.dropout(h, p=self.dropout, training=self.training)
        
        # Last layer (no activation, no dropout)
        h = self.layers[-1](adj, h)
        
        return h
    
    def get_embeddings(self, adj: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Get node embeddings from second-to-last layer.
        
        Args:
            adj: Normalized sparse adjacency
            features: Node features
            
        Returns:
            Node embeddings [N x hidden_dim]
        """
        h = features
        
        for i, layer in enumerate(self.layers[:-1]):
            h = layer(adj, h)
            if i < len(self.layers) - 2:  # Not the embedding layer yet
                h = self.activation(h)
                h = F_functional.dropout(h, p=self.dropout, training=self.training)
        
        return h


def test_base_gcn():
    """Test BaseGCN model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    # Create dummy data
    num_nodes = 100
    num_features = 32
    num_classes = 7
    hidden_dim = 64
    
    # Random sparse adjacency
    density = 0.1
    adj_dense = (torch.rand(num_nodes, num_nodes) < density).float()
    adj_dense = (adj_dense + adj_dense.T) / 2  # Symmetric
    
    from utils.sparse_ops import SparseOperations
    row, col, val = SparseOperations.dense_to_coo(adj_dense)
    adj_sparse = SparseOperations.coo_to_sparse_tensor(row, col, val, 
                                                        (num_nodes, num_nodes), device)
    
    # Normalize adjacency
    adj_norm = SparseOperations.normalize_adjacency(adj_sparse, add_self_loops=True)
    
    # Random features
    features = torch.randn(num_nodes, num_features).to(device)
    
    # Create model
    model = BaseGCN(
        num_features=num_features,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=3,
        dropout=0.5
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(adj_norm, features)
        embeddings = model.get_embeddings(adj_norm, features)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Embeddings shape: {embeddings.shape}")
    
    print("✅ BaseGCN test passed!")


if __name__ == "__main__":
    test_base_gcn()
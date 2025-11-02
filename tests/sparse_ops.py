"""
Sparse matrix operations using triplet (COO) format for GPU acceleration.
"""

import torch
import torch.sparse as sparse
import numpy as np
from typing import Tuple, Optional


class SparseOperations:
    """
    GPU-accelerated sparse matrix operations in COO (triplet) format.
    """
    
    @staticmethod
    def dense_to_coo(adj_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert dense adjacency matrix to COO (triplet) format.
        
        Args:
            adj_matrix: Dense adjacency matrix [N x N]
            
        Returns:
            row_indices: Row indices of non-zero elements
            col_indices: Column indices of non-zero elements
            values: Values of non-zero elements
        """
        if adj_matrix.is_sparse:
            adj_coo = adj_matrix.coalesce()
            indices = adj_coo.indices()
            return indices[0], indices[1], adj_coo.values()
        
        nonzero = adj_matrix.nonzero(as_tuple=False)
        row_indices = nonzero[:, 0]
        col_indices = nonzero[:, 1]
        values = adj_matrix[row_indices, col_indices]
        
        return row_indices, col_indices, values
    
    @staticmethod
    def coo_to_sparse_tensor(row_indices: torch.Tensor, 
                            col_indices: torch.Tensor,
                            values: torch.Tensor,
                            shape: Tuple[int, int],
                            device: torch.device = None) -> torch.Tensor:
        """
        Convert COO format to PyTorch sparse tensor.
        
        Args:
            row_indices: Row indices
            col_indices: Column indices  
            values: Values
            shape: Shape of the matrix (N, M)
            device: Target device
            
        Returns:
            Sparse tensor in COO format
        """
        if device is None:
            device = row_indices.device
            
        indices = torch.stack([row_indices, col_indices], dim=0).to(device)
        values = values.to(device)
        
        return torch.sparse_coo_tensor(indices, values, shape, device=device)
    
    @staticmethod
    def sparse_dense_mm(sparse_A: torch.Tensor, 
                       dense_B: torch.Tensor) -> torch.Tensor:
        """
        Sparse-Dense matrix multiplication (GPU optimized).
        
        Args:
            sparse_A: Sparse matrix [N x M] in COO format
            dense_B: Dense matrix [M x K]
            
        Returns:
            Result matrix [N x K]
        """
        return torch.sparse.mm(sparse_A, dense_B)
    
    @staticmethod
    def add_sparse_tensors(sparse_A: torch.Tensor,
                          sparse_B: torch.Tensor) -> torch.Tensor:
        """
        Add two sparse tensors.
        
        Args:
            sparse_A: First sparse tensor
            sparse_B: Second sparse tensor
            
        Returns:
            Sum of sparse tensors
        """
        # Ensure both are coalesced
        sparse_A = sparse_A.coalesce()
        sparse_B = sparse_B.coalesce()
        
        # Simple addition
        result = sparse_A + sparse_B
        return result.coalesce()
    
    @staticmethod
    def expand_sparse_tensor(sparse_tensor: torch.Tensor,
                            old_size: Tuple[int, int],
                            new_size: Tuple[int, int]) -> torch.Tensor:
        """
        Expand sparse tensor dimensions (for node addition).
        
        Args:
            sparse_tensor: Original sparse tensor
            old_size: Original size (N, N)
            new_size: New size (N+k, N+k)
            
        Returns:
            Expanded sparse tensor with zeros in new dimensions
        """
        sparse_tensor = sparse_tensor.coalesce()
        indices = sparse_tensor.indices()
        values = sparse_tensor.values()
        
        # Create new sparse tensor with expanded size
        expanded = torch.sparse_coo_tensor(
            indices, values, new_size, 
            device=sparse_tensor.device,
            dtype=sparse_tensor.dtype
        )
        
        return expanded.coalesce()
    
    @staticmethod
    def compute_delta_sparse(old_adj: torch.Tensor,
                            new_adj: torch.Tensor,
                            new_nodes: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute delta (difference) between two sparse adjacency matrices.
        Handles dimension mismatch when nodes are added.
        
        Args:
            old_adj: Original sparse adjacency [N x N]
            new_adj: Updated sparse adjacency [M x M], M >= N
            new_nodes: Indices of newly added nodes (optional)
            
        Returns:
            Delta sparse tensor [M x M]
        """
        old_size = old_adj.size()
        new_size = new_adj.size()
        
        # If dimensions differ, expand old_adj
        if old_size != new_size:
            old_adj_expanded = SparseOperations.expand_sparse_tensor(
                old_adj, old_size, new_size
            )
        else:
            old_adj_expanded = old_adj
        
        # Compute delta: new - old
        delta = new_adj - old_adj_expanded
        
        return delta.coalesce()
    
    @staticmethod
    def normalize_adjacency(adj: torch.Tensor,
                           add_self_loops: bool = True) -> torch.Tensor:
        """
        Normalize adjacency matrix: D^(-1/2) @ A @ D^(-1/2)
        Following GCN paper normalization.
        
        Args:
            adj: Sparse adjacency matrix [N x N]
            add_self_loops: Whether to add self-loops
            
        Returns:
            Normalized sparse adjacency matrix
        """
        adj = adj.coalesce()
        N = adj.size(0)
        
        # Add self-loops
        if add_self_loops:
            indices = adj.indices()
            values = adj.values()
            
            # Add diagonal indices
            self_loop_indices = torch.arange(N, device=adj.device).unsqueeze(0).repeat(2, 1)
            all_indices = torch.cat([indices, self_loop_indices], dim=1)
            all_values = torch.cat([values, torch.ones(N, device=adj.device)])
            
            adj = torch.sparse_coo_tensor(
                all_indices, all_values, (N, N), device=adj.device
            ).coalesce()
        
        # Compute degree
        degrees = torch.sparse.sum(adj, dim=1).to_dense()
        
        # D^(-1/2)
        deg_inv_sqrt = torch.pow(degrees, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        
        # Create diagonal matrix D^(-1/2)
        indices = torch.arange(N, device=adj.device).unsqueeze(0).repeat(2, 1)
        D_inv_sqrt = torch.sparse_coo_tensor(
            indices, deg_inv_sqrt, (N, N), device=adj.device
        )
        
        # D^(-1/2) @ A @ D^(-1/2)
        adj_normalized = torch.sparse.mm(D_inv_sqrt, adj)
        adj_normalized = torch.sparse.mm(adj_normalized, D_inv_sqrt)
        
        return adj_normalized.coalesce()
    
    @staticmethod
    def get_nnz(sparse_tensor: torch.Tensor) -> int:
        """
        Get number of non-zero elements in sparse tensor.
        
        Args:
            sparse_tensor: Sparse tensor
            
        Returns:
            Number of non-zero elements
        """
        return sparse_tensor._nnz()
    
    @staticmethod
    def sparse_to_device(sparse_tensor: torch.Tensor, 
                        device: torch.device) -> torch.Tensor:
        """
        Move sparse tensor to device.
        
        Args:
            sparse_tensor: Sparse tensor
            device: Target device
            
        Returns:
            Sparse tensor on target device
        """
        sparse_tensor = sparse_tensor.coalesce()
        indices = sparse_tensor.indices().to(device)
        values = sparse_tensor.values().to(device)
        size = sparse_tensor.size()
        
        return torch.sparse_coo_tensor(indices, values, size, device=device)


def test_sparse_operations():
    """Test sparse operations on GPU."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    # Create test adjacency matrix
    N = 100
    density = 0.1
    adj_dense = (torch.rand(N, N) < density).float().to(device)
    adj_dense = (adj_dense + adj_dense.T) / 2  # Make symmetric
    
    # Convert to sparse
    row, col, val = SparseOperations.dense_to_coo(adj_dense)
    adj_sparse = SparseOperations.coo_to_sparse_tensor(row, col, val, (N, N), device)
    
    print(f"Original nnz: {SparseOperations.get_nnz(adj_sparse)}")
    
    # Test normalization
    adj_norm = SparseOperations.normalize_adjacency(adj_sparse, add_self_loops=True)
    print(f"Normalized nnz: {SparseOperations.get_nnz(adj_norm)}")
    
    # Test sparse-dense multiplication
    features = torch.randn(N, 32).to(device)
    result = SparseOperations.sparse_dense_mm(adj_norm, features)
    print(f"SpMM result shape: {result.shape}")
    
    # Test expansion
    adj_expanded = SparseOperations.expand_sparse_tensor(adj_sparse, (N, N), (N+10, N+10))
    print(f"Expanded shape: {adj_expanded.shape}")
    
    print("✅ All tests passed!")


if __name__ == "__main__":
    test_sparse_operations()
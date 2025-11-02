"""
Debug script to check delta graph computation.
"""

import torch
from data.download import DatasetLoader
from data.graph_updater import GraphUpdater
from data.preprocessor import DataPreprocessor
from utils.sparse_ops import SparseOperations
from train.trainer_exi import ExiGCNTrainer

# Load dataset
loader = DatasetLoader(root='./data')
adj, features, labels, train_mask, val_mask, test_mask = loader.load_cora_full()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Create incremental scenario
updater = GraphUpdater(
    adj=adj,
    features=features,
    labels=labels,
    initial_ratio=0.9,
    n_buckets=5,
    seed=42
)

buckets, core_adj, core_features, core_labels, node_mapping = updater.create_incremental_scenario()

# Preprocessor
preprocessor = DataPreprocessor(
    normalize_features=True,
    normalize_adj=True,
    add_self_loops=True
)

# Preprocess core graph
core_adj_norm, core_features_norm = preprocessor.preprocess(
    core_adj, core_features, device
)

print("="*70)
print("STEP 1: Initial Graph (90%)")
print("="*70)
print(f"Core nodes: {core_adj_norm.size(0)}")
print(f"Core adj nnz (original): {core_adj._nnz()}")
print(f"Core adj nnz (normalized): {core_adj_norm._nnz()}")
print(f"Core features shape: {core_features_norm.shape}")
print(f"Core features non-zero: {(core_features_norm != 0).sum().item()}")

# Add bucket A
print("\n" + "="*70)
print("STEP 2: Adding Bucket A (+2%)")
print("="*70)

bucket_A_nodes = buckets['A']
print(f"Bucket A nodes (original indices): {bucket_A_nodes[:10]}")

# Add bucket
current_adj, current_features, current_labels = updater.add_bucket_to_graph(
    core_adj, core_features, core_labels, bucket_A_nodes
)

# Preprocess updated graph
current_adj_norm, current_features_norm = preprocessor.preprocess(
    current_adj, current_features, device
)

print(f"Updated nodes: {current_adj_norm.size(0)}")
print(f"Updated adj nnz (original): {current_adj._nnz()}")
print(f"Updated adj nnz (normalized): {current_adj_norm._nnz()}")
print(f"Updated features shape: {current_features_norm.shape}")
print(f"Updated features non-zero: {(current_features_norm != 0).sum().item()}")

# Compute deltas (same as trainer does)
print("\n" + "="*70)
print("STEP 3: Delta Computation")
print("="*70)

# Check: Are initial_adj and initial_features stored correctly?
print("\n--- Checking stored initial graph ---")
initial_adj_stored = core_adj_norm.clone()
initial_features_stored = core_features_norm.clone()
print(f"Stored initial adj shape: {initial_adj_stored.shape}")
print(f"Stored initial adj nnz: {initial_adj_stored._nnz()}")
print(f"Stored initial features shape: {initial_features_stored.shape}")

# Compute delta adjacency (same as trainer_exi._compute_deltas)
print("\n--- Computing delta adjacency ---")
delta_adj = SparseOperations.compute_delta_sparse(
    initial_adj_stored,
    current_adj_norm
)
print(f"Delta adj shape: {delta_adj.shape}")
print(f"Delta adj nnz: {delta_adj._nnz()}")

# Expected: Delta should be much smaller than full graph
print(f"\nDelta adj nnz / Updated adj nnz: {delta_adj._nnz() / current_adj_norm._nnz():.4f}")
print(f"Delta adj nnz / Initial adj nnz: {delta_adj._nnz() / initial_adj_stored._nnz():.4f}")

# Compute delta features (same as trainer_exi._compute_deltas)
print("\n--- Computing delta features ---")
if current_features_norm.size(0) > initial_features_stored.size(0):
    # Pad initial features with zeros
    padding = torch.zeros(
        current_features_norm.size(0) - initial_features_stored.size(0),
        initial_features_stored.size(1),
        device=initial_features_stored.device
    )
    initial_features_padded = torch.cat([initial_features_stored, padding], dim=0)
else:
    initial_features_padded = initial_features_stored

delta_features = current_features_norm - initial_features_padded

print(f"Delta features shape: {delta_features.shape}")
print(f"Delta features non-zero: {(delta_features != 0).sum().item()}")
print(f"Delta features non-zero / Updated features non-zero: {(delta_features != 0).sum().item() / (current_features_norm != 0).sum().item():.4f}")

# Check: Is delta actually the difference?
print("\n" + "="*70)
print("STEP 4: Verification")
print("="*70)

# Expand initial adj to match current size
initial_adj_expanded = SparseOperations.expand_sparse_tensor(
    initial_adj_stored,
    (initial_adj_stored.size(0), initial_adj_stored.size(0)),
    (current_adj_norm.size(0), current_adj_norm.size(0))
)

# Reconstruct: initial + delta = current?
reconstructed_adj = initial_adj_expanded + delta_adj
reconstructed_adj = reconstructed_adj.coalesce()

print(f"Initial adj (expanded) nnz: {initial_adj_expanded._nnz()}")
print(f"Delta adj nnz: {delta_adj._nnz()}")
print(f"Reconstructed adj nnz: {reconstructed_adj._nnz()}")
print(f"Current adj nnz: {current_adj_norm._nnz()}")
print(f"Match: {reconstructed_adj._nnz() == current_adj_norm._nnz()}")

# Check feature reconstruction
reconstructed_features = initial_features_padded + delta_features
print(f"\nInitial features (padded) non-zero: {(initial_features_padded != 0).sum().item()}")
print(f"Delta features non-zero: {(delta_features != 0).sum().item()}")
print(f"Reconstructed features non-zero: {(reconstructed_features != 0).sum().item()}")
print(f"Current features non-zero: {(current_features_norm != 0).sum().item()}")
print(f"Match: {torch.allclose(reconstructed_features, current_features_norm, atol=1e-6)}")

# Check: What percentage of edges are actually new?
print("\n" + "="*70)
print("STEP 5: Delta Analysis")
print("="*70)

# Get actual new edges (those in current but not in initial)
initial_indices_set = set()
initial_adj_coo = initial_adj_expanded.coalesce()
initial_indices = initial_adj_coo.indices().cpu().numpy()
for i in range(initial_indices.shape[1]):
    src, dst = int(initial_indices[0, i]), int(initial_indices[1, i])
    initial_indices_set.add((src, dst))

current_indices_set = set()
current_adj_coo = current_adj_norm.coalesce()
current_indices = current_adj_coo.indices().cpu().numpy()
for i in range(current_indices.shape[1]):
    src, dst = int(current_indices[0, i]), int(current_indices[1, i])
    current_indices_set.add((src, dst))

new_edges = current_indices_set - initial_indices_set
print(f"New edges count: {len(new_edges)}")
print(f"Delta adj nnz: {delta_adj._nnz()}")
print(f"Match: {len(new_edges) == delta_adj._nnz()}")

# Check: Are new edges mostly in the added nodes?
added_node_start = initial_adj_stored.size(0)
print(f"\nAdded nodes range: {added_node_start} to {current_adj_norm.size(0)-1}")
new_edges_in_added = [e for e in new_edges if e[0] >= added_node_start or e[1] >= added_node_start]
print(f"New edges involving added nodes: {len(new_edges_in_added)}")
print(f"New edges only in initial nodes: {len(new_edges) - len(new_edges_in_added)}")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("Delta computation verification complete.")
print("If delta contains too many edges, it might be computing the wrong thing.")


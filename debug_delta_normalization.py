"""
Debug script to check delta computation with normalization.
"""

import torch
from data.download import DatasetLoader
from data.graph_updater import GraphUpdater
from data.preprocessor import DataPreprocessor
from utils.sparse_ops import SparseOperations

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

print("="*70)
print("COMPARING: Delta before vs after normalization")
print("="*70)

# First, add bucket A to get current_adj
bucket_A_nodes = buckets['A']
current_adj, current_features, current_labels = updater.add_bucket_to_graph(
    core_adj, core_features, core_labels, bucket_A_nodes
)

# Approach 1: Compute delta on original (non-normalized) adj, then normalize
print("\n--- APPROACH 1: Delta on original, then normalize ---")
print("\nStep 1: Compute delta on original adjacency")
delta_adj_original = SparseOperations.compute_delta_sparse(
    core_adj,
    current_adj
)
print(f"Delta adj (original) nnz: {delta_adj_original._nnz()}")
print(f"Expected: Should be small (~458 edges)")

# Step 2: Normalize delta separately
print("\nStep 2: Normalize delta")
# But wait, we can't just normalize delta - it doesn't have the right structure
# We need to compute: normalized(current) - normalized(initial)

# Approach 2: Normalize separately, then compute delta
print("\n--- APPROACH 2: Normalize separately, then compute delta ---")
core_adj_norm, core_features_norm = preprocessor.preprocess(
    core_adj, core_features, device
)

current_adj_norm, current_features_norm = preprocessor.preprocess(
    current_adj, current_features, device
)

print("\nStep 1: Normalize separately")
print(f"Core adj (normalized) nnz: {core_adj_norm._nnz()}")
print(f"Current adj (normalized) nnz: {current_adj_norm._nnz()}")

print("\nStep 2: Compute delta on normalized")
delta_adj_normalized = SparseOperations.compute_delta_sparse(
    core_adj_norm,
    current_adj_norm
)
print(f"Delta adj (normalized) nnz: {delta_adj_normalized._nnz()}")
print(f"Current adj nnz: {current_adj_norm._nnz()}")
print(f"Delta / Current: {delta_adj_normalized._nnz() / current_adj_norm._nnz():.4f}")

# Check actual new edges
print("\n--- Actual new edges ---")
core_adj_coo = core_adj.coalesce()
core_indices = core_adj_coo.indices().cpu().numpy()
core_set = set()
for i in range(core_indices.shape[1]):
    src, dst = int(core_indices[0, i]), int(core_indices[1, i])
    core_set.add((src, dst))

current_adj_coo = current_adj.coalesce()
current_indices = current_adj_coo.indices().cpu().numpy()
current_set = set()
for i in range(current_indices.shape[1]):
    src, dst = int(current_indices[0, i]), int(current_indices[1, i])
    current_set.add((src, dst))

actual_new_edges = current_set - core_set
print(f"Actual new edges (original): {len(actual_new_edges)}")

# Check: Why does normalized delta have so many edges?
print("\n--- Why normalized delta is huge ---")
print("The problem: Normalization D^(-1/2) * A * D^(-1/2) depends on degrees.")
print("When nodes are added:")
print("1. New nodes get added with their edges")
print("2. Existing nodes' degrees may change")
print("3. All normalized edge values change!")
print("4. So delta = normalized(current) - normalized(initial) contains")
print("   ALL edges whose normalized values changed, not just new edges!")

# Verify: Check how many edges have different normalized values
print("\n--- Checking edge value changes ---")
core_adj_norm_coo = core_adj_norm.coalesce()
current_adj_norm_coo = current_adj_norm.coalesce()

# Expand core to match current size
core_adj_expanded = SparseOperations.expand_sparse_tensor(
    core_adj_norm,
    (core_adj_norm.size(0), core_adj_norm.size(0)),
    (current_adj_norm.size(0), current_adj_norm.size(0))
)

core_expanded_coo = core_adj_expanded.coalesce()
core_expanded_indices = core_expanded_coo.indices().cpu().numpy()
core_expanded_values = core_expanded_coo.values().cpu().numpy()

current_indices = current_adj_norm_coo.indices().cpu().numpy()
current_values = current_adj_norm_coo.values().cpu().numpy()

# Create dictionaries for quick lookup
core_dict = {}
for i in range(core_expanded_indices.shape[1]):
    src, dst = int(core_expanded_indices[0, i]), int(core_expanded_indices[1, i])
    core_dict[(src, dst)] = float(core_expanded_values[i])

current_dict = {}
for i in range(current_indices.shape[1]):
    src, dst = int(current_indices[0, i]), int(current_indices[1, i])
    current_dict[(src, dst)] = float(current_values[i])

# Count edges with different values
edges_with_changed_values = 0
for (src, dst), val in current_dict.items():
    if (src, dst) in core_dict:
        if abs(val - core_dict[(src, dst)]) > 1e-6:
            edges_with_changed_values += 1
    else:
        # New edge
        edges_with_changed_values += 1

print(f"Edges with changed normalized values: {edges_with_changed_values}")
print(f"Delta adj nnz: {delta_adj_normalized._nnz()}")
print(f"Match: {edges_with_changed_values == delta_adj_normalized._nnz()}")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("The delta contains ALL edges whose normalized values changed,")
print("not just the new edges. This is because normalization depends")
print("on degrees, which change when new nodes are added.")
print("\nThis is actually CORRECT mathematically, but it means the delta")
print("is much larger than just the new edges - it includes all edges")
print("whose normalized values changed due to degree changes.")


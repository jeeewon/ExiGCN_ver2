"""
Debug script to check mask/label mapping issues.
Updated to match run_incremental.py logic.
"""

import torch
from data.download import DatasetLoader
from data.graph_updater import GraphUpdater
from data.preprocessor import DataPreprocessor

# Load dataset
loader = DatasetLoader(root='./data')
adj, features, labels, train_mask, val_mask, test_mask = loader.load_cora_full()

print("="*70)
print("STEP 1: Original Dataset")
print("="*70)
print(f"Total nodes: {adj.size(0)}")
print(f"Train mask True: {train_mask.sum().item()} (indices: {train_mask.nonzero().flatten()[:10].tolist()})")
print(f"Val mask True: {val_mask.sum().item()}")
print(f"Test mask True: {test_mask.sum().item()}")

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

print("\n" + "="*70)
print("STEP 2: Core Graph (90%)")
print("="*70)
print(f"Core nodes: {core_adj.size(0)}")
print(f"Node mapping type: {type(node_mapping)}")
print(f"Node mapping length: {len(node_mapping)}")
print(f"Node mapping sample (first 5): {list(node_mapping.items())[:5]}")

# Check: What are the original indices in core graph?
core_original_indices = list(node_mapping.keys())
print(f"Core graph contains original nodes: {core_original_indices[:10]}")

# Current (WRONG) approach: simple slicing
wrong_core_train_mask = train_mask[:core_adj.size(0)]
print(f"\n[WRONG] train_mask[:core_size] True count: {wrong_core_train_mask.sum().item()}")
print(f"  This assumes core nodes are 0 to {core_adj.size(0)-1}")

# Correct approach (NEW - matches run_incremental.py)
core_size = core_adj.size(0)
correct_core_train_mask = torch.zeros(core_size, dtype=torch.bool)
for old_idx, new_idx in node_mapping.items():
    correct_core_train_mask[new_idx] = train_mask[old_idx]

print(f"\n[CORRECT] Mapped train_mask True count: {correct_core_train_mask.sum().item()}")

# Compare
print(f"\nDifference: {wrong_core_train_mask.sum().item()} vs {correct_core_train_mask.sum().item()}")
print(f"Are they equal? {torch.equal(wrong_core_train_mask, correct_core_train_mask)}")

# Now check bucket A
print("\n" + "="*70)
print("STEP 3: Adding Bucket A")
print("="*70)

bucket_A_nodes = buckets['A']
print(f"Bucket A nodes (original indices): {bucket_A_nodes[:10]}")

# Add bucket
current_adj, current_features, current_labels = updater.add_bucket_to_graph(
    core_adj, core_features, core_labels, bucket_A_nodes
)

current_size = current_adj.size(0)
print(f"Updated graph size: {current_size}")

# Current (WRONG) approach: simple slicing
wrong_current_train_mask = train_mask[:current_size]
print(f"\n[WRONG] train_mask[:current_size] True count: {wrong_current_train_mask.sum().item()}")
print(f"  This assumes nodes are 0 to {current_size-1}")

# Correct approach (NEW - matches run_incremental.py)
bucket_names = ['A', 'B', 'C', 'D', 'E']
stage_idx = 0  # Stage A is index 0

current_node_to_original = {}

# Add core nodes (already mapped)
for old_idx, new_idx in node_mapping.items():
    current_node_to_original[new_idx] = old_idx

# Add all buckets up to and including current stage
offset = core_size
for i, prev_bucket_name in enumerate(bucket_names):
    if i <= stage_idx:  # Include current stage (A)
        for j, old_idx in enumerate(buckets[prev_bucket_name]):
            new_idx = offset + j
            current_node_to_original[new_idx] = old_idx
        offset += len(buckets[prev_bucket_name])

print(f"\nCurrent graph node mapping:")
print(f"  Total mapped nodes: {len(current_node_to_original)}")
print(f"  Expected nodes: {current_size}")
print(f"  Match: {len(current_node_to_original) == current_size}")
print(f"  First 5: {dict(list(current_node_to_original.items())[:5])}")
print(f"  Last 5: {dict(list(current_node_to_original.items())[-5:])}")

# Build correct mask
correct_current_train_mask = torch.zeros(current_size, dtype=torch.bool)
for new_idx, old_idx in current_node_to_original.items():
    correct_current_train_mask[new_idx] = train_mask[old_idx]

print(f"\n[CORRECT] Mapped train_mask True count: {correct_current_train_mask.sum().item()}")

# Compare
print(f"\nDifference: {wrong_current_train_mask.sum().item()} vs {correct_current_train_mask.sum().item()}")

# Now test Stage B
print("\n" + "="*70)
print("STEP 4: Adding Bucket B")
print("="*70)

bucket_B_nodes = buckets['B']
current_adj, current_features, current_labels = updater.add_bucket_to_graph(
    current_adj, current_features, current_labels, bucket_B_nodes
)

current_size = current_adj.size(0)
print(f"Updated graph size (after B): {current_size}")

# Wrong approach
wrong_B_train_mask = train_mask[:current_size]
print(f"\n[WRONG] train_mask[:current_size] True count: {wrong_B_train_mask.sum().item()}")

# Correct approach (Stage B, stage_idx = 1)
stage_idx = 1  # Stage B

current_node_to_original = {}
for old_idx, new_idx in node_mapping.items():
    current_node_to_original[new_idx] = old_idx

offset = core_size
for i, prev_bucket_name in enumerate(bucket_names):
    if i <= stage_idx:  # Include A and B
        for j, old_idx in enumerate(buckets[prev_bucket_name]):
            new_idx = offset + j
            current_node_to_original[new_idx] = old_idx
        offset += len(buckets[prev_bucket_name])

print(f"\nStage B node mapping:")
print(f"  Total mapped nodes: {len(current_node_to_original)}")
print(f"  Expected nodes: {current_size}")
print(f"  Match: {len(current_node_to_original) == current_size}")

correct_B_train_mask = torch.zeros(current_size, dtype=torch.bool)
for new_idx, old_idx in current_node_to_original.items():
    correct_B_train_mask[new_idx] = train_mask[old_idx]

print(f"\n[CORRECT] Mapped train_mask True count: {correct_B_train_mask.sum().item()}")
print(f"\nDifference: {wrong_B_train_mask.sum().item()} vs {correct_B_train_mask.sum().item()}")

# Check label alignment
print("\n" + "="*70)
print("STEP 5: Label Alignment Check")
print("="*70)

# For core graph
print("\nCore graph (first 5):")
for old_idx, new_idx in list(node_mapping.items())[:5]:
    core_label = core_labels[new_idx]
    original_label = labels[old_idx]
    print(f"  new_idx={new_idx} (old_idx={old_idx}): core={core_label.item()}, orig={original_label.item()}, match={core_label==original_label}")

# For stage B graph
print("\nStage B graph (last 5):")
for new_idx, old_idx in list(current_node_to_original.items())[-5:]:
    current_label = current_labels[new_idx]
    original_label = labels[old_idx]
    match = current_label.item() == original_label.item()
    print(f"  new_idx={new_idx} (old_idx={old_idx}): current={current_label.item()}, orig={original_label.item()}, match={match}")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("✅ If all mapping counts match, the fix is CORRECT!")
print("❌ If mapping counts don't match, there's still a bug!")
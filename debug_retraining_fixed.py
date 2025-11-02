"""
Debug script to check ExiGCN retraining forward pass.
Fixed version with proper error handling.
"""

import torch
import torch.nn.functional as F
from data.download import DatasetLoader
from data.graph_updater import GraphUpdater
from data.preprocessor import DataPreprocessor
from models.exigcn import ExiGCN
from train.trainer_exi import ExiGCNTrainer

# Load dataset
loader = DatasetLoader(root='./data')
adj, features, labels, train_mask, val_mask, test_mask = loader.load_cora_full()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

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
core_labels = core_labels.to(device)

# Create masks for core graph
core_size = core_adj.size(0)
core_train_mask = torch.zeros(core_size, dtype=torch.bool)
core_val_mask = torch.zeros(core_size, dtype=torch.bool)

for old_idx, new_idx in node_mapping.items():
    core_train_mask[new_idx] = train_mask[old_idx]
    core_val_mask[new_idx] = val_mask[old_idx]

core_train_mask = core_train_mask.to(device)
core_val_mask = core_val_mask.to(device)

# Create model
num_features = features.size(1)
num_classes = labels.max().item() + 1

model = ExiGCN(
    num_features=num_features,
    hidden_dim=128,
    num_classes=num_classes,
    num_layers=2,
    dropout=0.5
).to(device)

print("="*70)
print("STEP 1: Initial Training (90%)")
print("="*70)

trainer = ExiGCNTrainer(
    model=model,
    device=device,
    learning_rate=0.01,
    weight_decay=0.0005,
    epochs=10,
    verbose=False
)

# Initial training
trainer.train_initial(
    core_adj_norm, core_features_norm, core_labels,
    core_train_mask, core_val_mask, None
)

print(f"Initial training complete")
print(f"  Cached Z: {len(model.cached_Z)} layers")
print(f"  Cached H: {len(model.cached_H)} layers")

# Get initial accuracy
model.eval()
with torch.no_grad():
    initial_logits = model.forward_initial(core_adj_norm, core_features_norm, cache=False)
    initial_val_acc = (initial_logits[core_val_mask].argmax(dim=1) == core_labels[core_val_mask]).float().mean().item()
print(f"Initial Val Acc: {initial_val_acc:.4f}")

# Store initial state
initial_state = {k: v.clone() for k, v in model.state_dict().items()}

print("\n" + "="*70)
print("STEP 2: Add Bucket A (90% -> 92%)")
print("="*70)

# Add bucket A
current_adj, current_features, current_labels = updater.add_bucket_to_graph(
    core_adj, core_features, core_labels, buckets['A']
)

# Preprocess
current_adj_norm, current_features_norm = preprocessor.preprocess(
    current_adj, current_features, device
)
current_labels = current_labels.to(device)

# Create masks
current_size = current_adj.size(0)
current_node_to_original = {}
for old_idx, new_idx in node_mapping.items():
    current_node_to_original[new_idx] = old_idx
for i, old_idx in enumerate(buckets['A']):
    new_idx = core_size + i
    current_node_to_original[new_idx] = old_idx

current_train_mask = torch.zeros(current_size, dtype=torch.bool)
current_val_mask = torch.zeros(current_size, dtype=torch.bool)

for new_idx, old_idx in current_node_to_original.items():
    current_train_mask[new_idx] = train_mask[old_idx]
    current_val_mask[new_idx] = val_mask[old_idx]

current_train_mask = current_train_mask.to(device)
current_val_mask = current_val_mask.to(device)

print(f"Updated graph size: {current_size}")
print(f"Added nodes: {current_size - core_size}")

# Compute deltas
delta_adj, delta_features = trainer._compute_deltas(
    current_adj_norm, current_features_norm
)

print(f"Delta adj nnz: {delta_adj._nnz()}")
print(f"Delta features non-zero: {(delta_features != 0).sum().item()}")

print("\n" + "="*70)
print("STEP 3: Compare Forward Passes")
print("="*70)

# Method 1: Full forward (ground truth)
print("\n--- Full Forward (Ground Truth) ---")
model_full = ExiGCN(
    num_features=num_features,
    hidden_dim=128,
    num_classes=num_classes,
    num_layers=2,
    dropout=0.5
).to(device)
model_full.load_state_dict(initial_state)
model_full.eval()

with torch.no_grad():
    full_logits = model_full.forward_initial(current_adj_norm, current_features_norm, cache=False)
    full_val_acc = (full_logits[current_val_mask].argmax(dim=1) == current_labels[current_val_mask]).float().mean().item()

print(f"Val Acc: {full_val_acc:.4f}")

# Method 2: Retraining forward
print("\n--- Retraining Forward (ExiGCN) ---")
model.load_state_dict(initial_state)
model.prepare_for_retraining()
model.eval()

with torch.no_grad():
    retraining_logits = model.forward_retraining(
        current_adj_norm, current_features_norm, delta_adj, delta_features
    )
    retraining_val_acc = (retraining_logits[current_val_mask].argmax(dim=1) == current_labels[current_val_mask]).float().mean().item()

print(f"Val Acc: {retraining_val_acc:.4f}")

# Compare
print("\n--- Comparison ---")
logit_diff_mean = torch.abs(full_logits - retraining_logits).mean().item()
logit_diff_max = torch.abs(full_logits - retraining_logits).max().item()

print(f"Mean difference: {logit_diff_mean:.6f}")
print(f"Max difference: {logit_diff_max:.6f}")

# Check old vs new nodes
added_node_start = core_size
diff_old = torch.abs(full_logits[:added_node_start] - retraining_logits[:added_node_start]).mean().item()
diff_new = torch.abs(full_logits[added_node_start:] - retraining_logits[added_node_start:]).mean().item()

print(f"\nOld nodes (0-{core_size-1}): {diff_old:.6f}")
print(f"New nodes ({core_size}-{current_size-1}): {diff_new:.6f}")

# Check if close
are_close = torch.allclose(full_logits, retraining_logits, atol=1e-4)

if are_close:
    print("\n✅ Forward pass is CORRECT!")
else:
    print("\n❌ Forward pass is WRONG!")
    if diff_new > 1.0:
        print(f"   New nodes have large error ({diff_new:.2f})")
        print("   Likely cause: cached_F or cached_B not expanded properly")

print("\n" + "="*70)
print("STEP 4: Test Training")
print("="*70)

if are_close:
    print("\n✅ Forward pass correct. Testing training...")
    
    model.load_state_dict(initial_state)
    model.prepare_for_retraining()
    
    delta_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(delta_params, lr=0.01, weight_decay=0.0005)
    
    print("\nTraining for 5 epochs:")
    for epoch in range(5):
        model.train()
        optimizer.zero_grad()
        logits = model.forward_retraining(
            current_adj_norm, current_features_norm, delta_adj, delta_features
        )
        loss = F.cross_entropy(logits[current_train_mask], current_labels[current_train_mask])
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 1 == 0:
            model.eval()
            with torch.no_grad():
                val_logits = model.forward_retraining(
                    current_adj_norm, current_features_norm, delta_adj, delta_features
                )
                val_acc = (val_logits[current_val_mask].argmax(dim=1) == current_labels[current_val_mask]).float().mean().item()
            print(f"  Epoch {epoch+1}: Loss={loss.item():.4f}, Val Acc={val_acc:.4f}")
    
    print("\n✅ Training works!")
else:
    print("\n❌ Skipping training test (forward pass broken)")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Forward pass correct: {are_close}")
print(f"Old nodes diff: {diff_old:.6f}")
print(f"New nodes diff: {diff_new:.6f}")

if are_close:
    print("\n✅ All checks passed! Ready for full experiment.")
else:
    print("\n❌ Fix needed:")
    if diff_new > 1.0:
        print("  - Check cached_F expansion in ExiGCNLayer")
        print("  - Check cached_B expansion in ExiGCNLayer")
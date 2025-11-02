"""
Debug script to check ExiGCN retraining forward pass.
"""

import torch
import torch.nn.functional as F_functional
from data.download import DatasetLoader
from data.graph_updater import GraphUpdater
from data.preprocessor import DataPreprocessor
from models.exigcn import ExiGCN
from train.trainer_exi import ExiGCNTrainer
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

# Preprocess core graph
core_adj_norm, core_features_norm = preprocessor.preprocess(
    core_adj, core_features, device
)
core_labels = core_labels.to(device)

# Create masks
core_train_mask = torch.zeros(len(node_mapping), dtype=torch.bool)
core_val_mask = torch.zeros(len(node_mapping), dtype=torch.bool)
core_test_mask = torch.zeros(len(node_mapping), dtype=torch.bool)
for old_idx, new_idx in node_mapping.items():
    core_train_mask[new_idx] = train_mask[old_idx]
    core_val_mask[new_idx] = val_mask[old_idx]
    core_test_mask[new_idx] = test_mask[old_idx]
core_train_mask = core_train_mask.to(device)
core_val_mask = core_val_mask.to(device)
core_test_mask = core_test_mask.to(device)

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

trainer = ExiGCNTrainer(
    model=model,
    device=device,
    learning_rate=0.01,
    weight_decay=0.0005,
    epochs=50,  # Short test
    verbose=False
)

print("="*70)
print("STEP 1: Initial Training (90%)")
print("="*70)

# Initial training
trainer.train_initial(
    core_adj_norm, core_features_norm, core_labels,
    core_train_mask, core_val_mask, core_test_mask
)

print(f"Initial training complete")
print(f"Cached Z layers: {len(model.cached_Z)}")
print(f"Cached H layers: {len(model.cached_H)}")

# Store initial model state
initial_state = {k: v.clone() for k, v in model.state_dict().items()}

# Get initial predictions
model.eval()
with torch.no_grad():
    initial_logits = model.forward_initial(core_adj_norm, core_features_norm, cache=False)
    initial_pred = initial_logits.argmax(dim=1)
    initial_val_acc = (initial_pred[core_val_mask] == core_labels[core_val_mask]).float().mean().item()
print(f"Initial Val Acc: {initial_val_acc:.4f}")

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

# Create masks for current graph
current_node_to_original = {}
for old_idx, new_idx in node_mapping.items():
    current_node_to_original[new_idx] = old_idx
for i, old_idx in enumerate(buckets['A']):
    new_idx = core_adj.size(0) + i
    current_node_to_original[new_idx] = old_idx

current_train_mask = torch.zeros(current_adj.size(0), dtype=torch.bool)
current_val_mask = torch.zeros(current_adj.size(0), dtype=torch.bool)
current_test_mask = torch.zeros(current_adj.size(0), dtype=torch.bool)
for new_idx, old_idx in current_node_to_original.items():
    current_train_mask[new_idx] = train_mask[old_idx]
    current_val_mask[new_idx] = val_mask[old_idx]
    current_test_mask[new_idx] = test_mask[old_idx]
current_train_mask = current_train_mask.to(device)
current_val_mask = current_val_mask.to(device)
current_test_mask = current_test_mask.to(device)

print(f"Updated graph size: {current_adj_norm.size(0)}")
print(f"Added nodes: {current_adj_norm.size(0) - core_adj_norm.size(0)}")

# Compute deltas
trainer.initial_adj = core_adj_norm.clone()
trainer.initial_features = core_features_norm.clone()
trainer.initial_num_nodes = core_adj_norm.size(0)

delta_adj, delta_features = trainer._compute_deltas(
    current_adj_norm, current_features_norm
)

print(f"\nDelta adj nnz: {delta_adj._nnz()}")
print(f"Delta features non-zero: {(delta_features != 0).sum().item()}")

print("\n" + "="*70)
print("STEP 3: Check Forward Pass Issues")
print("="*70)

# Prepare for retraining
model.prepare_for_retraining()

# Check: Can we reconstruct the correct output?
print("\n--- Testing Forward Pass Reconstruction ---")

# Method 1: Full forward (ground truth)
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
print(f"Full forward (ground truth) Val Acc: {full_val_acc:.4f}")

# Method 2: Retraining forward (what ExiGCN does)
model.eval()
with torch.no_grad():
    retraining_logits = model.forward_retraining(
        current_adj_norm, current_features_norm, delta_adj, delta_features
    )
    retraining_val_acc = (retraining_logits[current_val_mask].argmax(dim=1) == current_labels[current_val_mask]).float().mean().item()
print(f"Retraining forward (with deltas) Val Acc: {retraining_val_acc:.4f}")

# Check difference
logit_diff = torch.abs(full_logits - retraining_logits).mean().item()
print(f"\nLogit difference (abs mean): {logit_diff:.4f}")

# Check if outputs are close
are_close = torch.allclose(full_logits, retraining_logits, atol=0.1)
print(f"Outputs are close (atol=0.1): {are_close}")

# Check: What happens with training?
print("\n--- Testing Training Step ---")

# Reset model
model.load_state_dict(initial_state)
model.prepare_for_retraining()

# Create optimizer for deltas
delta_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(delta_params, lr=0.01, weight_decay=0.0005)

print("\nBefore training:")
model.eval()
with torch.no_grad():
    logits_before = model.forward_retraining(
        current_adj_norm, current_features_norm, delta_adj, delta_features
    )
    acc_before = (logits_before[current_val_mask].argmax(dim=1) == current_labels[current_val_mask]).float().mean().item()
    loss_before = torch.nn.functional.cross_entropy(logits_before[current_train_mask], current_labels[current_train_mask]).item()
print(f"  Val Acc: {acc_before:.4f}")
print(f"  Train Loss: {loss_before:.4f}")

# One training step
model.train()
optimizer.zero_grad()
logits = model.forward_retraining(
    current_adj_norm, current_features_norm, delta_adj, delta_features
)
loss = torch.nn.functional.cross_entropy(logits[current_train_mask], current_labels[current_train_mask])
loss.backward()

# Check gradients
print("\nGradient check:")
for name, param in model.named_parameters():
    if param.requires_grad and param.grad is not None:
        grad_norm = param.grad.norm().item()
        print(f"  {name}: grad_norm={grad_norm:.6f}")

optimizer.step()

print("\nAfter 1 training step:")
model.eval()
with torch.no_grad():
    logits_after = model.forward_retraining(
        current_adj_norm, current_features_norm, delta_adj, delta_features
    )
    acc_after = (logits_after[current_val_mask].argmax(dim=1) == current_labels[current_val_mask]).float().mean().item()
    loss_after = torch.nn.functional.cross_entropy(logits_after[current_train_mask], current_labels[current_train_mask]).item()
print(f"  Val Acc: {acc_after:.4f}")
print(f"  Train Loss: {loss_after:.4f}")
print(f"  Loss change: {loss_after - loss_before:.6f}")

# Check: What if we train for a few epochs?
print("\n--- Training for 10 epochs ---")
for epoch in range(10):
    model.train()
    optimizer.zero_grad()
    logits = model.forward_retraining(
        current_adj_norm, current_features_norm, delta_adj, delta_features
    )
    loss = torch.nn.functional.cross_entropy(logits[current_train_mask], current_labels[current_train_mask])
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 2 == 0:
        model.eval()
        with torch.no_grad():
            val_logits = model.forward_retraining(
                current_adj_norm, current_features_norm, delta_adj, delta_features
            )
            val_acc = (val_logits[current_val_mask].argmax(dim=1) == current_labels[current_val_mask]).float().mean().item()
            val_loss = torch.nn.functional.cross_entropy(val_logits[current_val_mask], current_labels[current_val_mask]).item()
        print(f"Epoch {epoch+1:2d}: Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

print("\n" + "="*70)
print("STEP 4: Detailed Forward Pass Analysis")
print("="*70)

# At initialization (delta weights = 0)
model.load_state_dict(initial_state)
model.prepare_for_retraining()

added_node_start = core_adj_norm.size(0)
added_nodes_mask = torch.arange(current_adj_norm.size(0), device=device) >= added_node_start

print(f"\nAdded node range: {added_node_start} to {current_adj_norm.size(0)-1}")
print(f"Number of added nodes: {added_nodes_mask.sum().item()}")

# Check layer-by-layer reconstruction
print("\n--- Layer-by-Layer Analysis ---")

# Get intermediate values from full forward
model_full.eval()
with torch.no_grad():
    # Manually compute layer outputs to inspect
    h_full = current_features_norm
    h_full_intermediate = []
    z_full_intermediate = []
    
    for i, layer in enumerate(model_full.layers[:-1]):
        z_full = layer.forward_initial(current_adj_norm, h_full)
        z_full_intermediate.append(z_full.clone())
        h_full = model_full.activation(z_full)
        h_full = F_functional.dropout(h_full, p=0.5, training=False)
        h_full_intermediate.append(h_full.clone())
    
    # Last layer
    h_full_last = h_full_intermediate[-1] if len(h_full_intermediate) > 0 else current_features_norm
    z_full_last = model_full.layers[-1].forward_initial(current_adj_norm, h_full_last)

# Check retraining forward layer by layer
model.eval()
with torch.no_grad():
    h_retraining = current_features_norm
    delta_h = delta_features  # For first layer, use delta_features
    
    for i, layer in enumerate(model.layers[:-1]):
        cached_Z = model.cached_Z[i] if i < len(model.cached_Z) else None
        cached_F = None  # Force recompute
        cached_B = None  # Force recompute
        
        # Get delta_h
        if i == 0:
            # First layer: delta_h is delta_features
            delta_h = delta_features
        elif i < len(model.cached_H):
            # Later layers: compare with previous layer's cached output
            cached_h = model.cached_H[i-1]  # Use previous layer's cached H
            if cached_h.size(0) < h_retraining.size(0):
                padding = torch.zeros(h_retraining.size(0) - cached_h.size(0),
                                    cached_h.size(1),
                                    device=cached_h.device)
                cached_h = torch.cat([cached_h, padding], dim=0)
            delta_h = h_retraining - cached_h
        else:
            delta_h = torch.zeros_like(h_retraining)
        
        # Forward retraining
        z_retraining, fixed_term, B = layer.forward_retraining(
            current_adj_norm, h_retraining, delta_adj, delta_h,
            cached_Z, cached_F, cached_B
        )
        
        # Update h_retraining for next layer
        h_retraining = model.activation(z_retraining)
        h_retraining = F_functional.dropout(h_retraining, p=0.5, training=False)
        
        # Compare with full forward
        # h_full_intermediate stores h after each layer
        # For layer i, we should compare with h_full_intermediate[i*2+1] 
        # (since we store both z and h, but actually we only store h)
        if i < len(h_full_intermediate):
            h_full = h_full_intermediate[i]
            
            # Compare
            diff_h = torch.abs(h_full - h_retraining).mean().item()
            diff_h_added = torch.abs(h_full[added_nodes_mask] - h_retraining[added_nodes_mask]).mean().item()
            diff_h_old = torch.abs(h_full[:added_node_start] - h_retraining[:added_node_start]).mean().item()
            
            print(f"\nLayer {i+1} (Hidden):")
            print(f"  Full H mean: {h_full.mean().item():.4f}, std: {h_full.std().item():.4f}")
            print(f"  Retraining H mean: {h_retraining.mean().item():.4f}, std: {h_retraining.std().item():.4f}")
            print(f"  Overall difference: {diff_h:.4f}")
            print(f"  Difference on added nodes: {diff_h_added:.4f}")
            print(f"  Difference on old nodes: {diff_h_old:.4f}")
            
            # Check cached_Z, fixed_term, B sizes
            if cached_Z is not None:
                print(f"  cached_Z shape: {cached_Z.shape}")
            print(f"  fixed_term shape: {fixed_term.shape}")
            print(f"  B shape: {B.shape}")
            print(f"  z_retraining shape: {z_retraining.shape}")
            
            # Check if cached_Z padding is the issue
            if cached_Z is not None:
                cached_Z_expanded = torch.cat([
                    cached_Z,
                    torch.zeros(z_retraining.size(0) - cached_Z.size(0), cached_Z.size(1), device=device)
                ], dim=0)
                print(f"  cached_Z_expanded[added_nodes] mean: {cached_Z_expanded[added_nodes_mask].mean().item():.4f}")
                print(f"  B_delta_W[added_nodes] mean (before W): {B[added_nodes_mask].mean().item():.4f}")
                print(f"  fixed_term[added_nodes] mean: {fixed_term[added_nodes_mask].mean().item():.4f}")

# Check final layer
print("\n--- Final Layer Analysis ---")
model.eval()
with torch.no_grad():
    # Recompute h_retraining for last layer (from previous loop)
    h_retraining_for_last = current_features_norm
    delta_h_for_last = delta_features
    
    for i in range(len(model.layers) - 1):
        cached_Z_i = model.cached_Z[i] if i < len(model.cached_Z) else None
        cached_F_i = None
        cached_B_i = None
        
        # Get delta_h
        if i == 0:
            delta_h_for_last = delta_features
        elif i-1 < len(model.cached_H):
            cached_h_i = model.cached_H[i-1]  # Previous layer's cached H
            if cached_h_i.size(0) < h_retraining_for_last.size(0):
                padding = torch.zeros(h_retraining_for_last.size(0) - cached_h_i.size(0),
                                    cached_h_i.size(1),
                                    device=cached_h_i.device)
                cached_h_i = torch.cat([cached_h_i, padding], dim=0)
            delta_h_for_last = h_retraining_for_last - cached_h_i
        else:
            delta_h_for_last = torch.zeros_like(h_retraining_for_last)
        
        z_i, _, _ = model.layers[i].forward_retraining(
            current_adj_norm, h_retraining_for_last, delta_adj, delta_h_for_last,
            cached_Z_i, cached_F_i, cached_B_i
        )
        h_retraining_for_last = model.activation(z_i)
        h_retraining_for_last = F_functional.dropout(h_retraining_for_last, p=0.5, training=False)
    
    # Compute delta_h for last layer
    if len(model.cached_H) >= model.num_layers - 1:
        cached_h_last = model.cached_H[-1]  # Last hidden layer's output
        if cached_h_last.size(0) < h_retraining_for_last.size(0):
            padding = torch.zeros(h_retraining_for_last.size(0) - cached_h_last.size(0),
                                cached_h_last.size(1),
                                device=cached_h_last.device)
            cached_h_last = torch.cat([cached_h_last, padding], dim=0)
        delta_h_last = h_retraining_for_last - cached_h_last
    else:
        delta_h_last = torch.zeros_like(h_retraining_for_last)
    
    cached_Z_last = model.cached_Z[-1] if len(model.cached_Z) >= model.num_layers else None
    
    z_retraining_last, fixed_term_last, B_last = model.layers[-1].forward_retraining(
        current_adj_norm, h_retraining_for_last, delta_adj, delta_h_last,
        cached_Z_last, None, None
    )
    
    # Compare with full forward
    diff_final = torch.abs(z_full_last - z_retraining_last).mean().item()
    diff_final_added = torch.abs(z_full_last[added_nodes_mask] - z_retraining_last[added_nodes_mask]).mean().item()
    diff_final_old = torch.abs(z_full_last[:added_node_start] - z_retraining_last[:added_node_start]).mean().item()
    
    print(f"Final layer Z comparison:")
    print(f"  Overall difference: {diff_final:.4f}")
    print(f"  Difference on added nodes: {diff_final_added:.4f}")
    print(f"  Difference on old nodes: {diff_final_old:.4f}")
    
    # Detailed breakdown
    if cached_Z_last is not None:
        cached_Z_last_expanded = torch.cat([
            cached_Z_last,
            torch.zeros(z_retraining_last.size(0) - cached_Z_last.size(0), cached_Z_last.size(1), device=device)
        ], dim=0)
        
        # Check what B_delta_W should be (with ΔW = 0, it should be 0)
        # But B itself should give us ÂH for added nodes
        B_delta_W_last = torch.mm(B_last, model.layers[-1].delta_W)  # ΔW = 0 initially
        
        print(f"\n  Component analysis (added nodes):")
        print(f"    cached_Z_last_expanded mean: {cached_Z_last_expanded[added_nodes_mask].mean().item():.6f}")
        print(f"    fixed_term_last mean: {fixed_term_last[added_nodes_mask].mean().item():.6f}")
        print(f"    B_delta_W_last mean: {B_delta_W_last[added_nodes_mask].mean().item():.6f}")
        print(f"    Sum (Z + F + BΔW) mean: {(cached_Z_last_expanded + fixed_term_last + B_delta_W_last)[added_nodes_mask].mean().item():.6f}")
        print(f"    Full forward Z mean: {z_full_last[added_nodes_mask].mean().item():.6f}")
        
        # Check what B gives us (should be ÂH for added nodes when ΔW=0)
        # But wait, B = ÂH + Â∆H + ∆ÂH + ∆Â∆H
        # For added nodes, what should ÂH be?
        print(f"\n  B matrix analysis (added nodes):")
        print(f"    B_last[added_nodes] mean: {B_last[added_nodes_mask].mean().item():.6f}")
        
        # Check: What are the components of B for added nodes?
        # B = ÂH + Â∆H + ∆ÂH + ∆Â∆H
        term1_B = SparseOperations.sparse_dense_mm(current_adj_norm, h_retraining_for_last)  # ÂH
        term2_B = SparseOperations.sparse_dense_mm(current_adj_norm, delta_h_last)  # Â∆H
        term3_B = SparseOperations.sparse_dense_mm(delta_adj, h_retraining_for_last)  # ∆ÂH
        term4_B = SparseOperations.sparse_dense_mm(delta_adj, delta_h_last)  # ∆Â∆H
        
        print(f"\n  B component analysis (added nodes):")
        print(f"    ÂH mean: {term1_B[added_nodes_mask].mean().item():.6f}")
        print(f"    Â∆H mean: {term2_B[added_nodes_mask].mean().item():.6f}")
        print(f"    ∆ÂH mean: {term3_B[added_nodes_mask].mean().item():.6f}")
        print(f"    ∆Â∆H mean: {term4_B[added_nodes_mask].mean().item():.6f}")
        print(f"    B sum mean: {(term1_B + term2_B + term3_B + term4_B)[added_nodes_mask].mean().item():.6f}")
        print(f"    B_last mean: {B_last[added_nodes_mask].mean().item():.6f}")
        
        # Check: What should ÂH*W be for added nodes?
        # Compute ÂH*W directly for comparison
        AH_support = term1_B  # ÂH (18210, 128)
        AH_support_W = torch.mm(AH_support, model.layers[-1].W)  # ÂH * W (18210, 70)
        
        # Also compute full Â_new * H_new * W for added nodes
        support_full = torch.mm(h_retraining_for_last, model.layers[-1].W)
        AH_full = SparseOperations.sparse_dense_mm(current_adj_norm, support_full)
        print(f"\n  Direct computation (added nodes):")
        print(f"    ÂH * W mean: {AH_support_W[added_nodes_mask].mean().item():.6f}")
        print(f"    Â_new * H_new * W mean: {AH_full[added_nodes_mask].mean().item():.6f}")
        
        # Check the issue: Why is there a difference?
        # Sum should equal Full forward Z when ΔW=0
        print(f"\n  Why is there a difference?")
        print(f"    Sum (Z+F+BΔW) = {0.042574:.6f}")
        print(f"    Full forward Z = {-0.000487:.6f}")
        print(f"    Difference = {0.042574 - (-0.000487):.6f}")
        print(f"    This suggests cached_Z or fixed_term is wrong!")
        
        # Check: What should cached_Z be for added nodes?
        # For added nodes, there's no cached Z, so it's 0 (padding)
        # But what should it be? It should be Â_old * H_old * W for added nodes
        # But added nodes don't exist in old graph, so cached_Z is correctly 0
        
        # Check: What should fixed_term be?
        # F = (Â∆H + ∆ÂH + ∆Â∆H)W
        # For added nodes, H_old = 0 (they didn't exist), so:
        # ∆H = H_new - H_old = H_new - 0 = H_new
        # F = (Â * H_new + ∆Â * 0 + ∆Â * H_new)W
        #   = (Â * H_new + ∆Â * H_new)W
        #   = (Â + ∆Â) * H_new * W
        #   = Â_new * H_new * W
        # So fixed_term should give us the base term for added nodes!
        
        print(f"\n  Theoretical check:")
        print(f"    For added nodes, fixed_term should approximate Â_new * H_new * W")
        print(f"    fixed_term_last[added_nodes] mean = {fixed_term_last[added_nodes_mask].mean().item():.6f}")
        print(f"    Full forward Z[added_nodes] mean = {z_full_last[added_nodes_mask].mean().item():.6f}")
        print(f"    Difference = {torch.abs(fixed_term_last[added_nodes_mask] - z_full_last[added_nodes_mask]).mean().item():.6f}")

print("\n" + "="*70)
print("STEP 5: Check Delta vs Full Comparison")
print("="*70)

# The key question: Does retraining forward give similar output to full forward?
print("\nComparing outputs at initialization (delta=0):")

# At initialization (delta weights = 0)
model.load_state_dict(initial_state)
model.prepare_for_retraining()

model.eval()
with torch.no_grad():
    retraining_init = model.forward_retraining(
        current_adj_norm, current_features_norm, delta_adj, delta_features
    )
    
    # Compare with full forward
    diff_init = torch.abs(full_logits - retraining_init).mean().item()
    print(f"Difference at initialization (delta=0): {diff_init:.4f}")
    
    # Check just the added nodes
    diff_added = torch.abs(full_logits[added_nodes_mask] - retraining_init[added_nodes_mask]).mean().item()
    print(f"Difference on added nodes: {diff_added:.4f}")
    
    diff_old = torch.abs(full_logits[:added_node_start] - retraining_init[:added_node_start]).mean().item()
    print(f"Difference on old nodes: {diff_old:.4f}")
    
    # Check max difference
    max_diff = torch.abs(full_logits - retraining_init).max().item()
    max_diff_added = torch.abs(full_logits[added_nodes_mask] - retraining_init[added_nodes_mask]).max().item()
    print(f"\nMax difference overall: {max_diff:.4f}")
    print(f"Max difference on added nodes: {max_diff_added:.4f}")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("If forward pass reconstruction is wrong, that's the problem.")
print("If gradients are too small, training won't work.")
print("If loss doesn't decrease, optimizer or learning rate might be wrong.")


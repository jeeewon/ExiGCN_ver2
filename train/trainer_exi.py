"""
ExiGCN trainer with efficient retraining.
"""

import torch
from train.trainer_base import BaseTrainer
from models.exigcn import ExiGCN
from utils.sparse_ops import SparseOperations
from utils.metrics import evaluate_model as compute_metrics, compute_loss
from typing import Dict, Optional


class ExiGCNTrainer(BaseTrainer):
    """
    Trainer for ExiGCN with efficient retraining.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_phase = "initial"  # "initial" or "retraining"
        
        # Store initial graph for delta computation
        self.initial_adj = None
        self.initial_features = None
        self.initial_num_nodes = None
        
        # CRITICAL: Store initial weights W_init for restoration at each stage
        self.initial_weights = None
    
    def train_initial(self,
                     adj: torch.Tensor,
                     features: torch.Tensor,
                     labels: torch.Tensor,
                     train_mask: torch.Tensor,
                     val_mask: torch.Tensor,
                     test_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Initial training on core graph (e.g., 90%).
        
        Args:
            adj: Normalized adjacency of core graph
            features: Node features of core graph
            labels: Node labels
            train_mask: Training mask
            val_mask: Validation mask
            test_mask: Test mask
            
        Returns:
            Final metrics
        """
        self.training_phase = "initial"
        self.model.is_initial_training = True
        
        # Store initial graph for later delta computation
        self.initial_adj = adj.clone()
        self.initial_features = features.clone()
        self.initial_num_nodes = adj.size(0)
        
        if self.verbose:
            print("\n" + "="*70)
            print("INITIAL TRAINING (Core Graph)")
            print("="*70)
            print(f"  Graph size: {adj.size(0)} nodes")
            print(f"  Edges: {adj._nnz()}")
        
        # Standard training with caching
        results = self.train(adj, features, labels, train_mask, val_mask, test_mask)
        
        # CRITICAL: Save initial weights for future stages!
        # Each stage should start from W_init, not W_init + accumulated deltas
        self.initial_weights = []
        for layer in self.model.layers:
            self.initial_weights.append({
                'W': layer.W.data.clone(),
                'bias': layer.bias.data.clone() if layer.bias is not None else None
            })
        
        if self.verbose:
            print("\n✅ Initial training complete")
            print(f"   Cached Z layers: {len(self.model.cached_Z)}")
            print(f"   Cached H layers: {len(self.model.cached_H)}")
            print(f"   Initial weights saved: {len(self.initial_weights)} layers")
        
        return results
    
    def train_retraining(self,
                        adj_updated: torch.Tensor,
                        features_updated: torch.Tensor,
                        labels_updated: torch.Tensor,
                        train_mask: torch.Tensor,
                        val_mask: torch.Tensor,
                        test_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Efficient retraining on updated graph.
        
        Args:
            adj_updated: Normalized adjacency of updated graph
            features_updated: Node features of updated graph
            labels_updated: Node labels (updated)
            train_mask: Training mask
            val_mask: Validation mask
            test_mask: Test mask
            
        Returns:
            Final metrics
        """
        if self.initial_adj is None:
            raise ValueError("Must call train_initial() before train_retraining()")
        
        self.training_phase = "retraining"
        
        if self.verbose:
            print("\n" + "="*70)
            print("EFFICIENT RETRAINING (ExiGCN)")
            print("="*70)
            print(f"  Initial nodes: {self.initial_num_nodes}")
            print(f"  Updated nodes: {adj_updated.size(0)}")
            print(f"  Added nodes: {adj_updated.size(0) - self.initial_num_nodes}")
        
        # Compute deltas
        with self.timer.measure_graph_update():
            delta_adj, delta_features = self._compute_deltas(
                adj_updated, features_updated
            )
        
        if self.verbose:
            print(f"\n  Delta adjacency nnz: {delta_adj._nnz()}")
            print(f"  Delta features non-zero: {(delta_features != 0).sum().item()}")
        
        # PAPER'S METHOD: Reset delta_W to zero
        # W itself should be already reset to W_init by run_incremental.py
        for layer in self.model.layers:
            layer.delta_W.data.zero_()
            if layer.delta_bias is not None:
                layer.delta_bias.data.zero_()
        
        if self.verbose:
            print(f"  ✅ Using fixed baseline (paper's method)")
        
        # Prepare model for retraining
        self.model.prepare_for_retraining()
        
        # Use retraining_lr (can be different from initial training LR)
        retraining_lr = self.retraining_lr
        
        # Create optimizer for delta parameters only
        delta_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(
            delta_params,
            lr=retraining_lr,
            weight_decay=self.weight_decay
        )
        
        if self.verbose:
            print(f"  Retraining LR: {retraining_lr:.6f}")
        
        # Reset trackers
        self.timer.reset()
        self.metrics_tracker = type(self.metrics_tracker)()
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.best_model_state = None
        
        # Store delta for forward pass
        self.delta_adj = delta_adj
        self.delta_features = delta_features
        
        # Training loop
        self.timer.start_training()
        
        epochs_without_improvement = 0
        
        for epoch in range(self.epochs):
            self.timer.start_epoch()
            
            # Train epoch with retraining forward
            train_loss = self._train_epoch_retraining(
                adj_updated, features_updated, labels_updated, train_mask
            )
            
            # Evaluate (uses retraining forward)
            metrics = self._evaluate_retraining(
                adj_updated, features_updated, labels_updated, val_mask, test_mask
            )
            
            # Update tracker
            self.metrics_tracker.update(
                train_loss=train_loss,
                val_loss=metrics['val_loss'],
                val_metrics={
                    'accuracy': metrics['val_acc'],
                    'f1_micro': metrics['val_f1_micro'],
                    'f1_macro': metrics['val_f1_macro']
                }
            )
            
            # Check improvement
            if metrics['val_acc'] > self.best_val_acc:
                self.best_val_acc = metrics['val_acc']
                self.best_epoch = epoch
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            # Print progress
            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{self.epochs} | "
                      f"Loss: {train_loss:.4f} | "
                      f"Val Acc: {metrics['val_acc']:.4f} | "
                      f"Best: {self.best_val_acc:.4f} @ {self.best_epoch+1}")
            
            # Early stopping
            if self.early_stopping and epochs_without_improvement >= self.patience:
                if self.verbose:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                break
            
            self.timer.stop_epoch()
        
        self.timer.stop_training()
        
        # Load best model (contains deltas)
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # Final evaluation BEFORE merge (model still has deltas)
        final_metrics = self._evaluate_retraining(
            adj_updated, features_updated, labels_updated, val_mask, test_mask
        )
        
        # NOW merge deltas into W
        if self.verbose:
            print("\nMerging delta weights...")
        self.model.merge_all_deltas()
        
        # PAPER'S METHOD: Keep initial baseline FIXED at core
        # DO NOT update initial_adj or initial_features
        # They should remain as the original core graph
        
        if self.verbose:
            print("Keeping baseline fixed (paper's method)")
        
        # Recache Z and H for the current graph size
        # Use merged W for cache computation, but baseline stays at core
        self.model.eval()
        with torch.no_grad():
            _ = self.model.forward_initial(adj_updated, features_updated, cache=True)
        
        # Set to initial training mode for next stage
        self.model.is_initial_training = True
        
        if self.verbose:
            print("\n" + "="*70)
            print("RETRAINING COMPLETE")
            print("="*70)
            print(f"Best Epoch: {self.best_epoch + 1}")
            print(f"Best Val Acc: {self.best_val_acc:.4f}")
            if test_mask is not None:
                print(f"Test Acc: {final_metrics['test_acc']:.4f}")
            
            # DEBUG: Print B computation stats
            b_count = sum(getattr(layer, '_b_compute_count', 0) for layer in self.model.layers)
            print(f"\n[DEBUG] B matrix computed {b_count} times (should be ~{self.model.num_layers} for efficiency)")
            
            # Print timing
            self.timer.print_report()
        
        return final_metrics
    
    def _compute_deltas(self,
                       adj_updated: torch.Tensor,
                       features_updated: torch.Tensor) -> tuple:
        """
        Compute delta adjacency and features.
        
        Args:
            adj_updated: Updated normalized adjacency
            features_updated: Updated features
            
        Returns:
            delta_adj, delta_features
        """
        # Compute delta adjacency
        delta_adj = SparseOperations.compute_delta_sparse(
            self.initial_adj,
            adj_updated
        )
        
        # Compute delta features
        if features_updated.size(0) > self.initial_features.size(0):
            # Pad initial features with zeros
            padding = torch.zeros(
                features_updated.size(0) - self.initial_features.size(0),
                self.initial_features.size(1),
                device=self.initial_features.device
            )
            initial_features_padded = torch.cat([self.initial_features, padding], dim=0)
        else:
            initial_features_padded = self.initial_features
        
        delta_features = features_updated - initial_features_padded
        
        return delta_adj, delta_features
    
    def _train_epoch_retraining(self,
                               adj: torch.Tensor,
                               features: torch.Tensor,
                               labels: torch.Tensor,
                               train_mask: torch.Tensor) -> float:
        """Train one epoch using ExiGCN retraining forward."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward with retraining
        with self.timer.measure_forward():
            logits = self.model.forward_retraining(
                adj, features, self.delta_adj, self.delta_features
            )
        
        # Compute loss
        from utils.metrics import compute_loss
        loss = compute_loss(logits, labels, train_mask)
        
        # Backward
        with self.timer.measure_backward():
            loss.backward()
        
        # Optimizer step
        with self.timer.measure_optimizer_step():
            self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def _evaluate_retraining(self,
                            adj: torch.Tensor,
                            features: torch.Tensor,
                            labels: torch.Tensor,
                            val_mask: torch.Tensor,
                            test_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Evaluate using ExiGCN retraining forward."""
        self.model.eval()
        
        from utils.metrics import evaluate_model, compute_loss
        
        with self.timer.measure_evaluation():
            logits = self.model.forward_retraining(
                adj, features, self.delta_adj, self.delta_features
            )
            
            val_metrics = evaluate_model(logits, labels, val_mask)
            
            results = {
                'val_loss': compute_loss(logits, labels, val_mask).item(),
                'val_acc': val_metrics['accuracy'],
                'val_f1_micro': val_metrics['f1_micro'],
                'val_f1_macro': val_metrics['f1_macro']
            }
            
            if test_mask is not None:
                test_metrics = evaluate_model(logits, labels, test_mask)
                results.update({
                    'test_loss': compute_loss(logits, labels, test_mask).item(),
                    'test_acc': test_metrics['accuracy'],
                    'test_f1_micro': test_metrics['f1_micro'],
                    'test_f1_macro': test_metrics['f1_macro']
                })
        
        return results


def test_exigcn_trainer():
    """Test ExiGCN trainer."""
    print("Testing ExiGCNTrainer on Cora-Full...")
    
    from data.download import DatasetLoader
    from data.preprocessor import DataPreprocessor
    from data.graph_updater import GraphUpdater
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load dataset
    loader = DatasetLoader(root='./data')
    adj, features, labels, train_mask, val_mask, test_mask = loader.load_cora_full()
    
    # Create graph updater
    updater = GraphUpdater(
        adj=adj,
        features=features,
        labels=labels,
        initial_ratio=0.9,
        n_buckets=5,
        seed=42
    )
    
    # Create incremental scenario
    print("\nCreating incremental scenario (90% core + 5 buckets)...")
    buckets, core_adj, core_features, core_labels, node_mapping = updater.create_incremental_scenario()
    
    # Preprocess core graph
    preprocessor = DataPreprocessor()
    core_adj_norm, core_features_norm = preprocessor.preprocess(core_adj, core_features, device)
    core_labels = core_labels.to(device)
    
    # Use same masks (simplified for testing)
    core_train_mask = train_mask[:core_adj.size(0)].to(device)
    core_val_mask = val_mask[:core_adj.size(0)].to(device)
    core_test_mask = test_mask[:core_adj.size(0)].to(device)
    
    # Create model
    num_features = features.size(1)
    num_classes = labels.max().item() + 1
    
    model = ExiGCN(
        num_features=num_features,
        hidden_dim=128,
        num_classes=num_classes,
        num_layers=2,
        dropout=0.5
    )
    
    # Create trainer
    trainer = ExiGCNTrainer(
        model=model,
        device=device,
        learning_rate=0.01,
        weight_decay=0.0005,
        epochs=200,
        verbose=True
    )
    
    # Phase 1: Initial training (90%)
    print("\n" + "="*70)
    print("PHASE 1: Initial Training (90%)")
    print("="*70)
    
    initial_results = trainer.train_initial(
        core_adj_norm, core_features_norm, core_labels,
        core_train_mask, core_val_mask, core_test_mask
    )
    
    initial_time = trainer.get_training_time()
    
    print(f"\nInitial Training Results:")
    print(f"  Val Acc: {initial_results['val_acc']:.4f}")
    print(f"  Test Acc: {initial_results['test_acc']:.4f}")
    print(f"  Training Time: {initial_time:.2f}s")
    
    # Phase 2: Add bucket A (90% → 92%)
    print("\n" + "="*70)
    print("PHASE 2: Retraining (90% → 92%)")
    print("="*70)
    
    # Add bucket A
    updated_adj, updated_features, updated_labels = updater.add_bucket_to_graph(
        core_adj, core_features, core_labels, buckets['A']
    )
    
    # Preprocess updated graph
    updated_adj_norm, updated_features_norm = preprocessor.preprocess(
        updated_adj, updated_features, device
    )
    updated_labels = updated_labels.to(device)
    
    # Update masks
    updated_train_mask = train_mask[:updated_adj.size(0)].to(device)
    updated_val_mask = val_mask[:updated_adj.size(0)].to(device)
    updated_test_mask = test_mask[:updated_adj.size(0)].to(device)
    
    # Retraining
    retraining_results = trainer.train_retraining(
        updated_adj_norm, updated_features_norm, updated_labels,
        updated_train_mask, updated_val_mask, updated_test_mask
    )
    
    retraining_time = trainer.get_training_time()
    
    print(f"\nRetraining Results:")
    print(f"  Val Acc: {retraining_results['val_acc']:.4f}")
    print(f"  Test Acc: {retraining_results['test_acc']:.4f}")
    print(f"  Training Time: {retraining_time:.2f}s")
    
    print("\n✅ ExiGCNTrainer test passed!")
    
    return trainer, initial_results, retraining_results


if __name__ == "__main__":
    test_exigcn_trainer()
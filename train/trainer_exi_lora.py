"""
ExiGCN LoRA Trainer
"""

from train.trainer_exi import ExiGCNTrainer
from models.exigcn_lora import ExiGCNLoRA
import torch


class ExiGCNLoRATrainer(ExiGCNTrainer):
    """
    Trainer for ExiGCN with LoRA.
    Only difference: optimizer uses LoRA params only during retraining.
    """
    
    def __init__(self, model: ExiGCNLoRA, *args, **kwargs):
        # Call parent with all args
        super().__init__(model, *args, **kwargs)
        
        # Store LoRA rank for logging
        self.lora_rank = model.lora_rank if hasattr(model, 'lora_rank') else 8
    
    def train_initial(self, adj, features, labels, train_mask, val_mask, test_mask=None):
        """Initial training - same as base ExiGCN."""
        if self.verbose:
            print("\n" + "="*70)
            print("INITIAL TRAINING (LoRA)")
            print("="*70)
            
            # Print parameter stats
            stats = self.model.get_num_params()
            print(f"  Total params: {stats['total']:,}")
            print(f"  Main params: {stats['main']:,}")
            print(f"  LoRA params: {stats['lora']:,}")
            print(f"  Reduction: {stats['reduction']:.1%}")
        
        return super().train_initial(adj, features, labels, train_mask, val_mask, test_mask)
    
    def train_retraining(self, adj_updated, features_updated, labels_updated,
                        train_mask, val_mask, test_mask=None):
        """Retraining with LoRA - only train LoRA parameters."""
        if self.verbose:
            print("\n" + "="*70)
            print("EFFICIENT RETRAINING (ExiGCN + LoRA)")
            print("="*70)
            print(f"  Initial nodes: {self.initial_num_nodes}")
            print(f"  Updated nodes: {adj_updated.size(0)}")
            print(f"  Added nodes: {adj_updated.size(0) - self.initial_num_nodes}")
            print(f"  LoRA rank: {self.lora_rank}")
        
        # Compute deltas
        with self.timer.measure_graph_update():
            delta_adj, delta_features = self._compute_deltas(
                adj_updated, features_updated
            )
        
        if self.verbose:
            print(f"\n  Delta adjacency nnz: {delta_adj._nnz()}")
            print(f"  Delta features non-zero: {(delta_features != 0).sum().item()}")
        
        # Prepare model (only once per experiment)
        if not hasattr(self.model, '_retraining_prepared') or not self.model._retraining_prepared:
            if self.verbose:
                print("  Preparing for retraining (resetting LoRA)...")
            self.model.prepare_for_retraining()
            
            # Freeze main weights, only train LoRA
            self.model.freeze_main_weights()
            
            self.model._retraining_prepared = True
            
            # Print trainable params
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            if self.verbose:
                print(f"  Trainable params: {trainable:,} (LoRA only)")
        else:
            if self.verbose:
                print("  Model already prepared, skipping reset...")
                print(f"  Existing cached_B layers: {len(self.model.cached_B)}")
        
        # Create optimizer for LoRA parameters only
        lora_params = self.model.get_lora_params()
        self.optimizer = torch.optim.Adam(
            lora_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Reset trackers
        self.timer.reset()
        self.metrics_tracker = type(self.metrics_tracker)()
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.best_model_state = None
        
        # Store delta
        self.delta_adj = delta_adj
        self.delta_features = delta_features
        
        # Training loop
        self.timer.start_training()
        
        epochs_without_improvement = 0
        
        for epoch in range(self.epochs):
            self.timer.start_epoch()
            
            train_loss = self._train_epoch_retraining(
                adj_updated, features_updated, labels_updated, train_mask
            )
            
            val_metrics = self._evaluate_retraining(
                adj_updated, features_updated, labels_updated, val_mask, None
            )
            
            if val_metrics['val_acc'] > self.best_val_acc:
                self.best_val_acc = val_metrics['val_acc']
                self.best_epoch = epoch
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{self.epochs} | "
                      f"Loss: {train_loss:.4f} | "
                      f"Val Acc: {val_metrics['val_acc']:.4f} | "
                      f"Best: {self.best_val_acc:.4f} @ {self.best_epoch+1}")
            
            if epochs_without_improvement >= self.patience:
                if self.verbose:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                break
            
            self.timer.stop_epoch()
        
        self.timer.stop_training()
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # Merge LoRA into main weights
        if self.verbose:
            print("\nMerging LoRA weights into main weights...")
        self.model.merge_all_lora()
        
        # Final evaluation
        final_metrics = self._evaluate_retraining(
            adj_updated, features_updated, labels_updated, val_mask, test_mask
        )
        
        if self.verbose:
            print("\n" + "="*70)
            print("RETRAINING COMPLETE")
            print("="*70)
            print(f"Best Epoch: {self.best_epoch + 1}")
            print(f"Best Val Acc: {self.best_val_acc:.4f}")
            if test_mask is not None:
                print(f"Test Acc: {final_metrics['test_acc']:.4f}")
            
            self.timer.print_report()
        
        return final_metrics


if __name__ == "__main__":
    print("✅ ExiGCN LoRA Trainer - Ready to use!")
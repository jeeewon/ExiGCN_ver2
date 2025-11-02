"""
Full Retraining trainer (baseline).
"""

import torch
from train.trainer_base import BaseTrainer
from models.base_gcn import BaseGCN
from typing import Dict


class FullRetrainingTrainer(BaseTrainer):
    """
    Trainer for Full Retraining baseline.
    Retrains from scratch on updated graph.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_phase = "initial"  # "initial" or "retraining"
    
    def train_on_updated_graph(self,
                               adj: torch.Tensor,
                               features: torch.Tensor,
                               labels: torch.Tensor,
                               train_mask: torch.Tensor,
                               val_mask: torch.Tensor,
                               test_mask: torch.Tensor = None) -> Dict[str, float]:
        """
        Train on updated graph (full retraining).
        
        Args:
            adj: Updated normalized adjacency
            features: Updated node features
            labels: Updated node labels
            train_mask: Training mask
            val_mask: Validation mask
            test_mask: Test mask
            
        Returns:
            Final metrics
        """
        self.training_phase = "retraining"
        
        # Reset model parameters (full retraining!)
        self._reset_model()
        
        # Reset optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Reset trackers
        self.timer.reset()
        self.metrics_tracker = type(self.metrics_tracker)()
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.best_model_state = None
        
        if self.verbose:
            print("\n" + "="*70)
            print("FULL RETRAINING (FROM SCRATCH)")
            print("="*70)
            print(f"  Graph size: {adj.size(0)} nodes")
            print(f"  Edges: {adj._nnz()}")
        
        # Train from scratch
        results = self.train(adj, features, labels, train_mask, val_mask, test_mask)
        
        return results
    
    def _reset_model(self):
        """Reset model parameters to random initialization."""
        for layer in self.model.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


if __name__ == "__main__":
    print("FullRetrainingTrainer - use in experiments")
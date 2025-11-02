"""
Base trainer class with common training logic.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional
from utils.timer import ExperimentTimer
from utils.metrics import evaluate_model, compute_loss, MetricsTracker
import os


class BaseTrainer:
    """
    Base trainer for GCN models.
    """
    
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 learning_rate: float = 0.01,
                 retraining_lr: Optional[float] = None,  # NEW!
                 weight_decay: float = 0.0005,
                 epochs: int = 200,
                 early_stopping: bool = False,
                 patience: int = 50,
                 verbose: bool = True):
        """
        Args:
            model: GCN model
            device: Training device
            learning_rate: Learning rate for initial training
            retraining_lr: Learning rate for retraining (default: same as learning_rate)
            weight_decay: L2 regularization
            epochs: Number of training epochs
            early_stopping: Enable early stopping
            patience: Early stopping patience
            verbose: Print training progress
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.retraining_lr = retraining_lr if retraining_lr is not None else learning_rate  # NEW!
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.verbose = verbose
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Trackers
        self.timer = ExperimentTimer(use_cuda=(device.type == 'cuda'))
        self.metrics_tracker = MetricsTracker()
        
        # Best model state
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.best_model_state = None
    
    def train_epoch(self,
                   adj: torch.Tensor,
                   features: torch.Tensor,
                   labels: torch.Tensor,
                   train_mask: torch.Tensor) -> float:
        """
        Train for one epoch.
        
        Args:
            adj: Normalized adjacency
            features: Node features
            labels: Node labels
            train_mask: Training mask
            
        Returns:
            Training loss
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward
        with self.timer.measure_forward():
            logits = self.model(adj, features)
        
        # Compute loss
        loss = compute_loss(logits, labels, train_mask)
        
        # Backward
        with self.timer.measure_backward():
            loss.backward()
        
        # Optimizer step
        with self.timer.measure_optimizer_step():
            self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self,
                adj: torch.Tensor,
                features: torch.Tensor,
                labels: torch.Tensor,
                val_mask: torch.Tensor,
                test_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Evaluate model.
        
        Args:
            adj: Normalized adjacency
            features: Node features
            labels: Node labels
            val_mask: Validation mask
            test_mask: Test mask (optional)
            
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        
        with self.timer.measure_evaluation():
            logits = self.model(adj, features)
            
            # Validation metrics
            val_metrics = evaluate_model(logits, labels, val_mask)
            
            results = {
                'val_loss': compute_loss(logits, labels, val_mask).item(),
                'val_acc': val_metrics['accuracy'],
                'val_f1_micro': val_metrics['f1_micro'],
                'val_f1_macro': val_metrics['f1_macro']
            }
            
            # Test metrics if provided
            if test_mask is not None:
                test_metrics = evaluate_model(logits, labels, test_mask)
                results.update({
                    'test_loss': compute_loss(logits, labels, test_mask).item(),
                    'test_acc': test_metrics['accuracy'],
                    'test_f1_micro': test_metrics['f1_micro'],
                    'test_f1_macro': test_metrics['f1_macro']
                })
        
        return results
    
    def train(self,
             adj: torch.Tensor,
             features: torch.Tensor,
             labels: torch.Tensor,
             train_mask: torch.Tensor,
             val_mask: torch.Tensor,
             test_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Full training loop.
        
        Args:
            adj: Normalized adjacency
            features: Node features
            labels: Node labels
            train_mask: Training mask
            val_mask: Validation mask
            test_mask: Test mask (optional)
            
        Returns:
            Final metrics
        """
        if self.verbose:
            print("\n" + "="*70)
            print("TRAINING")
            print("="*70)
        
        self.timer.start_training()
        
        epochs_without_improvement = 0
        
        for epoch in range(self.epochs):
            self.timer.start_epoch()
            
            # Train
            train_loss = self.train_epoch(adj, features, labels, train_mask)
            
            # Evaluate
            metrics = self.evaluate(adj, features, labels, val_mask, test_mask)
            
            # Update tracker
            self.metrics_tracker.update(
                train_loss=train_loss,
                train_acc=None,  # Could compute if needed
                val_loss=metrics['val_loss'],
                val_metrics={
                    'accuracy': metrics['val_acc'],
                    'f1_micro': metrics['val_f1_micro'],
                    'f1_macro': metrics['val_f1_macro']
                }
            )
            
            # Check for improvement
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
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # Final evaluation
        final_metrics = self.evaluate(adj, features, labels, val_mask, test_mask)
        
        if self.verbose:
            print("\n" + "="*70)
            print("TRAINING COMPLETE")
            print("="*70)
            print(f"Best Epoch: {self.best_epoch + 1}")
            print(f"Best Val Acc: {self.best_val_acc:.4f}")
            if test_mask is not None:
                print(f"Test Acc: {final_metrics['test_acc']:.4f}")
            
            # Print timing
            self.timer.print_report()
        
        return final_metrics
    
    def get_training_time(self) -> float:
        """Get training time (논문 보고용)."""
        return self.timer.get_training_time_only()
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
        }, path)
        if self.verbose:
            print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        self.best_epoch = checkpoint['best_epoch']
        if self.verbose:
            print(f"Checkpoint loaded from {path}")


if __name__ == "__main__":
    print("BaseTrainer - use in experiments")
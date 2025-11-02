"""
Evaluation metrics for node classification.
"""

import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from typing import Dict


def accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute classification accuracy.
    
    Args:
        predictions: Predicted labels [N]
        labels: Ground truth labels [N]
        
    Returns:
        Accuracy score
    """
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    return correct / total


def f1_micro(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute micro-averaged F1 score.
    
    Args:
        predictions: Predicted labels [N]
        labels: Ground truth labels [N]
        
    Returns:
        Micro F1 score
    """
    preds_np = predictions.cpu().numpy()
    labels_np = labels.cpu().numpy()
    return f1_score(labels_np, preds_np, average='micro')


def f1_macro(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute macro-averaged F1 score.
    
    Args:
        predictions: Predicted labels [N]
        labels: Ground truth labels [N]
        
    Returns:
        Macro F1 score
    """
    preds_np = predictions.cpu().numpy()
    labels_np = labels.cpu().numpy()
    return f1_score(labels_np, preds_np, average='macro', zero_division=0)


def evaluate_model(logits: torch.Tensor, 
                   labels: torch.Tensor,
                   mask: torch.Tensor = None) -> Dict[str, float]:
    """
    Comprehensive evaluation of model predictions.
    
    Args:
        logits: Model output logits [N x C]
        labels: Ground truth labels [N]
        mask: Optional mask for evaluation [N]
        
    Returns:
        Dictionary of metrics
    """
    # Get predictions
    predictions = logits.argmax(dim=1)
    
    # Apply mask if provided
    if mask is not None:
        predictions = predictions[mask]
        labels = labels[mask]
    
    # Compute metrics
    acc = accuracy(predictions, labels)
    f1_mic = f1_micro(predictions, labels)
    f1_mac = f1_macro(predictions, labels)
    
    return {
        'accuracy': acc,
        'f1_micro': f1_mic,
        'f1_macro': f1_mac
    }


def compute_loss(logits: torch.Tensor,
                labels: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
    """
    Compute cross-entropy loss.
    
    Args:
        logits: Model output logits [N x C]
        labels: Ground truth labels [N]
        mask: Optional mask for loss computation [N]
        
    Returns:
        Loss value
    """
    if mask is not None:
        logits = logits[mask]
        labels = labels[mask]
    
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss


class MetricsTracker:
    """
    Track metrics over training epochs.
    """
    
    def __init__(self):
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1_micro': [],
            'val_f1_macro': []
        }
    
    def update(self, 
               train_loss: float = None,
               train_acc: float = None,
               val_loss: float = None,
               val_metrics: Dict[str, float] = None):
        """
        Update metrics for current epoch.
        
        Args:
            train_loss: Training loss
            train_acc: Training accuracy
            val_loss: Validation loss
            val_metrics: Dictionary of validation metrics
        """
        if train_loss is not None:
            self.history['train_loss'].append(train_loss)
        if train_acc is not None:
            self.history['train_acc'].append(train_acc)
        if val_loss is not None:
            self.history['val_loss'].append(val_loss)
        if val_metrics is not None:
            self.history['val_acc'].append(val_metrics.get('accuracy', 0))
            self.history['val_f1_micro'].append(val_metrics.get('f1_micro', 0))
            self.history['val_f1_macro'].append(val_metrics.get('f1_macro', 0))
    
    def get_best_epoch(self, metric: str = 'val_acc') -> int:
        """
        Get epoch with best validation metric.
        
        Args:
            metric: Metric name to use
            
        Returns:
            Epoch index (0-based)
        """
        if metric not in self.history or len(self.history[metric]) == 0:
            return 0
        return int(np.argmax(self.history[metric]))
    
    def get_best_value(self, metric: str = 'val_acc') -> float:
        """
        Get best validation metric value.
        
        Args:
            metric: Metric name to use
            
        Returns:
            Best metric value
        """
        if metric not in self.history or len(self.history[metric]) == 0:
            return 0.0
        return float(np.max(self.history[metric]))
    
    def summary(self) -> Dict[str, float]:
        """
        Get summary of all metrics.
        
        Returns:
            Dictionary with best/final values
        """
        return {
            'best_val_acc': self.get_best_value('val_acc'),
            'best_val_f1_micro': self.get_best_value('val_f1_micro'),
            'best_val_f1_macro': self.get_best_value('val_f1_macro'),
            'final_train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else 0,
            'final_val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else 0
        }


def test_metrics():
    """Test metrics functions."""
    # Create dummy data
    num_samples = 100
    num_classes = 7
    
    logits = torch.randn(num_samples, num_classes)
    labels = torch.randint(0, num_classes, (num_samples,))
    
    # Test evaluation
    metrics = evaluate_model(logits, labels)
    print("Metrics:", metrics)
    
    # Test loss
    loss = compute_loss(logits, labels)
    print(f"Loss: {loss.item():.4f}")
    
    # Test tracker
    tracker = MetricsTracker()
    for epoch in range(10):
        tracker.update(
            train_loss=1.0 / (epoch + 1),
            train_acc=0.5 + epoch * 0.05,
            val_loss=1.2 / (epoch + 1),
            val_metrics={'accuracy': 0.4 + epoch * 0.05, 'f1_micro': 0.4 + epoch * 0.04}
        )
    
    print("\nTracker summary:", tracker.summary())
    print(f"Best epoch: {tracker.get_best_epoch('val_acc')}")
    
    print("\n✅ Metrics test passed!")


if __name__ == "__main__":
    test_metrics()
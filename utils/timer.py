"""
Enhanced timer for comprehensive time measurement.
"""

import time
import torch
from typing import Dict, List
from contextlib import contextmanager
from collections import defaultdict


class ComprehensiveTimer:
    """
    Comprehensive timer measuring all aspects of training.
    """
    
    def __init__(self, use_cuda: bool = True):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        
        # Multiple timers for different phases
        self.timers = defaultdict(list)
        self.current_measurements = {}
        
    def _sync(self):
        """Synchronize CUDA if needed."""
        if self.use_cuda:
            torch.cuda.synchronize()
    
    @contextmanager
    def measure(self, name: str):
        """
        Context manager for timing a block.
        
        Usage:
            with timer.measure("forward"):
                output = model(input)
        """
        self._sync()
        start = time.time()
        
        try:
            yield
        finally:
            self._sync()
            elapsed = time.time() - start
            self.timers[name].append(elapsed)
    
    def start(self, name: str):
        """Start measuring for a named phase."""
        self._sync()
        self.current_measurements[name] = time.time()
    
    def stop(self, name: str):
        """Stop measuring for a named phase."""
        if name not in self.current_measurements:
            raise ValueError(f"Timer '{name}' was not started")
        
        self._sync()
        elapsed = time.time() - self.current_measurements[name]
        self.timers[name].append(elapsed)
        del self.current_measurements[name]
    
    def get_times(self, name: str) -> List[float]:
        """Get all recorded times for a name."""
        return self.timers.get(name, [])
    
    def get_total(self, name: str) -> float:
        """Get total time for a name."""
        times = self.get_times(name)
        return sum(times) if times else 0.0
    
    def get_average(self, name: str) -> float:
        """Get average time for a name."""
        times = self.get_times(name)
        return sum(times) / len(times) if times else 0.0
    
    def get_count(self, name: str) -> int:
        """Get number of measurements for a name."""
        return len(self.timers.get(name, []))
    
    def reset(self):
        """Reset all timers."""
        self.timers.clear()
        self.current_measurements.clear()
    
    def summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get comprehensive summary.
        
        Returns:
            Dictionary with statistics for each timer
        """
        summary = {}
        
        for name, times in self.timers.items():
            if times:
                summary[name] = {
                    'total': sum(times),
                    'average': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                    'count': len(times)
                }
        
        return summary
    
    def print_summary(self):
        """Print formatted summary."""
        print("\n" + "="*70)
        print("COMPREHENSIVE TIMER SUMMARY")
        print("="*70)
        
        summary = self.summary()
        
        if not summary:
            print("No measurements recorded.")
            return
        
        # Sort by total time (descending)
        sorted_items = sorted(summary.items(), 
                            key=lambda x: x[1]['total'], 
                            reverse=True)
        
        print(f"{'Phase':<30} {'Total':>10} {'Average':>10} {'Count':>8}")
        print("-"*70)
        
        for name, stats in sorted_items:
            print(f"{name:<30} {stats['total']:>9.3f}s {stats['average']:>9.4f}s {stats['count']:>8d}")
        
        print("="*70)


class ExperimentTimer:
    """
    Timer for full experiment with structured phases.
    """
    
    def __init__(self, use_cuda: bool = True):
        self.timer = ComprehensiveTimer(use_cuda)
        self.phase_stack = []
    
    # ==== Data Loading ====
    def start_data_loading(self):
        """Start data loading phase."""
        self.timer.start("data_loading")
    
    def stop_data_loading(self):
        """Stop data loading phase."""
        self.timer.stop("data_loading")
    
    # ==== Preprocessing ====
    def start_preprocessing(self):
        """Start preprocessing phase."""
        self.timer.start("preprocessing")
    
    def stop_preprocessing(self):
        """Stop preprocessing phase."""
        self.timer.stop("preprocessing")
    
    # ==== Initialization ====
    def start_initialization(self):
        """Start model initialization."""
        self.timer.start("initialization")
    
    def stop_initialization(self):
        """Stop model initialization."""
        self.timer.stop("initialization")
    
    # ==== Training (Overall) ====
    def start_training(self):
        """Start training phase."""
        self.timer.start("training_total")
    
    def stop_training(self):
        """Stop training phase."""
        self.timer.stop("training_total")
    
    # ==== Epoch ====
    def start_epoch(self):
        """Start single epoch."""
        self.timer.start("epoch")
    
    def stop_epoch(self):
        """Stop single epoch."""
        self.timer.stop("epoch")
    
    # ==== Forward ====
    @contextmanager
    def measure_forward(self):
        """Measure forward pass."""
        with self.timer.measure("forward"):
            yield
    
    # ==== Backward ====
    @contextmanager
    def measure_backward(self):
        """Measure backward pass."""
        with self.timer.measure("backward"):
            yield
    
    # ==== Optimizer Step ====
    @contextmanager
    def measure_optimizer_step(self):
        """Measure optimizer step."""
        with self.timer.measure("optimizer_step"):
            yield
    
    # ==== ExiGCN Specific ====
    @contextmanager
    def measure_cache_computation(self):
        """Measure fixed term F computation (ExiGCN only)."""
        with self.timer.measure("cache_computation"):
            yield
    
    # ==== Evaluation ====
    @contextmanager
    def measure_evaluation(self):
        """Measure validation/test evaluation."""
        with self.timer.measure("evaluation"):
            yield
    
    # ==== Graph Update ====
    @contextmanager
    def measure_graph_update(self):
        """Measure graph update (delta computation)."""
        with self.timer.measure("graph_update"):
            yield
    
    # ==== Results ====
    def get_training_time_only(self) -> float:
        """
        Get training time (논문 보고용).
        Forward + Backward + Optimizer step only.
        """
        return (self.timer.get_total("forward") +
                self.timer.get_total("backward") +
                self.timer.get_total("optimizer_step"))
    
    def get_total_time(self) -> float:
        """Get total experiment time."""
        return self.timer.get_total("training_total")
    
    def get_speedup(self, baseline_training_time: float) -> float:
        """
        Calculate speedup vs baseline.
        Uses training_time_only (논문 방식).
        """
        current_time = self.get_training_time_only()
        if current_time == 0:
            return 0.0
        return baseline_training_time / current_time
    
    def get_report(self) -> Dict[str, float]:
        """
        Generate comprehensive report.
        """
        return {
            # Paper metric (main)
            'training_time': self.get_training_time_only(),
            
            # Breakdown
            'forward_time': self.timer.get_total("forward"),
            'backward_time': self.timer.get_total("backward"),
            'optimizer_time': self.timer.get_total("optimizer_step"),
            
            # Full pipeline
            'total_time': self.get_total_time(),
            'data_loading_time': self.timer.get_total("data_loading"),
            'preprocessing_time': self.timer.get_total("preprocessing"),
            'initialization_time': self.timer.get_total("initialization"),
            'evaluation_time': self.timer.get_total("evaluation"),
            
            # ExiGCN specific
            'cache_computation_time': self.timer.get_total("cache_computation"),
            'graph_update_time': self.timer.get_total("graph_update"),
            
            # Statistics
            'num_epochs': self.timer.get_count("epoch"),
            'avg_epoch_time': self.timer.get_average("epoch"),
        }
    
    def print_report(self):
        """Print formatted report."""
        report = self.get_report()
        
        print("\n" + "="*70)
        print("EXPERIMENT TIME REPORT")
        print("="*70)
        
        print("\n[논문 보고용 - Training Time Only]")
        print(f"  Training Time: {report['training_time']:.3f}s")
        print(f"    - Forward:   {report['forward_time']:.3f}s")
        print(f"    - Backward:  {report['backward_time']:.3f}s")
        print(f"    - Optimizer: {report['optimizer_time']:.3f}s")
        
        print("\n[전체 파이프라인]")
        print(f"  Total Time:         {report['total_time']:.3f}s")
        print(f"  Data Loading:       {report['data_loading_time']:.3f}s")
        print(f"  Preprocessing:      {report['preprocessing_time']:.3f}s")
        print(f"  Initialization:     {report['initialization_time']:.3f}s")
        print(f"  Evaluation:         {report['evaluation_time']:.3f}s")
        
        if report['cache_computation_time'] > 0:
            print("\n[ExiGCN Specific]")
            print(f"  Cache Computation:  {report['cache_computation_time']:.3f}s")
            print(f"  Graph Update:       {report['graph_update_time']:.3f}s")
        
        print("\n[통계]")
        print(f"  Total Epochs:       {report['num_epochs']}")
        print(f"  Avg Epoch Time:     {report['avg_epoch_time']:.4f}s")
        
        print("="*70)
    
    def reset(self):
        """Reset all timers."""
        self.timer.reset()
        self.phase_stack.clear()


def test_comprehensive_timer():
    """Test comprehensive timer."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    timer = ExperimentTimer(use_cuda=(device.type == 'cuda'))
    
    # Simulate full experiment
    
    # 1. Data loading
    timer.start_data_loading()
    time.sleep(0.1)
    timer.stop_data_loading()
    
    # 2. Preprocessing
    timer.start_preprocessing()
    time.sleep(0.05)
    timer.stop_preprocessing()
    
    # 3. Initialization
    timer.start_initialization()
    model = torch.nn.Linear(100, 10).to(device)
    timer.stop_initialization()
    
    # 4. Training
    timer.start_training()
    
    for epoch in range(5):
        timer.start_epoch()
        
        # Forward
        with timer.measure_forward():
            x = torch.randn(32, 100, device=device)
            y = model(x)
            time.sleep(0.01)
        
        # Backward
        with timer.measure_backward():
            loss = y.sum()
            loss.backward()
            time.sleep(0.01)
        
        # Optimizer
        with timer.measure_optimizer_step():
            # optimizer.step() would go here
            time.sleep(0.005)
        
        # Evaluation
        with timer.measure_evaluation():
            time.sleep(0.02)
        
        timer.stop_epoch()
    
    timer.stop_training()
    
    # Print results
    timer.print_report()
    
    # Test speedup calculation
    baseline_time = 0.5
    speedup = timer.get_speedup(baseline_time)
    print(f"\nSpeedup vs baseline ({baseline_time}s): {speedup:.2f}x")
    
    print("\n✅ Comprehensive timer test passed!")


if __name__ == "__main__":
    test_comprehensive_timer()
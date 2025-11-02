"""
Incremental experiment: Compare Full Retraining vs ExiGCN
"""

import torch
import numpy as np
import pandas as pd
import os
import yaml
from typing import Dict, List
import time

from data.download import DatasetLoader
from data.preprocessor import DataPreprocessor
from data.graph_updater import GraphUpdater
from models.base_gcn import BaseGCN
from models.exigcn import ExiGCN
from train.trainer_full import FullRetrainingTrainer
from train.trainer_exi import ExiGCNTrainer


class IncrementalExperiment:
    """
    Run incremental experiment comparing Full Retraining vs ExiGCN.
    """
    
    def __init__(self, config_path: str = None):
        """
        Args:
            config_path: Path to config file (optional)
        """
        # Load config
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        # Results storage
        self.results = []
    
    def _default_config(self) -> Dict:
        """Default configuration."""
        return {
            'model': {
                'num_layers': 2,
                'hidden_dim': 128,
                'dropout': 0.5,
                'activation': 'relu'
            },
            'training': {
                'epochs': 200,
                'learning_rate': 0.01,
                'weight_decay': 0.0005,
                'optimizer': 'adam',
                'early_stopping': False
            },
            'experiment': {
                'initial_ratio': 0.9,
                'update_stages': [0.02, 0.04, 0.06, 0.08, 0.10],
                'num_runs': 1,  # Set to 10 for full experiment
                'seed': 42
            },
            'data': {
                'root': './data',
                'normalize_features': True,
                'normalize_adj': True,
                'add_self_loops': True
            }
        }
    
    def run_experiment(self, dataset_name: str = 'cora_full'):
        """
        Run full incremental experiment.
        
        Args:
            dataset_name: Name of dataset to use
        """
        print("\n" + "="*70)
        print(f"INCREMENTAL EXPERIMENT: {dataset_name.upper()}")
        print("="*70)
        
        # Load dataset
        loader = DatasetLoader(root=self.config['data']['root'])
        adj, features, labels, train_mask, val_mask, test_mask = loader.load_dataset(dataset_name)
        
        # Print dataset info
        print(f"\nDataset: {dataset_name}")
        print(f"  Nodes: {adj.size(0):,}")
        print(f"  Edges: {adj._nnz():,}")
        print(f"  Features: {features.size(1)}")
        print(f"  Classes: {labels.max().item() + 1}")
        
        # Run multiple times
        num_runs = self.config['experiment']['num_runs']
        
        for run in range(num_runs):
            print(f"\n{'='*70}")
            print(f"RUN {run + 1}/{num_runs}")
            print("="*70)
            
            torch.manual_seed(self.config['experiment']['seed'] + run)
            np.random.seed(self.config['experiment']['seed'] + run)
            
            self._run_single_experiment(
                dataset_name, adj, features, labels,
                train_mask, val_mask, test_mask, run
            )
        
        # Save results
        self._save_results(dataset_name)
        
        # Print summary
        self._print_summary(dataset_name)
    
    def _run_single_experiment(self,
                              dataset_name: str,
                              adj: torch.Tensor,
                              features: torch.Tensor,
                              labels: torch.Tensor,
                              train_mask: torch.Tensor,
                              val_mask: torch.Tensor,
                              test_mask: torch.Tensor,
                              run: int):
        """Run single experiment iteration."""
        
        # Create graph updater
        updater = GraphUpdater(
            adj=adj,
            features=features,
            labels=labels,
            initial_ratio=self.config['experiment']['initial_ratio'],
            n_buckets=len(self.config['experiment']['update_stages']),
            seed=self.config['experiment']['seed'] + run
        )
        
        # Create incremental scenario
        print("\nCreating incremental scenario...")
        buckets, core_adj, core_features, core_labels, node_mapping = \
            updater.create_incremental_scenario()
        
        # Preprocessor
        preprocessor = DataPreprocessor(
            normalize_features=self.config['data']['normalize_features'],
            normalize_adj=self.config['data']['normalize_adj'],
            add_self_loops=self.config['data']['add_self_loops']
        )
        
        # Preprocess core graph
        core_adj_norm, core_features_norm = preprocessor.preprocess(
            core_adj, core_features, self.device
        )
        core_labels = core_labels.to(self.device)
        
        # Get masks for core graph
        core_size = core_adj.size(0)
        
        # DEBUG: Print mask information
        print(f"\n[DEBUG] Mask Information:")
        print(f"  Original dataset nodes: {adj.size(0)}")
        print(f"  Core graph nodes: {core_size}")
        print(f"  Original train_mask length: {len(train_mask)}")
        print(f"  Original train_mask True count: {train_mask.sum()}")
        print(f"  Original val_mask True count: {val_mask.sum()}")
        print(f"  Original test_mask True count: {test_mask.sum()}")
        print(f"  Node mapping available: {node_mapping is not None}")
        if node_mapping is not None:
            print(f"  Node mapping length: {len(node_mapping)}")
            print(f"  Node mapping sample (first 10): {list(node_mapping.keys())[:10]}")
        
        core_train_mask = torch.zeros(core_size, dtype=torch.bool)
        core_val_mask = torch.zeros(core_size, dtype=torch.bool)
        core_test_mask = torch.zeros(core_size, dtype=torch.bool)
        
        # Map masks using node_mapping
        for old_idx, new_idx in node_mapping.items():
            core_train_mask[new_idx] = train_mask[old_idx]
            core_val_mask[new_idx] = val_mask[old_idx]
            core_test_mask[new_idx] = test_mask[old_idx]
        
        core_train_mask = core_train_mask.to(self.device)
        core_val_mask = core_val_mask.to(self.device)
        core_test_mask = core_test_mask.to(self.device)
        
        # DEBUG: Print resulting mask sizes
        print(f"  Core train_mask True count: {core_train_mask.sum()}")
        print(f"  Core val_mask True count: {core_val_mask.sum()}")
        print(f"  Core test_mask True count: {core_test_mask.sum()}")
        print(f"  Core mask total coverage: {(core_train_mask.sum() + core_val_mask.sum() + core_test_mask.sum()).item()} / {core_size}")
        
        num_features = features.size(1)
        num_classes = labels.max().item() + 1
        
        # ================================================================
        # PHASE 1: Initial Training (90%)
        # ================================================================
        
        print("\n" + "="*70)
        print("PHASE 1: INITIAL TRAINING (90%)")
        print("="*70)
        
        # Full Retraining Model
        model_full = BaseGCN(
            num_features=num_features,
            hidden_dim=self.config['model']['hidden_dim'],
            num_classes=num_classes,
            num_layers=self.config['model']['num_layers'],
            dropout=self.config['model']['dropout']
        )
        
        trainer_full = FullRetrainingTrainer(
            model=model_full,
            device=self.device,
            learning_rate=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            epochs=self.config['training']['epochs'],
            verbose=True
        )
        
        print("\n[Full Retraining] Initial Training...")
        initial_results_full = trainer_full.train(
            core_adj_norm, core_features_norm, core_labels,
            core_train_mask, core_val_mask, core_test_mask
        )
        initial_time_full = trainer_full.get_training_time()
        
        # ExiGCN Model
        model_exi = ExiGCN(
            num_features=num_features,
            hidden_dim=self.config['model']['hidden_dim'],
            num_classes=num_classes,
            num_layers=self.config['model']['num_layers'],
            dropout=self.config['model']['dropout']
        )
        
        trainer_exi = ExiGCNTrainer(
            model=model_exi,
            device=self.device,
            learning_rate=self.config['training']['learning_rate'],
            retraining_lr=self.config['training'].get('retraining_lr', self.config['training']['learning_rate']),
            weight_decay=self.config['training']['weight_decay'],
            epochs=self.config['training']['epochs'],
            verbose=True
        )
        
        print("\n[ExiGCN] Initial Training...")
        initial_results_exi = trainer_exi.train_initial(
            core_adj_norm, core_features_norm, core_labels,
            core_train_mask, core_val_mask, core_test_mask
        )
        initial_time_exi = trainer_exi.get_training_time()
        
        # Record initial results
        self.results.append({
            'dataset': dataset_name,
            'run': run,
            'stage': 'initial',
            'ratio': self.config['experiment']['initial_ratio'],
            'nodes': core_size,
            'method': 'Full',
            'train_time': initial_time_full,
            'val_acc': initial_results_full['val_acc'],
            'test_acc': initial_results_full['test_acc'],
            'val_f1_micro': initial_results_full['val_f1_micro'],
            'test_f1_micro': initial_results_full['test_f1_micro']
        })
        
        self.results.append({
            'dataset': dataset_name,
            'run': run,
            'stage': 'initial',
            'ratio': self.config['experiment']['initial_ratio'],
            'nodes': core_size,
            'method': 'ExiGCN',
            'train_time': initial_time_exi,
            'val_acc': initial_results_exi['val_acc'],
            'test_acc': initial_results_exi['test_acc'],
            'val_f1_micro': initial_results_exi['val_f1_micro'],
            'test_f1_micro': initial_results_exi['test_f1_micro']
        })
        
        # ================================================================
        # PHASE 2: Incremental Updates
        # ================================================================
        
        current_adj = core_adj
        current_features = core_features
        current_labels = core_labels
        
        bucket_names = ['A', 'B', 'C', 'D', 'E']
        
        for stage_idx, (bucket_name, update_ratio) in enumerate(
            zip(bucket_names, self.config['experiment']['update_stages'])
        ):
            print(f"\n{'='*70}")
            print(f"PHASE 2.{stage_idx + 1}: UPDATE STAGE {bucket_name} "
                  f"(+{update_ratio*100:.0f}%)")
            print("="*70)
            
            # Add bucket
            current_adj, current_features, current_labels = \
                updater.add_bucket_to_graph(
                    current_adj, current_features, current_labels,
                    buckets[bucket_name]
                )
            
            # Preprocess
            current_adj_norm, current_features_norm = preprocessor.preprocess(
                current_adj, current_features, self.device
            )
            current_labels = current_labels.to(self.device)
            
            # Update masks using node_mapping
            current_size = current_adj.size(0)
            
            # Build cumulative node mapping
            # Stage A: core + A
            # Stage B: core + A + B
            # Stage C: core + A + B + C
            # etc.
            current_node_to_original = {}
            
            # Add core nodes (already mapped)
            for old_idx, new_idx in node_mapping.items():
                current_node_to_original[new_idx] = old_idx
            
            # Add all buckets up to and including current stage
            offset = core_size
            for i, prev_bucket_name in enumerate(bucket_names):
                if i <= stage_idx:  # Include current stage
                    for j, old_idx in enumerate(buckets[prev_bucket_name]):
                        new_idx = offset + j
                        current_node_to_original[new_idx] = old_idx
                    offset += len(buckets[prev_bucket_name])
            
            # DEBUG: Print mask information for this stage
            print(f"\n[DEBUG] Stage {bucket_name} Mask Information:")
            print(f"  Current graph nodes: {current_size}")
            print(f"  Node mapping built with {len(current_node_to_original)} nodes")
            if len(current_node_to_original) != current_size:
                print(f"  ⚠️  WARNING: Mapping size mismatch!")
            
            # Build masks using mapping
            current_train_mask = torch.zeros(current_size, dtype=torch.bool)
            current_val_mask = torch.zeros(current_size, dtype=torch.bool)
            current_test_mask = torch.zeros(current_size, dtype=torch.bool)
            
            for new_idx, old_idx in current_node_to_original.items():
                current_train_mask[new_idx] = train_mask[old_idx]
                current_val_mask[new_idx] = val_mask[old_idx]
                current_test_mask[new_idx] = test_mask[old_idx]
            
            current_train_mask = current_train_mask.to(self.device)
            current_val_mask = current_val_mask.to(self.device)
            current_test_mask = current_test_mask.to(self.device)
            
            print(f"  Current train_mask True count: {current_train_mask.sum()}")
            print(f"  Current val_mask True count: {current_val_mask.sum()}")
            print(f"  Current test_mask True count: {current_test_mask.sum()}")
            print(f"  Current mask total: {(current_train_mask.sum() + current_val_mask.sum() + current_test_mask.sum()).item()} / {current_size}")
            
            print(f"\nGraph updated: {current_size:,} nodes")
            
            # Full Retraining
            print("\n[Full Retraining] Retraining from scratch...")
            full_results = trainer_full.train_on_updated_graph(
                current_adj_norm, current_features_norm, current_labels,
                current_train_mask, current_val_mask, current_test_mask
            )
            full_time = trainer_full.get_training_time()
            
            # ExiGCN
            print("\n[ExiGCN] Efficient retraining...")
            exi_results = trainer_exi.train_retraining(
                current_adj_norm, current_features_norm, current_labels,
                current_train_mask, current_val_mask, current_test_mask
            )
            exi_time = trainer_exi.get_training_time()
            
            # Speedup
            speedup = full_time / exi_time if exi_time > 0 else 0
            acc_diff = abs(full_results['test_acc'] - exi_results['test_acc'])
            
            print(f"\n{'='*70}")
            print(f"STAGE {bucket_name} RESULTS")
            print("="*70)
            print(f"Full Retraining:")
            print(f"  Time: {full_time:.2f}s")
            print(f"  Test Acc: {full_results['test_acc']:.4f}")
            print(f"\nExiGCN:")
            print(f"  Time: {exi_time:.2f}s")
            print(f"  Test Acc: {exi_results['test_acc']:.4f}")
            print(f"\nSpeedup: {speedup:.2f}x")
            print(f"Acc Diff: {acc_diff:.4f}")
            
            # Record results
            cumulative_ratio = self.config['experiment']['initial_ratio'] + \
                             sum(self.config['experiment']['update_stages'][:stage_idx+1])
            
            self.results.append({
                'dataset': dataset_name,
                'run': run,
                'stage': bucket_name,
                'ratio': cumulative_ratio,
                'nodes': current_size,
                'method': 'Full',
                'train_time': full_time,
                'val_acc': full_results['val_acc'],
                'test_acc': full_results['test_acc'],
                'val_f1_micro': full_results['val_f1_micro'],
                'test_f1_micro': full_results['test_f1_micro']
            })
            
            self.results.append({
                'dataset': dataset_name,
                'run': run,
                'stage': bucket_name,
                'ratio': cumulative_ratio,
                'nodes': current_size,
                'method': 'ExiGCN',
                'train_time': exi_time,
                'val_acc': exi_results['val_acc'],
                'test_acc': exi_results['test_acc'],
                'val_f1_micro': exi_results['val_f1_micro'],
                'test_f1_micro': exi_results['test_f1_micro'],
                'speedup': speedup,
                'acc_diff': acc_diff
            })
    
    def _save_results(self, dataset_name: str):
        """Save results to CSV."""
        df = pd.DataFrame(self.results)
        
        os.makedirs('results/tables', exist_ok=True)
        
        # Use timestamp to avoid file lock issues
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f'results/tables/{dataset_name}_incremental_{timestamp}.csv'
        
        df.to_csv(output_path, index=False)
        print(f"\n✅ Results saved to {output_path}")
    
    def _print_summary(self, dataset_name: str):
        """Print summary of results."""
        df = pd.DataFrame(self.results)
        
        print("\n" + "="*70)
        print("EXPERIMENT SUMMARY")
        print("="*70)
        
        # Average results across runs
        summary = df.groupby(['stage', 'method']).agg({
            'train_time': ['mean', 'std'],
            'test_acc': ['mean', 'std'],
            'speedup': 'mean',
            'acc_diff': 'mean'
        }).reset_index()
        
        print("\n", summary.to_string())
        
        # Overall speedup
        exi_times = df[df['method'] == 'ExiGCN']['train_time'].values
        full_times = df[df['method'] == 'Full']['train_time'].values
        
        if len(exi_times) > 0 and len(full_times) > 0:
            avg_speedup = np.mean(full_times) / np.mean(exi_times)
            print(f"\nAverage Speedup: {avg_speedup:.2f}x")
            print(f"Average Accuracy Difference: {df['acc_diff'].mean():.4f}")


def main():
    """Main experiment entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run incremental experiment')
    parser.add_argument('--dataset', type=str, default='cora_full',
                       help='Dataset name')
    parser.add_argument('--config', type=str, default='config/cora_full.yaml',
                       help='Config file path')
    parser.add_argument('--num_runs', type=int, default=1,
                       help='Number of runs')
    parser.add_argument('--model', type=str, default=None,
                       choices=['exigcn', 'exigcn_lora'],
                       help='Model type: exigcn or exigcn_lora (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    
    args = parser.parse_args()
    
    # Run experiment
    experiment = IncrementalExperiment(config_path=args.config)
    
    # Override model type if specified
    if args.model:
        experiment.config['model']['type'] = args.model
        print(f"Model type overridden to: {args.model}")
    
    # Override epochs if specified
    if args.epochs:
        experiment.config['training']['epochs'] = args.epochs
        print(f"Epochs overridden to: {args.epochs}")
    
    # Override num_runs if specified
    if args.num_runs:
        experiment.config['experiment']['num_runs'] = args.num_runs
    
    experiment.run_experiment(dataset_name=args.dataset)


if __name__ == "__main__":
    main()
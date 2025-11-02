"""
Deletion experiment runner for ExiGCN - CORRECT VERSION WITH CACHING
Uses ExiGCNTrainer just like incremental experiments.
"""

import torch
import torch.nn as nn
import numpy as np
import yaml
import os
import pandas as pd
from typing import Dict, List, Tuple

from data.download import DatasetLoader
from data.preprocessor import DataPreprocessor
from data.graph_updater import GraphUpdater
from models.base_gcn import BaseGCN
from models.exigcn import ExiGCN
from train.trainer_full import FullRetrainingTrainer
from train.trainer_exi import ExiGCNTrainer
from utils.summary_writer import SummaryWriter


class DeletionExperiment:
    """Run deletion experiments comparing Full Retraining vs ExiGCN with caching."""
    
    def __init__(self, config_path: str = None):
        """Initialize experiment runner."""
        print(f"\n[DEBUG] Loading config from: {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        print(f"[DEBUG] Config loaded successfully")
        print(f"[DEBUG] Experiment num_runs: {self.config['experiment']['num_runs']}")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() and 
                                  self.config['device']['use_cuda'] else 'cpu')
        print(f"Device: {self.device}")
        
        self.results = []
        self.summary_writer = None
    
    def run_experiment(self, dataset_name: str = 'cora_full'):
        """Run full deletion experiment."""
        print("\n" + "="*70)
        print(f"DELETION EXPERIMENT: {dataset_name.upper()}")
        print("="*70)
        
        # Load dataset
        loader = DatasetLoader(root=self.config['data']['root'])
        adj, features, labels, train_mask, val_mask, test_mask = loader.load_dataset(dataset_name)
        
        print(f"\nDataset: {dataset_name}")
        print(f"  Nodes: {adj.size(0):,}")
        print(f"  Edges: {adj._nnz():,}")
        print(f"  Features: {features.size(1)}")
        print(f"  Classes: {labels.max().item() + 1}")
        
        # Initialize summary writer
        self.summary_writer = SummaryWriter(
            dataset_name=dataset_name,
            experiment_type='deletion'
        )
        self.summary_writer.write_configuration(self.config)
        
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
        
        # Create graph updater (start from 100%)
        updater = GraphUpdater(
            adj=adj,
            features=features,
            labels=labels,
            initial_ratio=1.0,  # Start from 100%!
            seed=self.config['experiment']['seed'] + run
        )
        
        # Create deletion scenario
        deletion_ratios = self.config['experiment'].get('deletion_stages', 
                                                        [0.02, 0.04, 0.06, 0.08, 0.10])
        deletion_stages = updater.create_deletion_scenario(
            deletion_ratios=deletion_ratios,
            strategy='stratified'
        )
        
        # Preprocess full graph
        preprocessor = DataPreprocessor()
        adj_norm, features_norm = preprocessor.preprocess(adj, features, self.device)
        labels = labels.to(self.device)
        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device)
        test_mask = test_mask.to(self.device)
        
        num_features = features.size(1)
        num_classes = labels.max().item() + 1
        
        # ================================================================
        # PHASE 1: Initial Training on FULL graph (100%)
        # ================================================================
        
        print("\n" + "="*70)
        print("PHASE 1: INITIAL TRAINING (100%)")
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
            adj_norm, features_norm, labels,
            train_mask, val_mask, test_mask
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
            retraining_lr=self.config['training'].get('retraining_lr', 
                                                     self.config['training']['learning_rate']),
            weight_decay=self.config['training']['weight_decay'],
            epochs=self.config['training']['epochs'],
            verbose=True
        )
        
        print("\n[ExiGCN] Initial Training with caching...")
        initial_results_exi = trainer_exi.train_initial(
            adj_norm, features_norm, labels,
            train_mask, val_mask, test_mask
        )
        initial_time_exi = trainer_exi.get_training_time()
        
        print(f"\n✅ Initial training complete")
        print(f"   Full - Test Acc: {initial_results_full['test_acc']:.4f}, Time: {initial_time_full:.2f}s")
        print(f"   ExiGCN - Test Acc: {initial_results_exi['test_acc']:.4f}, Time: {initial_time_exi:.2f}s")
        
        # Record initial results
        self.results.append({
            'dataset': dataset_name,
            'run': run,
            'stage': 'initial',
            'ratio': 1.0,
            'nodes': adj.size(0),
            'method': 'Full',
            'train_time': initial_time_full,
            'val_acc': initial_results_full['val_acc'],
            'test_acc': initial_results_full['test_acc'],
            'val_f1_micro': initial_results_full.get('val_f1_micro', 0),
            'test_f1_micro': initial_results_full.get('test_f1_micro', 0),
            'test_f1_macro': initial_results_full.get('test_f1_macro', 0),
            'test_f1_weighted': initial_results_full.get('test_f1_weighted', 0)
        })
        
        self.results.append({
            'dataset': dataset_name,
            'run': run,
            'stage': 'initial',
            'ratio': 1.0,
            'nodes': adj.size(0),
            'method': 'ExiGCN',
            'train_time': initial_time_exi,
            'val_acc': initial_results_exi['val_acc'],
            'test_acc': initial_results_exi['test_acc'],
            'val_f1_micro': initial_results_exi.get('val_f1_micro', 0),
            'test_f1_micro': initial_results_exi.get('test_f1_micro', 0),
            'test_f1_macro': initial_results_exi.get('test_f1_macro', 0),
            'test_f1_weighted': initial_results_exi.get('test_f1_weighted', 0)
        })
        
        # ================================================================
        # PHASE 2: Deletion Stages (with caching!)
        # ================================================================
        
        for stage_name in sorted(deletion_stages.keys()):
            ratio = deletion_stages[stage_name]['ratio']
            remaining_nodes = deletion_stages[stage_name]['remaining_nodes']
            
            print(f"\n{'='*70}")
            print(f"PHASE 2.{ord(stage_name)-64}: DELETION SCENARIO {stage_name} (-{ratio*100:.0f}%)")
            print("="*70)
            print(f"Remaining nodes: {len(remaining_nodes):,}")
            
            # Get deletion stage graph
            adj_del, features_del, labels_del, train_mask_del, val_mask_del, test_mask_del = \
                updater.get_deletion_stage(stage_name)
            
            # Preprocess
            adj_del_norm, features_del_norm = preprocessor.preprocess(
                adj_del, features_del, self.device
            )
            labels_del = labels_del.to(self.device)
            train_mask_del = train_mask_del.to(self.device)
            val_mask_del = val_mask_del.to(self.device)
            test_mask_del = test_mask_del.to(self.device)
            
            print(f"Graph after deletion: {adj_del.size(0):,} nodes, {adj_del._nnz():,} edges")
            
            # Full Retraining (from scratch)
            print("\n[Full Retraining] Retraining from scratch...")
            model_full_new = BaseGCN(
                num_features=num_features,
                hidden_dim=self.config['model']['hidden_dim'],
                num_classes=num_classes,
                num_layers=self.config['model']['num_layers'],
                dropout=self.config['model']['dropout']
            )
            
            trainer_full_new = FullRetrainingTrainer(
                model=model_full_new,
                device=self.device,
                learning_rate=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay'],
                epochs=self.config['training']['epochs'],
                verbose=True
            )
            
            full_metrics = trainer_full_new.train(
                adj_del_norm, features_del_norm, labels_del,
                train_mask_del, val_mask_del, test_mask_del
            )
            full_time = trainer_full_new.get_training_time()
            
            # ExiGCN (with caching!)
            print("\n[ExiGCN] Efficient retraining with cached activations...")
            print("✅ Using cached Z and H from initial training")
            print("✅ Reusing W_core baseline")
            
            exi_metrics = trainer_exi.retrain(
                adj_del_norm, features_del_norm, labels_del,
                train_mask_del, val_mask_del, test_mask_del
            )
            exi_time = trainer_exi.get_training_time()
            
            # Compute metrics
            speedup = full_time / exi_time if exi_time > 0 else 1.0
            acc_diff = full_metrics['test_acc'] - exi_metrics['test_acc']
            
            print(f"\n{'='*70}")
            print(f"STAGE {stage_name} RESULTS")
            print("="*70)
            print(f"Full Retraining:")
            print(f"  Time: {full_time:.2f}s")
            print(f"  Test Acc: {full_metrics['test_acc']:.4f}")
            print(f"\nExiGCN:")
            print(f"  Time: {exi_time:.2f}s")
            print(f"  Test Acc: {exi_metrics['test_acc']:.4f}")
            print(f"\nSpeedup: {speedup:.2f}x")
            print(f"Acc Diff: {acc_diff:.4f}")
            
            # Store results
            self.results.append({
                'dataset': dataset_name,
                'run': run,
                'stage': stage_name,
                'ratio': 1.0 - ratio,
                'nodes': adj_del.size(0),
                'method': 'Full',
                'train_time': full_time,
                'val_acc': full_metrics['val_acc'],
                'test_acc': full_metrics['test_acc'],
                'val_f1_micro': full_metrics.get('val_f1_micro', 0),
                'test_f1_micro': full_metrics.get('test_f1_micro', 0),
                'test_f1_macro': full_metrics.get('test_f1_macro', 0),
                'test_f1_weighted': full_metrics.get('test_f1_weighted', 0)
            })
            
            self.results.append({
                'dataset': dataset_name,
                'run': run,
                'stage': stage_name,
                'ratio': 1.0 - ratio,
                'nodes': adj_del.size(0),
                'method': 'ExiGCN',
                'train_time': exi_time,
                'val_acc': exi_metrics['val_acc'],
                'test_acc': exi_metrics['test_acc'],
                'val_f1_micro': exi_metrics.get('val_f1_micro', 0),
                'test_f1_micro': exi_metrics.get('test_f1_micro', 0),
                'test_f1_macro': exi_metrics.get('test_f1_macro', 0),
                'test_f1_weighted': exi_metrics.get('test_f1_weighted', 0),
                'speedup': speedup,
                'acc_diff': acc_diff
            })
    
    def _save_results(self, dataset_name: str):
        """Save results to CSV."""
        df = pd.DataFrame(self.results)
        os.makedirs('results/tables', exist_ok=True)
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f'results/tables/{dataset_name}_deletion_{timestamp}.csv'
        
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
            'test_f1_macro': ['mean', 'std'],
            'speedup': 'mean',
            'acc_diff': 'mean'
        }).reset_index()
        
        print("\n", summary.to_string())
        
        # Overall speedup
        exi_data = df[df['method'] == 'ExiGCN']
        if 'speedup' in exi_data.columns:
            speedup_values = exi_data['speedup'].dropna()
            if len(speedup_values) > 0:
                print(f"\nAverage Speedup: {speedup_values.mean():.2f}x")
                print(f"Average Accuracy Difference: {exi_data['acc_diff'].mean():.4f}")
        
        # Generate paper-ready table
        self._print_paper_table(df)
        
        # Write to summary file
        if self.summary_writer is not None:
            self.summary_writer.write_paper_format(df)
            
            speedup_values = df[df['method'] == 'ExiGCN']['speedup'].dropna()
            acc_diff_values = df[df['method'] == 'ExiGCN']['acc_diff'].dropna()
            
            if len(speedup_values) > 0:
                self.summary_writer.write_overall_results(
                    avg_speedup=speedup_values.mean(),
                    avg_acc_diff=acc_diff_values.mean()
                )
            
            self.summary_writer.close()
    
    def _print_paper_table(self, df: pd.DataFrame):
        """Print paper-ready table."""
        print("\n" + "="*70)
        print("PAPER-READY TABLE (Deletion Scenario)")
        print("="*70)
        
        df_stages = df[df['stage'] != 'initial'].copy()
        stages = sorted(df_stages['stage'].unique())
        
        print(f"\n{'Stage':<8} {'Method':<8} {'Remaining %':<12} {'Test Acc (%)':<20} {'Speedup':<10}")
        print("-" * 70)
        
        for stage in stages:
            stage_data = df_stages[df_stages['stage'] == stage]
            
            for method in ['Full', 'ExiGCN']:
                method_data = stage_data[stage_data['method'] == method]
                
                if len(method_data) == 0:
                    continue
                
                ratio = method_data['ratio'].mean() * 100
                acc_mean = method_data['test_acc'].mean() * 100
                acc_std = method_data['test_acc'].std() * 100
                
                if pd.isna(acc_std) or acc_std == 0:
                    acc_str = f"{acc_mean:.2f}"
                else:
                    acc_str = f"{acc_mean:.2f}±{acc_std:.2f}"
                
                if method == 'ExiGCN':
                    speedup = method_data['speedup'].mean()
                    speedup_str = f"{speedup:.2f}x" if not pd.isna(speedup) else "-"
                else:
                    speedup_str = "-"
                
                print(f"{stage:<8} {method:<8} {ratio:<12.0f} {acc_str:<20} {speedup_str:<10}")
        
        print("\n" + "="*70)


def main():
    """Main experiment entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run deletion experiment')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Dataset name (if not specified, read from config)')
    parser.add_argument('--config', type=str, default='configs/cora_full_deletion.yaml',
                       help='Config file path')
    parser.add_argument('--num_runs', type=int, default=None,
                       help='Number of runs (overrides config)')
    
    args = parser.parse_args()
    
    # Run experiment
    experiment = DeletionExperiment(config_path=args.config)
    
    # Get dataset name from config if not specified
    if args.dataset is None:
        args.dataset = experiment.config.get('dataset', {}).get('name', 'cora_full')
        print(f"Dataset name from config: {args.dataset}")
    
    # Override num_runs if specified
    if args.num_runs is not None:
        experiment.config['experiment']['num_runs'] = args.num_runs
        print(f"Num runs overridden to: {args.num_runs}")
    
    experiment.run_experiment(dataset_name=args.dataset)


if __name__ == "__main__":
    main()
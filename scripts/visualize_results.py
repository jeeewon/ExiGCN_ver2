"""
Visualization script for ExiGCN experiment results.
Generates publication-quality plots from CSV results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import os
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12

def load_latest_results(dataset_name, exp_type='incremental'):
    """Load latest CSV results for a dataset."""
    pattern = f'results/tables/{dataset_name}_{exp_type}_*.csv'
    files = glob.glob(pattern)
    
    if not files:
        print(f"Warning: No results found for {dataset_name}")
        return None
    
    # Get latest file
    latest = max(files, key=os.path.getctime)
    print(f"Loading: {latest}")
    
    df = pd.read_csv(latest)
    return df

def create_speedup_comparison(datasets):
    """Create speedup comparison bar plot across datasets."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    speedups = {}
    for dataset in datasets:
        df = load_latest_results(dataset)
        if df is not None:
            # Get ExiGCN speedups (exclude initial stage)
            exi_data = df[(df['method'] == 'ExiGCN') & (df['stage'] != 'initial')]
            if 'speedup' in exi_data.columns:
                speedups[dataset] = exi_data['speedup'].mean()
    
    if not speedups:
        print("No speedup data found!")
        return
    
    # Create bar plot
    x = np.arange(len(speedups))
    bars = ax.bar(x, speedups.values(), color=['#2ecc71', '#3498db', '#e74c3c', '#9b59b6'])
    
    ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Speedup (×)', fontsize=14, fontweight='bold')
    ax.set_title('ExiGCN Speedup vs Full Retraining', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace('_', '\n').title() for d in speedups.keys()])
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, speedups.values())):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}×',
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Add horizontal line at y=1
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='No speedup')
    
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    os.makedirs('results/figures', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f'results/figures/speedup_comparison_{timestamp}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    plt.close()

def create_accuracy_comparison(datasets):
    """Create accuracy comparison plot."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    all_data = []
    for dataset in datasets:
        df = load_latest_results(dataset)
        if df is not None:
            df_stages = df[df['stage'] != 'initial'].copy()
            df_stages['dataset'] = dataset.replace('_', ' ').title()
            all_data.append(df_stages)
    
    if not all_data:
        print("No accuracy data found!")
        return
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Plot 1: Test Accuracy
    for dataset in combined['dataset'].unique():
        data = combined[combined['dataset'] == dataset]
        
        full_data = data[data['method'] == 'Full']
        exi_data = data[data['method'] == 'ExiGCN']
        
        stages = sorted(full_data['stage'].unique())
        x = np.arange(len(stages))
        
        axes[0].plot(x, full_data.groupby('stage')['test_acc'].mean() * 100, 
                    'o-', label=f'{dataset} (Full)', linewidth=2, markersize=8)
        axes[0].plot(x, exi_data.groupby('stage')['test_acc'].mean() * 100, 
                    's--', label=f'{dataset} (ExiGCN)', linewidth=2, markersize=8)
    
    axes[0].set_xlabel('Incremental Stage', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
    axes[0].set_title('Test Accuracy Across Stages', fontsize=16, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(stages)
    axes[0].legend(loc='best', ncol=2)
    axes[0].grid(alpha=0.3)
    
    # Plot 2: Accuracy Difference (Full - ExiGCN)
    for dataset in combined['dataset'].unique():
        data = combined[combined['dataset'] == dataset]
        exi_data = data[data['method'] == 'ExiGCN']
        
        if 'acc_diff' in exi_data.columns:
            stages = sorted(exi_data['stage'].unique())
            x = np.arange(len(stages))
            
            acc_diffs = exi_data.groupby('stage')['acc_diff'].mean() * 100
            axes[1].plot(x, acc_diffs, 'o-', label=dataset, linewidth=2, markersize=8)
    
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Incremental Stage', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Accuracy Difference (%)', fontsize=14, fontweight='bold')
    axes[1].set_title('Accuracy Gap (Full - ExiGCN)', fontsize=16, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(stages)
    axes[1].legend(loc='best')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f'results/figures/accuracy_comparison_{timestamp}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    plt.close()

def create_training_time_comparison(datasets):
    """Create training time comparison plot."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    all_data = []
    for dataset in datasets:
        df = load_latest_results(dataset)
        if df is not None:
            df_stages = df[df['stage'] != 'initial'].copy()
            df_stages['dataset'] = dataset.replace('_', ' ').title()
            all_data.append(df_stages)
    
    if not all_data:
        print("No training time data found!")
        return
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Group by dataset and method
    grouped = combined.groupby(['dataset', 'method'])['train_time'].mean().reset_index()
    
    # Create grouped bar plot
    datasets_list = grouped['dataset'].unique()
    x = np.arange(len(datasets_list))
    width = 0.35
    
    full_times = [grouped[(grouped['dataset'] == d) & (grouped['method'] == 'Full')]['train_time'].values[0] 
                  if len(grouped[(grouped['dataset'] == d) & (grouped['method'] == 'Full')]) > 0 else 0
                  for d in datasets_list]
    
    exi_times = [grouped[(grouped['dataset'] == d) & (grouped['method'] == 'ExiGCN')]['train_time'].values[0]
                 if len(grouped[(grouped['dataset'] == d) & (grouped['method'] == 'ExiGCN')]) > 0 else 0
                 for d in datasets_list]
    
    bars1 = ax.bar(x - width/2, full_times, width, label='Full Retraining', color='#e74c3c')
    bars2 = ax.bar(x + width/2, exi_times, width, label='ExiGCN', color='#2ecc71')
    
    ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Training Time (s)', fontsize=14, fontweight='bold')
    ax.set_title('Training Time Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace(' ', '\n') for d in datasets_list])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}s',
                       ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f'results/figures/training_time_comparison_{timestamp}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    plt.close()

def create_comprehensive_plots(datasets):
    """Create all visualization plots."""
    print("\n" + "="*70)
    print("Generating Visualizations")
    print("="*70)
    
    print("\n1. Speedup Comparison...")
    create_speedup_comparison(datasets)
    
    print("\n2. Accuracy Comparison...")
    create_accuracy_comparison(datasets)
    
    print("\n3. Training Time Comparison...")
    create_training_time_comparison(datasets)
    
    print("\n" + "="*70)
    print("All visualizations saved to: results/figures/")
    print("="*70)

if __name__ == "__main__":
    # Datasets to visualize
    datasets = ['cora_full', 'reddit', 'amazon_computer', 'ogbn_arxiv']
    
    create_comprehensive_plots(datasets)
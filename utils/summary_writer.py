"""
Summary extractor for experiment results.
Saves clean, readable summaries separate from full logs.
"""

import os
from datetime import datetime
from typing import Dict, List


class SummaryWriter:
    """Write clean experiment summaries."""
    
    def __init__(self, dataset_name: str, experiment_type: str = 'incremental'):
        """
        Initialize summary writer.
        
        Args:
            dataset_name: Name of dataset
            experiment_type: 'incremental' or 'deletion'
        """
        self.dataset_name = dataset_name
        self.experiment_type = experiment_type
        
        # Create directories
        os.makedirs('results/summaries', exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.summary_path = f'results/summaries/{dataset_name}_{experiment_type}_{timestamp}_summary.txt'
        
        # Open file
        self.file = open(self.summary_path, 'w', encoding='utf-8')
        
        # Write header
        self._write_header()
    
    def _write_header(self):
        """Write file header."""
        self.write("="*70)
        self.write(f"EXPERIMENT SUMMARY: {self.dataset_name.upper()}")
        self.write(f"Type: {self.experiment_type.upper()}")
        self.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.write("="*70)
        self.write("")
    
    def write(self, text: str = ""):
        """Write line to summary file."""
        self.file.write(text + "\n")
        self.file.flush()  # Ensure immediate write
    
    def write_section(self, title: str):
        """Write section header."""
        self.write("")
        self.write("="*70)
        self.write(title)
        self.write("="*70)
        self.write("")
    
    def write_stage_result(self, stage: str, full_time: float, full_acc: float,
                          exi_time: float, exi_acc: float, speedup: float):
        """Write single stage result."""
        self.write(f"STAGE {stage} RESULTS:")
        self.write(f"  Full Retraining: {full_time:.2f}s, Acc {full_acc*100:.2f}%")
        self.write(f"  ExiGCN:          {exi_time:.2f}s, Acc {exi_acc*100:.2f}%")
        self.write(f"  Speedup:         {speedup:.2f}x")
        self.write(f"  Acc Diff:        {abs(full_acc - exi_acc)*100:.2f}%")
        self.write("")
    
    def write_overall_results(self, avg_speedup: float, avg_acc_diff: float):
        """Write overall experiment results."""
        self.write_section("OVERALL RESULTS")
        self.write(f"Average Speedup:     {avg_speedup:.2f}x")
        self.write(f"Average Acc Diff:    {avg_acc_diff:.4f} ({avg_acc_diff*100:.2f}%)")
        self.write("")
    
    def write_latex_table(self, latex_code: str):
        """Write LaTeX table."""
        self.write_section("LATEX TABLE (Copy to Paper)")
        self.write(latex_code)
        self.write("")
    
    def write_plain_table(self, table_lines: List[str]):
        """Write plain text table."""
        self.write_section("PLAIN TEXT TABLE")
        for line in table_lines:
            self.write(line)
        self.write("")
    
    def write_paper_format(self, df):
        """Write paper-ready format with mean±std."""
        import pandas as pd
        
        self.write_section("PAPER-READY FORMAT (Mean ± Std)")
        
        # Filter stages
        df_stages = df[df['stage'] != 'initial'].copy()
        
        if len(df_stages) == 0:
            self.write("No stage data available.")
            return
        
        stages = sorted(df_stages['stage'].unique())
        
        # Header
        header = f"{'Stage':<8} {'Method':<10} {'Test Acc (%)':<25} {'Test F1-Macro (%)':<25} {'Speedup':<10}"
        self.write(header)
        self.write("-" * len(header))
        
        # Data rows
        for stage in stages:
            stage_data = df_stages[df_stages['stage'] == stage]
            
            for method in ['Full', 'ExiGCN']:
                method_data = stage_data[stage_data['method'] == method]
                
                if len(method_data) == 0:
                    continue
                
                # Accuracy
                acc_mean = method_data['test_acc'].mean() * 100
                acc_std = method_data['test_acc'].std() * 100
                
                if pd.isna(acc_std) or acc_std == 0 or len(method_data) == 1:
                    acc_str = f"{acc_mean:.2f}"
                else:
                    acc_str = f"{acc_mean:.2f} ± {acc_std:.2f}"
                
                # F1
                if 'test_f1_macro' in method_data.columns:
                    f1_mean = method_data['test_f1_macro'].mean() * 100
                    f1_std = method_data['test_f1_macro'].std() * 100
                    
                    if pd.isna(f1_std) or f1_std == 0 or len(method_data) == 1:
                        f1_str = f"{f1_mean:.2f}"
                    else:
                        f1_str = f"{f1_mean:.2f} ± {f1_std:.2f}"
                else:
                    f1_str = "N/A"
                
                # Speedup
                speedup = method_data['speedup'].mean()
                if method == 'ExiGCN' and not pd.isna(speedup):
                    speedup_str = f"{speedup:.2f}x"
                else:
                    speedup_str = "-"
                
                row = f"{stage:<8} {method:<10} {acc_str:<25} {f1_str:<25} {speedup_str:<10}"
                self.write(row)
        
        self.write("")
        
        # Add statistics
        if 'speedup' in df_stages.columns:
            speedup_values = df_stages[df_stages['method'] == 'ExiGCN']['speedup'].dropna()
            if len(speedup_values) > 0:
                self.write(f"Average Speedup: {speedup_values.mean():.2f}x (± {speedup_values.std():.2f})")
        
        if 'acc_diff' in df_stages.columns:
            acc_diff_values = df_stages[df_stages['method'] == 'ExiGCN']['acc_diff'].dropna()
            if len(acc_diff_values) > 0:
                self.write(f"Average Acc Drop: {acc_diff_values.mean()*100:.2f}% (± {acc_diff_values.std()*100:.2f}%)")
        
        self.write("")
    
    def write_configuration(self, config: Dict):
        """Write experiment configuration."""
        self.write_section("EXPERIMENT CONFIGURATION")
        
        self.write("Model:")
        self.write(f"  Type: {config.get('model', {}).get('type', 'N/A')}")
        self.write(f"  Layers: {config.get('model', {}).get('num_layers', 'N/A')}")
        self.write(f"  Hidden: {config.get('model', {}).get('hidden_dim', 'N/A')}")
        self.write(f"  Dropout: {config.get('model', {}).get('dropout', 'N/A')}")
        self.write("")
        
        self.write("Training:")
        self.write(f"  Epochs: {config.get('training', {}).get('epochs', 'N/A')}")
        self.write(f"  LR: {config.get('training', {}).get('learning_rate', 'N/A')}")
        self.write(f"  Retraining LR: {config.get('training', {}).get('retraining_lr', 'N/A')}")
        self.write(f"  Weight Decay: {config.get('training', {}).get('weight_decay', 'N/A')}")
        self.write(f"  Patience: {config.get('training', {}).get('patience', 'N/A')}")
        self.write("")
        
        self.write("Experiment:")
        self.write(f"  Num Runs: {config.get('experiment', {}).get('num_runs', 'N/A')}")
        self.write(f"  Seed: {config.get('experiment', {}).get('seed', 'N/A')}")
        
        if self.experiment_type == 'incremental':
            stages = config.get('experiment', {}).get('update_stages', [])
            self.write(f"  Update Stages: {stages}")
        elif self.experiment_type == 'deletion':
            stages = config.get('experiment', {}).get('deletion_stages', [])
            self.write(f"  Deletion Stages: {stages}")
        
        self.write("")
    
    def close(self):
        """Close summary file."""
        self.write("")
        self.write("="*70)
        self.write(f"Summary saved to: {self.summary_path}")
        self.write("="*70)
        self.file.close()
        print(f"\n✅ Summary saved to: {self.summary_path}")


def extract_summary_from_log(log_path: str, output_path: str = None):
    """
    Extract summary from full log file.
    
    Args:
        log_path: Path to full log file
        output_path: Path to save summary (optional)
    """
    if output_path is None:
        output_path = log_path.replace('.log', '_summary.txt')
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    # Extract important sections
    summary_lines = []
    in_summary = False
    in_table = False
    
    for line in lines:
        # Start capturing at summary sections
        if 'EXPERIMENT SUMMARY' in line or 'PAPER-READY TABLE' in line or 'PLAIN TEXT TABLE' in line:
            in_summary = True
        
        # Capture summary content
        if in_summary:
            summary_lines.append(line)
            
            # Stop at certain markers
            if line.strip().endswith('='*70) and len(summary_lines) > 10:
                in_summary = False
    
    # Write extracted summary
    with open(output_path, 'w') as f:
        f.writelines(summary_lines)
    
    print(f"✅ Summary extracted to: {output_path}")


if __name__ == "__main__":
    print("SummaryWriter utility - use in experiments")
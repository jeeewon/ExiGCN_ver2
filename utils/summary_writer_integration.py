"""
Example: How to integrate SummaryWriter into experiments.

This shows where to add summary writer calls in run_incremental.py or run_deletion.py.
"""

# ============================================================================
# STEP 1: Import at the top of the file
# ============================================================================

from utils.summary_writer import SummaryWriter


# ============================================================================
# STEP 2: Initialize in __init__ method
# ============================================================================

class IncrementalExperiment:
    def __init__(self, config_path: str = None):
        # ... existing code ...
        
        self.results = []
        # Add this:
        self.summary_writer = None  # Will be created when experiment starts
    

# ============================================================================
# STEP 3: Create writer when experiment starts
# ============================================================================

    def run_experiment(self, dataset_name: str = 'cora_full'):
        # At the beginning of run_experiment:
        
        # Create summary writer
        self.summary_writer = SummaryWriter(
            dataset_name=dataset_name,
            experiment_type='incremental'  # or 'deletion'
        )
        
        # Write configuration
        self.summary_writer.write_configuration(self.config)
        
        # ... rest of experiment code ...
        
        # At the end, before _print_summary:
        self._save_results(dataset_name)
        self._print_summary(dataset_name)  # This will use summary writer


# ============================================================================
# STEP 4: Modify _print_summary to use summary writer
# ============================================================================

    def _print_summary(self, dataset_name: str):
        """Print summary of results."""
        df = pd.DataFrame(self.results)
        
        # ... existing summary printing code ...
        
        # ADD THIS at the end:
        if self.summary_writer is not None:
            # Write paper-ready format
            self.summary_writer.write_paper_format(df)
            
            # Write LaTeX table (if you have the code)
            # self.summary_writer.write_latex_table(latex_code)
            
            # Write plain table (if you have the lines)
            # self.summary_writer.write_plain_table(table_lines)
            
            # Write overall results
            if 'speedup' in df.columns:
                speedup_values = df[df['method'] == 'ExiGCN']['speedup'].dropna()
                acc_diff_values = df[df['method'] == 'ExiGCN']['acc_diff'].dropna()
                
                if len(speedup_values) > 0:
                    self.summary_writer.write_overall_results(
                        avg_speedup=speedup_values.mean(),
                        avg_acc_diff=acc_diff_values.mean()
                    )
            
            # Close summary file
            self.summary_writer.close()


# ============================================================================
# STEP 5: Optionally write stage results in real-time
# ============================================================================

    def _run_single_experiment(self, ...):
        # ... after each stage ...
        
        # Optionally write stage result immediately
        if self.summary_writer is not None:
            self.summary_writer.write_stage_result(
                stage=stage_name,
                full_time=full_time,
                full_acc=full_results['test_acc'],
                exi_time=exi_time,
                exi_acc=exi_results['test_acc'],
                speedup=speedup
            )


# ============================================================================
# COMPLETE EXAMPLE: Modified _print_summary method
# ============================================================================

def _print_summary_with_summary_writer(self, dataset_name: str):
    """Print summary of results WITH summary writer integration."""
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
    exi_times = df[df['method'] == 'ExiGCN']['train_time'].values
    full_times = df[df['method'] == 'Full']['train_time'].values
    
    if len(exi_times) > 0 and len(full_times) > 0:
        avg_speedup = np.mean(full_times) / np.mean(exi_times)
        print(f"\nAverage Speedup: {avg_speedup:.2f}x")
        print(f"Average Accuracy Difference: {df['acc_diff'].mean():.4f}")
    
    # Generate paper-ready table (existing code)
    self._print_paper_table(df)
    
    # ========================================================================
    # ADD THIS: Write to summary file
    # ========================================================================
    if self.summary_writer is not None:
        # Write paper-ready format
        self.summary_writer.write_paper_format(df)
        
        # Write overall results
        speedup_values = df[df['method'] == 'ExiGCN']['speedup'].dropna()
        acc_diff_values = df[df['method'] == 'ExiGCN']['acc_diff'].dropna()
        
        if len(speedup_values) > 0:
            self.summary_writer.write_overall_results(
                avg_speedup=speedup_values.mean(),
                avg_acc_diff=acc_diff_values.mean()
            )
        
        # Close summary file
        self.summary_writer.close()


# ============================================================================
# USAGE
# ============================================================================

if __name__ == "__main__":
    # Just run as normal, summary will be automatically generated
    # python -m experiments.run_incremental --config configs/cora_full.yaml
    pass
#!/usr/bin/env python3
"""
Complete Pipeline Runner with Logging
Runs the full pipeline and all evaluation scripts, logging everything to output.log
"""

import sys
import subprocess
from pathlib import Path
import time
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_LOG = PROJECT_ROOT / "output.log"

def log_message(message, log_file=OUTPUT_LOG):
    """Write message to both console and log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}\n"
    print(message, flush=True)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_entry)
        f.flush()

def run_command(cmd, description, log_file=OUTPUT_LOG):
    """Run a command and log output."""
    log_message(f"\n{'='*60}")
    log_message(f" {description}")
    log_message(f"{'='*60}")
    log_message(f"Command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    log_message("")
    
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            result = subprocess.run(
                cmd if isinstance(cmd, list) else cmd.split(),
                cwd=PROJECT_ROOT,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                check=False
            )
        
        if result.returncode == 0:
            log_message(f" {description} completed successfully!")
            return True
        else:
            log_message(f"  {description} completed with exit code {result.returncode}")
            return False
    except Exception as e:
        log_message(f" Error running {description}: {e}")
        import traceback
        with open(log_file, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        return False

def main():
    """Main execution function."""
    # Initialize log file
    with open(OUTPUT_LOG, 'w', encoding='utf-8') as f:
        f.write(f"Complete Pipeline Run - Started at {datetime.now()}\n")
        f.write("="*60 + "\n\n")
    
    log_message(" Starting Complete Pipeline and Evaluation")
    log_message(f" All output will be logged to: {OUTPUT_LOG}")
    log_message(f"⏰ Start time: {datetime.now()}")
    
    # Step 1: Run Full Pipeline
    log_message("\n" + "="*60)
    log_message("STEP 1: Running Full Pipeline (Phases 1-6)")
    log_message("="*60)
    run_command(
        [sys.executable, "run_full_pipeline.py"],
        "Full Pipeline (Phases 1-6)"
    )
    
    # Step 2: Run All Scripts
    log_message("\n" + "="*60)
    log_message("STEP 2: Running All Scripts")
    log_message("="*60)
    
    scripts_to_run = [
        ("scripts/generate_report_figures.py", "Generate Report Figures"),
        ("scripts/create_additional_figures.py", "Create Additional Figures"),
        ("scripts/analyze_ic_deep.py", "Deep IC Analysis"),
        ("scripts/run_baseline_comparison.py", "Baseline Model Comparison"),
        ("scripts/run_improved_ablation.py", "Improved Ablation Study"),
        ("scripts/run_marl_ablation.py", "MARL Ablation Study"),
        ("scripts/experiment_lookahead_horizons.py", "Lookahead Horizon Experiment"),
        ("scripts/experiment_graph_sparsification.py", "Graph Sparsification Experiment"),
        ("scripts/experiment_robustness_checks.py", "Robustness Checks Experiment"),
    ]
    
    for script_path, description in scripts_to_run:
        script_full_path = PROJECT_ROOT / script_path
        if script_full_path.exists():
            run_command(
                [sys.executable, str(script_full_path)],
                description
            )
            time.sleep(1)  # Small delay between scripts
        else:
            log_message(f"  Script not found: {script_path}")
    
    # Step 3: Run All Evaluation Modules
    log_message("\n" + "="*60)
    log_message("STEP 3: Running All Evaluation Modules")
    log_message("="*60)
    
    evaluation_modules = [
        ("src.evaluation.evaluation", "Main Evaluation Pipeline"),
        ("src.evaluation.ablation", "Ablation Studies"),
        ("src.evaluation.enhanced_evaluation", "Enhanced Evaluation"),
        ("src.evaluation.enhanced_ablation", "Enhanced Ablation"),
        ("src.evaluation.deep_analysis", "Deep Analysis"),
        ("src.evaluation.statistical_tests", "Statistical Tests"),
        ("src.evaluation.visualization", "Visualization"),
    ]
    
    for module_path, description in evaluation_modules:
        run_command(
            [sys.executable, "-m", module_path],
            description
        )
        time.sleep(1)  # Small delay between modules
    
    # Step 4: Run Enhancement Modules
    log_message("\n" + "="*60)
    log_message("STEP 4: Running Enhancement Modules")
    log_message("="*60)
    
    enhancement_modules = [
        ("src.evaluation.enhancements.edge_importance", "Edge Importance Analysis"),
        ("src.evaluation.enhancements.cross_period_validation", "Cross-Period Validation"),
        ("src.evaluation.enhancements.multi_agent_analysis", "Multi-Agent Analysis"),
        ("src.evaluation.enhancements.sensitivity_analysis", "Sensitivity Analysis"),
        ("src.evaluation.enhancements.failure_analysis", "Failure Analysis"),
    ]
    
    for module_path, description in enhancement_modules:
        run_command(
            [sys.executable, "-m", module_path],
            description
        )
        time.sleep(1)  # Small delay between modules
    
    # Final Summary
    log_message("\n" + "="*60)
    log_message(" Complete Pipeline and Evaluation Finished!")
    log_message("="*60)
    log_message(f" All output has been logged to: {OUTPUT_LOG}")
    log_message(f"⏰ Completed at: {datetime.now()}")
    
    # Show log file size
    if OUTPUT_LOG.exists():
        size_mb = OUTPUT_LOG.stat().st_size / (1024 * 1024)
        log_message(f" Log file size: {size_mb:.2f} MB")
        log_message(f" Log file lines: {sum(1 for _ in open(OUTPUT_LOG))}")

if __name__ == "__main__":
    main()


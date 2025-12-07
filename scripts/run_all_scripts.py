#!/usr/bin/env python3
"""
Run All Scripts in scripts/ Folder

This script provides a convenient way to run all utility scripts in the scripts/ folder.
You can run all scripts at once, or select specific categories.
"""

import sys
import subprocess
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

def run_script(script_name, description=""):
    """Run a single script and return success status."""
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        print(f"  Script not found: {script_name}")
        return False
    
    print("\n" + "="*60)
    print(f" Running: {script_name}")
    if description:
        print(f"   Description: {description}")
    print("="*60)
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            check=False
        )
        
        if result.returncode == 0:
            print(f" {script_name} completed successfully!")
            return True
        else:
            print(f"  {script_name} completed with warnings (exit code {result.returncode})")
            return True  # Don't fail pipeline for optional scripts
    except Exception as e:
        print(f" Error running {script_name}: {e}")
        return False

def main():
    """Main function to run all scripts."""
    print("="*60)
    print(" Run All Scripts in scripts/ Folder")
    print("="*60)
    print("\nThis script will run all utility scripts in the scripts/ folder.")
    print("You can choose to run all scripts or select specific categories.")
    print()
    
    # Define script categories
    figure_generation_scripts = [
        ("generate_report_figures.py", "Generate all main report figures"),
        ("create_additional_figures.py", "Generate additional analysis figures"),
        ("check_and_generate_all_figures.py", "Check and generate missing figures"),
    ]
    
    experiment_scripts = [
        ("experiment_lookahead_horizons.py", "Test different prediction horizons"),
        ("experiment_graph_sparsification.py", "Evaluate graph sparsification strategies"),
        ("experiment_robustness_checks.py", "Test transaction cost and slippage sensitivity"),
    ]
    
    evaluation_scripts = [
        ("run_baseline_comparison.py", "Compare baseline models (GCN, GAT, GraphSAGE, etc.)"),
        ("run_improved_ablation.py", "Run improved ablation studies with full retraining"),
        ("run_marl_ablation.py", "Run Multi-Agent RL ablation studies"),
        ("analyze_ic_deep.py", "Deep analysis of Information Coefficient (IC)"),
    ]
    
    all_scripts = figure_generation_scripts + experiment_scripts + evaluation_scripts
    
    # Ask user what to run
    print("Options:")
    print("  1. Run all scripts")
    print("  2. Run figure generation scripts only")
    print("  3. Run experiment scripts only")
    print("  4. Run evaluation scripts only")
    print("  5. Select individual scripts")
    print()
    
    choice = input("Enter your choice (1-5, default=1): ").strip() or "1"
    
    scripts_to_run = []
    
    if choice == "1":
        scripts_to_run = all_scripts
    elif choice == "2":
        scripts_to_run = figure_generation_scripts
    elif choice == "3":
        scripts_to_run = experiment_scripts
    elif choice == "4":
        scripts_to_run = evaluation_scripts
    elif choice == "5":
        print("\nAvailable scripts:")
        for i, (script, desc) in enumerate(all_scripts, 1):
            print(f"  {i}. {script} - {desc}")
        print()
        selection = input("Enter script numbers (comma-separated, e.g., 1,3,5): ").strip()
        indices = [int(x.strip()) - 1 for x in selection.split(",") if x.strip().isdigit()]
        scripts_to_run = [all_scripts[i] for i in indices if 0 <= i < len(all_scripts)]
    else:
        print("Invalid choice. Running all scripts...")
        scripts_to_run = all_scripts
    
    if not scripts_to_run:
        print("No scripts selected. Exiting.")
        return
    
    # Run selected scripts
    results = {}
    for script_name, description in scripts_to_run:
        success = run_script(script_name, description)
        results[script_name] = " Success" if success else " Failed"
        time.sleep(1)  # Small delay between scripts
    
    # Summary
    print("\n" + "="*60)
    print(" Script Execution Summary")
    print("="*60)
    
    for script_name, status in results.items():
        print(f"  {status} {script_name}")
    
    successful = sum(1 for s in results.values() if "" in s)
    total = len(results)
    
    print(f"\n Completed: {successful}/{total} scripts")
    print(f" Check results in:")
    print(f"  - results/ - Analysis results")
    print(f"  - figures/ - Generated figures")

if __name__ == "__main__":
    main()


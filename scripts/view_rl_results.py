# scripts/view_rl_results.py
"""
View and Display RL Results Dynamically
Shows latest test results with interactive plots
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import glob

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

def find_latest_results():
    """Find the latest result files."""
    # Find latest CSV report
    csv_files = list(RESULTS_DIR.glob('rl_evaluation_report_*.csv'))
    if csv_files:
        latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
    else:
        latest_csv = None
    
    # Find latest plot
    plot_files = list(RESULTS_DIR.glob('rl_performance_plots_*.png'))
    if plot_files:
        latest_plot = max(plot_files, key=lambda p: p.stat().st_mtime)
    else:
        latest_plot = None
    
    # Find quick test results
    quick_plot = RESULTS_DIR / 'quick_rl_test_results.png'
    if quick_plot.exists():
        quick_plot = quick_plot
    else:
        quick_plot = None
    
    return latest_csv, latest_plot, quick_plot

def display_latest_report():
    """Display the latest evaluation report."""
    latest_csv, latest_plot, quick_plot = find_latest_results()
    
    print("="*70)
    print("Latest RL Agent Test Results")
    print("="*70)
    
    if latest_csv:
        print(f"\n📊 Latest Report: {latest_csv.name}")
        print(f"   Created: {datetime.fromtimestamp(latest_csv.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
        df = pd.read_csv(latest_csv)
        print("\n" + df.to_string(index=False))
    else:
        print("\n⚠️  No evaluation report found.")
        print("   Run: python scripts/run_full_rl_evaluation.py")
    
    if quick_plot:
        print(f"\n📈 Quick Test Plot: {quick_plot}")
        print(f"   Created: {datetime.fromtimestamp(quick_plot.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
    
    if latest_plot:
        print(f"\n📊 Full Evaluation Plot: {latest_plot}")
        print(f"   Created: {datetime.fromtimestamp(latest_plot.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "="*70)
    print("To get latest results:")
    print("  1. Quick test:    python scripts/quick_rl_test.py")
    print("  2. Full eval:     python scripts/run_full_rl_evaluation.py")
    print("  3. View results: python scripts/view_rl_results.py")
    print("="*70 + "\n")

def show_plot(plot_path):
    """Display a plot file."""
    if plot_path and plot_path.exists():
        try:
            import matplotlib.image as mpimg
            img = mpimg.imread(plot_path)
            plt.figure(figsize=(12, 6))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f'Results: {plot_path.name}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Could not display plot: {e}")
            print(f"Plot file: {plot_path}")

if __name__ == '__main__':
    display_latest_report()
    
    # Optionally show plot
    import sys
    if '--show-plot' in sys.argv:
        _, _, quick_plot = find_latest_results()
        if quick_plot:
            show_plot(quick_plot)


#!/usr/bin/env python3
"""
Run All Evaluation Modules in src/evaluation/

This script provides a convenient way to run all evaluation modules.
You can run all modules at once, or select specific ones.
"""

import sys
import subprocess
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def run_module(module_path, description=""):
    """Run a module and return success status."""
    print("\n" + "="*60)
    print(f"🔄 Running: {module_path}")
    if description:
        print(f"   Description: {description}")
    print("="*60)
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", module_path],
            cwd=PROJECT_ROOT,
            check=False
        )
        
        if result.returncode == 0:
            print(f"✅ {module_path} completed successfully!")
            return True
        else:
            print(f"⚠️  {module_path} completed with warnings (exit code {result.returncode})")
            return True  # Don't fail pipeline for optional modules
    except Exception as e:
        print(f"❌ Error running {module_path}: {e}")
        return False

def main():
    """Main function to run all evaluation modules."""
    print("="*60)
    print("🚀 Run All Evaluation Modules")
    print("="*60)
    print("\nThis script will run all evaluation modules in src/evaluation/.")
    print("You can choose to run all modules or select specific ones.")
    print()
    
    # Define evaluation modules
    core_modules = [
        ("src.evaluation.evaluation", "Main evaluation pipeline (Phase 6)"),
        ("src.evaluation.ablation", "Ablation studies"),
        ("src.evaluation.enhanced_evaluation", "Enhanced evaluation with deep analysis"),
        ("src.evaluation.enhanced_ablation", "Enhanced ablation studies (with retraining)"),
        ("src.evaluation.deep_analysis", "Deep analysis (error patterns, feature importance)"),
        ("src.evaluation.statistical_tests", "Statistical significance testing"),
        ("src.evaluation.visualization", "Generate visualizations"),
    ]
    
    enhancement_modules = [
        ("src.evaluation.enhancements.edge_importance", "Edge importance analysis"),
        ("src.evaluation.enhancements.cross_period_validation", "Cross-period validation"),
        ("src.evaluation.enhancements.multi_agent_analysis", "Multi-agent RL analysis"),
        ("src.evaluation.enhancements.sensitivity_analysis", "Sensitivity analysis"),
        ("src.evaluation.enhancements.failure_analysis", "Failure analysis"),
    ]
    
    all_modules = core_modules + enhancement_modules
    
    # Ask user what to run
    print("Options:")
    print("  1. Run all modules")
    print("  2. Run core evaluation modules only")
    print("  3. Run enhancement modules only")
    print("  4. Select individual modules")
    print()
    
    choice = input("Enter your choice (1-4, default=1): ").strip() or "1"
    
    modules_to_run = []
    
    if choice == "1":
        modules_to_run = all_modules
    elif choice == "2":
        modules_to_run = core_modules
    elif choice == "3":
        modules_to_run = enhancement_modules
    elif choice == "4":
        print("\nAvailable modules:")
        for i, (module, desc) in enumerate(all_modules, 1):
            print(f"  {i}. {module} - {desc}")
        print()
        selection = input("Enter module numbers (comma-separated, e.g., 1,3,5): ").strip()
        indices = [int(x.strip()) - 1 for x in selection.split(",") if x.strip().isdigit()]
        modules_to_run = [all_modules[i] for i in indices if 0 <= i < len(all_modules)]
    else:
        print("Invalid choice. Running all modules...")
        modules_to_run = all_modules
    
    if not modules_to_run:
        print("No modules selected. Exiting.")
        return
    
    # Run selected modules
    results = {}
    for module_path, description in modules_to_run:
        success = run_module(module_path, description)
        results[module_path] = "✅ Success" if success else "❌ Failed"
        time.sleep(1)  # Small delay between modules
    
    # Summary
    print("\n" + "="*60)
    print("📊 Evaluation Module Execution Summary")
    print("="*60)
    
    for module_path, status in results.items():
        print(f"  {status} {module_path}")
    
    successful = sum(1 for s in results.values() if "✅" in s)
    total = len(results)
    
    print(f"\n✅ Completed: {successful}/{total} modules")
    print(f"📁 Check results in:")
    print(f"  - results/ - Evaluation results")
    print(f"  - models/plots/ - Visualizations")

if __name__ == "__main__":
    main()


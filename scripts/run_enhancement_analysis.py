#!/usr/bin/env python3
"""
Run all enhancement analysis modules
Includes: cross-period validation, failure analysis, edge importance, multi-agent analysis, sensitivity analysis
"""

import sys
from pathlib import Path
import subprocess
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ENHANCEMENTS_DIR = PROJECT_ROOT / "src" / "evaluation" / "enhancements"
RESULTS_DIR = PROJECT_ROOT / "results"

def run_enhancement_module(module_name, description):
    """Run an enhancement analysis module."""
    print("\n" + "=" * 60)
    print(f"Running: {description}")
    print("=" * 60)
    
    module_path = ENHANCEMENTS_DIR / f"{module_name}.py"
    
    if not module_path.exists():
        print(f"  Module not found: {module_path}")
        return False
    
    try:
        # Run the module
        result = subprocess.run(
            [sys.executable, str(module_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes timeout
        )
        
        if result.returncode == 0:
            print(f" {description} completed successfully")
            if result.stdout:
                print(result.stdout[-500:])  # Print last 500 chars
            return True
        else:
            print(f" {description} failed with return code {result.returncode}")
            if result.stderr:
                print("Error output:")
                print(result.stderr[-500:])
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏱  {description} timed out after 30 minutes")
        return False
    except Exception as e:
        print(f" Error running {description}: {e}")
        return False

def main():
    """Run all enhancement analysis modules."""
    print("=" * 60)
    print("Enhancement Analysis Pipeline")
    print("=" * 60)
    print("\nThis script runs all enhancement analysis modules:")
    print("  1. Cross-Period Validation")
    print("  2. Failure Analysis")
    print("  3. Edge Importance Analysis")
    print("  4. Multi-Agent Analysis")
    print("  5. Sensitivity Analysis")
    print()
    
    modules = [
        ("cross_period_validation", "Cross-Period Validation"),
        ("failure_analysis", "Failure Analysis"),
        ("edge_importance", "Edge Importance Analysis"),
        ("multi_agent_analysis", "Multi-Agent Analysis"),
        ("sensitivity_analysis", "Sensitivity Analysis"),
    ]
    
    results = {}
    start_time = time.time()
    
    for module_name, description in modules:
        module_start = time.time()
        success = run_enhancement_module(module_name, description)
        module_time = time.time() - module_start
        results[description] = {
            'success': success,
            'time': module_time
        }
        
        if success:
            print(f"⏱  Time taken: {module_time:.1f} seconds")
        print()
    
    total_time = time.time() - start_time
    
    # Summary
    print("=" * 60)
    print("Enhancement Analysis Summary")
    print("=" * 60)
    
    successful = sum(1 for r in results.values() if r['success'])
    total = len(results)
    
    for desc, result in results.items():
        status = "" if result['success'] else ""
        print(f"{status} {desc}: {result['time']:.1f}s")
    
    print(f"\nTotal: {successful}/{total} modules completed successfully")
    print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    
    if successful == total:
        print("\n All enhancement analysis modules completed!")
    else:
        print(f"\n  {total - successful} module(s) failed. Check logs above for details.")

if __name__ == "__main__":
    main()


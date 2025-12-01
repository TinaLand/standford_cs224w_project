#!/usr/bin/env python3
"""
Full Pipeline Runner
Runs the complete CS224W Stock RL GNN project pipeline from Phase 1 to Phase 6.
"""

import sys
from pathlib import Path
import subprocess
import time
import os

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

# Add src to path
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

def run_phase(phase_name, module_path, description):
    """Run a phase script using subprocess."""
    print("\n" + "="*60)
    print(f"🔄 Phase: {phase_name}")
    print("="*60)
    print(f"Description: {description}")
    print(f"Module: {module_path}")
    print()
    
    try:
        # Change to project root
        original_cwd = os.getcwd()
        os.chdir(PROJECT_ROOT)
        
        # Run as module
        cmd = [sys.executable, "-m", module_path]
        print(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=False
        )
        
        os.chdir(original_cwd)
        
        if result.returncode == 0:
            print(f"\n✅ {phase_name} completed successfully!")
            return True
        else:
            print(f"\n❌ {phase_name} failed with exit code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running {phase_name}: {e}")
        import traceback
        traceback.print_exc()
        os.chdir(original_cwd)
        return False

def run_optional_experiment(experiment_name, script_path, description):
    """Run an optional experiment script."""
    print("\n" + "="*60)
    print(f"🔬 Optional Experiment: {experiment_name}")
    print("="*60)
    print(f"Description: {description}")
    print(f"Script: {script_path}")
    print()
    
    try:
        original_cwd = os.getcwd()
        os.chdir(PROJECT_ROOT)
        
        cmd = [sys.executable, str(script_path)]
        print(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=False
        )
        
        os.chdir(original_cwd)
        
        if result.returncode == 0:
            print(f"\n✅ {experiment_name} completed successfully!")
            return True
        else:
            print(f"\n⚠️  {experiment_name} completed with warnings (exit code {result.returncode})")
            return True  # Don't fail pipeline for optional experiments
            
    except Exception as e:
        print(f"⚠️  Error running {experiment_name}: {e}")
        print("Continuing pipeline...")
        os.chdir(original_cwd)
        return True  # Don't fail pipeline for optional experiments

def main():
    """Main pipeline runner."""
    print("="*60)
    print("🚀 CS224W Stock RL GNN - Full Pipeline")
    print("="*60)
    print("\nThis script will run all phases of the project:")
    print("  Phase 1: Data Collection & Feature Engineering")
    print("  Phase 2: Graph Construction")
    print("  Phase 3: Baseline GNN Training")
    print("  Phase 4: Core Transformer Training")
    print("  Phase 5: RL Integration")
    print("  Phase 6: Evaluation")
    print()
    print("Optional experiments (proposal-aligned) can be run separately:")
    print("  - Lookahead Horizon: python scripts/experiment_lookahead_horizons.py")
    print("  - Graph Sparsification: python scripts/experiment_graph_sparsification.py")
    print("  - Robustness Checks: python scripts/experiment_robustness_checks.py")
    print()
    
    phases = [
        {
            'name': 'Phase 1: Data Collection',
            'module': 'src.data.collection',
            'description': 'Download and collect raw stock data'
        },
        {
            'name': 'Phase 1: Feature Engineering',
            'module': 'src.data.feature_engineering',
            'description': 'Calculate technical indicators and normalize features'
        },
        {
            'name': 'Phase 1: Edge Parameters',
            'module': 'src.data.edge_parameters',
            'description': 'Calculate rolling correlations and fundamental similarity'
        },
        {
            'name': 'Phase 2: Graph Construction',
            'module': 'src.data.graph_construction',
            'description': 'Build daily graph snapshots'
        },
        {
            'name': 'Phase 3: Baseline Training',
            'module': 'src.training.baseline_trainer',
            'description': 'Train baseline GAT model'
        },
        {
            'name': 'Phase 4: Transformer Training',
            'module': 'src.training.transformer_trainer',
            'description': 'Train Role-Aware Graph Transformer'
        },
        {
            'name': 'Phase 5: RL Integration',
            'module': 'src.rl.integration',
            'description': 'Integrate GNN with RL agent'
        },
        {
            'name': 'Phase 6: Evaluation',
            'module': 'src.evaluation.evaluation',
            'description': 'Evaluate models and generate metrics'
        }
    ]
    
    results = {}
    
    for i, phase in enumerate(phases, 1):
        success = run_phase(phase['name'], phase['module'], phase['description'])
        results[phase['name']] = 'Success' if success else 'Failed'
        
        if not success:
            print(f"\n⚠️  {phase['name']} failed.")
            print("Continuing to next phase...")
            # Auto-continue instead of asking for input
            # response = input().strip().lower()
            # if response != 'y':
            #     print("\n❌ Pipeline stopped by user")
            #     break
        
        # Small delay between phases
        time.sleep(1)
    
    # Summary
    print("\n" + "="*60)
    print("📊 Pipeline Summary")
    print("="*60)
    
    for phase_name, status in results.items():
        status_icon = "✅" if status == "Success" else "❌" if status == "Failed" else "⚠️"
        print(f"  {status_icon} {phase_name}: {status}")
    
    print("\n✅ Full pipeline execution complete!")
    print("\n📁 Check results in:")
    print("  - models/ - Trained models")
    print("  - results/ - Evaluation results")
    print("  - models/plots/ - Visualizations")
    
    print("\n" + "="*60)
    print("🔬 Optional Research Experiments")
    print("="*60)
    print("\nThe following experiments are available (proposal-aligned):")
    print("  1. Lookahead Horizon Analysis")
    print("     python scripts/experiment_lookahead_horizons.py")
    print("     - Tests: 1, 3, 5, 7, 10 day horizons")
    print("     - Results: results/lookahead_horizon_results.csv")
    print()
    print("  2. Graph Sparsification Experiments")
    print("     python scripts/experiment_graph_sparsification.py")
    print("     - Tests: Various Top-K and correlation thresholds")
    print("     - Results: results/graph_sparsification_results.csv")
    print()
    print("  3. Robustness Checks")
    print("     python scripts/experiment_robustness_checks.py")
    print("     - Tests: Transaction costs and slippage sensitivity")
    print("     - Results: results/robustness_checks_results.csv")
    print()
    print("  4. Statistical Significance Testing")
    print("     - Automatically included in Phase 6 evaluation")
    print("     - Results: results/statistical_tests.csv")
    print()
    print("Note: These experiments are optional but recommended for comprehensive analysis.")

if __name__ == "__main__":
    main()


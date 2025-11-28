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
            print(f"\n⚠️  {phase['name']} failed. Continue? (y/n): ", end='')
            response = input().strip().lower()
            if response != 'y':
                print("\n❌ Pipeline stopped by user")
                break
        
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

if __name__ == "__main__":
    main()


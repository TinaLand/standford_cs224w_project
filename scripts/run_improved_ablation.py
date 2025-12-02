"""
Improved Ablation Study Script
Re-runs ablation studies with proper model retraining to show real differences
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.enhanced_ablation import run_enhanced_ablation_studies

if __name__ == "__main__":
    print("=" * 60)
    print("Running Improved Ablation Studies")
    print("=" * 60)
    print("This will retrain models for each configuration to show real differences.")
    print("Expected runtime: 2-3 hours")
    print()
    
    results = run_enhanced_ablation_studies()
    
    print("\n" + "=" * 60)
    print("Ablation Studies Complete!")
    print("=" * 60)
    print(f"Results saved to: results/ablation_results.csv")
    print("\nSummary:")
    for config_name, metrics in results.items():
        print(f"  {config_name}:")
        print(f"    Accuracy: {metrics.get('accuracy', 0):.4f}")
        print(f"    F1 Score: {metrics.get('f1_score', 0):.4f}")
        print(f"    Precision@Top-10: {metrics.get('precision_at_top10', 0):.4f}")


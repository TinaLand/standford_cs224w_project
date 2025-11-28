# run_all_enhancements.py
"""
Run all enhancement scripts to achieve A+ grade.
This script orchestrates all enhancement analyses.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / 'scripts'))

def main():
    """Run all enhancement scripts."""
    print("="*60)
    print("🚀 Running All Enhancement Scripts for A+ Grade")
    print("="*60)
    
    enhancements = [
        {
            'name': 'Multi-Agent Decision Analysis',
            'script': 'enhancement_multi_agent_analysis',
            'description': 'Agent disagreements, sector performance, mixing network analysis'
        },
        {
            'name': 'Failure Analysis',
            'script': 'enhancement_failure_analysis',
            'description': 'Worst periods, error patterns, drawdown analysis'
        },
        {
            'name': 'Edge Importance Analysis',
            'script': 'enhancement_edge_importance',
            'description': 'Edge importance rankings, sector subgraphs, correlation vs fundamental'
        },
        {
            'name': 'Cross-Period Validation',
            'script': 'enhancement_cross_period_validation',
            'description': 'Performance across different market regimes'
        },
        {
            'name': 'Sensitivity Analysis',
            'script': 'enhancement_sensitivity_analysis',
            'description': 'Transaction costs, parameters, slippage sensitivity'
        }
    ]
    
    results = {}
    
    for i, enhancement in enumerate(enhancements, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(enhancements)}] {enhancement['name']}")
        print(f"{'='*60}")
        print(f"Description: {enhancement['description']}")
        
        try:
            module = __import__(enhancement['script'])
            if hasattr(module, 'main'):
                module.main()
                results[enhancement['name']] = '✅ Success'
            else:
                results[enhancement['name']] = '⚠️  No main function'
        except Exception as e:
            print(f"❌ Error: {e}")
            results[enhancement['name']] = f'❌ Failed: {str(e)[:50]}'
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 Enhancement Summary")
    print("="*60)
    
    for name, status in results.items():
        print(f"  {status} {name}")
    
    print(f"\n✅ All enhancements completed!")
    print(f"📄 Results saved in: results/")
    print(f"📊 Visualizations saved in: models/plots/")


if __name__ == "__main__":
    main()


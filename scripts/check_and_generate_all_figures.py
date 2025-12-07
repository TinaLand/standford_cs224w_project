"""
Comprehensive Figure Generation Script
Ensures all required figures for A+ grade are generated
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FIGS_DIR = PROJECT_ROOT / "figures"
FIGS_DIR.mkdir(exist_ok=True)

# Required figures for A+ grade
REQUIRED_FIGURES = {
    # Core Figures (5)
    'figure1_system_architecture.png': 'System Architecture Diagram',
    'figure4_portfolio_performance.png': 'Portfolio Performance (with baselines)',
    'figure5_ablation_study.png': 'Ablation Study (dual metrics)',
    'figure9_gnn_architecture.png': 'GNN Message Passing & Innovation',
    'figure10_multitask_loss.png': 'Multi-Task Loss Structure',
    
    # Supporting Figures (5)
    'figure6_attention_heatmap.png': 'GNN Attention Heatmap',
    'figure8_regime_performance.png': 'Regime Performance',
    'figure_precision_topk_curve.png': 'Precision@Top-K Curve',
    'figure_pearl_embedding_visualization.png': 'PEARL Embedding Visualization',
    'figure_marl_decision_flow.png': 'MARL Decision Flow',
    
    # Additional Figures
    'figure2_training_curves.png': 'Training Curves',
    'figure3_model_comparison.png': 'Model Comparison',
    'figure_ic_analysis.png': 'IC Analysis',
    'figure7a_graph_structure_overview.png': 'Graph Structure Overview',
}

def check_figures():
    """Check which figures exist and which are missing."""
    print("=" * 60)
    print("Checking Required Figures for A+ Grade")
    print("=" * 60)
    print()
    
    existing = []
    missing = []
    
    for fig_name, description in REQUIRED_FIGURES.items():
        fig_path = FIGS_DIR / fig_name
        if fig_path.exists():
            size_kb = fig_path.stat().st_size / 1024
            existing.append((fig_name, description, size_kb))
            print(f" {fig_name:50s} ({size_kb:.1f} KB) - {description}")
        else:
            missing.append((fig_name, description))
            print(f" {fig_name:50s} MISSING - {description}")
    
    print()
    print("=" * 60)
    print(f"Summary: {len(existing)}/{len(REQUIRED_FIGURES)} figures exist")
    print("=" * 60)
    
    if missing:
        print(f"\n  Missing {len(missing)} figures:")
        for fig_name, description in missing:
            print(f"   - {fig_name}: {description}")
        return False
    else:
        print("\n All required figures exist!")
        return True

def generate_missing_figures():
    """Generate missing figures."""
    print("\n" + "=" * 60)
    print("Generating Missing Figures")
    print("=" * 60)
    print()
    
    # Import figure generation functions
    from scripts.generate_report_figures import (
        create_architecture_diagram,
        create_training_curves,
        create_model_comparison,
        create_portfolio_performance,
        create_ablation_study,
        create_attention_heatmap,
        create_graph_structure_overview,
        create_gnn_architecture_diagram,
        create_multitask_loss_diagram,
        create_regime_performance,
    )
    
    from scripts.create_additional_figures import (
        create_pearl_embedding_visualization,
        create_precision_topk_curve,
        create_marl_decision_flow,
    )
    
    from scripts.analyze_ic_deep import analyze_ic_deep
    
    # Check what's missing and generate
    missing_figures = []
    for fig_name, _ in REQUIRED_FIGURES.items():
        if not (FIGS_DIR / fig_name).exists():
            missing_figures.append(fig_name)
    
    if not missing_figures:
        print(" All figures already exist!")
        return
    
    # Generate based on missing figures
    figure_generators = {
        'figure1_system_architecture.png': create_architecture_diagram,
        'figure2_training_curves.png': create_training_curves,
        'figure3_model_comparison.png': create_model_comparison,
        'figure4_portfolio_performance.png': create_portfolio_performance,
        'figure5_ablation_study.png': create_ablation_study,
        'figure6_attention_heatmap.png': create_attention_heatmap,
        'figure7a_graph_structure_overview.png': create_graph_structure_overview,
        'figure8_regime_performance.png': create_regime_performance,
        'figure9_gnn_architecture.png': create_gnn_architecture_diagram,
        'figure10_multitask_loss.png': create_multitask_loss_diagram,
        'figure_precision_topk_curve.png': create_precision_topk_curve,
        'figure_pearl_embedding_visualization.png': create_pearl_embedding_visualization,
        'figure_marl_decision_flow.png': create_marl_decision_flow,
        'figure_ic_analysis.png': analyze_ic_deep,
    }
    
    for fig_name in missing_figures:
        if fig_name in figure_generators:
            print(f"\nGenerating {fig_name}...")
            try:
                figure_generators[fig_name]()
                print(f" Generated: {fig_name}")
            except Exception as e:
                print(f" Error generating {fig_name}: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    # Check existing figures
    all_exist = check_figures()
    
    # Generate missing ones
    if not all_exist:
        response = input("\nGenerate missing figures? (y/n): ").strip().lower()
        if response == 'y':
            generate_missing_figures()
            print("\n" + "=" * 60)
            print("Re-checking figures...")
            print("=" * 60)
            check_figures()
    else:
        print("\n All figures are ready!")


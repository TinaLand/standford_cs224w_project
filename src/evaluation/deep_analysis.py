"""
Deep Analysis Module for A+ Grade Enhancements
Includes error pattern analysis, feature importance, and trading behavior analysis
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def analyze_error_patterns(
    predictions: np.ndarray,
    targets: np.ndarray,
    dates: List[pd.Timestamp],
    tickers: List[str]
) -> Dict[str, Any]:
    """
    Analyze error patterns in predictions.
    
    Returns:
        Dictionary with error analysis results
    """
    print("\n--- Error Pattern Analysis ---")
    
    # Confusion matrix
    cm = confusion_matrix(targets.flatten(), predictions.flatten())
    tn, fp, fn, tp = cm.ravel()
    
    # Error rates
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    # Per-day error analysis
    daily_errors = []
    for day_idx in range(len(dates)):
        day_preds = predictions[day_idx]
        day_targets = targets[day_idx]
        errors = np.sum(day_preds != day_targets)
        error_rate = errors / len(day_preds) if len(day_preds) > 0 else 0.0
        daily_errors.append({
            'date': dates[day_idx],
            'error_count': errors,
            'error_rate': error_rate,
            'false_positives': np.sum((day_preds == 1) & (day_targets == 0)),
            'false_negatives': np.sum((day_preds == 0) & (day_targets == 1))
        })
    
    daily_errors_df = pd.DataFrame(daily_errors)
    
    # Worst performing days
    worst_days = daily_errors_df.nlargest(10, 'error_rate')
    
    # Best performing days
    best_days = daily_errors_df.nsmallest(10, 'error_rate')
    
    results = {
        'confusion_matrix': cm.tolist(),
        'false_positive_rate': float(false_positive_rate),
        'false_negative_rate': float(false_negative_rate),
        'total_errors': int(fp + fn),
        'worst_days': worst_days.to_dict('records'),
        'best_days': best_days.to_dict('records'),
        'mean_daily_error_rate': float(daily_errors_df['error_rate'].mean()),
        'std_daily_error_rate': float(daily_errors_df['error_rate'].std())
    }
    
    print(f"   False Positive Rate: {false_positive_rate:.4f}")
    print(f"   False Negative Rate: {false_negative_rate:.4f}")
    print(f"   Mean Daily Error Rate: {results['mean_daily_error_rate']:.4f}")
    print(f"   Worst Day Error Rate: {worst_days['error_rate'].max():.4f}")
    print(f"   Best Day Error Rate: {best_days['error_rate'].min():.4f}")
    
    return results


def analyze_feature_importance(
    model: torch.nn.Module,
    sample_data,
    device: torch.device
) -> Dict[str, Any]:
    """
    Analyze feature importance using gradient-based methods.
    """
    print("\n--- Feature Importance Analysis ---")
    
    try:
        model.eval()
        sample_data = sample_data.to(device)
        
        # Get input features (before positional encoding)
        x = sample_data['stock'].x.clone()
        x.requires_grad = True
        
        # Create a modified data object with grad-enabled features
        from torch_geometric.data import HeteroData
        modified_data = HeteroData()
        modified_data['stock'].x = x
        modified_data.edge_index_dict = sample_data.edge_index_dict
        
        # Forward pass
        with torch.enable_grad():
            # Check if model needs date_tensor
            if hasattr(model, 'enable_time_aware') and model.enable_time_aware:
                import pandas as pd
                REFERENCE_DATE = pd.to_datetime('2015-01-01')
                # Use a dummy date for feature importance
                days_since_ref = 1000  # Dummy value
                num_nodes = x.shape[0]
                date_tensor = torch.full((num_nodes,), days_since_ref, dtype=torch.float32).to(device)
                out = model(modified_data, date_tensor=date_tensor)
            else:
                out = model(modified_data)
            
            # Use classification output
            if isinstance(out, tuple):
                out = out[0]
            
            # Calculate gradient w.r.t. input features
            loss = out.sum()
            loss.backward()
            
            # Feature importance = absolute gradient values
            feature_importance = x.grad.abs().mean(dim=0).cpu().numpy()
        
        # Get top important features
        top_k = min(20, len(feature_importance))
        top_indices = np.argsort(feature_importance)[-top_k:][::-1]
        
        results = {
            'feature_importance': feature_importance.tolist(),
            'top_features': {
                f'feature_{i}': float(feature_importance[i]) 
                for i in top_indices
            },
            'mean_importance': float(feature_importance.mean()),
            'std_importance': float(feature_importance.std())
        }
        
        print(f"   Top {top_k} Important Features:")
        for idx in top_indices[:10]:
            print(f"      Feature {idx}: {feature_importance[idx]:.6f}")
        
        return results
    
    except Exception as e:
        print(f"     Feature importance analysis failed: {e}")
        # Return empty results
        return {
            'feature_importance': [],
            'top_features': {},
            'mean_importance': 0.0,
            'std_importance': 0.0,
            'error': str(e)
        }


def visualize_feature_importance(
    feature_importance: Dict[str, Any],
    feature_names: List[str] = None,
    results_dir: Path = RESULTS_DIR,
    top_k: int = 20
):
    """
    Create visualizations for feature importance analysis.
    
    Args:
        feature_importance: Dictionary from analyze_feature_importance()
        feature_names: Optional list of feature names (if None, uses indices)
        results_dir: Directory to save plots
        top_k: Number of top features to visualize
    """
    print("\n--- Creating Feature Importance Visualizations ---")
    
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    if 'feature_importance' not in feature_importance or len(feature_importance['feature_importance']) == 0:
        print("     No feature importance data to visualize")
        return
    
    importance_array = np.array(feature_importance['feature_importance'])
    
    # Get top K features
    top_indices = np.argsort(importance_array)[-top_k:][::-1]
    top_importance = importance_array[top_indices]
    
    # Create feature labels
    if feature_names is None or len(feature_names) != len(importance_array):
        feature_labels = [f'Feature {i}' for i in top_indices]
    else:
        feature_labels = [feature_names[i] for i in top_indices]
    
    # 1. Horizontal Bar Chart (Top K Features)
    plt.figure(figsize=(12, max(8, top_k * 0.4)))
    colors = plt.cm.viridis(np.linspace(0, 1, top_k))
    bars = plt.barh(range(top_k), top_importance, color=colors)
    plt.yticks(range(top_k), feature_labels)
    plt.xlabel('Importance Score (|Gradient|)', fontsize=12, fontweight='bold')
    plt.title(f'Top {top_k} Most Important Features', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(plots_dir / 'feature_importance_topk.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Top-{top_k} feature importance chart saved")
    
    # 2. Feature Importance Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(importance_array, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
    plt.axvline(importance_array.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {importance_array.mean():.6f}')
    plt.axvline(importance_array.std(), color='orange', linestyle='--', linewidth=2, 
                label=f'Std: {importance_array.std():.6f}')
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Feature Importance Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'feature_importance_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    Feature importance distribution chart saved")
    
    # 3. Cumulative Importance (Top Features)
    cumulative_importance = np.cumsum(np.sort(importance_array)[::-1])
    cumulative_pct = cumulative_importance / cumulative_importance[-1] * 100
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, min(50, len(cumulative_pct)) + 1), cumulative_pct[:50], 
             linewidth=2, marker='o', markersize=4)
    plt.xlabel('Number of Top Features', fontsize=12)
    plt.ylabel('Cumulative Importance (%)', fontsize=12)
    plt.title('Cumulative Feature Importance', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axhline(80, color='red', linestyle='--', alpha=0.5, label='80% Threshold')
    plt.axhline(90, color='orange', linestyle='--', alpha=0.5, label='90% Threshold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / 'feature_importance_cumulative.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    Cumulative feature importance chart saved")
    
    print(f"    All feature importance visualizations saved to: {plots_dir}")


def get_feature_names_from_engineering() -> List[str]:
    """
    Get feature names from feature engineering module.
    This is a helper function to map feature indices to meaningful names.
    """
    # Common feature groups (approximate order from feature_engineering.py)
    feature_groups = [
        # Price features
        'Close', 'Open', 'High', 'Low', 'Volume',
        # Returns
        'Daily_Return', 'Log_Return',
        # Technical Indicators - Momentum
        'RSI', 'Momentum', 'ROC',
        # Technical Indicators - Trend
        'MACD', 'MACD_Signal', 'MACD_Hist',
        'SMA_20', 'SMA_50', 'SMA_200',
        'EMA_12', 'EMA_26',
        'Price_to_SMA20', 'Price_to_SMA200',
        'SMA20_to_SMA50', 'EMA12_to_EMA26',
        # Technical Indicators - Volatility
        'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width', 'BB_Position',
        'ATR', 'True_Range', 'Historical_Volatility',
        # Technical Indicators - Oscillators
        'Stochastic_K', 'Stochastic_D',
        'Williams_R', 'CCI', 'ADX',
        # Technical Indicators - Volume
        'OBV', 'AD', 'Volume_ROC', 'Volume_to_MA20', 'PVT',
        # Structural features (if included)
        'PageRank', 'Degree_Centrality', 'Betweenness_Centrality',
        'Closeness_Centrality', 'Clustering_Coefficient',
        'Core_Number', 'Avg_Neighbor_Degree', 'Triangle_Count'
    ]
    
    # Note: Actual feature count may vary. This is a template.
    # In practice, you'd extract actual feature names from feature_engineering.py
    return feature_groups


def analyze_trading_behavior(
    portfolio_values: List[float],
    trades: List[Dict],
    dates: List[pd.Timestamp]
) -> Dict[str, Any]:
    """
    Analyze trading behavior and patterns.
    """
    print("\n--- Trading Behavior Analysis ---")
    
    if len(portfolio_values) < 2:
        return {}
    
    portfolio_series = pd.Series(portfolio_values, index=dates[:len(portfolio_values)])
    returns = portfolio_series.pct_change().dropna()
    
    # Win rate
    positive_returns = returns[returns > 0]
    win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0.0
    
    # Average return per trade
    avg_return = returns.mean()
    
    # Volatility
    volatility = returns.std()
    
    # Maximum consecutive wins/losses
    win_loss_sequence = (returns > 0).astype(int)
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    current_wins = 0
    current_losses = 0
    
    for val in win_loss_sequence:
        if val == 1:
            current_wins += 1
            current_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_consecutive_losses = max(max_consecutive_losses, current_losses)
    
    # Turnover rate (if trades data available)
    turnover_rate = 0.0
    if trades and len(trades) > 0:
        try:
            # Handle different trade data formats
            total_trades = 0
            for t in trades:
                # Skip if t is not a dict (e.g., numpy.float64)
                if not isinstance(t, dict):
                    continue
                
                # Try to get trades from various possible keys
                trades_data = None
                for key in ['trades', 'action', 'num_trades', 'trade_count']:
                    if key in t:
                        trades_data = t[key]
                        break
                
                if trades_data is not None:
                    if isinstance(trades_data, (list, tuple)):
                        total_trades += len(trades_data)
                    elif isinstance(trades_data, (int, float, np.integer, np.floating)):
                        total_trades += int(trades_data)
            
            turnover_rate = total_trades / len(portfolio_values) if len(portfolio_values) > 0 and total_trades > 0 else 0.0
        except Exception as e:
            # If we can't calculate turnover, just set to 0
            turnover_rate = 0.0
    
    results = {
        'win_rate': float(win_rate),
        'average_return_per_trade': float(avg_return),
        'volatility': float(volatility),
        'max_consecutive_wins': int(max_consecutive_wins),
        'max_consecutive_losses': int(max_consecutive_losses),
        'turnover_rate': float(turnover_rate),
        'total_trading_days': len(returns)
    }
    
    print(f"   Win Rate: {win_rate:.4f}")
    print(f"   Average Return per Trade: {avg_return:.6f}")
    print(f"   Volatility: {volatility:.6f}")
    print(f"   Max Consecutive Wins: {max_consecutive_wins}")
    print(f"   Max Consecutive Losses: {max_consecutive_losses}")
    print(f"   Turnover Rate: {turnover_rate:.4f}")
    
    return results


def create_visualizations(
    portfolio_values: List[float],
    dates: List[pd.Timestamp],
    daily_returns: List[float],
    results_dir: Path = RESULTS_DIR
):
    """
    Create comprehensive visualizations for A+ grade.
    """
    print("\n--- Creating Visualizations ---")
    
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Portfolio Value Over Time
    plt.figure(figsize=(12, 6))
    portfolio_series = pd.Series(portfolio_values, index=dates[:len(portfolio_values)])
    plt.plot(portfolio_series.index, portfolio_series.values, linewidth=2)
    plt.title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'portfolio_value_over_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    Portfolio value chart saved")
    
    # 2. Drawdown Analysis
    if len(portfolio_values) > 1:
        portfolio_series = pd.Series(portfolio_values, index=dates[:len(portfolio_values)])
        cumulative_max = portfolio_series.cummax()
        drawdown = (cumulative_max - portfolio_series) / cumulative_max
        
        plt.figure(figsize=(12, 6))
        plt.fill_between(drawdown.index, 0, drawdown.values, alpha=0.3, color='red')
        plt.plot(drawdown.index, drawdown.values, linewidth=2, color='red')
        plt.title('Drawdown Analysis', fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Drawdown', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / 'drawdown_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("    Drawdown chart saved")
    
    # 3. Daily Returns Distribution
    if daily_returns:
        plt.figure(figsize=(10, 6))
        plt.hist(daily_returns, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(daily_returns), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(daily_returns):.4f}')
        plt.title('Daily Returns Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Daily Return', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / 'daily_returns_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("    Returns distribution chart saved")
    
    print(f"    All visualizations saved to: {plots_dir}")


def generate_comprehensive_report(
    gnn_metrics: Dict,
    rl_metrics: Dict,
    error_analysis: Dict,
    feature_importance: Dict,
    trading_behavior: Dict,
    output_path: Path = RESULTS_DIR / "comprehensive_analysis_report.json"
):
    """
    Generate comprehensive analysis report.
    """
    import json
    
    report = {
        'gnn_metrics': gnn_metrics,
        'rl_metrics': rl_metrics,
        'error_analysis': error_analysis,
        'feature_importance': feature_importance,
        'trading_behavior': trading_behavior,
        'summary': {
            'gnn_f1_score': gnn_metrics.get('f1_score', 0),
            'rl_sharpe_ratio': rl_metrics.get('Sharpe_Ratio', 0),
            'rl_cumulative_return': rl_metrics.get('Cumulative_Return', 0),
            'win_rate': trading_behavior.get('win_rate', 0),
            'mean_daily_error_rate': error_analysis.get('mean_daily_error_rate', 0)
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n Comprehensive analysis report saved to: {output_path}")
    return report


"""
Enhanced Evaluation Module for A+ Grade
Adds deep analysis, visualizations, and comprehensive reporting
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from src.evaluation.evaluation import (
    evaluate_gnn_metrics,
    run_final_backtest,
    calculate_financial_metrics
)
from src.evaluation.deep_analysis import (
    analyze_error_patterns,
    analyze_feature_importance,
    analyze_trading_behavior,
    create_visualizations,
    generate_comprehensive_report
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def run_enhanced_evaluation():
    """
    Run enhanced evaluation with deep analysis for A+ grade.
    """
    print("=" * 60)
    print("🚀 Enhanced Evaluation for A+ Grade")
    print("=" * 60)
    
    # Import necessary modules
    from src.training.transformer_trainer import (
        RoleAwareGraphTransformer,
        load_graph_data,
        create_target_labels,
        _read_time_series_csv,
        OHLCV_RAW_FILE,
        HIDDEN_CHANNELS,
        OUT_CHANNELS,
        NUM_LAYERS,
        NUM_HEADS
    )
    from src.rl.integration import load_gnn_model_for_rl
    
    # Load model
    print("\n--- Loading Models ---")
    gnn_model = load_gnn_model_for_rl()
    
    # Get test dates and tickers
    graph_files = sorted(list((PROJECT_ROOT / "data" / "graphs").glob('graph_t_*.pt')))
    all_dates = [pd.to_datetime(f.stem.split('_')[-1]) for f in graph_files]
    
    split_85_idx = int(len(all_dates) * 0.85)
    test_dates = all_dates[split_85_idx:]
    
    # Get tickers
    sample_graph = torch.load(graph_files[0], weights_only=False)
    if 'tickers' in sample_graph:
        tickers = sample_graph['tickers']
    else:
        ohlcv_df = _read_time_series_csv(OHLCV_RAW_FILE)
        tickers = [col.replace('Close_', '') for col in ohlcv_df.columns if col.startswith('Close_')]
    
    # Create targets
    targets_class_dict, targets_reg_dict = create_target_labels(tickers, all_dates, lookahead_days=5)
    
    # 1. GNN Metrics with Error Analysis
    print("\n" + "=" * 60)
    print("📊 Step 1: Enhanced GNN Evaluation")
    print("=" * 60)
    
    # Collect predictions for error analysis
    all_predictions = []
    all_targets = []
    all_probs = []
    all_dates_list = []
    
    REFERENCE_DATE = pd.to_datetime('2015-01-01')
    
    for date in test_dates:
        data = load_graph_data(date)
        target = targets_class_dict.get(date)
        
        if data and target is not None:
            gnn_model.eval()
            with torch.no_grad():
                days_since_ref = (date - REFERENCE_DATE).days
                num_nodes = data['stock'].x.shape[0]
                date_tensor = torch.full((num_nodes,), days_since_ref, dtype=torch.float32).to(DEVICE) if hasattr(gnn_model, 'enable_time_aware') and gnn_model.enable_time_aware else None
                out = gnn_model(data.to(DEVICE), date_tensor=date_tensor)
                probs = torch.nn.functional.softmax(out, dim=1)
                preds = out.argmax(dim=1).cpu().numpy()
            
            all_predictions.append(preds)
            all_targets.append(target.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_dates_list.append(date)
    
    predictions_array = np.array(all_predictions)
    targets_array = np.array(all_targets)
    probs_array = np.array(all_probs)
    
    # Error analysis
    error_analysis = analyze_error_patterns(
        predictions_array,
        targets_array,
        all_dates_list,
        tickers
    )
    
    # Feature importance (using first test date)
    if test_dates:
        sample_data = load_graph_data(test_dates[0])
        if sample_data:
            feature_importance = analyze_feature_importance(
                gnn_model,
                sample_data,
                DEVICE
            )
        else:
            feature_importance = {}
    else:
        feature_importance = {}
    
    # Standard GNN metrics
    gnn_metrics = evaluate_gnn_metrics(gnn_model, test_dates, targets_class_dict, tickers)
    gnn_metrics.update(error_analysis)
    gnn_metrics['feature_importance'] = feature_importance
    
    # 2. RL Metrics with Trading Behavior Analysis
    print("\n" + "=" * 60)
    print("📊 Step 2: Enhanced RL Evaluation")
    print("=" * 60)
    
    rl_agent_file = PROJECT_ROOT / "models" / "rl_ppo_agent_model" / "ppo_stock_agent.zip"
    if rl_agent_file.exists():
        # Run backtest and collect trading data
        from src.rl.environment import StockTradingEnv
        from src.rl.agent import StockTradingAgent
        
        # Setup environment
        def make_test_env():
            return StockTradingEnv(
                start_date=test_dates[0] if test_dates else pd.to_datetime('2023-01-01'),
                end_date=test_dates[-1] if test_dates else pd.to_datetime('2024-12-31'),
                gnn_model=gnn_model,
                device=DEVICE
            )
        
        # Load agent
        agent = StockTradingAgent(
            gnn_model=gnn_model,
            env_factory=make_test_env,
            device=DEVICE,
            learning_rate=1e-5,
            tensorboard_log=None,
            policy="MlpPolicy",
            verbose=0
        )
        agent.load(rl_agent_file)
        
        # Run backtest
        test_env = make_test_env()
        obs, info = test_env.reset()
        portfolio_values = [test_env.initial_cash]
        daily_returns = []
        trades_history = []
        done = False
        
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            
            portfolio_values.append(info['portfolio_value'])
            if len(portfolio_values) > 1:
                daily_returns.append((portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2])
            trades_history.append(info)
        
        # Calculate metrics
        rl_metrics = calculate_financial_metrics(portfolio_values, len(portfolio_values) - 1)
        rl_metrics['final_value'] = portfolio_values[-1]
        rl_metrics['total_days'] = len(portfolio_values) - 1
        
        # Trading behavior analysis
        trading_behavior = analyze_trading_behavior(
            portfolio_values,
            trades_history,
            test_dates[:len(portfolio_values)]
        )
        rl_metrics.update(trading_behavior)
        
        # Create visualizations
        create_visualizations(
            portfolio_values,
            test_dates[:len(portfolio_values)],
            daily_returns
        )
        
    else:
        print("⚠️  RL agent not found, skipping RL analysis")
        rl_metrics = {}
        trading_behavior = {}
    
    # 3. Generate Comprehensive Report
    print("\n" + "=" * 60)
    print("📊 Step 3: Generating Comprehensive Report")
    print("=" * 60)
    
    comprehensive_report = generate_comprehensive_report(
        gnn_metrics,
        rl_metrics,
        error_analysis,
        feature_importance,
        trading_behavior
    )
    
    # Save to CSV for easy viewing
    summary_df = pd.DataFrame([{
        'GNN_F1_Score': gnn_metrics.get('f1_score', 0),
        'GNN_Accuracy': gnn_metrics.get('accuracy', 0),
        'Precision@Top10': gnn_metrics.get('Precision@Top-10', 0),
        'RL_Sharpe_Ratio': rl_metrics.get('Sharpe_Ratio', 0),
        'RL_Cumulative_Return': rl_metrics.get('Cumulative_Return', 0),
        'RL_Max_Drawdown': rl_metrics.get('Max_Drawdown', 0),
        'Win_Rate': trading_behavior.get('win_rate', 0),
        'Avg_Return_per_Trade': trading_behavior.get('average_return_per_trade', 0),
        'Turnover_Rate': trading_behavior.get('turnover_rate', 0),
        'False_Positive_Rate': error_analysis.get('false_positive_rate', 0),
        'False_Negative_Rate': error_analysis.get('false_negative_rate', 0)
    }])
    summary_df.to_csv(RESULTS_DIR / 'enhanced_evaluation_summary.csv', index=False)
    
    print("\n" + "=" * 60)
    print("✅ Enhanced Evaluation Complete!")
    print("=" * 60)
    print(f"\n📁 Results saved to:")
    print(f"  - {RESULTS_DIR / 'comprehensive_analysis_report.json'}")
    print(f"  - {RESULTS_DIR / 'enhanced_evaluation_summary.csv'}")
    print(f"  - {PLOTS_DIR} (visualizations)")
    
    return comprehensive_report


if __name__ == '__main__':
    run_enhanced_evaluation()


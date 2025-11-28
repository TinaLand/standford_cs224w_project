# scripts/phase6_evaluation.py

import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, List

# Import necessary modules from previous phases
from phase4_core_training import RoleAwareGraphTransformer 
from phase5_rl_integration import load_gnn_model_for_rl 
from rl_environment import StockTradingEnv
from rl_agent import StockTradingAgent 

# --- Configuration & Setup ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RL_SAVE_PATH = MODELS_DIR / "rl_ppo_agent_model"
RESULTS_DIR = PROJECT_ROOT / "results"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Evaluation Period: Should be the final segment (Test Set)
START_DATE_TEST = pd.to_datetime('2023-01-01') # Placeholder: Use actual test split start
END_DATE_TEST = pd.to_datetime('2024-12-31')   # Placeholder

# Ensure results directory exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --- 1. Performance Metric Calculation ---

def calculate_financial_metrics(portfolio_values: List[float], trading_days: int) -> Dict[str, float]:
    """
    Calculates key financial metrics including Cumulative Return, Sharpe Ratio, and Max Drawdown.
    """
    if len(portfolio_values) < 2:
        return {'Cumulative_Return': 0, 'Sharpe_Ratio': 0, 'Max_Drawdown': 0}

    # Convert to pandas Series for easier calculations
    portfolio_series = pd.Series(portfolio_values)
    
    # Convert to daily returns
    returns = portfolio_series.pct_change().dropna()
    
    # Cumulative Return
    cumulative_return = (portfolio_values[-1] / portfolio_values[0]) - 1

    # Sharpe Ratio (assuming Risk-Free Rate R_f = 0 for simplicity)
    # Annualization factor = 252 (trading days)
    annualization_factor = np.sqrt(252)
    sharpe_ratio = returns.mean() / returns.std() * annualization_factor if returns.std() != 0 else 0

    # Max Drawdown
    cumulative_max = portfolio_series.cummax()
    drawdown = (cumulative_max - portfolio_series) / cumulative_max
    max_drawdown = drawdown.max()
    
    return {
        'Cumulative_Return': cumulative_return,
        'Sharpe_Ratio': sharpe_ratio,
        'Max_Drawdown': max_drawdown
    }


def calculate_precision_at_topk(predictions: np.ndarray, targets: np.ndarray, k: int = 10) -> float:
    """
    Calculate Precision@Top-K metric.
    
    For each day, select the top-K stocks with highest predicted probabilities for "Up" class.
    Then calculate the precision: how many of these top-K stocks actually went up.
    
    Args:
        predictions: Array of shape (num_days, num_stocks, num_classes) - predicted probabilities
                     or (num_days, num_stocks) - predicted class labels
        targets: Array of shape (num_days, num_stocks) - true class labels (0=Down, 1=Up)
        k: Number of top stocks to consider (default: 10)
    
    Returns:
        Precision@Top-K score (float between 0 and 1)
    """
    if len(predictions.shape) == 3:
        # If predictions are probabilities, extract probability of "Up" class (class 1)
        pred_probs = predictions[:, :, 1]  # Shape: (num_days, num_stocks)
    else:
        # If predictions are class labels, convert to probabilities (simplified)
        pred_probs = predictions.astype(float)
    
    num_days, num_stocks = pred_probs.shape
    if k > num_stocks:
        k = num_stocks
    
    correct_predictions = 0
    total_selected = 0
    
    for day_idx in range(num_days):
        # Get top-K stocks for this day (highest predicted probability of going up)
        topk_indices = np.argsort(pred_probs[day_idx])[-k:][::-1]  # Descending order
        
        # Count how many of these top-K actually went up
        topk_targets = targets[day_idx, topk_indices]
        correct_predictions += np.sum(topk_targets == 1)
        total_selected += len(topk_indices)
    
    precision_at_k = correct_predictions / total_selected if total_selected > 0 else 0.0
    return precision_at_k


def calculate_information_coefficient(predictions: np.ndarray, actual_returns: np.ndarray) -> Dict[str, float]:
    """
    Calculate Information Coefficient (IC) - correlation between predictions and actual returns.
    
    IC measures the predictive power of the model:
    - IC = correlation(predicted_returns, actual_returns)
    - IC > 0: Positive predictive power
    - IC ≈ 0: No predictive power (random)
    - IC < 0: Negative predictive power
    
    Args:
        predictions: Array of shape (num_days, num_stocks) - predicted class probabilities or logits
        actual_returns: Array of shape (num_days, num_stocks) - actual forward returns (continuous values)
    
    Returns:
        Dictionary with:
        - 'IC_mean': Mean IC across all days
        - 'IC_std': Standard deviation of IC
        - 'IC_IR': Information Ratio (IC_mean / IC_std) - measures consistency
    """
    from scipy.stats import pearsonr
    
    num_days, num_stocks = predictions.shape
    
    # If predictions are probabilities, convert to expected returns
    # For binary classification: expected_return = prob_up - prob_down
    if predictions.max() <= 1.0 and predictions.min() >= 0.0:
        # Probabilities: convert to expected returns signal
        pred_returns = predictions * 2 - 1  # Map [0,1] to [-1,1]
    else:
        # Logits or raw predictions: use as-is
        pred_returns = predictions
    
    ic_values = []
    
    # Calculate IC for each day
    for day_idx in range(num_days):
        pred_day = pred_returns[day_idx]
        actual_day = actual_returns[day_idx]
        
        # Remove NaN values
        valid_mask = ~(np.isnan(pred_day) | np.isnan(actual_day))
        if np.sum(valid_mask) < 2:  # Need at least 2 points for correlation
            continue
        
        pred_valid = pred_day[valid_mask]
        actual_valid = actual_day[valid_mask]
        
        # Calculate Pearson correlation
        try:
            ic, p_value = pearsonr(pred_valid, actual_valid)
            if not np.isnan(ic):
                ic_values.append(ic)
        except:
            continue
    
    if len(ic_values) == 0:
        return {'IC_mean': 0.0, 'IC_std': 0.0, 'IC_IR': 0.0}
    
    ic_mean = np.mean(ic_values)
    ic_std = np.std(ic_values)
    ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0
    
    return {
        'IC_mean': float(ic_mean),
        'IC_std': float(ic_std),
        'IC_IR': float(ic_ir),
        'IC_values': ic_values  # For detailed analysis
    }


# --- 2. Node-Level Metrics Calculation (GNN Model) ---

def evaluate_gnn_metrics(gnn_model, test_dates, targets_dict, tickers) -> Dict[str, Any]:
    """
    Evaluate GNN model on test set and calculate node-level metrics including Precision@Top-K and IC.
    """
    print("\n--- Evaluating GNN Model Metrics ---")
    
    from phase4_core_training import load_graph_data, evaluate
    
    all_predictions = []
    all_targets = []
    all_probs = []
    
    # Collect predictions for all test dates
    for date in tqdm(test_dates, desc="Evaluating GNN"):
        data = load_graph_data(date)
        target = targets_dict.get(date)
        
        if data and target is not None:
            # Get model predictions
            gnn_model.eval()
            with torch.no_grad():
                out = gnn_model(data.to(DEVICE))  # Shape: (num_stocks, 2)
                probs = F.softmax(out, dim=1)  # Convert logits to probabilities
                preds = out.argmax(dim=1).cpu().numpy()
            
            all_predictions.append(preds)
            all_targets.append(target.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    if len(all_predictions) == 0:
        print("❌ No predictions collected")
        return {}
    
    # Convert to numpy arrays
    predictions_array = np.array(all_predictions)  # Shape: (num_days, num_stocks)
    targets_array = np.array(all_targets)  # Shape: (num_days, num_stocks)
    probs_array = np.array(all_probs)  # Shape: (num_days, num_stocks, 2)
    
    # Calculate Precision@Top-K for different K values
    precision_at_k_results = {}
    for k in [5, 10, 20]:
        precision_k = calculate_precision_at_topk(probs_array, targets_array, k=k)
        precision_at_k_results[f'Precision@Top-{k}'] = precision_k
        print(f"   Precision@Top-{k}: {precision_k:.4f}")
    
    # Calculate IC (need actual returns, not just binary labels)
    # For IC, we need to load actual forward returns
    try:
        from phase4_core_training import _read_time_series_csv, OHLCV_RAW_FILE
        ohlcv_df = _read_time_series_csv(OHLCV_RAW_FILE)
        
        # Calculate 5-day forward returns
        close_cols = [col for col in ohlcv_df.columns if col.startswith('Close_')]
        close_prices = ohlcv_df[close_cols].copy()
        forward_returns = (close_prices.shift(-5) - close_prices) / close_prices
        
        # Extract returns for test dates
        actual_returns_list = []
        for date in test_dates:
            date_str = date.strftime('%Y-%m-%d')
            if date_str in forward_returns.index:
                returns_vector = []
                for ticker in tickers:
                    col_name = f'Close_{ticker}'
                    if col_name in forward_returns.columns:
                        ret = forward_returns.loc[date_str, col_name]
                        returns_vector.append(ret if not pd.isna(ret) else 0.0)
                    else:
                        returns_vector.append(0.0)
                actual_returns_list.append(returns_vector)
            else:
                actual_returns_list.append([0.0] * len(tickers))
        
        actual_returns_array = np.array(actual_returns_list)  # Shape: (num_days, num_stocks)
        
        # Calculate IC using probability of "Up" class as prediction signal
        pred_probs_up = probs_array[:, :, 1]  # Probability of "Up" class
        ic_results = calculate_information_coefficient(pred_probs_up, actual_returns_array)
        
        print(f"   IC Mean: {ic_results['IC_mean']:.4f}")
        print(f"   IC Std: {ic_results['IC_std']:.4f}")
        print(f"   IC IR (Information Ratio): {ic_results['IC_IR']:.4f}")
        
    except Exception as e:
        print(f"⚠️  Could not calculate IC: {e}")
        ic_results = {'IC_mean': 0.0, 'IC_std': 0.0, 'IC_IR': 0.0}
    
    # Calculate standard metrics
    from sklearn.metrics import accuracy_score, f1_score
    flat_predictions = predictions_array.flatten()
    flat_targets = targets_array.flatten()
    
    accuracy = accuracy_score(flat_targets, flat_predictions)
    f1 = f1_score(flat_targets, flat_predictions, average='macro')
    
    results = {
        'accuracy': accuracy,
        'f1_score': f1,
        **precision_at_k_results,
        **ic_results
    }
    
    return results


# --- 3. Final Backtesting (Evaluation Run) ---

def run_final_backtest(gnn_model, rl_agent_path: Path) -> Dict[str, Any]:
    """
    Runs the final backtest on the held-out test set using the trained RL agent.
    """
    print("\n--- 2. Final Backtesting on Test Set ---")
    
    # Setup environment factory for the evaluation period
    def make_test_env():
        return StockTradingEnv(
            start_date=START_DATE_TEST,
            end_date=END_DATE_TEST,
            gnn_model=gnn_model,
            device=DEVICE
        )
    
    # Load the trained RL agent using Agent class
    agent_path = rl_agent_path
    if not agent_path.exists():
        # Try with .zip extension
        agent_path = Path(str(rl_agent_path) + ".zip")
        if not agent_path.exists():
            raise FileNotFoundError(f"RL Agent not found at: {rl_agent_path}. Ensure Phase 5 finished.")
    
    # Create agent wrapper and load trained weights
    agent = StockTradingAgent(
        gnn_model=gnn_model,
        env_factory=make_test_env,
        device=DEVICE,
        learning_rate=1e-5,  # Not used for inference
        tensorboard_log=None,
        policy="MlpPolicy",
        verbose=0
    )
    
    # Load trained agent
    agent.load(agent_path)
    
    # Create test environment
    test_env = make_test_env()

    # Initialize environment and run simulation
    obs, info = test_env.reset()
    portfolio_values = [test_env.initial_cash]
    done = False
    
    print(f"Starting value: ${test_env.initial_cash:.2f}")

    while not done:
        # RL agent determines action
        action, _ = agent.predict(obs, deterministic=True) 
        
        # Step environment
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated

        portfolio_values.append(info['portfolio_value'])
        
        # Progress check
        if test_env.current_step % 100 == 0:
            print(f"  Step {test_env.current_step}: Value ${info['portfolio_value']:.2f}")

    # Calculate final metrics
    metrics = calculate_financial_metrics(portfolio_values, test_env.current_step)
    
    print("\n✅ Backtest Complete.")
    print(f"Final Value: ${portfolio_values[-1]:.2f}")
    
    return {
        'strategy': 'Core_GNN_RL',
        'final_value': portfolio_values[-1],
        'total_days': test_env.current_step,
        **metrics
    }


# --- 3. Ablation Studies (The Research Component) ---

def modify_graph_for_ablation(data, config: Dict) -> torch.Tensor:
    """
    Modify graph data for ablation study by removing specific edge types.
    
    Args:
        data: HeteroData graph object
        config: Dictionary with ablation configuration
            - 'remove_edges': List of edge types to remove (e.g., ['rolling_correlation'])
            - 'disable_pearl': Boolean to disable PEARL (handled at model level)
    
    Returns:
        Modified HeteroData object
    """
    from torch_geometric.data import HeteroData
    
    # Create a copy of the data
    modified_data = HeteroData()
    modified_data['stock'].x = data['stock'].x.clone()
    
    # Copy tickers if present
    if 'tickers' in data:
        modified_data['tickers'] = data['tickers']
    
    # Add edges based on config
    remove_edges = config.get('remove_edges', [])
    
    for edge_type in data.edge_index_dict.keys():
        if edge_type not in remove_edges:
            # Keep this edge type
            modified_data[edge_type].edge_index = data[edge_type].edge_index.clone()
            if hasattr(data[edge_type], 'edge_attr'):
                modified_data[edge_type].edge_attr = data[edge_type].edge_attr.clone()
    
    return modified_data


def evaluate_ablation_gnn(gnn_model, test_dates, targets_dict, tickers, config: Dict) -> Dict[str, Any]:
    """
    Evaluate GNN model with ablation configuration (modified graph structure).
    
    This is a faster alternative to full retraining - we evaluate the existing model
    on modified graphs to see the impact of removing edge types.
    """
    from phase4_core_training import load_graph_data
    
    all_predictions = []
    all_targets = []
    all_probs = []
    
    gnn_model.eval()
    with torch.no_grad():
        for date in tqdm(test_dates, desc=f"Ablation: {config.get('name', 'unknown')}"):
            data = load_graph_data(date)
            target = targets_dict.get(date)
            
            if data and target is not None:
                # Modify graph according to ablation config
                modified_data = modify_graph_for_ablation(data, config)
                
                # Get predictions
                out = gnn_model(modified_data.to(DEVICE))
                probs = F.softmax(out, dim=1)
                preds = out.argmax(dim=1).cpu().numpy()
                
                all_predictions.append(preds)
                all_targets.append(target.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
    
    if len(all_predictions) == 0:
        return {}
    
    # Calculate metrics
    predictions_array = np.array(all_predictions)
    targets_array = np.array(all_targets)
    probs_array = np.array(all_probs)
    
    from sklearn.metrics import accuracy_score, f1_score
    flat_predictions = predictions_array.flatten()
    flat_targets = targets_array.flatten()
    
    accuracy = accuracy_score(flat_targets, flat_predictions)
    f1 = f1_score(flat_targets, flat_predictions, average='macro')
    
    # Calculate Precision@Top-K
    precision_k = calculate_precision_at_topk(probs_array, targets_array, k=10)
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision_at_top10': precision_k
    }


def train_and_evaluate_ablation(ablation_name: str, config_modifier: Dict) -> Dict[str, Any]:
    """
    Evaluate ablation study by modifying graph structure and evaluating performance.
    
    This version uses the existing trained model and evaluates on modified graphs,
    which is faster than full retraining but still provides meaningful insights.
    
    Args:
        ablation_name: Name of the ablation study
        config_modifier: Configuration dict with:
            - 'remove_edges': List of edge types to remove
            - 'disable_pearl': Boolean (requires model modification, not implemented here)
    
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n--- Running Ablation: {ablation_name} ---")
    print(f"   Config: {config_modifier}")
    
    # Load model and test data
    gnn_model = load_gnn_model_for_rl()
    
    # Get test dates and tickers
    from phase4_core_training import create_target_labels, _read_time_series_csv, OHLCV_RAW_FILE
    import pandas as pd
    
    graph_files = sorted(list((PROJECT_ROOT / "data" / "graphs").glob('graph_t_*.pt')))
    all_dates = [pd.to_datetime(f.stem.split('_')[-1]) for f in graph_files]
    split_85_idx = int(len(all_dates) * 0.85)
    test_dates = all_dates[split_85_idx:]
    
    sample_graph = torch.load(graph_files[0], weights_only=False)
    if 'tickers' in sample_graph:
        tickers = sample_graph['tickers']
    else:
        ohlcv_df = _read_time_series_csv(OHLCV_RAW_FILE)
        tickers = [col.replace('Close_', '') for col in ohlcv_df.columns if col.startswith('Close_')]
    
    targets_dict = create_target_labels(tickers, all_dates, lookahead_days=5)
    
    # Add name to config
    config_modifier['name'] = ablation_name
    
    # Evaluate with modified graphs
    metrics = evaluate_ablation_gnn(gnn_model, test_dates, targets_dict, tickers, config_modifier)
    
    print(f"   Results: Accuracy={metrics.get('accuracy', 0):.4f}, "
          f"F1={metrics.get('f1_score', 0):.4f}, "
          f"Precision@Top-10={metrics.get('precision_at_top10', 0):.4f}")
    
    return {
        'strategy': ablation_name,
        **metrics
    }

def run_ablation_studies():
    """Defines and runs the full set of ablation studies from the proposal."""
    
    print("\n" + "=" * 50)
    print("🔬 Starting Ablation Studies")
    print("=" * 50)
    print("Note: Using existing model on modified graphs (faster than full retraining)")
    print("=" * 50)
    
    # Define ablation studies based on edge types available
    # Edge types: 'rolling_correlation', 'fund_similarity', 'sector_industry', 'supply_competitor'
    ablations = [
        {
            'name': 'Full_Model', 
            'config': {'remove_edges': []}  # No edges removed - baseline
        },
        {
            'name': 'Abl_NoCorrelation', 
            'config': {'remove_edges': ['rolling_correlation']}  # Remove dynamic correlation edges
        },
        {
            'name': 'Abl_NoFundSim', 
            'config': {'remove_edges': ['fund_similarity']}  # Remove fundamental similarity edges
        },
        {
            'name': 'Abl_NoStatic', 
            'config': {'remove_edges': ['sector_industry', 'supply_competitor']}  # Remove all static edges
        },
        {
            'name': 'Abl_OnlyCorrelation', 
            'config': {'remove_edges': ['fund_similarity', 'sector_industry', 'supply_competitor']}  # Only correlation
        },
        {
            'name': 'Abl_OnlyFundSim', 
            'config': {'remove_edges': ['rolling_correlation', 'sector_industry', 'supply_competitor']}  # Only fund sim
        },
    ]
    
    ablation_results = []
    
    # Execute each ablation
    for abl in ablations:
        try:
        result = train_and_evaluate_ablation(abl['name'], abl['config'])
        ablation_results.append(result)
        except Exception as e:
            print(f"⚠️  Error in ablation {abl['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if ablation_results:
    ablation_df = pd.DataFrame(ablation_results)
    ablation_df.to_csv(RESULTS_DIR / 'ablation_results.csv', index=False)
    
        print("\n" + "=" * 50)
        print("✅ Ablation Studies Complete. Results saved.")
        print("=" * 50)
        print("\n📊 Ablation Results Summary:")
        print(ablation_df[['strategy', 'accuracy', 'f1_score', 'precision_at_top10']].to_string(index=False))
        print(f"\n📁 Results saved to: {RESULTS_DIR / 'ablation_results.csv'}")
        
    return ablation_df
    else:
        print("❌ No ablation results generated")
        return pd.DataFrame()


# --- 4. Main Execution ---

def main():
    """Orchestrates the final evaluation and analysis pipeline."""
    
    print("🚀 Starting Phase 6: Final Evaluation and Ablation Studies")
    print("=" * 50)

    # Load the core GNN model structure (needed for environment setup)
    gnn_model = load_gnn_model_for_rl()

    # Get test dates and tickers
    from phase4_core_training import load_graph_data, create_target_labels, _read_time_series_csv, OHLCV_RAW_FILE
    import pandas as pd
    
    # Get all graph dates
    graph_files = sorted(list((PROJECT_ROOT / "data" / "graphs").glob('graph_t_*.pt')))
    all_dates = [pd.to_datetime(f.stem.split('_')[-1]) for f in graph_files]
    
    # Use same split as Phase 4 (70/15/15)
    split_70_idx = int(len(all_dates) * 0.70)
    split_85_idx = int(len(all_dates) * 0.85)
    test_dates = all_dates[split_85_idx:]
    
    # Get tickers
    sample_graph = torch.load(graph_files[0], weights_only=False)
    if 'tickers' in sample_graph:
        tickers = sample_graph['tickers']
    else:
        # Fallback: extract from OHLCV data
        ohlcv_df = _read_time_series_csv(OHLCV_RAW_FILE)
        tickers = [col.replace('Close_', '') for col in ohlcv_df.columns if col.startswith('Close_')]
    
    # Create targets
    targets_dict = create_target_labels(tickers, all_dates, lookahead_days=5)

    # 1. Evaluate GNN Model Metrics (Precision@Top-K, IC)
    print("\n" + "=" * 50)
    print("📊 Step 1: GNN Model Node-Level Metrics")
    print("=" * 50)
    gnn_metrics = evaluate_gnn_metrics(gnn_model, test_dates, targets_dict, tickers)
    
    # Save GNN metrics
    gnn_metrics_df = pd.DataFrame([gnn_metrics])
    gnn_metrics_df.to_csv(RESULTS_DIR / 'gnn_node_metrics.csv', index=False)
    print("\n✅ GNN Metrics saved to: results/gnn_node_metrics.csv")

    # 2. Run Final Backtest (RL Agent Performance)
    print("\n" + "=" * 50)
    print("📊 Step 2: RL Agent Portfolio-Level Metrics")
    print("=" * 50)
    rl_agent_file = RL_SAVE_PATH / "ppo_stock_agent.zip"
    if rl_agent_file.exists():
        core_metrics = run_final_backtest(gnn_model, rl_agent_file)
        
        # Save core metrics
        metrics_df = pd.DataFrame([core_metrics])
        metrics_df.to_csv(RESULTS_DIR / 'final_metrics.csv', index=False)
        print("\n🏆 RL AGENT PERFORMANCE:")
        print(metrics_df[['Sharpe_Ratio', 'Cumulative_Return', 'Max_Drawdown']].iloc[0])
    else:
        print("❌ Cannot run final backtest: RL agent not found.")
        metrics_df = pd.DataFrame()

    # 3. Run Ablation Studies
    print("\n" + "=" * 50)
    print("📊 Step 3: Ablation Studies")
    print("=" * 50)
    try:
        ablation_df = run_ablation_studies()
        if not ablation_df.empty:
            print(f"\n📈 ABLATION STUDY RESULTS:\n{ablation_df}")
    except Exception as e:
        print(f"⚠️  Ablation studies failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("✅ Project Execution Finalized!")
    print("🎯 Project is ready for Report Generation and Interpretability Analysis (t-SNE/UMAP).")
    print(f"📁 Results saved to: {RESULTS_DIR}")

if __name__ == '__main__':
    # Dependencies: pip install pandas numpy stable-baselines3 tqdm
    # Note: Requires the final RL agent to be trained via phase5_rl_integration.py
    main()
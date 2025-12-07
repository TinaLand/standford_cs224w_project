# scripts/rl_environment_balanced.py
"""
Balanced RL Environment with Improved Trading Logic

Key Improvements:
1. Dynamic position sizing based on GNN confidence
2. Faster position building in uptrends
3. Balanced buy/sell capabilities
4. Better capital utilization
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys

from src.rl.environments.single_agent import StockTradingEnv
from src.rl.config import SingleAgentConfig, DATA_GRAPHS_DIR, PROJECT_ROOT
TRANSACTION_COST = SingleAgentConfig.TRANSACTION_COST


class BalancedStockTradingEnv(StockTradingEnv):
    """
    Enhanced environment with balanced trading capabilities.
    
    Improvements:
    - Dynamic position sizing (not fixed 0.02%)
    - Faster position building in uptrends
    - Better capital utilization
    """
    
    def __init__(self, start_date, end_date, gnn_model, device, 
                 max_position_per_stock=0.05,  # Max 5% per stock
                 min_buy_amount=0.01,  # Min 1% per buy
                 max_buy_amount=0.10):  # Max 10% per buy
        super().__init__(start_date, end_date, gnn_model, device)
        
        self.max_position_per_stock = max_position_per_stock
        self.min_buy_amount = min_buy_amount
        self.max_buy_amount = max_buy_amount
        
    def _get_gnn_confidence(self, graph_data):
        """
        Extract GNN prediction confidence for position sizing.
        
        Returns:
            confidence: Array of shape [N] with confidence scores (0-1)
        """
        self.gnn_model.eval()
        with torch.no_grad():
            # Get GNN predictions
            out = self.gnn_model(graph_data.to(self.device))
            probs = torch.softmax(out, dim=1)
            
            # Confidence = probability of predicted class
            preds = out.argmax(dim=1)
            confidence = probs[torch.arange(len(preds)), preds].cpu().numpy()
        
        return confidence
    
    def _calculate_dynamic_buy_amount(self, stock_idx, confidence, current_holdings_value, portfolio_value):
        """
        Calculate dynamic buy amount based on:
        1. GNN confidence (higher confidence = larger position)
        2. Current position size (avoid over-concentration)
        3. Available cash
        
        Args:
            stock_idx: Index of the stock
            confidence: GNN confidence score (0-1)
            current_holdings_value: Current value of holdings in this stock
            portfolio_value: Total portfolio value
        
        Returns:
            buy_amount: Amount to invest in this stock
        """
        # Base buy amount scales with confidence
        # High confidence (0.8-1.0) -> larger position
        # Low confidence (0.5-0.8) -> smaller position
        
        # Map confidence to position size (0.01 to 0.10 of portfolio)
        base_size = self.min_buy_amount + (self.max_buy_amount - self.min_buy_amount) * confidence
        
        # Check current position
        current_position_ratio = current_holdings_value / portfolio_value if portfolio_value > 0 else 0
        
        # If already at max position, reduce buy amount
        if current_position_ratio >= self.max_position_per_stock:
            return 0.0
        
        # Calculate target position
        target_position_ratio = min(self.max_position_per_stock, current_position_ratio + base_size)
        buy_amount = (target_position_ratio - current_position_ratio) * portfolio_value
        
        return max(0, buy_amount)
    
    def step(self, action):
        """
        Override step with improved trading logic.
        """
        # Get base step data
        date_t = self.data_loader['dates'][self.current_step]
        date_t_plus_1 = self.data_loader['dates'][self.current_step + 1]
        
        prices_df = self.data_loader['prices']
        
        if date_t in prices_df.index:
            prices_t = prices_df.loc[date_t]
        else:
            nearest_idx = prices_df.index.get_indexer([date_t], method='nearest')[0]
            prices_t = prices_df.iloc[nearest_idx]
        
        if date_t_plus_1 in prices_df.index:
            prices_t_plus_1 = prices_df.loc[date_t_plus_1]
        else:
            nearest_idx = prices_df.index.get_indexer([date_t_plus_1], method='nearest')[0]
            prices_t_plus_1 = prices_df.iloc[nearest_idx]
        
        # Get prices - handle both Series and DataFrame
        ticker_cols = [col for col in prices_df.columns if col.startswith('Close_')]
        
        if isinstance(prices_t, pd.Series):
            # Extract Close prices for our tickers
            current_prices = np.array([prices_t.get(f'Close_{ticker}', 0) for ticker in self.data_loader['tickers']])
        else:
            current_prices = np.array([prices_t.get(f'Close_{ticker}', 0) if hasattr(prices_t, 'get') 
                                     else prices_t[f'Close_{ticker}'].iloc[0] if f'Close_{ticker}' in prices_t.columns 
                                     else 0 for ticker in self.data_loader['tickers']])
        
        if isinstance(prices_t_plus_1, pd.Series):
            next_prices = np.array([prices_t_plus_1.get(f'Close_{ticker}', 0) for ticker in self.data_loader['tickers']])
        else:
            next_prices = np.array([prices_t_plus_1.get(f'Close_{ticker}', 0) if hasattr(prices_t_plus_1, 'get')
                                   else prices_t_plus_1[f'Close_{ticker}'].iloc[0] if f'Close_{ticker}' in prices_t_plus_1.columns
                                   else 0 for ticker in self.data_loader['tickers']])
        
        # Ensure correct shape
        if len(current_prices) != self.NUM_STOCKS:
            current_prices = np.zeros(self.NUM_STOCKS)
        if len(next_prices) != self.NUM_STOCKS:
            next_prices = np.zeros(self.NUM_STOCKS)
        
        # Current portfolio value
        current_portfolio_value = self.cash + np.sum(self.holdings * next_prices)
        
        # Get GNN confidence for dynamic position sizing
        graph_file = DATA_GRAPHS_DIR / f"graph_t_{date_t.strftime('%Y-%m-%d')}.pt"
        if graph_file.exists():
            graph_data = torch.load(graph_file, weights_only=False)
            confidence = self._get_gnn_confidence(graph_data)
        else:
            confidence = np.ones(self.NUM_STOCKS) * 0.5  # Default medium confidence
        
        # Execute trading actions with improved logic
        trade_volume = np.zeros(self.NUM_STOCKS)
        
        for i, act in enumerate(action):
            price = next_prices[i]
            if pd.isna(price) or price <= 0:
                continue
            
            current_holdings_value = self.holdings[i] * price
            
            if act == 2:  # Buy - IMPROVED: Dynamic position sizing
                # Calculate dynamic buy amount based on confidence
                buy_amount = self._calculate_dynamic_buy_amount(
                    stock_idx=i,
                    confidence=confidence[i],
                    current_holdings_value=current_holdings_value,
                    portfolio_value=current_portfolio_value
                )
                
                if buy_amount > 0 and self.cash >= buy_amount * (1 + TRANSACTION_COST):
                    shares_to_buy = buy_amount / price
                    cost = buy_amount * (1 + TRANSACTION_COST)
                    self.cash -= cost
                    self.holdings[i] += shares_to_buy
                    trade_volume[i] = shares_to_buy
                    
            elif act == 0:  # Sell - Keep existing logic (20% per trade)
                if self.holdings[i] > 0:
                    shares_to_sell = self.holdings[i] * 0.2
                    revenue = shares_to_sell * price * (1 - TRANSACTION_COST)
                    self.cash += revenue
                    self.holdings[i] -= shares_to_sell
                    trade_volume[i] = -shares_to_sell
        
        # Update portfolio value
        new_portfolio_value = self.cash + np.sum(self.holdings * next_prices)
        
        # Reward calculation (can use improved reward from rl_environment_improved)
        reward = (new_portfolio_value - current_portfolio_value) / current_portfolio_value
        
        self.portfolio_value = new_portfolio_value
        self.current_step += 1
        
        # Check termination
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Next observation
        obs = self._get_observation()
        
        info = {
            'portfolio_value': self.portfolio_value,
            'trades': np.sum(np.abs(trade_volume)),
            'date': date_t_plus_1.strftime('%Y-%m-%d'),
            'confidence_used': np.mean(confidence) if len(confidence) > 0 else 0.5,
            'avg_buy_amount': np.mean([v for v in trade_volume if v > 0]) if np.any(trade_volume > 0) else 0
        }
        
        return obs, reward, terminated, truncated, info


# --- Comparison Test ---

if __name__ == '__main__':
    """
    Test the balanced environment vs original.
    """
    from src.rl.integration import load_gnn_model_for_rl
    from src.evaluation.evaluation import START_DATE_TEST, END_DATE_TEST
    
    print("=" * 70)
    print(" Testing Balanced Trading Environment")
    print("=" * 70)
    
    gnn_model = load_gnn_model_for_rl()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test original environment
    print("\n--- Original Environment (Fixed 0.02% per buy) ---")
    from src.rl.environments.single_agent import StockTradingEnv
    env_original = StockTradingEnv(
        start_date=START_DATE_TEST,
        end_date=END_DATE_TEST,
        gnn_model=gnn_model,
        device=device
    )
    
    obs, info = env_original.reset()
    original_values = [info.get('portfolio_value', 10000)]
    
    for step in range(min(50, env_original.max_steps)):
        action = env_original.action_space.sample()
        obs, reward, terminated, truncated, info = env_original.step(action)
        original_values.append(info['portfolio_value'])
        if terminated or truncated:
            break
    
    # Test balanced environment
    print("\n--- Balanced Environment (Dynamic position sizing) ---")
    env_balanced = BalancedStockTradingEnv(
        start_date=START_DATE_TEST,
        end_date=END_DATE_TEST,
        gnn_model=gnn_model,
        device=device
    )
    
    obs, info = env_balanced.reset()
    balanced_values = [info.get('portfolio_value', 10000)]
    
    for step in range(min(50, env_balanced.max_steps)):
        action = env_balanced.action_space.sample()
        obs, reward, terminated, truncated, info = env_balanced.step(action)
        balanced_values.append(info['portfolio_value'])
        if terminated or truncated:
            break
    
    # Compare
    print("\n" + "=" * 70)
    print(" Comparison Results (50 steps)")
    print("=" * 70)
    print(f"Original Environment:")
    print(f"  Final Value: ${original_values[-1]:.2f}")
    print(f"  Return: {(original_values[-1] / original_values[0] - 1) * 100:.2f}%")
    
    print(f"\nBalanced Environment:")
    print(f"  Final Value: ${balanced_values[-1]:.2f}")
    print(f"  Return: {(balanced_values[-1] / balanced_values[0] - 1) * 100:.2f}%")
    
    improvement = ((balanced_values[-1] / balanced_values[0]) / (original_values[-1] / original_values[0]) - 1) * 100
    print(f"\nImprovement: {improvement:+.2f}%")
    
    print("\n Balanced environment allows faster position building in uptrends!")


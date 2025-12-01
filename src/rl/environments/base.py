# src/rl/environments/base.py
"""
Base trading environment implementation
Shared functionality for single-agent and multi-agent environments
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import gymnasium as gym
from gymnasium import spaces

from ..config import SingleAgentConfig, GNNConfig, PROJECT_ROOT


class BaseTradingEnv(gym.Env):
    """
    Base class for trading environments.
    Provides common functionality for portfolio management and GNN integration.
    """
    
    def __init__(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        gnn_model: torch.nn.Module,
        device: torch.device,
        initial_balance: float = 100000.0
    ):
        """
        Initialize base trading environment.
        
        Args:
            start_date: Trading start date
            end_date: Trading end date
            gnn_model: Trained GNN model for state representation
            device: PyTorch device
            initial_balance: Initial portfolio balance
        """
        super().__init__()
        
        self.start_date = start_date
        self.end_date = end_date
        self.gnn_model = gnn_model
        self.device = device
        self.initial_balance = initial_balance
        
        # Load available graph files
        self.data_dir = PROJECT_ROOT / "data" / "graphs"
        self.graph_files = sorted(list(self.data_dir.glob('graph_t_*.pt')))
        
        if not self.graph_files:
            raise ValueError(f"No graph files found in {self.data_dir}")
        
        # Filter by date range
        self.valid_files = self._filter_files_by_date()
        if not self.valid_files:
            raise ValueError(f"No graph files found for date range {start_date} to {end_date}")
        
        # Load first graph to get dimensions
        sample_data = torch.load(self.valid_files[0], weights_only=False)
        self.tickers = getattr(sample_data, 'tickers', list(range(sample_data['stock'].x.shape[0])))
        self.num_stocks = len(self.tickers)
        
        # Get embedding dimension from GNN model
        self.embedding_dim = self._get_embedding_dim()
        
        # Portfolio state
        self.reset_portfolio()
        
        # Current step
        self.current_step = 0
        
        # Define action and observation spaces (to be overridden by subclasses)
        self.action_space = None
        self.observation_space = None
    
    def _filter_files_by_date(self) -> List[Path]:
        """Filter graph files by date range."""
        valid_files = []
        
        for file_path in self.graph_files:
            # Extract date from filename: graph_t_YYYYMMDD.pt
            date_str = file_path.stem.split('_')[-1]
            try:
                file_date = pd.to_datetime(date_str)
                if self.start_date <= file_date <= self.end_date:
                    valid_files.append(file_path)
            except (ValueError, IndexError):
                continue
        
        return valid_files
    
    def _get_embedding_dim(self) -> int:
        """Get embedding dimension from GNN model."""
        try:
            # Load sample data and get embedding
            sample_data = torch.load(self.valid_files[0], weights_only=False)
            sample_data = sample_data.to(self.device)
            
            # Create proper date tensor - match number of nodes
            num_nodes = sample_data['stock'].x.shape[0]
            date_tensor = torch.zeros(num_nodes, device=self.device)
            
            with torch.no_grad():
                embeddings = self.gnn_model.get_embeddings(sample_data, date_tensor)
                return embeddings.shape[1]
        except Exception as e:
            print(f"Warning: Could not determine embedding dimension: {e}")
            return GNNConfig.OUT_CHANNELS  # Fallback to config
    
    def reset_portfolio(self):
        """Reset portfolio to initial state."""
        self.balance = self.initial_balance
        self.holdings = np.zeros(self.num_stocks, dtype=np.float32)
        self.portfolio_value = self.initial_balance
        self.transaction_costs = 0.0
        self.trade_history = []
    
    def get_current_data(self) -> torch.Tensor:
        """Get current graph data."""
        if self.current_step >= len(self.valid_files):
            return None
        
        file_path = self.valid_files[self.current_step]
        data = torch.load(file_path, weights_only=False)
        return data.to(self.device)
    
    def get_gnn_embeddings(self, data) -> np.ndarray:
        """Get GNN embeddings for current state."""
        if data is None:
            return np.zeros((self.num_stocks, self.embedding_dim), dtype=np.float32)
        
        try:
            # Extract date from current file
            current_file = self.valid_files[self.current_step]
            date_str = current_file.stem.split('_')[-1]
            file_date = pd.to_datetime(date_str)
            
            # Create date tensor - match number of nodes
            num_nodes = data['stock'].x.shape[0]
            date_value = file_date.dayofyear / 365.0
            date_tensor = torch.full((num_nodes,), date_value, device=self.device)
            
            with torch.no_grad():
                embeddings = self.gnn_model.get_embeddings(data, date_tensor)
                return embeddings.cpu().numpy().astype(np.float32)
        
        except Exception as e:
            print(f"Warning: Could not get GNN embeddings: {e}")
            return np.zeros((self.num_stocks, self.embedding_dim), dtype=np.float32)
    
    def calculate_portfolio_value(self, data) -> float:
        """Calculate current portfolio value."""
        if data is None:
            return self.balance
        
        try:
            # Get current prices (use closing prices from node features)
            prices = data['stock'].x[:, 3].cpu().numpy()  # Assuming close price is at index 3
            stock_values = self.holdings * prices
            total_value = self.balance + np.sum(stock_values)
            return total_value
        except Exception:
            return self.balance + np.sum(self.holdings)  # Fallback
    
    def execute_trades(self, actions: np.ndarray, data) -> float:
        """
        Execute trades based on actions.
        
        Args:
            actions: Trading actions for each stock
            data: Current graph data
            
        Returns:
            Transaction costs incurred
        """
        if data is None:
            return 0.0
        
        try:
            prices = data['stock'].x[:, 3].cpu().numpy()
        except Exception:
            prices = np.ones(self.num_stocks)  # Fallback
        
        total_cost = 0.0
        
        for i, action in enumerate(actions):
            if action == 1:  # Buy
                # Buy with available balance (simple strategy)
                max_shares = self.balance / (prices[i] * self.num_stocks)
                shares_to_buy = max_shares * 0.1  # Buy 10% of max possible
                
                cost = shares_to_buy * prices[i]
                transaction_cost = cost * SingleAgentConfig.TRANSACTION_COST
                
                if self.balance >= cost + transaction_cost:
                    self.holdings[i] += shares_to_buy
                    self.balance -= (cost + transaction_cost)
                    total_cost += transaction_cost
                    
                    self.trade_history.append({
                        'step': self.current_step,
                        'ticker': self.tickers[i] if isinstance(self.tickers, list) else i,
                        'action': 'buy',
                        'shares': shares_to_buy,
                        'price': prices[i],
                        'cost': transaction_cost
                    })
            
            elif action == 2:  # Sell
                # Sell 10% of holdings
                shares_to_sell = self.holdings[i] * 0.1
                
                if shares_to_sell > 0:
                    revenue = shares_to_sell * prices[i]
                    transaction_cost = revenue * SingleAgentConfig.TRANSACTION_COST
                    
                    self.holdings[i] -= shares_to_sell
                    self.balance += (revenue - transaction_cost)
                    total_cost += transaction_cost
                    
                    self.trade_history.append({
                        'step': self.current_step,
                        'ticker': self.tickers[i] if isinstance(self.tickers, list) else i,
                        'action': 'sell',
                        'shares': shares_to_sell,
                        'price': prices[i],
                        'cost': transaction_cost
                    })
        
        self.transaction_costs += total_cost
        return total_cost
    
    def calculate_reward(self, prev_value: float, current_value: float, transaction_cost: float) -> float:
        """
        Calculate reward for current step.
        
        Args:
            prev_value: Previous portfolio value
            current_value: Current portfolio value
            transaction_cost: Transaction costs incurred
            
        Returns:
            Reward value
        """
        # Simple return-based reward
        return_pct = (current_value - prev_value) / prev_value if prev_value > 0 else 0
        
        # Subtract transaction costs
        cost_penalty = transaction_cost / prev_value if prev_value > 0 else 0
        
        reward = return_pct - cost_penalty
        
        # Optional: Add Sharpe ratio component
        if SingleAgentConfig.USE_SHARPE_REWARD and len(self.trade_history) >= SingleAgentConfig.SHARPE_WINDOW:
            # Calculate rolling Sharpe ratio (simplified)
            recent_returns = [trade.get('return', 0) for trade in self.trade_history[-SingleAgentConfig.SHARPE_WINDOW:]]
            if len(recent_returns) > 1:
                mean_return = np.mean(recent_returns)
                std_return = np.std(recent_returns)
                if std_return > 0:
                    sharpe = (mean_return - SingleAgentConfig.RISK_FREE_RATE/252) / std_return
                    reward += sharpe * 0.1  # Small Sharpe bonus
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation (to be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement _get_observation")
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.reset_portfolio()
        
        # Get initial observation
        data = self.get_current_data()
        if data is not None:
            self.portfolio_value = self.calculate_portfolio_value(data)
        
        observation = self._get_observation()
        info = {
            'portfolio_value': self.portfolio_value,
            'balance': self.balance,
            'step': self.current_step
        }
        
        return observation, info
    
    def step(self, action):
        """Execute one step in the environment (to be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement step")
    
    def close(self):
        """Clean up environment."""
        pass
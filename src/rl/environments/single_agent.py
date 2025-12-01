# src/rl/environments/single_agent.py
"""
Single-Agent Trading Environment
Enhanced version of the original trading environment with improved structure
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, Tuple

from .base import BaseTradingEnv
from ..config import SingleAgentConfig


class StockTradingEnv(BaseTradingEnv):
    """
    Single-agent trading environment for stock portfolio management.
    Extends BaseTradingEnv with single-agent specific functionality.
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
        Initialize single-agent trading environment.
        
        Args:
            start_date: Trading start date
            end_date: Trading end date
            gnn_model: Trained GNN model for state representation
            device: PyTorch device
            initial_balance: Initial portfolio balance
        """
        super().__init__(start_date, end_date, gnn_model, device, initial_balance)
        
        # Define action and observation spaces
        self._setup_action_observation_spaces()
        
        # Portfolio tracking
        self.returns_history = []
        
    def _setup_action_observation_spaces(self):
        """Setup action and observation spaces for single-agent environment."""
        # Action Space: Buy/Sell/Hold for each stock
        # 0: Sell, 1: Hold, 2: Buy
        self.action_space = spaces.MultiDiscrete([3] * self.num_stocks)
        
        # Observation Space: [Holdings (N)] + [GNN Embeddings (N * H)]
        state_dim = self.num_stocks + (self.num_stocks * self.embedding_dim)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(state_dim,), 
            dtype=np.float32
        )
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation for single agent."""
        # Get current graph data
        data = self.get_current_data()
        
        if data is None:
            # Fallback observation
            holdings_norm = np.zeros(self.num_stocks, dtype=np.float32)
            embeddings_flat = np.zeros(self.num_stocks * self.embedding_dim, dtype=np.float32)
            return np.concatenate([holdings_norm, embeddings_flat])
        
        # Get GNN embeddings
        embeddings = self.get_gnn_embeddings(data)
        embeddings_flat = embeddings.flatten()
        
        # Normalize holdings
        total_value = self.portfolio_value if self.portfolio_value > 0 else 1.0
        holdings_norm = self.holdings / total_value
        
        # Concatenate observation
        observation = np.concatenate([holdings_norm, embeddings_flat]).astype(np.float32)
        
        return observation
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Trading actions for each stock
            
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        if self.current_step >= len(self.valid_files) - 1:
            # End of data
            obs = self._get_observation()
            info = {
                'portfolio_value': self.portfolio_value,
                'total_return': (self.portfolio_value - self.initial_balance) / self.initial_balance,
                'step': self.current_step
            }
            return obs, 0.0, True, False, info
        
        # Get current data
        current_data = self.get_current_data()
        prev_value = self.portfolio_value
        
        # Execute trades
        transaction_cost = self.execute_trades(action, current_data)
        
        # Move to next step
        self.current_step += 1
        next_data = self.get_current_data()
        
        # Calculate new portfolio value
        new_value = self.calculate_portfolio_value(next_data)
        self.portfolio_value = new_value
        
        # Calculate reward
        reward = self.calculate_reward(prev_value, new_value, transaction_cost)
        
        # Track returns for Sharpe calculation
        if prev_value > 0:
            period_return = (new_value - prev_value) / prev_value
            self.returns_history.append(period_return)
            
            # Keep only recent returns for Sharpe calculation
            if len(self.returns_history) > SingleAgentConfig.SHARPE_WINDOW:
                self.returns_history.pop(0)
        
        # Get next observation
        obs = self._get_observation()
        
        # Check termination
        terminated = self.current_step >= len(self.valid_files) - 1
        truncated = False
        
        # Info
        info = {
            'portfolio_value': self.portfolio_value,
            'balance': self.balance,
            'total_return': (self.portfolio_value - self.initial_balance) / self.initial_balance,
            'transaction_cost': transaction_cost,
            'step': self.current_step
        }
        
        # Add Sharpe ratio if enough data
        if len(self.returns_history) >= 10:
            returns_array = np.array(self.returns_history)
            if np.std(returns_array) > 0:
                excess_returns = returns_array - SingleAgentConfig.RISK_FREE_RATE / 252
                sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
                info['sharpe_ratio'] = sharpe_ratio
        
        return obs, reward, terminated, truncated, info
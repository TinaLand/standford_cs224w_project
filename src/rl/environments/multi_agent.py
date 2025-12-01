# src/rl/environments/multi_agent.py
"""
Multi-Agent Trading Environment
Environment wrapper for sector-based multi-agent trading
"""

import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from gymnasium import spaces

from .base import BaseTradingEnv
from ..config import MultiAgentConfig


class MultiAgentTradingEnv(BaseTradingEnv):
    """
    Multi-Agent Trading Environment.
    Supports sector-based agents with coordinated trading.
    """
    
    def __init__(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        gnn_model: torch.nn.Module,
        sector_groups: Dict[str, List[str]],
        device: torch.device,
        initial_balance: float = 100000.0
    ):
        """
        Initialize multi-agent trading environment.
        
        Args:
            start_date: Trading start date
            end_date: Trading end date
            gnn_model: Trained GNN model for state representation
            sector_groups: Dictionary mapping sector names to ticker lists
            device: PyTorch device
            initial_balance: Initial portfolio balance
        """
        super().__init__(start_date, end_date, gnn_model, device, initial_balance)
        
        self.sector_groups = sector_groups
        
        # Create ticker to sector mapping
        self.ticker_to_sector = {}
        self.sector_indices = {}
        
        for sector, tickers in sector_groups.items():
            for ticker in tickers:
                if ticker in self.tickers:
                    self.ticker_to_sector[ticker] = sector
            
            # Map sector to stock indices
            indices = [i for i, ticker in enumerate(self.tickers) if ticker in tickers]
            self.sector_indices[sector] = indices
        
        # Setup action and observation spaces
        self._setup_spaces()
        
        # Coordinator reference (set externally)
        self.coordinator = None
    
    def _setup_spaces(self):
        """Setup action and observation spaces for multi-agent environment."""
        # Action space is handled by individual agents
        # This environment receives pre-processed actions
        total_actions = sum(len(indices) for indices in self.sector_indices.values())
        self.action_space = spaces.MultiDiscrete([3] * total_actions)
        
        # Observation space: global state
        state_dim = self.num_stocks + (self.num_stocks * self.embedding_dim)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )
    
    def get_sector_observations(self, global_obs: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Split global observation into sector-specific observations.
        
        Args:
            global_obs: Global observation vector
            
        Returns:
            Dictionary mapping sector names to sector observations
        """
        sector_obs = {}
        
        # Split observation: [holdings] + [embeddings]
        holdings = global_obs[:self.num_stocks]
        embeddings_flat = global_obs[self.num_stocks:]
        
        # Reshape embeddings
        embeddings = embeddings_flat.reshape(self.num_stocks, self.embedding_dim)
        
        for sector, indices in self.sector_indices.items():
            if len(indices) > 0:
                # Extract sector-specific data
                sector_holdings = holdings[indices]
                sector_embeddings = embeddings[indices]
                
                # Flatten and concatenate
                sector_embeddings_flat = sector_embeddings.flatten()
                sector_observation = np.concatenate([
                    sector_holdings, 
                    sector_embeddings_flat
                ]).astype(np.float32)
                
                # Pad or truncate to expected size
                expected_size = len(indices) * (1 + self.embedding_dim)
                if len(sector_observation) < expected_size:
                    padding = np.zeros(expected_size - len(sector_observation))
                    sector_observation = np.concatenate([sector_observation, padding])
                elif len(sector_observation) > expected_size:
                    sector_observation = sector_observation[:expected_size]
                
                sector_obs[sector] = sector_observation
        
        return sector_obs
    
    def merge_actions(
        self,
        actions_dict: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Merge sector-specific actions into global action vector.
        
        Args:
            actions_dict: Dictionary mapping sector names to actions
            
        Returns:
            Combined action vector for all stocks
        """
        combined_actions = np.zeros(self.num_stocks, dtype=np.int32)
        
        for sector, sector_actions in actions_dict.items():
            indices = self.sector_indices.get(sector, [])
            for i, action in enumerate(sector_actions):
                if i < len(indices):
                    combined_actions[indices[i]] = action
        
        return combined_actions
    
    def _get_observation(self) -> np.ndarray:
        """Get global observation for multi-agent environment."""
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
    
    def step(self, actions_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, float], bool, bool, Dict[str, Any]]:
        """
        Step environment with actions from multiple agents.
        
        Args:
            actions_dict: Dictionary mapping sector names to actions
            
        Returns:
            (global_obs, sector_rewards, terminated, truncated, info)
        """
        if self.current_step >= len(self.valid_files) - 1:
            # End of data
            obs = self._get_observation()
            sector_obs = self.get_sector_observations(obs)
            sector_rewards = {sector: 0.0 for sector in self.sector_groups.keys()}
            info = {
                'portfolio_value': self.portfolio_value,
                'total_return': (self.portfolio_value - self.initial_balance) / self.initial_balance,
                'step': self.current_step,
                'sector_observations': sector_obs
            }
            return obs, sector_rewards, True, False, info
        
        # Get current data
        current_data = self.get_current_data()
        prev_value = self.portfolio_value
        prev_sector_values = self._calculate_sector_values(current_data)
        
        # Merge actions from all agents
        combined_actions = self.merge_actions(actions_dict)
        
        # Execute trades
        transaction_cost = self.execute_trades(combined_actions, current_data)
        
        # Move to next step
        self.current_step += 1
        next_data = self.get_current_data()
        
        # Calculate new portfolio value
        new_value = self.calculate_portfolio_value(next_data)
        self.portfolio_value = new_value
        
        # Calculate sector-specific rewards
        new_sector_values = self._calculate_sector_values(next_data)
        sector_rewards = {}
        
        for sector in self.sector_groups.keys():
            prev_sector_val = prev_sector_values.get(sector, 0)
            new_sector_val = new_sector_values.get(sector, 0)
            
            if prev_sector_val > 0:
                sector_return = (new_sector_val - prev_sector_val) / prev_sector_val
                # Add coordination bonus based on global performance
                global_return = (new_value - prev_value) / prev_value if prev_value > 0 else 0
                coordination_bonus = global_return * 0.1  # Small coordination incentive
                sector_rewards[sector] = sector_return + coordination_bonus
            else:
                sector_rewards[sector] = 0.0
        
        # Get next observation
        obs = self._get_observation()
        sector_obs = self.get_sector_observations(obs)
        
        # Check termination
        terminated = self.current_step >= len(self.valid_files) - 1
        truncated = False
        
        # Info
        info = {
            'portfolio_value': self.portfolio_value,
            'balance': self.balance,
            'total_return': (self.portfolio_value - self.initial_balance) / self.initial_balance,
            'transaction_cost': transaction_cost,
            'step': self.current_step,
            'sector_observations': sector_obs,
            'sector_values': new_sector_values
        }
        
        return obs, sector_rewards, terminated, truncated, info
    
    def _calculate_sector_values(self, data) -> Dict[str, float]:
        """Calculate portfolio value for each sector."""
        if data is None:
            return {sector: 0.0 for sector in self.sector_groups.keys()}
        
        sector_values = {}
        
        try:
            prices = data['stock'].x[:, 3].cpu().numpy()  # Assuming close price at index 3
            
            for sector, indices in self.sector_indices.items():
                if len(indices) > 0:
                    sector_holdings = self.holdings[indices]
                    sector_prices = prices[indices]
                    sector_value = np.sum(sector_holdings * sector_prices)
                else:
                    sector_value = 0.0
                
                sector_values[sector] = sector_value
                
        except Exception:
            sector_values = {sector: 0.0 for sector in self.sector_groups.keys()}
        
        return sector_values
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        obs, info = super().reset(seed, options)
        
        # Add sector observations to info
        sector_obs = self.get_sector_observations(obs)
        info['sector_observations'] = sector_obs
        
        return obs, info
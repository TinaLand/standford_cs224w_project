# src/rl/agents/sector_agent.py
"""
Sector-Specific Agent Implementation
Individual agent for managing stocks in a specific sector
"""

import torch
import numpy as np
from typing import List, Dict, Any
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv

from .base import BaseTradingAgent
from ..config import MultiAgentConfig


class SectorAgent(BaseTradingAgent):
    """
    Individual agent for managing stocks in a specific sector.
    Each agent has its own PPO policy network.
    """
    
    def __init__(
        self,
        sector_name: str,
        tickers: List[str],
        gnn_model: torch.nn.Module,
        device: torch.device,
        embedding_dim: int = 64,
        learning_rate: float = MultiAgentConfig.LEARNING_RATE,
        policy: str = "MlpPolicy"
    ):
        """
        Initialize sector agent.
        
        Args:
            sector_name: Name of the sector this agent manages
            tickers: List of stock tickers in this sector
            gnn_model: Trained GNN model for state embedding
            device: PyTorch device
            learning_rate: Learning rate for PPO
            policy: Policy type
        """
        super().__init__(f"sector_{sector_name.lower()}", gnn_model, device)
        self.sector_name = sector_name
        self.tickers = tickers
        self.num_stocks = len(tickers)
        self.embedding_dim = embedding_dim
        
        # Initialize PPO agent
        self.agent = self._create_agent(learning_rate, policy)
    
    def _create_agent(self, learning_rate: float, policy: str) -> PPO:
        """Create PPO agent for this sector."""
        
        num_stocks = self.num_stocks
        embedding_dim = self.embedding_dim  # Capture embedding_dim in closure
        
        def dummy_env_factory():
            from gymnasium import spaces
            import gymnasium as gym
            
            class DummyEnv(gym.Env):
                def __init__(self, num_stocks):
                    super().__init__()
                    self.num_stocks = num_stocks
                    # Correct observation space: holdings + embeddings for each stock
                    obs_dim = self.num_stocks * (1 + embedding_dim)
                    self.observation_space = spaces.Box(
                        low=-np.inf, high=np.inf, 
                        shape=(obs_dim,), dtype=np.float32
                    )
                    self.action_space = spaces.MultiDiscrete([3] * self.num_stocks)
                
                def reset(self, seed=None):
                    return np.zeros(self.observation_space.shape), {}
                
                def step(self, action):
                    return np.zeros(self.observation_space.shape), 0.0, False, False, {}
            
            return DummyEnv(num_stocks=num_stocks)
        
        vec_env = DummyVecEnv([dummy_env_factory])
        
        return PPO(
            policy=policy,
            env=vec_env,
            learning_rate=learning_rate,
            verbose=0,
            device="cpu"
        )
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple:
        """Predict actions for this sector's stocks."""
        return self.agent.predict(observation, deterministic=deterministic)
    
    def train(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Train the sector agent.
        Note: In multi-agent setting, training is typically handled by coordinator.
        """
        # This will be called by the coordinator during centralized training
        return {
            "sector": self.sector_name,
            "num_stocks": self.num_stocks,
            "status": "training_delegated_to_coordinator"
        }
    
    def train_step(self, obs, actions, rewards, dones, values=None):
        """Perform one training step (for centralized training)."""
        # This will be called by the coordinator during centralized training
        pass
    
    def get_portfolio_allocation(self, observation: np.ndarray) -> np.ndarray:
        """
        Get portfolio allocation weights for this sector.
        
        Args:
            observation: Sector-specific observation
            
        Returns:
            Portfolio allocation weights for stocks in this sector
        """
        action, _ = self.predict(observation, deterministic=True)
        
        # Convert discrete actions to weights
        weights = np.zeros(self.num_stocks)
        
        for i, a in enumerate(action):
            if a == 0:  # Hold
                weights[i] = 1.0 / self.num_stocks
            elif a == 1:  # Buy
                weights[i] = 1.5 / self.num_stocks
            else:  # Sell
                weights[i] = 0.5 / self.num_stocks
        
        # Normalize
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        
        return weights
    
    def save(self, path):
        """Save sector agent to file."""
        agent_path = path / f"sector_agent_{self.sector_name.lower()}.zip"
        self.agent.save(agent_path.with_suffix(''))
        print(f" Sector agent '{self.sector_name}' saved to: {agent_path}")
    
    def load(self, path):
        """Load sector agent from file."""
        agent_path = path / f"sector_agent_{self.sector_name.lower()}.zip"
        
        if not agent_path.exists():
            raise FileNotFoundError(f"Sector agent file not found: {agent_path}")
        
        self.agent = PPO.load(agent_path.with_suffix(''), device="cpu")
        self.is_trained = True
        print(f" Sector agent '{self.sector_name}' loaded from: {agent_path}")
    
    def __repr__(self):
        return f"SectorAgent(sector={self.sector_name}, stocks={self.num_stocks}, trained={self.is_trained})"
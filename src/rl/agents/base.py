# src/rl/agents/base.py
"""
Abstract base classes for RL agents
Defines common interfaces for single-agent and multi-agent systems
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch

class BaseAgent(ABC):
    """Abstract base class for all RL agents."""
    
    def __init__(self, agent_id: str, device: torch.device):
        self.agent_id = agent_id
        self.device = device
        self.is_trained = False
    
    @abstractmethod
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict action given observation.
        
        Args:
            observation: Current state observation
            deterministic: Use deterministic policy
            
        Returns:
            (action, state) tuple
        """
        pass
    
    @abstractmethod
    def train(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Train the agent.
        
        Returns:
            Dictionary with training statistics
        """
        pass
    
    @abstractmethod
    def save(self, path):
        """Save agent to file."""
        pass
    
    @abstractmethod
    def load(self, path):
        """Load agent from file."""
        pass
    
    def evaluate(self, env, n_episodes: int = 10, deterministic: bool = True) -> Dict[str, float]:
        """
        Evaluate agent performance on environment.
        
        Args:
            env: Trading environment
            n_episodes: Number of episodes to evaluate
            deterministic: Use deterministic policy
            
        Returns:
            Dictionary with evaluation metrics
        """
        episode_returns = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            obs, info = env.reset()
            done = False
            episode_return = 0
            episode_length = 0
            
            while not done:
                action, _ = self.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_return += reward
                episode_length += 1
            
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
        
        return {
            "mean_return": np.mean(episode_returns),
            "std_return": np.std(episode_returns),
            "mean_length": np.mean(episode_lengths),
            "n_episodes": n_episodes
        }
    
    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.agent_id}, trained={self.is_trained})"


class BaseTradingAgent(BaseAgent):
    """Base class for trading-specific agents."""
    
    def __init__(self, agent_id: str, gnn_model: torch.nn.Module, device: torch.device):
        super().__init__(agent_id, device)
        self.gnn_model = gnn_model
        
        # Freeze GNN parameters
        for param in self.gnn_model.parameters():
            param.requires_grad = False
    
    @abstractmethod
    def get_portfolio_allocation(self, observation: np.ndarray) -> np.ndarray:
        """
        Get portfolio allocation weights.
        
        Args:
            observation: Current state observation
            
        Returns:
            Portfolio allocation weights
        """
        pass
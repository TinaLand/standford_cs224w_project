# src/rl/agents/single_agent.py
"""
Single-Agent RL Implementation
PPO-based agent for stock trading with GNN state representation
"""

import torch
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional, Dict, Any, Tuple
import numpy as np

from .base import BaseTradingAgent
from ..config import SingleAgentConfig

# Try to import tensorboard
try:
    import tensorboard
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class StockTradingAgent(BaseTradingAgent):
    """
    Single-agent RL wrapper for stock trading using PPO algorithm.
    Integrates GNN model for state representation.
    """
    
    def __init__(
        self,
        gnn_model: torch.nn.Module,
        env_factory,
        device: torch.device,
        learning_rate: float = SingleAgentConfig.LEARNING_RATE,
        tensorboard_log: Optional[Path] = None,
        policy: str = SingleAgentConfig.POLICY,
        verbose: int = SingleAgentConfig.VERBOSE,
        agent_id: str = "single_agent"
    ):
        """
        Initialize the RL Agent.
        
        Args:
            gnn_model: Trained GNN model for state embedding
            env_factory: Function that creates the trading environment
            device: PyTorch device (cuda/cpu)
            learning_rate: PPO learning rate
            tensorboard_log: Path for tensorboard logging (optional)
            policy: Policy type (MlpPolicy, CnnPolicy, etc.)
            verbose: Verbosity level
            agent_id: Unique identifier for this agent
        """
        super().__init__(agent_id, gnn_model, device)
        self.learning_rate = learning_rate
        self.policy = policy
        self.verbose = verbose
        
        # Create vectorized environment
        self.vec_env = make_vec_env(env_factory, n_envs=1)
        
        # Setup PPO agent
        ppo_kwargs = {
            "policy": policy,
            "env": self.vec_env,
            "verbose": verbose,
            "learning_rate": learning_rate,
            "device": "cpu",  # PPO with MLP works better on CPU
        }
        
        if tensorboard_log and TENSORBOARD_AVAILABLE:
            ppo_kwargs["tensorboard_log"] = tensorboard_log
        
        self.agent = PPO(**ppo_kwargs)
    
    def predict(
        self,
        observation: Optional[np.ndarray] = None,
        deterministic: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict action given observation.
        
        Args:
            observation: Current state observation (if None, uses environment's current state)
            deterministic: Use deterministic policy (True) or stochastic (False)
            
        Returns:
            (action, state) tuple
        """
        if observation is None:
            # Use environment's current observation
            return self.agent.predict(observation, deterministic=deterministic)
        else:
            return self.agent.predict(observation, deterministic=deterministic)
    
    def train(
        self,
        total_timesteps: int = SingleAgentConfig.TOTAL_TIMESTEPS,
        callback: Optional[BaseCallback] = None,
        progress_bar: bool = False
    ) -> Dict[str, Any]:
        """
        Train the RL agent.
        
        Args:
            total_timesteps: Total number of training steps
            callback: Optional callback for monitoring
            progress_bar: Show progress bar
            
        Returns:
            Dictionary with training statistics
        """
        print(f"\n Starting RL Agent Training ({total_timesteps} timesteps)...")
        
        try:
            self.agent.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                progress_bar=progress_bar
            )
            self.is_trained = True
            
            # Get training statistics
            stats = {
                "total_timesteps": total_timesteps,
                "learning_rate": self.learning_rate,
                "policy": self.policy,
                "status": "completed",
                "agent_id": self.agent_id
            }
            
            print(" Training completed successfully")
            return stats
            
        except Exception as e:
            print(f" Training failed: {e}")
            raise
    
    def get_portfolio_allocation(self, observation: np.ndarray) -> np.ndarray:
        """
        Get portfolio allocation weights.
        
        Args:
            observation: Current state observation
            
        Returns:
            Portfolio allocation weights
        """
        action, _ = self.predict(observation, deterministic=True)
        
        # Convert discrete actions to portfolio weights
        # Action space: [0: Hold, 1: Buy, 2: Sell] for each stock
        # Convert to weights: Hold=current, Buy=increase, Sell=decrease
        num_stocks = len(action)
        weights = np.zeros(num_stocks)
        
        for i, a in enumerate(action):
            if a == 0:  # Hold
                weights[i] = 1.0 / num_stocks  # Equal weight default
            elif a == 1:  # Buy
                weights[i] = 1.5 / num_stocks  # Increase weight
            else:  # Sell
                weights[i] = 0.5 / num_stocks  # Decrease weight
        
        # Normalize to sum to 1
        weights = weights / np.sum(weights)
        return weights
    
    def save(self, path: Path):
        """
        Save agent to file.
        
        Args:
            path: Path to save the agent (without extension, .zip will be added)
        """
        self.agent.save(path)
        print(f" Agent saved to: {path}.zip")
    
    def load(self, path: Path):
        """
        Load agent from file.
        
        Args:
            path: Path to load the agent from (with or without .zip extension)
        """
        if not path.suffix:
            path = Path(str(path) + ".zip")
        
        if not path.exists():
            raise FileNotFoundError(f"Agent file not found: {path}")
        
        self.agent = PPO.load(path, env=self.vec_env, device="cpu")
        self.is_trained = True
        print(f" Agent loaded from: {path}")
    
    def get_action_distribution(self, observation: np.ndarray) -> np.ndarray:
        """
        Get action probability distribution (for analysis).
        
        Args:
            observation: Current state observation
            
        Returns:
            Action probability distribution
        """
        # This requires accessing the policy network directly
        # For now, return deterministic action
        action, _ = self.predict(observation, deterministic=False)
        return action
    
    def __repr__(self):
        return f"StockTradingAgent(id={self.agent_id}, policy={self.policy}, lr={self.learning_rate}, trained={self.is_trained})"
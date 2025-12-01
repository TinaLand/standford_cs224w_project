# src/rl/coordination/mixing.py
"""
QMIX-style Mixing Networks
Value decomposition for multi-agent coordination
"""

import torch
import torch.nn as nn
from typing import Tuple

from ..config import MultiAgentConfig


class MixingNetwork(nn.Module):
    """
    QMIX-style Mixing Network for combining Q-values from multiple agents.
    
    Architecture:
    - Takes individual Q-values from each agent
    - Mixes them using a hypernetwork
    - Outputs global Q-value for centralized training
    """
    
    def __init__(
        self, 
        num_agents: int, 
        state_dim: int, 
        hidden_dim: int = MultiAgentConfig.MIXING_HIDDEN_DIM
    ):
        """
        Initialize mixing network.
        
        Args:
            num_agents: Number of agents
            state_dim: Global state dimension
            hidden_dim: Hidden dimension for mixing network
        """
        super().__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # Hypernetwork: generates mixing network weights from global state
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_agents * hidden_dim)
        )
        
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, q_values: torch.Tensor, global_state: torch.Tensor) -> torch.Tensor:
        """
        Mix individual Q-values into global Q-value.
        
        Args:
            q_values: [batch_size, num_agents] - Q-values from each agent
            global_state: [batch_size, state_dim] - Global state
        
        Returns:
            global_q: [batch_size, 1] - Mixed global Q-value
        """
        batch_size = q_values.size(0)
        
        # Generate mixing weights from global state
        w1 = torch.abs(self.hyper_w1(global_state))  # [batch, num_agents * hidden]
        w1 = w1.view(batch_size, self.num_agents, self.hidden_dim)
        
        w2 = torch.abs(self.hyper_w2(global_state))  # [batch, hidden]
        w2 = w2.view(batch_size, self.hidden_dim, 1)
        
        b1 = self.hyper_b1(global_state)  # [batch, hidden]
        b1 = b1.view(batch_size, 1, self.hidden_dim)
        
        b2 = self.hyper_b2(global_state)  # [batch, 1]
        
        # Mix Q-values
        hidden = torch.bmm(q_values.unsqueeze(1), w1) + b1  # [batch, 1, hidden]
        hidden = torch.relu(hidden)
        q_total = torch.bmm(hidden, w2) + b2  # [batch, 1, 1]
        
        return q_total.squeeze(-1)  # [batch, 1]


class AttentionMixingNetwork(nn.Module):
    """
    Attention-based mixing network for dynamic agent coordination.
    Alternative to hypernetwork approach.
    """
    
    def __init__(
        self, 
        num_agents: int, 
        state_dim: int, 
        hidden_dim: int = MultiAgentConfig.MIXING_HIDDEN_DIM
    ):
        """
        Initialize attention-based mixing network.
        
        Args:
            num_agents: Number of agents
            state_dim: Global state dimension  
            hidden_dim: Hidden dimension
        """
        super().__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # Attention mechanism for agent weighting
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Agent value transformation
        self.agent_transform = nn.Linear(1, hidden_dim)
        
        # State transformation for attention keys/values
        self.state_transform = nn.Linear(state_dim, hidden_dim)
        
        # Final mixing layers
        self.mixing_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, q_values: torch.Tensor, global_state: torch.Tensor) -> torch.Tensor:
        """
        Mix Q-values using attention mechanism.
        
        Args:
            q_values: [batch_size, num_agents] - Q-values from each agent
            global_state: [batch_size, state_dim] - Global state
        
        Returns:
            global_q: [batch_size, 1] - Mixed global Q-value
        """
        batch_size = q_values.size(0)
        
        # Transform agent Q-values
        agent_values = self.agent_transform(q_values.unsqueeze(-1))  # [batch, num_agents, hidden]
        
        # Transform global state for attention
        state_context = self.state_transform(global_state)  # [batch, hidden]
        state_context = state_context.unsqueeze(1).repeat(1, self.num_agents, 1)  # [batch, num_agents, hidden]
        
        # Apply attention
        attended_values, attention_weights = self.attention(
            query=agent_values,
            key=state_context,
            value=agent_values
        )
        
        # Aggregate attended values
        aggregated = torch.mean(attended_values, dim=1)  # [batch, hidden]
        
        # Final mixing
        global_q = self.mixing_layers(aggregated)  # [batch, 1]
        
        return global_q


def create_mixing_network(
    num_agents: int,
    state_dim: int,
    mixing_type: str = "hypernetwork",
    hidden_dim: int = MultiAgentConfig.MIXING_HIDDEN_DIM
) -> nn.Module:
    """
    Factory function to create mixing networks.
    
    Args:
        num_agents: Number of agents
        state_dim: Global state dimension
        mixing_type: Type of mixing network ("hypernetwork" or "attention")
        hidden_dim: Hidden dimension
        
    Returns:
        Mixing network instance
    """
    if mixing_type == "hypernetwork":
        return MixingNetwork(num_agents, state_dim, hidden_dim)
    elif mixing_type == "attention":
        return AttentionMixingNetwork(num_agents, state_dim, hidden_dim)
    else:
        raise ValueError(f"Unknown mixing type: {mixing_type}")
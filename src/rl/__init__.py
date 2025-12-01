# src/rl/__init__.py
"""
Reinforcement Learning Module
Complete RL system for stock trading with single-agent and multi-agent support

Optimized Structure:
- agents/: Base classes and specific agent implementations
- environments/: Trading environments for single and multi-agent systems  
- coordination/: Multi-agent coordination components (CTDE, mixing networks)
- training/: Training pipelines and utilities
- config.py: Centralized configuration and hyperparameters
"""

# Import main components for easy access
from .config import SingleAgentConfig, MultiAgentConfig, GNNConfig

# Agent imports
from .agents import BaseAgent, BaseTradingAgent, StockTradingAgent, SectorAgent

# Environment imports  
from .environments import BaseTradingEnv, StockTradingEnv, MultiAgentTradingEnv

# Coordination imports
from .coordination import MultiAgentCoordinator, create_multi_agent_system

# Training imports
from .training import run_single_agent_training, run_multi_agent_training

__all__ = [
    # Configuration
    'SingleAgentConfig',
    'MultiAgentConfig', 
    'GNNConfig',
    
    # Agents
    'BaseAgent',
    'BaseTradingAgent',
    'StockTradingAgent',
    'SectorAgent',
    
    # Environments
    'BaseTradingEnv',
    'StockTradingEnv', 
    'MultiAgentTradingEnv',
    
    # Coordination
    'MultiAgentCoordinator',
    'create_multi_agent_system',
    
    # Training
    'run_single_agent_training',
    'run_multi_agent_training'
]
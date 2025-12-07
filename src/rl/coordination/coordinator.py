# src/rl/coordination/coordinator.py
"""
Multi-Agent Coordinator
CTDE (Centralized Training, Decentralized Execution) implementation
Fixed performance bottlenecks from original coordinator
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from ..agents.sector_agent import SectorAgent
from ..environments.multi_agent import MultiAgentTradingEnv
from .mixing import create_mixing_network
from ..config import MultiAgentConfig, PROJECT_ROOT


class SectorGrouping:
    """Utility class for sector-based stock grouping."""
    
    @staticmethod
    def load_sector_mapping() -> Dict[str, List[str]]:
        """Load sector-to-tickers mapping from static data."""
        sector_file = PROJECT_ROOT / "data" / "raw" / "static_sector_industry.csv"
        
        if not sector_file.exists():
            return SectorGrouping._create_default_grouping()
        
        try:
            df = pd.read_csv(sector_file)
            sector_groups = defaultdict(list)
            
            for _, row in df.iterrows():
                ticker = row['Ticker']
                sector = row.get('Sector', 'Unknown')
                sector_groups[sector].append(ticker)
            
            return dict(sector_groups)
        except Exception:
            return SectorGrouping._create_default_grouping()
    
    @staticmethod
    def _create_default_grouping() -> Dict[str, List[str]]:
        """Create default sector grouping if data not available."""
        return MultiAgentConfig.DEFAULT_SECTORS


class MultiAgentCoordinator:
    """
    Coordinator for Multi-Agent RL system.
    Implements CTDE (Centralized Training, Decentralized Execution).
    
    Key Performance Improvements:
    - Pre-initialized mixing network (no recreation)
    - Cached sector mappings
    - Optimized action merging
    """
    
    def __init__(
        self,
        gnn_model: torch.nn.Module,
        sector_groups: Dict[str, List[str]],
        all_tickers: List[str],
        device: torch.device,
        embedding_dim: int = 64,
        learning_rate: float = MultiAgentConfig.LEARNING_RATE
    ):
        """
        Initialize multi-agent coordinator.
        
        Args:
            gnn_model: Trained GNN model
            sector_groups: Dictionary mapping sector names to ticker lists
            all_tickers: List of all tickers in the system
            device: PyTorch device
            learning_rate: Learning rate for coordination
        """
        self.gnn_model = gnn_model
        self.sector_groups = sector_groups
        self.all_tickers = all_tickers
        self.device = device
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        
        # Pre-compute mappings for performance
        self._precompute_mappings()
        
        # Create agents for each sector
        self.agents = {}
        self._create_sector_agents()
        
        # Create mixing network (FIXED: pre-initialized, not recreated)
        self.mixing_network = create_mixing_network(
            num_agents=len(self.agents),
            state_dim=MultiAgentConfig.GLOBAL_STATE_DIM,
            mixing_type="hypernetwork"
        ).to(device)
        
        # Optimizer for mixing network
        self.mixer_optimizer = torch.optim.Adam(
            self.mixing_network.parameters(), 
            lr=learning_rate
        )
        
        print(f" Multi-Agent Coordinator initialized with {len(self.agents)} sector agents")
        for sector, agent in self.agents.items():
            print(f"   - {sector}: {agent.num_stocks} stocks")
    
    def _precompute_mappings(self):
        """Pre-compute mappings for performance optimization."""
        # Ticker to sector mapping
        self.ticker_to_sector = {}
        self.sector_to_indices = {}
        
        for sector, tickers in self.sector_groups.items():
            valid_tickers = [t for t in tickers if t in self.all_tickers]
            
            for ticker in valid_tickers:
                self.ticker_to_sector[ticker] = sector
            
            # Pre-compute indices for fast lookup
            indices = [self.all_tickers.index(t) for t in valid_tickers]
            self.sector_to_indices[sector] = indices
    
    def _create_sector_agents(self):
        """Create sector-specific agents."""
        for sector_name, tickers in self.sector_groups.items():
            # Filter tickers that exist in all_tickers
            valid_tickers = [t for t in tickers if t in self.all_tickers]
            
            if len(valid_tickers) >= MultiAgentConfig.MIN_STOCKS_PER_SECTOR:
                self.agents[sector_name] = SectorAgent(
                    sector_name=sector_name,
                    tickers=valid_tickers,
                    gnn_model=self.gnn_model,
                    device=self.device,
                    embedding_dim=self.embedding_dim,
                    learning_rate=self.learning_rate
                )
    
    def get_agent_actions(
        self,
        observations: Dict[str, np.ndarray],
        deterministic: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Get actions from all agents (decentralized execution).
        
        Args:
            observations: Dictionary mapping sector_name -> observation
            deterministic: Use deterministic policy
        
        Returns:
            Dictionary mapping sector_name -> actions
        """
        actions = {}
        for sector_name, agent in self.agents.items():
            if sector_name in observations:
                action, _ = agent.predict(observations[sector_name], deterministic=deterministic)
                actions[sector_name] = action
        return actions
    
    def merge_actions(
        self,
        actions_dict: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Merge actions from all agents into a single action vector.
        
        PERFORMANCE OPTIMIZATION: Use pre-computed indices.
        
        Args:
            actions_dict: Dictionary mapping sector_name -> actions for that sector
        
        Returns:
            Combined action vector [num_all_stocks]
        """
        combined_actions = np.ones(len(self.all_tickers), dtype=np.int32)  # Default to "Hold"
        
        for sector_name, sector_actions in actions_dict.items():
            indices = self.sector_to_indices.get(sector_name, [])
            
            # Safely merge actions
            for i, action in enumerate(sector_actions):
                if i < len(indices):
                    combined_actions[indices[i]] = action
        
        return combined_actions
    
    def compute_global_q_value(
        self,
        q_values_dict: Dict[str, float],
        global_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute global Q-value using mixing network.
        
        Args:
            q_values_dict: Dictionary mapping sector_name -> Q-value
            global_state: Global state tensor
        
        Returns:
            Global Q-value
        """
        # Convert to tensor (maintain sector order)
        q_values_list = [q_values_dict.get(sector, 0.0) for sector in self.agents.keys()]
        q_values = torch.tensor(q_values_list, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        global_state = global_state.unsqueeze(0).to(self.device)
        
        # Mix Q-values using pre-initialized network
        global_q = self.mixing_network(q_values, global_state)
        
        return global_q
    
    def train_mixing_network(
        self,
        q_values_dict: Dict[str, float],
        global_state: torch.Tensor,
        target_global_q: torch.Tensor
    ) -> float:
        """
        Train mixing network using TD error.
        
        Args:
            q_values_dict: Individual Q-values from agents
            global_state: Global state
            target_global_q: Target global Q-value
            
        Returns:
            Training loss
        """
        # Compute current global Q
        current_global_q = self.compute_global_q_value(q_values_dict, global_state)
        
        # TD error
        td_error = target_global_q - current_global_q
        loss = td_error ** 2
        
        # Backward pass
        self.mixer_optimizer.zero_grad()
        loss.backward()
        self.mixer_optimizer.step()
        
        return loss.item()
    
    def coordinate_training_step(
        self,
        observations: Dict[str, np.ndarray],
        actions: Dict[str, np.ndarray],
        rewards: Dict[str, float],
        next_observations: Dict[str, np.ndarray],
        dones: Dict[str, bool]
    ):
        """
        Perform centralized training step for all agents.
        
        Args:
            observations: Current observations for each agent
            actions: Actions taken by each agent
            rewards: Rewards received by each agent
            next_observations: Next observations for each agent
            dones: Done flags for each agent
        """
        # This would implement the CTDE training loop
        # For now, individual agents handle their own training
        pass
    
    def save(self, path: Path):
        """Save coordinator and all agents."""
        coordinator_path = path / "multi_agent_coordinator"
        coordinator_path.mkdir(parents=True, exist_ok=True)
        
        # Save mixing network
        torch.save(self.mixing_network.state_dict(), coordinator_path / "mixing_network.pt")
        
        # Save all sector agents
        for sector_name, agent in self.agents.items():
            agent.save(coordinator_path)
        
        print(f" Multi-agent coordinator saved to: {coordinator_path}")
    
    def load(self, path: Path):
        """Load coordinator and all agents."""
        coordinator_path = path / "multi_agent_coordinator"
        
        if not coordinator_path.exists():
            raise FileNotFoundError(f"Coordinator path not found: {coordinator_path}")
        
        # Load mixing network
        mixing_path = coordinator_path / "mixing_network.pt"
        if mixing_path.exists():
            self.mixing_network.load_state_dict(torch.load(mixing_path, map_location=self.device))
        
        # Load all sector agents
        for sector_name, agent in self.agents.items():
            try:
                agent.load(coordinator_path)
            except FileNotFoundError as e:
                print(f"Warning: Could not load agent for {sector_name}: {e}")
        
        print(f" Multi-agent coordinator loaded from: {coordinator_path}")
    
    def __repr__(self):
        return f"MultiAgentCoordinator(sectors={len(self.agents)}, stocks={len(self.all_tickers)})"


def create_multi_agent_system(
    gnn_model: torch.nn.Module,
    device: torch.device,
    embedding_dim: int = 64,
    learning_rate: float = MultiAgentConfig.LEARNING_RATE,
    actual_tickers: Optional[List[str]] = None
) -> MultiAgentCoordinator:
    """
    Create multi-agent RL system with automatic sector grouping.
    
    Args:
        gnn_model: Trained GNN model
        device: PyTorch device
        learning_rate: Learning rate for coordination
    
    Returns:
        MultiAgentCoordinator instance
    """
    # Load sector grouping
    sector_groups = SectorGrouping.load_sector_mapping()
    
    # Use actual tickers from data if provided, otherwise use all configured tickers
    if actual_tickers is not None:
        all_tickers = actual_tickers
    else:
        # Get all tickers from sector groups (fallback)
        all_tickers = []
        for tickers in sector_groups.values():
            all_tickers.extend(tickers)
        all_tickers = sorted(list(set(all_tickers)))
    
    # Create coordinator
    coordinator = MultiAgentCoordinator(
        gnn_model=gnn_model,
        sector_groups=sector_groups,
        all_tickers=all_tickers,
        device=device,
        embedding_dim=embedding_dim,
        learning_rate=learning_rate
    )
    
    return coordinator
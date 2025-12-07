# multi_agent_rl_coordinator.py
"""
Multi-Agent RL Coordinator for Stock Trading
Implements Cooperative Multi-Agent RL with CTDE (Centralized Training, Decentralized Execution)
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import sys

sys.path.append(str(Path(__file__).resolve().parent))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from src.rl.agent import StockTradingAgent
from src.rl.environment import StockTradingEnv
from src.training.transformer_trainer import RoleAwareGraphTransformer, load_graph_data
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class SectorGrouping:
    """Group stocks by sector for multi-agent RL."""
    
    @staticmethod
    def load_sector_mapping() -> Dict[str, List[str]]:
        """Load sector-to-tickers mapping from static data."""
        sector_file = PROJECT_ROOT / "data" / "raw" / "static_sector_industry.csv"
        
        if not sector_file.exists():
            # Fallback: create default grouping
            return SectorGrouping._create_default_grouping()
        
        df = pd.read_csv(sector_file)
        sector_groups = defaultdict(list)
        
        for _, row in df.iterrows():
            ticker = row['Ticker']
            sector = row.get('Sector', 'Unknown')
            sector_groups[sector].append(ticker)
        
        return dict(sector_groups)
    
    @staticmethod
    def _create_default_grouping() -> Dict[str, List[str]]:
        """Create default sector grouping if data not available."""
        return {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMZN', 'AVGO', 'ADBE', 'CSCO', 'CRM'],
            'Financials': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'COF', 'SCHW', 'BLK'],
            'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN'],
            'Consumer Discretionary': ['HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'TJX', 'DG', 'ROST', 'BBY'],
            'Energy': ['XOM', 'CVX', 'SLB', 'EOG', 'COP', 'MPC', 'VLO', 'PSX', 'HAL', 'OXY']
        }


class SectorAgent:
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
        learning_rate: float = 1e-5,
        policy: str = "MlpPolicy"
    ):
        self.sector_name = sector_name
        self.tickers = tickers
        self.num_stocks = len(tickers)
        self.gnn_model = gnn_model
        self.device = device
        
        # Create environment factory for this sector
        self.env_factory = self._create_env_factory()
        
        # Initialize PPO agent
        self.agent = self._create_agent(learning_rate, policy)
        self.is_trained = False
    
    def _create_env_factory(self):
        """Create environment factory for this sector's stocks."""
        def env_factory():
            # This will be set during training
            return None  # Placeholder
        return env_factory
    
    def _create_agent(self, learning_rate: float, policy: str) -> PPO:
        """Create PPO agent for this sector."""
        # Note: Environment will be created during training
        # For now, we'll create a dummy env to initialize the agent
        from stable_baselines3.common.env_util import DummyVecEnv
        
        num_stocks = self.num_stocks  # Use instance variable
        
        def dummy_env_factory():
            from gymnasium import spaces
            import gymnasium as gym
            
            class DummyEnv(gym.Env):
                def __init__(self, num_stocks=50):
                    super().__init__()
                    self.num_stocks = num_stocks
                    self.observation_space = spaces.Box(
                        low=-np.inf, high=np.inf, 
                        shape=(self.num_stocks * 10,), dtype=np.float32
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
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Predict actions for this sector's stocks."""
        return self.agent.predict(observation, deterministic=deterministic)[0]
    
    def train_step(self, obs, actions, rewards, dones, values=None):
        """Perform one training step (for centralized training)."""
        # This will be called by the coordinator during centralized training
        pass


class MixingNetwork(torch.nn.Module):
    """
    QMIX-style Mixing Network for combining Q-values from multiple agents.
    
    Architecture:
    - Takes individual Q-values from each agent
    - Mixes them using a hypernetwork
    - Outputs global Q-value for centralized training
    """
    
    def __init__(self, num_agents: int, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # Hypernetwork: generates mixing network weights from global state
        self.hyper_w1 = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_agents * hidden_dim)
        )
        
        self.hyper_w2 = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.hyper_b1 = torch.nn.Linear(state_dim, hidden_dim)
        self.hyper_b2 = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, q_values: torch.Tensor, global_state: torch.Tensor) -> torch.Tensor:
        """
        Mix individual Q-values into global Q-value.
        
        Args:
            q_values: [batch_size, num_agents] - Q-values from each agent
            global_state: [batch_size, state_dim] - Global state (e.g., portfolio state)
        
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


class MultiAgentCoordinator:
    """
    Coordinator for Multi-Agent RL system.
    Implements CTDE (Centralized Training, Decentralized Execution).
    """
    
    def __init__(
        self,
        gnn_model: torch.nn.Module,
        sector_groups: Dict[str, List[str]],
        all_tickers: List[str],
        device: torch.device,
        learning_rate: float = 1e-5
    ):
        self.gnn_model = gnn_model
        self.sector_groups = sector_groups
        self.all_tickers = all_tickers
        self.device = device
        
        # Create agents for each sector
        self.agents = {}
        for sector_name, tickers in sector_groups.items():
            # Filter tickers that exist in all_tickers
            valid_tickers = [t for t in tickers if t in all_tickers]
            if len(valid_tickers) > 0:
                self.agents[sector_name] = SectorAgent(
                    sector_name=sector_name,
                    tickers=valid_tickers,
                    gnn_model=gnn_model,
                    device=device,
                    learning_rate=learning_rate
                )
        
        # Create mixing network for centralized training
        self.mixing_network = MixingNetwork(
            num_agents=len(self.agents),
            state_dim=256,  # Global state dimension (can be adjusted)
            hidden_dim=64
        ).to(device)
        
        # Optimizer for mixing network
        self.mixer_optimizer = torch.optim.Adam(self.mixing_network.parameters(), lr=learning_rate)
        
        print(f" Created {len(self.agents)} sector agents:")
        for sector, agent in self.agents.items():
            print(f"   - {sector}: {agent.num_stocks} stocks")
    
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
                action = agent.predict(observations[sector_name], deterministic=deterministic)
                actions[sector_name] = action
        return actions
    
    def merge_actions(
        self,
        actions_dict: Dict[str, np.ndarray],
        ticker_to_sector: Dict[str, str]
    ) -> np.ndarray:
        """
        Merge actions from all agents into a single action vector for all stocks.
        
        Args:
            actions_dict: Dictionary mapping sector_name -> actions for that sector
            ticker_to_sector: Dictionary mapping ticker -> sector_name
        
        Returns:
            Combined action vector [num_all_stocks]
        """
        combined_actions = np.zeros(len(self.all_tickers), dtype=np.int32)
        
        for sector_name, sector_actions in actions_dict.items():
            agent = self.agents[sector_name]
            for i, ticker in enumerate(agent.tickers):
                if ticker in self.all_tickers:
                    ticker_idx = self.all_tickers.index(ticker)
                    combined_actions[ticker_idx] = sector_actions[i]
        
        return combined_actions
    
    def compute_global_q_value(
        self,
        q_values_dict: Dict[str, float],
        global_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute global Q-value using mixing network (for centralized training).
        
        Args:
            q_values_dict: Dictionary mapping sector_name -> Q-value
            global_state: Global state tensor [state_dim]
        
        Returns:
            Global Q-value [1]
        """
        # Convert to tensor
        q_values = torch.tensor(
            [q_values_dict.get(sector, 0.0) for sector in self.agents.keys()],
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)  # [1, num_agents]
        
        global_state = global_state.unsqueeze(0).to(self.device)  # [1, state_dim]
        
        # Mix Q-values
        global_q = self.mixing_network(q_values, global_state)
        
        return global_q
    
    def train_mixing_network(
        self,
        q_values_dict: Dict[str, float],
        global_state: torch.Tensor,
        target_global_q: torch.Tensor
    ):
        """
        Train mixing network using TD error.
        
        Args:
            q_values_dict: Individual Q-values from agents
            global_state: Global state
            target_global_q: Target global Q-value (from Bellman equation)
        """
        # Compute current global Q
        current_global_q = self.compute_global_q_value(q_values_dict, global_state)
        
        # TD error
        td_error = target_global_q - current_global_q
        
        # Loss
        loss = td_error ** 2
        
        # Backward pass
        self.mixer_optimizer.zero_grad()
        loss.backward()
        self.mixer_optimizer.step()
        
        return loss.item()


class MultiAgentTradingEnv:
    """
    Multi-Agent Trading Environment.
    Extends StockTradingEnv to support multiple agents.
    """
    
    def __init__(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        gnn_model: torch.nn.Module,
        sector_groups: Dict[str, List[str]],
        all_tickers: List[str],
        device: torch.device
    ):
        # Create base environment (for data loading)
        self.base_env = StockTradingEnv(start_date, end_date, gnn_model, device)
        
        self.sector_groups = sector_groups
        self.all_tickers = all_tickers
        self.gnn_model = gnn_model
        self.device = device
        
        # Create ticker to sector mapping
        self.ticker_to_sector = {}
        for sector, tickers in sector_groups.items():
            for ticker in tickers:
                if ticker in all_tickers:
                    self.ticker_to_sector[ticker] = sector
        
        # Create sector to indices mapping
        self.sector_indices = {}
        for sector, tickers in sector_groups.items():
            indices = [all_tickers.index(t) for t in tickers if t in all_tickers]
            self.sector_indices[sector] = indices
    
    def get_sector_observations(self, global_obs: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Split global observation into sector-specific observations.
        
        Observation format: [holdings (N)] + [embeddings (N * H)]
        where N = num_stocks, H = embedding_dim
        
        Each sector agent expects observation shape: (num_stocks * 10,)
        where num_stocks is the number of stocks in that sector.
        
        Args:
            global_obs: Global observation [total_dim]
        
        Returns:
            Dictionary mapping sector_name -> sector observation
        """
        sector_obs = {}
        
        # Get observation dimensions from base environment
        num_stocks = len(self.all_tickers)
        holdings_dim = num_stocks
        embeddings_dim = len(global_obs) - holdings_dim
        embedding_dim = embeddings_dim // num_stocks if num_stocks > 0 else 0
        
        # Split holdings and embeddings
        holdings = global_obs[:holdings_dim]
        embeddings_flat = global_obs[holdings_dim:]
        embeddings = embeddings_flat.reshape(num_stocks, embedding_dim) if embedding_dim > 0 else embeddings_flat
        
        for sector, indices in self.sector_indices.items():
            if len(indices) > 0:
                num_sector_stocks = len(indices)
                expected_obs_dim = num_sector_stocks * 10  # Agent expects (num_stocks * 10,)
                
                # Extract sector-specific holdings and embeddings
                sector_holdings = holdings[indices]
                sector_embeddings = embeddings[indices] if embedding_dim > 0 else embeddings_flat
                
                # Flatten sector embeddings
                if embedding_dim > 0:
                    sector_embeddings_flat = sector_embeddings.flatten()
                else:
                    sector_embeddings_flat = sector_embeddings
                
                # Concatenate holdings and embeddings for this sector
                sector_obs_full = np.concatenate([sector_holdings, sector_embeddings_flat]).astype(np.float32)
                
                # Pad or truncate to match expected dimension
                if len(sector_obs_full) < expected_obs_dim:
                    # Pad with zeros
                    padding = np.zeros(expected_obs_dim - len(sector_obs_full), dtype=np.float32)
                    sector_obs[sector] = np.concatenate([sector_obs_full, padding])
                elif len(sector_obs_full) > expected_obs_dim:
                    # Truncate
                    sector_obs[sector] = sector_obs_full[:expected_obs_dim]
                else:
                    sector_obs[sector] = sector_obs_full
            else:
                # Fallback: create zero observation with expected shape
                # Use a default size based on typical sector size
                default_size = 10 * 10  # 10 stocks * 10 features
                sector_obs[sector] = np.zeros(default_size, dtype=np.float32)
        
        return sector_obs
    
    def reset(self, seed=None):
        """Reset environment."""
        return self.base_env.reset(seed=seed)
    
    def step(self, actions_dict: Dict[str, np.ndarray]):
        """
        Step environment with actions from multiple agents.
        
        Args:
            actions_dict: Dictionary mapping sector_name -> actions
        
        Returns:
            (next_obs, reward, done, info)
        """
        # Merge actions
        coordinator = MultiAgentCoordinator(
            self.gnn_model, self.sector_groups, self.all_tickers, self.device
        )
        combined_actions = coordinator.merge_actions(actions_dict, self.ticker_to_sector)
        
        # Step base environment
        return self.base_env.step(combined_actions)


def create_multi_agent_system(
    gnn_model: RoleAwareGraphTransformer,
    device: torch.device,
    learning_rate: float = 1e-5
) -> MultiAgentCoordinator:
    """
    Create multi-agent RL system.
    
    Args:
        gnn_model: Trained GNN model
        device: PyTorch device
        learning_rate: Learning rate for agents
    
    Returns:
        MultiAgentCoordinator instance
    """
    # Load sector grouping
    sector_groups = SectorGrouping.load_sector_mapping()
    
    # Get all tickers (from data or model)
    # This should match the tickers used in training
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
        learning_rate=learning_rate
    )
    
    return coordinator


if __name__ == "__main__":
    print(" Multi-Agent RL System")
    print("="*60)
    
    # Load GNN model
    gnn_model_path = MODELS_DIR / "core_transformer_model.pt"
    if not gnn_model_path.exists():
        print(f" GNN model not found: {gnn_model_path}")
        print("   Please train Phase 4 model first.")
        sys.exit(1)
    
    # Load model (simplified - actual loading needs proper initialization)
    print(" Loading GNN model...")
    # gnn_model = load_gnn_model_for_rl()  # Use existing function
    
    print("\n Multi-Agent RL system structure created!")
    print("\nNext steps:")
    print("1. Implement training loop with CTDE")
    print("2. Create sector-specific environments")
    print("3. Train and evaluate multi-agent system")


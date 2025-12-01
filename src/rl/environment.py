# scripts/rl_environment.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Fix PyTorch serialization for pandas timestamps (PyTorch 2.6+)
import torch.serialization
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([pd._libs.tslibs.timestamps._unpickle_timestamp])

# --- Configuration (Must match training script) ---
# NOTE: This file lives in `src/rl/`, so the project root is three levels up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_GRAPHS_DIR = PROJECT_ROOT / "data" / "graphs"
MODELS_DIR = PROJECT_ROOT / "models"
LOOKAHEAD_DAYS = 5
TRANSACTION_COST = 0.001 # 交易成本 (0.1% per trade)

# --- Environment Class ---

class StockTradingEnv(gym.Env):
    """
    Custom Gym Environment for Sequential Stock Portfolio Management.
    """
    def __init__(self, start_date, end_date, gnn_model, device):
        super(StockTradingEnv, self).__init__()
        
        # RL Integration Components
        self.gnn_model = gnn_model.to(device)
        self.device = device
        
        # Load all graph files and targets (simplified, assumes pre-loaded features/targets)
        # NOTE: In a real environment, you need a DataLoader that can fetch G_t and P_{t+1}
        self.data_loader = self._initialize_data_loader(start_date, end_date)
        
        if not self.data_loader:
             raise ValueError("Failed to load environment data.")

        # State Space: [Portfolio Holdings (N)] + [GNN Node Embeddings (N * H)]
        self.NUM_STOCKS = len(self.data_loader['tickers'])
        
        # Get actual embedding dimension from GNN model output
        # We need to compute this by running a sample through the model
        sample_graph = torch.load(list(DATA_GRAPHS_DIR.glob('graph_t_*.pt'))[0], weights_only=False)
        sample_graph_tensor = sample_graph.to(device)
        with torch.no_grad():
            gnn_model.eval()
            # Create date_tensor for time-aware encoding if enabled
            import pandas as pd
            sample_date_str = list(DATA_GRAPHS_DIR.glob('graph_t_*.pt'))[0].stem.split('_')[-1]
            sample_date = pd.to_datetime(sample_date_str)
            REFERENCE_DATE = pd.to_datetime('2015-01-01')
            days_since_ref = (sample_date - REFERENCE_DATE).days
            num_nodes = sample_graph_tensor['stock'].x.shape[0]
            date_tensor = torch.full((num_nodes,), days_since_ref, dtype=torch.float32).to(device) if hasattr(gnn_model, 'enable_time_aware') and gnn_model.enable_time_aware else None
            sample_embeddings = gnn_model.get_embeddings(sample_graph_tensor, date_tensor=date_tensor)
            self.EMBEDDING_DIM = sample_embeddings.shape[1]  # Actual GNN output dimension (256)

        # State dimension: holdings (N) + flattened embeddings (N * H)
        state_dim = self.NUM_STOCKS + (self.NUM_STOCKS * self.EMBEDDING_DIM)
        
        # Action Space: Buy/Sell/Hold for each stock (Discrete Multi-Dimensional)
        # N stocks, 3 actions per stock
        self.action_space = spaces.MultiDiscrete([3] * self.NUM_STOCKS) # 0: Sell, 1: Hold, 2: Buy
        
        # Observation Space (State)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)

        # Initial State
        self.initial_cash = 10000.0
        self.current_step = 0
        self.max_steps = len(self.data_loader['dates']) - 1
        
        # Advanced Portfolio Constraints
        self.MAX_POSITION_SIZE = 0.20  # Maximum 20% of portfolio in single stock
        self.MAX_LEVERAGE = 1.0  # No leverage (1.0 = 100% of cash)
        self.MIN_CASH_RESERVE = 0.05  # Keep at least 5% cash reserve
        self.MAX_TURNOVER = 0.50  # Maximum 50% portfolio turnover per step
        
        # Reward function configuration
        self.USE_SHARPE_REWARD = True  # Use Sharpe ratio in reward calculation
        self.RISK_FREE_RATE = 0.02  # 2% annual risk-free rate
        self.REWARD_LOOKBACK_WINDOW = 20  # Days to calculate Sharpe ratio
        
        # Track returns for Sharpe calculation
        self.portfolio_returns_history = []

    def _initialize_data_loader(self, start_date, end_date):
        """Placeholder for a robust data loading system."""
        # This function should load ALL required data (graphs, prices, targets) 
        # for the entire backtesting period and map them to dates.
        
        # Example data structure for demonstration
        sample_graph = torch.load(list(DATA_GRAPHS_DIR.glob('graph_t_*.pt'))[0], weights_only=False)
        
        # We need actual prices for calculating returns
        ohlcv_df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "stock_prices_ohlcv_raw.csv", index_col='Date', parse_dates=True)
        
        # Ensure index is datetime
        if not isinstance(ohlcv_df.index, pd.DatetimeIndex):
            ohlcv_df.index = pd.to_datetime(ohlcv_df.index)
        
        trading_dates = sorted(list(set([pd.to_datetime(f.stem.split('_')[-1]) for f in DATA_GRAPHS_DIR.glob('graph_t_*.pt')])))
        
        # Filter dates for backtesting period and ensure they exist in price data
        backtest_dates = []
        for d in trading_dates:
            if start_date <= d <= end_date:
                # Check if date exists in price data
                if d in ohlcv_df.index:
                    backtest_dates.append(d)
                else:
                    # Try to find nearest date
                    nearest_date = ohlcv_df.index[ohlcv_df.index.get_indexer([d], method='nearest')[0]]
                    if abs((nearest_date - d).days) <= 1:  # Within 1 day
                        backtest_dates.append(nearest_date)
        
        if not backtest_dates:
            raise ValueError(f"No valid trading dates found in range {start_date} to {end_date}")
        
        # Get all available tickers (don't limit here, will be limited later)
        all_tickers = [col.split('_')[-1] for col in ohlcv_df.columns if col.startswith('Close_')]
        
        return {
            'dates': backtest_dates,
            'prices': ohlcv_df,
            'tickers': all_tickers
            # 'graphs': Dictionary of all loaded graphs
        }

    def reset(self, seed=None, options=None):
        """Resets the environment to the initial state."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.portfolio_value = self.initial_cash
        self.cash = self.initial_cash
        self.holdings = np.zeros(self.NUM_STOCKS) # Number of shares held
        self.portfolio_returns_history = []  # Reset returns history
        
        # Initial State Generation
        obs = self._get_observation()
        info = {'portfolio_value': self.portfolio_value}
        
        return obs, info

    def _get_observation(self):
        """Generates the observation (state) for the current step."""
        date_t = self.data_loader['dates'][self.current_step]
        
        # 1. Load GNN Data (G_t)
        data_t = torch.load(DATA_GRAPHS_DIR / f"graph_t_{date_t.strftime('%Y%m%d')}.pt", weights_only=False)

        # 2. Generate GNN Embeddings (E_t) using the trained model
        with torch.no_grad():
            self.gnn_model.eval()
            # Use the get_embeddings() method to extract embeddings before final classifier
            data_t_tensor = data_t.to(self.device)
            # Create date_tensor for time-aware encoding if enabled
            import pandas as pd
            REFERENCE_DATE = pd.to_datetime('2015-01-01')
            days_since_ref = (date_t - REFERENCE_DATE).days
            num_nodes = data_t_tensor['stock'].x.shape[0]
            date_tensor = torch.full((num_nodes,), days_since_ref, dtype=torch.float32).to(self.device) if hasattr(self.gnn_model, 'enable_time_aware') and self.gnn_model.enable_time_aware else None
            embeddings_t = self.gnn_model.get_embeddings(data_t_tensor, date_tensor=date_tensor)
            embeddings_t = embeddings_t.cpu().numpy()
            
            # Flatten embeddings: [N * H]
            flat_embeddings = embeddings_t.flatten()
            
        # 3. Concatenate: [Holdings (N)] + [GNN Embeddings (N * H)]
        holdings_normalized = self.holdings / (self.cash + np.sum(self.holdings)) # Simple normalization
        
        observation = np.concatenate([holdings_normalized, flat_embeddings]).astype(np.float32)
        
        return observation

    def step(self, action):
        """Executes one trading day step (t -> t+1)."""
        
        # 1. Calculate Rewards (Price change from t to t+1)
        # Find prices P_t and P_t+1
        date_t = self.data_loader['dates'][self.current_step]
        date_t_plus_1 = self.data_loader['dates'][self.current_step + 1]

        # Get prices using datetime index (more robust)
        prices_df = self.data_loader['prices']
        
        # Try direct datetime access first
        if date_t in prices_df.index:
            prices_t = prices_df.loc[date_t]
        else:
            # Find nearest date
            nearest_idx = prices_df.index.get_indexer([date_t], method='nearest')[0]
            prices_t = prices_df.iloc[nearest_idx]
        
        if date_t_plus_1 in prices_df.index:
            prices_t_plus_1 = prices_df.loc[date_t_plus_1]
        else:
            # Find nearest date
            nearest_idx = prices_df.index.get_indexer([date_t_plus_1], method='nearest')[0]
            prices_t_plus_1 = prices_df.iloc[nearest_idx]
        
        # Get prices for current tickers (Close_TICKER)
        # Handle both Series and DataFrame cases
        if isinstance(prices_t, pd.Series):
            current_prices = np.array([prices_t.get(f'Close_{t}', 0.0) for t in self.data_loader['tickers']])
        else:
            current_prices = np.array([prices_t[f'Close_{t}'] for t in self.data_loader['tickers']])
        
        if isinstance(prices_t_plus_1, pd.Series):
            next_prices = np.array([prices_t_plus_1.get(f'Close_{t}', 0.0) for t in self.data_loader['tickers']])
        else:
            next_prices = np.array([prices_t_plus_1[f'Close_{t}'] for t in self.data_loader['tickers']])
        
        # Portfolio value before action (based on next day's price)
        current_portfolio_value = self.cash + np.sum(self.holdings * next_prices)

        # 2. Execute Trading Action at P_t+1 (Simplified)
        
        trade_volume = np.zeros(self.NUM_STOCKS)
        # Action space: 0: Sell, 1: Hold, 2: Buy
        for i, act in enumerate(action):
            price = next_prices[i]
            
            if act == 2: # Buy
                # Simplified: allocate a fixed fraction of total value to buy
                buy_amount = self.portfolio_value * 0.01 / self.NUM_STOCKS 
                shares_to_buy = buy_amount / price
                cost = buy_amount * (1 + TRANSACTION_COST)
                if self.cash >= cost:
                    self.cash -= cost
                    self.holdings[i] += shares_to_buy
                    trade_volume[i] = shares_to_buy
                    
            elif act == 0: # Sell
                # Simplified: sell a fixed fraction of current holdings
                shares_to_sell = self.holdings[i] * 0.2 
                revenue = shares_to_sell * price * (1 - TRANSACTION_COST)
                self.cash += revenue
                self.holdings[i] -= shares_to_sell
                trade_volume[i] = -shares_to_sell

        # 3. Advanced Portfolio Constraints
        # Enforce maximum position size per stock
        total_portfolio_value = self.cash + np.sum(self.holdings * next_prices)
        for i in range(self.NUM_STOCKS):
            position_value = self.holdings[i] * next_prices[i]
            position_pct = position_value / total_portfolio_value if total_portfolio_value > 0 else 0
            if position_pct > self.MAX_POSITION_SIZE:
                # Reduce position to max allowed
                max_shares = (total_portfolio_value * self.MAX_POSITION_SIZE) / next_prices[i]
                excess_shares = self.holdings[i] - max_shares
                self.holdings[i] = max_shares
                self.cash += excess_shares * next_prices[i] * (1 - TRANSACTION_COST)
        
        # Enforce cash reserve
        min_cash = total_portfolio_value * self.MIN_CASH_RESERVE
        if self.cash < min_cash:
            # Sell some holdings to maintain cash reserve
            needed_cash = min_cash - self.cash
            for i in range(self.NUM_STOCKS):
                if needed_cash <= 0:
                    break
                if self.holdings[i] > 0:
                    shares_to_sell = min(self.holdings[i], needed_cash / (next_prices[i] * (1 + TRANSACTION_COST)))
                    self.holdings[i] -= shares_to_sell
                    self.cash += shares_to_sell * next_prices[i] * (1 - TRANSACTION_COST)
                    needed_cash -= shares_to_sell * next_prices[i] * (1 - TRANSACTION_COST)
        
        # 4. Portfolio Value Update and Reward Calculation
        new_portfolio_value = self.cash + np.sum(self.holdings * next_prices)
        
        # Calculate portfolio return
        portfolio_return = (new_portfolio_value - current_portfolio_value) / current_portfolio_value if current_portfolio_value > 0 else 0.0
        
        # Update returns history for Sharpe calculation
        self.portfolio_returns_history.append(portfolio_return)
        if len(self.portfolio_returns_history) > self.REWARD_LOOKBACK_WINDOW:
            self.portfolio_returns_history.pop(0)
        
        # Calculate reward: Sharpe ratio if enabled, otherwise simple return
        if self.USE_SHARPE_REWARD and len(self.portfolio_returns_history) >= 10:
            # Calculate Sharpe ratio over lookback window
            returns_array = np.array(self.portfolio_returns_history)
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)
            
            if std_return > 1e-8:  # Avoid division by zero
                # Annualized Sharpe ratio (assuming daily returns)
                daily_rf_rate = self.RISK_FREE_RATE / 252  # 252 trading days per year
                sharpe_ratio = (mean_return - daily_rf_rate) / std_return * np.sqrt(252)
                # Use Sharpe ratio as reward (scaled)
                reward = sharpe_ratio / 10.0  # Scale to reasonable range
            else:
                reward = portfolio_return
        else:
            # Simple return-based reward
            reward = portfolio_return
        
        self.portfolio_value = new_portfolio_value
        self.current_step += 1
        
        # 4. Check Termination
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # 5. Generate next observation
        obs = self._get_observation()
        
        info = {
            'portfolio_value': self.portfolio_value, 
            'trades': np.sum(np.abs(trade_volume)),
            'date': date_t_plus_1.strftime('%Y-%m-%d')
        }
        
        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        # Not strictly needed for training but good for visualization
        print(f"Step {self.current_step}: Value ${self.portfolio_value:.2f}")

    def close(self):
        pass


# --- 💻 2. `scripts/phase5_rl_integration.py` (Main Training Script) ---

import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

# Import GNN model
from src.training.transformer_trainer import RoleAwareGraphTransformer
# Note: StockTradingEnv is defined in this same file, no need to import

# --- Configuration (Shared) ---
MODEL_PATH = MODELS_DIR / 'core_transformer_model.pt'
RL_LOG_PATH = PROJECT_ROOT / "logs" / "rl_logs"
RL_SAVE_PATH = MODELS_DIR / "rl_ppo_agent_model"
RL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
RL_LOG_PATH.mkdir(parents=True, exist_ok=True)


def load_gnn_model_for_rl():
    """Loads the trained GNN model and exposes the embedding layer."""
    # Temporarily load features to determine INPUT_DIM
    temp_data = torch.load(list(DATA_GRAPHS_DIR.glob('graph_t_*.pt'))[0], weights_only=False)
    INPUT_DIM = temp_data['stock'].x.shape[1]
    
    # Initialize the GNN model structure
    gnn_model = RoleAwareGraphTransformer(
        INPUT_DIM, 256, 2, 2, 4 # Use same dimensions as Phase 4 training
    )
    
    # Load trained weights
    gnn_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
    
    # Freeze GNN parameters (we only use it for feature extraction)
    for param in gnn_model.parameters():
        param.requires_grad = False
        
    print(f"✅ GNN Model loaded and frozen for embedding generation.")
    
    # EXPOSE THE EMBEDDING LAYER: In the GNN model, we need the output *before* the final classifier.
    # The embedding dimension is the input dimension of the final linear layer.
    
    return gnn_model

def run_rl_pipeline():
    """Runs the RL training pipeline."""
    
    # 1. Load GNN Model
    gnn_model = load_gnn_model_for_rl()
    
    # 2. Define Backtesting Period (Use the test split from Phase 3/4)
    # NOTE: This should be dynamically loaded from Phase 4 split results for consistency.
    # Placeholder dates are used here:
    START_DATE = pd.to_datetime('2022-01-01')
    END_DATE = pd.to_datetime('2024-12-31')
    
    # 3. Setup Environment
    env = StockTradingEnv(
        start_date=START_DATE, 
        end_date=END_DATE, 
        gnn_model=gnn_model, 
        device=DEVICE
    )
    
    # SB3 requires vectorized environments
    vec_env = make_vec_env(lambda: env, n_envs=1)
    
    # 4. RL Agent Setup (PPO)
    # Policy: MlpPolicy (Multi-Layer Perceptron) is standard for discrete action space
    model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=1e-4, device=DEVICE,
                tensorboard_log=RL_LOG_PATH)
    
    # 5. Training
    TOTAL_TIMESTEPS = 100000 
    print(f"\n🔨 Starting PPO RL Training for {TOTAL_TIMESTEPS} timesteps...")
    
    model.learn(total_timesteps=TOTAL_TIMESTEPS, reset_num_timesteps=False)
    
    # 6. Save Agent
    model.save(RL_SAVE_PATH / "ppo_stock_agent")
    print(f"\n✅ RL Agent trained and saved to: {RL_SAVE_PATH}")

    # 7. Final Backtesting (Evaluation)
    # The final step is to run the saved model on a separate test set (Phase 6)
    
if __name__ == '__main__':
    run_rl_pipeline()
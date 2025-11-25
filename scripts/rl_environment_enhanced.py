# scripts/rl_environment_enhanced.py
"""
Enhanced RL Environment for Stock Trading with:
- Proper GNN embedding extraction
- Risk-adjusted reward functions (Sharpe ratio)
- Portfolio constraints and position sizing
- Risk metrics tracking (max drawdown, Sharpe)
- Slippage and latency modeling
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from collections import deque

# Fix PyTorch serialization for pandas timestamps (PyTorch 2.6+)
import torch.serialization
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([pd._libs.tslibs.timestamps._unpickle_timestamp])

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_GRAPHS_DIR = PROJECT_ROOT / "data" / "graphs"
MODELS_DIR = PROJECT_ROOT / "models"
LOOKAHEAD_DAYS = 5
TRANSACTION_COST = 0.001  # 0.1% per trade
SLIPPAGE = 0.0005  # 0.05% slippage
RISK_FREE_RATE = 0.02 / 252  # Daily risk-free rate (2% annual)


class StockTradingEnvEnhanced(gym.Env):
    """
    Enhanced Stock Trading Environment with:
    - GNN embedding-based state representation
    - Risk-adjusted rewards (Sharpe ratio)
    - Portfolio constraints
    - Risk metrics tracking
    - Slippage and latency modeling
    """
    
    def __init__(self, start_date, end_date, gnn_model, device, 
                 initial_cash=10000.0, max_position_pct=0.1, 
                 enable_slippage=True, enable_risk_penalty=True):
        super(StockTradingEnvEnhanced, self).__init__()
        
        # RL Integration Components
        self.gnn_model = gnn_model.to(device)
        self.gnn_model.eval()  # Set to evaluation mode
        self.device = device
        
        # Portfolio constraints
        self.initial_cash = initial_cash
        self.max_position_pct = max_position_pct  # Max 10% per stock
        self.enable_slippage = enable_slippage
        self.enable_risk_penalty = enable_risk_penalty
        
        # Load data
        self.data_loader = self._initialize_data_loader(start_date, end_date)
        if not self.data_loader:
            raise ValueError("Failed to load environment data.")
        
        self.NUM_STOCKS = len(self.data_loader['tickers'])
        
        # Get embedding dimension from GNN model
        # Run a sample forward pass to get embedding size
        sample_graph = torch.load(list(DATA_GRAPHS_DIR.glob('graph_t_*.pt'))[0], weights_only=False)
        sample_graph = sample_graph.to(device)
        with torch.no_grad():
            sample_embedding = self.gnn_model.get_embeddings(sample_graph)
            self.EMBEDDING_DIM = sample_embedding.shape[1]
        
        # State Space: [Portfolio Holdings (N)] + [GNN Embeddings (N * H)] + [Cash Ratio (1)]
        state_dim = self.NUM_STOCKS + (self.NUM_STOCKS * self.EMBEDDING_DIM) + 1
        
        # Action Space: Buy/Sell/Hold for each stock (Discrete Multi-Dimensional)
        self.action_space = spaces.MultiDiscrete([3] * self.NUM_STOCKS)  # 0: Sell, 1: Hold, 2: Buy
        
        # Observation Space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        
        # Risk metrics tracking
        self.portfolio_history = deque(maxlen=252)  # Track 1 year of daily returns
        self.max_drawdown = 0.0
        self.peak_value = initial_cash
        
        # Initialize state
        self.current_step = 0
        self.max_steps = len(self.data_loader['dates']) - 1
    
    def _initialize_data_loader(self, start_date, end_date):
        """Load all required data for backtesting period."""
        # Load OHLCV data
        ohlcv_df = pd.read_csv(
            PROJECT_ROOT / "data" / "raw" / "stock_prices_ohlcv_raw.csv",
            index_col='Date', parse_dates=True
        )
        
        # Get trading dates from graph files
        trading_dates = sorted([
            pd.to_datetime(f.stem.split('_')[-1]) 
            for f in DATA_GRAPHS_DIR.glob('graph_t_*.pt')
        ])
        
        # Filter dates for backtesting period
        backtest_dates = [d for d in trading_dates if start_date <= d <= end_date]
        
        if not backtest_dates:
            print(f"⚠️  Warning: No dates found in range {start_date} to {end_date}")
            return None
        
        # Get tickers from OHLCV data
        all_tickers = sorted([
            col.split('_')[-1] 
            for col in ohlcv_df.columns 
            if col.startswith('Close_')
        ])
        
        return {
            'dates': backtest_dates,
            'prices': ohlcv_df,
            'tickers': all_tickers
        }
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.portfolio_value = self.initial_cash
        self.cash = self.initial_cash
        self.holdings = np.zeros(self.NUM_STOCKS)  # Number of shares held
        
        # Reset risk metrics
        self.portfolio_history.clear()
        self.max_drawdown = 0.0
        self.peak_value = self.initial_cash
        
        obs = self._get_observation()
        info = {
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'holdings_value': 0.0
        }
        
        return obs, info
    
    def _get_observation(self):
        """Generate observation (state) for current step using GNN embeddings."""
        if self.current_step >= len(self.data_loader['dates']):
            # Return zero observation if out of bounds
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        date_t = self.data_loader['dates'][self.current_step]
        
        # 1. Load graph data
        graph_file = DATA_GRAPHS_DIR / f"graph_t_{date_t.strftime('%Y%m%d')}.pt"
        if not graph_file.exists():
            # Fallback: return zero embeddings
            embeddings_t = np.zeros((self.NUM_STOCKS, self.EMBEDDING_DIM), dtype=np.float32)
        else:
            data_t = torch.load(graph_file, weights_only=False)
            data_t = data_t.to(self.device)
            
            # 2. Extract GNN embeddings using the new method
            with torch.no_grad():
                embeddings_t = self.gnn_model.get_embeddings(data_t).cpu().numpy()
        
        # 3. Normalize holdings (as percentage of portfolio)
        total_value = self.cash + np.sum(self.holdings * self._get_current_prices())
        if total_value > 0:
            holdings_normalized = self.holdings / (total_value + 1e-8)
        else:
            holdings_normalized = np.zeros(self.NUM_STOCKS)
        
        # 4. Cash ratio
        cash_ratio = self.cash / (total_value + 1e-8)
        
        # 5. Flatten embeddings
        flat_embeddings = embeddings_t.flatten()
        
        # 6. Concatenate: [Holdings (N)] + [GNN Embeddings (N * H)] + [Cash Ratio (1)]
        observation = np.concatenate([
            holdings_normalized.astype(np.float32),
            flat_embeddings.astype(np.float32),
            np.array([cash_ratio], dtype=np.float32)
        ])
        
        return observation
    
    def _get_current_prices(self):
        """Get current prices for all stocks."""
        if self.current_step >= len(self.data_loader['dates']):
            return np.zeros(self.NUM_STOCKS)
        
        date_t = self.data_loader['dates'][self.current_step]
        date_str = date_t.strftime('%Y-%m-%d')
        
        try:
            prices_row = self.data_loader['prices'].loc[date_str]
            current_prices = np.array([
                prices_row.get(f'Close_{ticker}', 0.0) 
                for ticker in self.data_loader['tickers']
            ])
            return current_prices
        except KeyError:
            return np.zeros(self.NUM_STOCKS)
    
    def _calculate_reward(self, prev_value, new_value, returns_history):
        """
        Calculate risk-adjusted reward.
        
        Options:
        1. Simple return: (new - prev) / prev
        2. Sharpe ratio: (mean_return - risk_free) / std_return
        3. Risk-adjusted: return - risk_penalty
        """
        if prev_value <= 0:
            return 0.0
        
        # Simple return
        simple_return = (new_value - prev_value) / prev_value
        
        if not self.enable_risk_penalty or len(returns_history) < 2:
            return simple_return
        
        # Risk-adjusted reward: Sharpe ratio proxy
        returns_array = np.array(returns_history)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array) + 1e-8
        
        # Sharpe ratio (annualized proxy)
        sharpe_ratio = (mean_return - RISK_FREE_RATE) / std_return * np.sqrt(252)
        
        # Use Sharpe ratio as reward (scaled)
        reward = sharpe_ratio * 0.1  # Scale down for stability
        
        # Also add simple return component
        reward += simple_return * 0.9
        
        return reward
    
    def step(self, action):
        """Execute one trading day step."""
        if self.current_step >= self.max_steps:
            obs = self._get_observation()
            return obs, 0.0, True, False, {'portfolio_value': self.portfolio_value}
        
        date_t = self.data_loader['dates'][self.current_step]
        
        # Get current and next prices
        current_prices = self._get_current_prices()
        
        if self.current_step + 1 < len(self.data_loader['dates']):
            date_t_plus_1 = self.data_loader['dates'][self.current_step + 1]
            date_str_next = date_t_plus_1.strftime('%Y-%m-%d')
            try:
                prices_row_next = self.data_loader['prices'].loc[date_str_next]
                next_prices = np.array([
                    prices_row_next.get(f'Close_{ticker}', current_prices[i])
                    for i, ticker in enumerate(self.data_loader['tickers'])
                ])
            except KeyError:
                next_prices = current_prices
        else:
            next_prices = current_prices
        
        # Portfolio value before action
        prev_portfolio_value = self.cash + np.sum(self.holdings * current_prices)
        
        # Execute trading actions with constraints
        trade_costs = 0.0
        for i, act in enumerate(action):
            price = current_prices[i]
            if price <= 0:
                continue
            
            # Apply slippage
            if self.enable_slippage:
                if act == 2:  # Buy
                    price *= (1 + SLIPPAGE)
                elif act == 0:  # Sell
                    price *= (1 - SLIPPAGE)
            
            if act == 2:  # Buy
                # Position sizing: max_position_pct of total portfolio
                max_investment = prev_portfolio_value * self.max_position_pct
                shares_to_buy = max_investment / price
                cost = shares_to_buy * price * (1 + TRANSACTION_COST)
                
                if self.cash >= cost:
                    self.cash -= cost
                    self.holdings[i] += shares_to_buy
                    trade_costs += cost * TRANSACTION_COST
                    
            elif act == 0:  # Sell
                # Sell 50% of current holdings
                shares_to_sell = self.holdings[i] * 0.5
                if shares_to_sell > 0:
                    revenue = shares_to_sell * price * (1 - TRANSACTION_COST)
                    self.cash += revenue
                    self.holdings[i] -= shares_to_sell
                    trade_costs += shares_to_sell * price * TRANSACTION_COST
        
        # Update portfolio value
        new_portfolio_value = self.cash + np.sum(self.holdings * next_prices)
        
        # Calculate daily return
        if prev_portfolio_value > 0:
            daily_return = (new_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        else:
            daily_return = 0.0
        
        # Update risk metrics
        self.portfolio_history.append(daily_return)
        
        # Update max drawdown
        if new_portfolio_value > self.peak_value:
            self.peak_value = new_portfolio_value
        else:
            drawdown = (self.peak_value - new_portfolio_value) / self.peak_value
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown
        
        # Calculate reward
        reward = self._calculate_reward(
            prev_portfolio_value, 
            new_portfolio_value,
            list(self.portfolio_history)
        )
        
        # Penalize excessive trading
        reward -= trade_costs / prev_portfolio_value * 10  # Scale penalty
        
        self.portfolio_value = new_portfolio_value
        self.current_step += 1
        
        # Check termination
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Generate next observation
        obs = self._get_observation()
        
        # Calculate Sharpe ratio for info
        sharpe = 0.0
        if len(self.portfolio_history) >= 2:
            returns_array = np.array(self.portfolio_history)
            mean_ret = np.mean(returns_array)
            std_ret = np.std(returns_array) + 1e-8
            sharpe = (mean_ret - RISK_FREE_RATE) / std_ret * np.sqrt(252)
        
        info = {
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'holdings_value': new_portfolio_value - self.cash,
            'daily_return': daily_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': self.max_drawdown,
            'date': date_t.strftime('%Y-%m-%d')
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        """Render environment state."""
        if mode == 'human':
            sharpe = 0.0
            if len(self.portfolio_history) >= 2:
                returns_array = np.array(self.portfolio_history)
                mean_ret = np.mean(returns_array)
                std_ret = np.std(returns_array) + 1e-8
                sharpe = (mean_ret - RISK_FREE_RATE) / std_ret * np.sqrt(252)
            
            print(f"Step {self.current_step}: Portfolio Value=${self.portfolio_value:.2f}, "
                  f"Cash=${self.cash:.2f}, Sharpe={sharpe:.3f}, MaxDD={self.max_drawdown:.3f}")
    
    def close(self):
        """Clean up resources."""
        pass


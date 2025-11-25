# scripts/dynamic_rl_visualization.py
"""
Dynamic Real-time Visualization for RL Training and Testing
Shows live updates during training and backtesting
"""
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from datetime import datetime
from collections import deque

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / 'scripts'))
sys.path.append(str(PROJECT_ROOT / 'scripts' / 'components'))

from phase5_rl_integration import load_gnn_model_for_rl
from rl_environment import StockTradingEnv
from rl_agent import StockTradingAgent
from phase6_evaluation import calculate_financial_metrics

MODELS_DIR = PROJECT_ROOT / "models"
DATA_GRAPHS_DIR = PROJECT_ROOT / "data" / "graphs"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RL_SAVE_PATH = MODELS_DIR / "rl_ppo_agent_model" / "ppo_stock_agent.zip"

class DynamicRLVisualizer:
    """Real-time visualization during RL training and testing."""
    
    def __init__(self, max_points=1000):
        self.max_points = max_points
        self.portfolio_values = deque(maxlen=max_points)
        self.rewards = deque(maxlen=max_points)
        self.steps = deque(maxlen=max_points)
        self.returns = deque(maxlen=max_points)
        self.drawdowns = deque(maxlen=max_points)
        
        # Setup figure
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('RL Agent - Real-time Performance Monitor', 
                         fontsize=16, fontweight='bold')
        
        # Initialize plots
        self.init_plots()
        
    def init_plots(self):
        """Initialize empty plots."""
        # 1. Portfolio Value
        self.ax1 = self.axes[0, 0]
        self.line1, = self.ax1.plot([], [], 'b-', linewidth=2, label='Portfolio Value')
        self.ax1.axhline(y=10000, color='r', linestyle='--', label='Initial Value')
        self.ax1.set_xlabel('Step')
        self.ax1.set_ylabel('Portfolio Value ($)')
        self.ax1.set_title('Portfolio Value Over Time')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_xlim(0, self.max_points)
        self.ax1.set_ylim(8000, 20000)
        
        # 2. Cumulative Returns
        self.ax2 = self.axes[0, 1]
        self.line2, = self.ax2.plot([], [], 'g-', linewidth=2, label='Returns')
        self.ax2.axhline(y=0, color='r', linestyle='--')
        self.ax2.set_xlabel('Step')
        self.ax2.set_ylabel('Cumulative Return (%)')
        self.ax2.set_title('Cumulative Returns')
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_xlim(0, self.max_points)
        self.ax2.set_ylim(-20, 100)
        
        # 3. Rewards
        self.ax3 = self.axes[1, 0]
        self.line3, = self.ax3.plot([], [], 'purple', linewidth=1, alpha=0.7, label='Reward')
        self.ax3.axhline(y=0, color='r', linestyle='--')
        self.ax3.set_xlabel('Step')
        self.ax3.set_ylabel('Reward')
        self.ax3.set_title('Reward Over Time')
        self.ax3.legend()
        self.ax3.grid(True, alpha=0.3)
        self.ax3.set_xlim(0, self.max_points)
        
        # 4. Drawdown
        self.ax4 = self.axes[1, 1]
        self.line4, = self.ax4.plot([], [], 'r-', linewidth=1, label='Drawdown')
        self.fill4 = None  # Will be created dynamically
        self.ax4.set_xlabel('Step')
        self.ax4.set_ylabel('Drawdown (%)')
        self.ax4.set_title('Drawdown Analysis')
        self.ax4.legend()
        self.ax4.grid(True, alpha=0.3)
        self.ax4.set_xlim(0, self.max_points)
        self.ax4.set_ylim(0, 20)
        
        # Text for metrics
        self.text_metrics = self.fig.text(0.5, 0.02, '', 
                                          ha='center', fontsize=10,
                                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    def update_data(self, step, portfolio_value, reward):
        """Update data for visualization."""
        self.steps.append(step)
        self.portfolio_values.append(portfolio_value)
        self.rewards.append(reward)
        
        if len(self.portfolio_values) > 1:
            initial = self.portfolio_values[0]
            return_pct = (portfolio_value / initial - 1) * 100
            self.returns.append(return_pct)
            
            # Calculate drawdown
            if len(self.portfolio_values) > 1:
                portfolio_array = np.array(self.portfolio_values)
                cumulative_max = np.maximum.accumulate(portfolio_array)
                drawdown = (cumulative_max - portfolio_array) / cumulative_max * 100
                self.drawdowns.append(drawdown[-1])
            else:
                self.drawdowns.append(0)
        else:
            self.returns.append(0)
            self.drawdowns.append(0)
    
    def update_plot(self, frame):
        """Update plots with latest data."""
        if len(self.steps) == 0:
            return [self.line1, self.line2, self.line3, self.line4, self.text_metrics]
        
        steps_array = np.array(self.steps)
        portfolio_array = np.array(self.portfolio_values)
        returns_array = np.array(self.returns)
        rewards_array = np.array(self.rewards)
        drawdowns_array = np.array(self.drawdowns)
        
        # Update portfolio value
        self.line1.set_data(steps_array, portfolio_array)
        if len(steps_array) > 0:
            self.ax1.set_xlim(0, max(steps_array[-1], 100))
            self.ax1.set_ylim(min(portfolio_array) * 0.9, max(portfolio_array) * 1.1)
        
        # Update returns
        self.line2.set_data(steps_array, returns_array)
        if len(steps_array) > 0:
            self.ax2.set_xlim(0, max(steps_array[-1], 100))
            if len(returns_array) > 0:
                self.ax2.set_ylim(min(returns_array) - 5, max(returns_array) + 5)
        
        # Update rewards
        self.line3.set_data(steps_array, rewards_array)
        if len(steps_array) > 0:
            self.ax3.set_xlim(0, max(steps_array[-1], 100))
            if len(rewards_array) > 0:
                self.ax3.set_ylim(min(rewards_array) - 0.01, max(rewards_array) + 0.01)
        
        # Update drawdown
        self.line4.set_data(steps_array, drawdowns_array)
        if len(steps_array) > 0:
            self.ax4.set_xlim(0, max(steps_array[-1], 100))
            # Update fill area
            if self.fill4 is not None:
                self.fill4.remove()
            self.fill4 = self.ax4.fill_between(steps_array, drawdowns_array, 0, alpha=0.3, color='red')
        
        # Update metrics text
        if len(portfolio_array) > 0:
            initial = portfolio_array[0]
            current = portfolio_array[-1]
            total_return = (current / initial - 1) * 100
            avg_reward = np.mean(rewards_array) if len(rewards_array) > 0 else 0
            max_dd = max(drawdowns_array) if len(drawdowns_array) > 0 else 0
            
            metrics_text = (
                f"Step: {steps_array[-1]} | "
                f"Value: ${current:,.2f} | "
                f"Return: {total_return:.2f}% | "
                f"Avg Reward: {avg_reward:.6f} | "
                f"Max Drawdown: {max_dd:.2f}%"
            )
            self.text_metrics.set_text(metrics_text)
        
        # Return all artists (excluding fill which is handled separately)
        artists = [self.line1, self.line2, self.line3, self.line4, self.text_metrics]
        if self.fill4 is not None:
            artists.append(self.fill4)
        return [a for a in artists if a is not None]
    
    def run_live_backtest(self, agent, gnn_model, start_date, end_date, max_steps=1000):
        """Run backtest with live visualization."""
        print("\n" + "="*70)
        print("Starting Live Backtest with Dynamic Visualization")
        print("="*70)
        print("Close the plot window to stop the backtest\n")
        
        # Create environment
        def make_env():
            return StockTradingEnv(
                start_date=start_date,
                end_date=end_date,
                gnn_model=gnn_model,
                device=DEVICE
            )
        
        env = make_env()
        obs, info = env.reset()
        
        # Start animation (blit=False to avoid matplotlib issues)
        ani = FuncAnimation(self.fig, self.update_plot, interval=100, blit=False)
        plt.show(block=False)
        
        step = 0
        done = False
        
        try:
            while not done and step < max_steps:
                # Get action
                action, _ = agent.predict(obs, deterministic=True)
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Update visualization
                portfolio_value = info.get('portfolio_value', env.portfolio_value)
                self.update_data(step, portfolio_value, reward)
                
                # Progress update
                if step % 10 == 0:
                    print(f"Step {step}: Value ${portfolio_value:.2f}, Reward: {reward:.6f}")
                
                step += 1
                time.sleep(0.01)  # Small delay for visualization
        
        except KeyboardInterrupt:
            print("\n\nBacktest interrupted by user")
        
        finally:
            # Final metrics
            if len(self.portfolio_values) > 1:
                metrics = calculate_financial_metrics(list(self.portfolio_values), step)
                print("\n" + "="*70)
                print("Final Results")
                print("="*70)
                print(f"Steps: {step}")
                print(f"Initial Value: ${self.portfolio_values[0]:,.2f}")
                print(f"Final Value: ${self.portfolio_values[-1]:,.2f}")
                print(f"Total Return: {(self.portfolio_values[-1]/self.portfolio_values[0]-1)*100:.2f}%")
                print(f"Sharpe Ratio: {metrics.get('Sharpe_Ratio', 0):.4f}")
                print(f"Max Drawdown: {metrics.get('Max_Drawdown', 0)*100:.2f}%")
                print("="*70)
            
            # Keep plot open
            plt.show(block=True)


def main():
    """Main function for dynamic visualization."""
    print("="*70)
    print("Dynamic RL Visualization - Live Backtest Monitor")
    print("="*70)
    
    # Load GNN model
    print("\n[1/3] Loading GNN model...")
    gnn_model = load_gnn_model_for_rl()
    
    # Get date range
    graph_files = list(DATA_GRAPHS_DIR.glob('graph_t_*.pt'))
    graph_dates = sorted([pd.to_datetime(f.stem.split('_')[-1]) for f in graph_files])
    graph_start = graph_dates[0]
    graph_end = graph_dates[-1]
    date_range_days = (graph_end - graph_start).days
    start_offset_days = int(date_range_days * 0.8)
    START_DATE = graph_start + pd.Timedelta(days=start_offset_days)
    END_DATE = graph_end
    
    print(f"\n[2/3] Setting up agent...")
    print(f"   Date range: {START_DATE.date()} to {END_DATE.date()}")
    
    def make_env():
        return StockTradingEnv(
            start_date=START_DATE,
            end_date=END_DATE,
            gnn_model=gnn_model,
            device=DEVICE
        )
    
    agent = StockTradingAgent(
        gnn_model=gnn_model,
        env_factory=make_env,
        device=DEVICE,
        learning_rate=1e-5,
        tensorboard_log=None,
        policy="MlpPolicy",
        verbose=0
    )
    
    # Load or train agent
    if RL_SAVE_PATH.exists():
        try:
            agent.load(RL_SAVE_PATH)
            print("   ✅ Loaded existing agent")
        except:
            print("   ⚠️  Training new agent (100 steps)...")
            agent.train(total_timesteps=100, progress_bar=False)
    else:
        print("   ⚠️  Training new agent (100 steps)...")
        agent.train(total_timesteps=100, progress_bar=False)
    
    # Run live visualization
    print(f"\n[3/3] Starting live backtest visualization...")
    print("   A window will open showing real-time performance")
    print("   Close the window to stop the backtest\n")
    
    visualizer = DynamicRLVisualizer(max_points=1000)
    visualizer.run_live_backtest(agent, gnn_model, START_DATE, END_DATE, max_steps=500)


if __name__ == '__main__':
    main()


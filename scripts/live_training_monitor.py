# scripts/live_training_monitor.py
"""
Live Training Monitor - Real-time visualization during RL training
"""
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / 'scripts'))
sys.path.append(str(PROJECT_ROOT / 'scripts' / 'components'))

from phase5_rl_integration import load_gnn_model_for_rl
from rl_environment import StockTradingEnv
from rl_agent import StockTradingAgent

MODELS_DIR = PROJECT_ROOT / "models"
DATA_GRAPHS_DIR = PROJECT_ROOT / "data" / "graphs"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TrainingMonitorCallback(BaseCallback):
    """Callback to monitor training progress."""
    
    def __init__(self, visualizer, verbose=0):
        super().__init__(verbose)
        self.visualizer = visualizer
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        # Get info from environment
        if 'episode' in self.locals.get('infos', [{}])[0]:
            episode_info = self.locals['infos'][0]['episode']
            self.episode_rewards.append(episode_info['r'])
            self.episode_lengths.append(episode_info['l'])
            self.episode_count += 1
            
            # Update visualization
            if self.visualizer:
                self.visualizer.update_training_data(
                    self.num_timesteps,
                    self.episode_rewards[-1] if self.episode_rewards else 0,
                    np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else 0
                )
        
        return True


class TrainingVisualizer:
    """Real-time visualization during training."""
    
    def __init__(self):
        self.timesteps = deque(maxlen=1000)
        self.episode_rewards = deque(maxlen=1000)
        self.avg_rewards = deque(maxlen=1000)
        
        # Setup figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(14, 5))
        self.fig.suptitle('RL Training - Real-time Monitor', fontsize=14, fontweight='bold')
        
        # Initialize plots
        self.line1, = self.ax1.plot([], [], 'b-', linewidth=2, label='Episode Reward')
        self.line2, = self.ax1.plot([], [], 'r--', linewidth=2, label='Avg Reward (10 episodes)')
        self.ax1.set_xlabel('Episode')
        self.ax1.set_ylabel('Reward')
        self.ax1.set_title('Episode Rewards')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        
        self.line3, = self.ax2.plot([], [], 'g-', linewidth=2)
        self.ax2.set_xlabel('Timestep')
        self.ax2.set_ylabel('Reward')
        self.ax2.set_title('Reward Over Time')
        self.ax2.grid(True, alpha=0.3)
        
        self.text_stats = self.fig.text(0.5, 0.02, '', ha='center', fontsize=10,
                                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    def update_training_data(self, timestep, episode_reward, avg_reward):
        """Update training data."""
        self.timesteps.append(timestep)
        self.episode_rewards.append(episode_reward)
        self.avg_rewards.append(avg_reward)
    
    def update_plot(self, frame):
        """Update plots."""
        if len(self.episode_rewards) == 0:
            return []
        
        episodes = np.arange(len(self.episode_rewards))
        
        # Update episode rewards
        self.line1.set_data(episodes, list(self.episode_rewards))
        self.line2.set_data(episodes, list(self.avg_rewards))
        if len(episodes) > 0:
            self.ax1.set_xlim(0, max(episodes[-1], 10))
            if len(self.episode_rewards) > 0:
                all_rewards = list(self.episode_rewards) + list(self.avg_rewards)
                self.ax1.set_ylim(min(all_rewards) * 1.1, max(all_rewards) * 1.1)
        
        # Update timestep rewards
        if len(self.timesteps) > 0:
            self.line3.set_data(list(self.timesteps), list(self.episode_rewards))
            self.ax2.set_xlim(0, max(self.timesteps[-1], 100))
            if len(self.episode_rewards) > 0:
                self.ax2.set_ylim(min(self.episode_rewards) * 1.1, max(self.episode_rewards) * 1.1)
        
        # Update stats
        if len(self.episode_rewards) > 0:
            stats_text = (
                f"Episodes: {len(self.episode_rewards)} | "
                f"Latest Reward: {self.episode_rewards[-1]:.4f} | "
                f"Avg Reward: {np.mean(list(self.episode_rewards)):.4f} | "
                f"Timesteps: {self.timesteps[-1] if self.timesteps else 0}"
            )
            self.text_stats.set_text(stats_text)
        
        return [self.line1, self.line2, self.line3, self.text_stats]
    
    def start_monitoring(self):
        """Start the animation."""
        ani = FuncAnimation(self.fig, self.update_plot, interval=500, blit=False)
        plt.show(block=False)
        return ani


def main():
    """Main function for live training monitor."""
    print("="*70)
    print("Live RL Training Monitor")
    print("="*70)
    print("\nThis will train the agent with real-time visualization.")
    print("A window will show training progress.\n")
    
    # Load GNN
    print("[1/3] Loading GNN model...")
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
    
    def make_env():
        return StockTradingEnv(
            start_date=START_DATE,
            end_date=END_DATE,
            gnn_model=gnn_model,
            device=DEVICE
        )
    
    # Create visualizer
    visualizer = TrainingVisualizer()
    ani = visualizer.start_monitoring()
    
    # Create agent with callback
    agent = StockTradingAgent(
        gnn_model=gnn_model,
        env_factory=make_env,
        device=DEVICE,
        learning_rate=1e-5,
        tensorboard_log=None,
        policy="MlpPolicy",
        verbose=0
    )
    
    callback = TrainingMonitorCallback(visualizer)
    
    print(f"\n[3/3] Starting training with live monitor...")
    print("   Training for 1000 timesteps...")
    print("   Watch the plot window for real-time updates\n")
    
    try:
        agent.train(total_timesteps=1000, callback=callback, progress_bar=False)
        print("\n✅ Training complete!")
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    finally:
        plt.show(block=True)


if __name__ == '__main__':
    main()


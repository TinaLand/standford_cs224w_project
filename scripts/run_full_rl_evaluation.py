# scripts/run_full_rl_evaluation.py
"""
Full RL Evaluation Pipeline with Dynamic Visualization
Runs complete training and evaluation, then displays results dynamically.
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

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / 'scripts'))
sys.path.append(str(PROJECT_ROOT / 'scripts' / 'components'))

# Import modules
from phase4_core_training import RoleAwareGraphTransformer
from phase5_rl_integration import load_gnn_model_for_rl, run_rl_pipeline
from rl_environment import StockTradingEnv
from rl_agent import StockTradingAgent
from phase6_evaluation import calculate_financial_metrics

# Configuration
MODELS_DIR = PROJECT_ROOT / "models"
DATA_GRAPHS_DIR = PROJECT_ROOT / "data" / "graphs"
RESULTS_DIR = PROJECT_ROOT / "results"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RL_SAVE_PATH = MODELS_DIR / "rl_ppo_agent_model" / "ppo_stock_agent.zip"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class RLResultsVisualizer:
    """Dynamic visualization of RL training and evaluation results."""
    
    def __init__(self):
        self.portfolio_values = []
        self.dates = []
        self.rewards = []
        self.actions_history = []
        self.start_time = None
        
    def update_training_progress(self, step, portfolio_value, reward, date_str):
        """Update training progress data."""
        self.portfolio_values.append(portfolio_value)
        self.dates.append(date_str)
        self.rewards.append(reward)
        if self.start_time is None:
            self.start_time = time.time()
    
    def run_backtest_with_visualization(self, agent, gnn_model, start_date, end_date):
        """Run backtest and collect data for visualization."""
        print("\n" + "="*70)
        print("Running Full Backtest with Data Collection")
        print("="*70)
        
        # Create environment
        def make_test_env():
            return StockTradingEnv(
                start_date=start_date,
                end_date=end_date,
                gnn_model=gnn_model,
                device=DEVICE
            )
        
        test_env = make_test_env()
        
        # Reset environment
        obs, info = test_env.reset()
        initial_value = test_env.initial_cash
        self.portfolio_values = [initial_value]
        self.dates = []
        self.rewards = []
        self.actions_history = []
        
        step = 0
        done = False
        
        print(f"Starting backtest: ${initial_value:.2f}")
        print(f"Date range: {start_date.date()} to {end_date.date()}")
        
        while not done:
            # Get action from agent
            action, _ = agent.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            
            # Collect data
            portfolio_value = info.get('portfolio_value', test_env.portfolio_value)
            date_str = info.get('date', f'Step_{step}')
            
            self.portfolio_values.append(portfolio_value)
            self.dates.append(date_str)
            self.rewards.append(reward)
            self.actions_history.append(action.copy())
            
            # Progress update
            if step % 50 == 0:
                print(f"  Step {step}: Value ${portfolio_value:.2f}, Reward: {reward:.6f}")
            
            step += 1
        
        print(f"\nBacktest complete: {step} steps")
        print(f"Final value: ${self.portfolio_values[-1]:.2f}")
        
        return step
    
    def create_static_plots(self, save_path=None):
        """Create static plots of results."""
        if not self.portfolio_values:
            print("No data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('RL Agent Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Portfolio Value Over Time
        ax1 = axes[0, 0]
        steps = range(len(self.portfolio_values))
        ax1.plot(steps, self.portfolio_values, 'b-', linewidth=2, label='Portfolio Value')
        ax1.axhline(y=self.portfolio_values[0], color='r', linestyle='--', label='Initial Value')
        ax1.set_xlabel('Trading Step')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_title('Portfolio Value Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Cumulative Returns
        ax2 = axes[0, 1]
        returns = np.array(self.portfolio_values) / self.portfolio_values[0] - 1
        ax2.plot(steps, returns * 100, 'g-', linewidth=2)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Trading Step')
        ax2.set_ylabel('Cumulative Return (%)')
        ax2.set_title('Cumulative Returns')
        ax2.grid(True, alpha=0.3)
        
        # 3. Reward Distribution
        ax3 = axes[1, 0]
        if self.rewards:
            ax3.hist(self.rewards, bins=50, alpha=0.7, color='purple', edgecolor='black')
            ax3.axvline(x=0, color='r', linestyle='--', label='Zero Reward')
            ax3.set_xlabel('Reward')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Reward Distribution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Drawdown Analysis
        ax4 = axes[1, 1]
        portfolio_series = pd.Series(self.portfolio_values)
        cumulative_max = portfolio_series.cummax()
        drawdown = (cumulative_max - portfolio_series) / cumulative_max * 100
        ax4.fill_between(steps, drawdown, 0, alpha=0.3, color='red')
        ax4.plot(steps, drawdown, 'r-', linewidth=1)
        ax4.set_xlabel('Trading Step')
        ax4.set_ylabel('Drawdown (%)')
        ax4.set_title('Drawdown Analysis')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✅ Plots saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_report(self, total_steps):
        """Generate comprehensive performance report."""
        if not self.portfolio_values:
            return None
        
        # Calculate metrics
        metrics = calculate_financial_metrics(self.portfolio_values, total_steps)
        
        # Additional metrics
        total_return = (self.portfolio_values[-1] / self.portfolio_values[0] - 1) * 100
        avg_reward = np.mean(self.rewards) if self.rewards else 0
        std_reward = np.std(self.rewards) if self.rewards else 0
        
        # Action statistics
        if self.actions_history:
            actions_array = np.array(self.actions_history)
            buy_count = np.sum(actions_array == 2)
            sell_count = np.sum(actions_array == 0)
            hold_count = np.sum(actions_array == 1)
            total_actions = len(actions_array) * len(actions_array[0])
        else:
            buy_count = sell_count = hold_count = total_actions = 0
        
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_steps': total_steps,
            'initial_value': self.portfolio_values[0],
            'final_value': self.portfolio_values[-1],
            'total_return_pct': total_return,
            'sharpe_ratio': metrics.get('Sharpe_Ratio', 0),
            'max_drawdown': metrics.get('Max_Drawdown', 0) * 100,
            'cumulative_return': metrics.get('Cumulative_Return', 0) * 100,
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'buy_actions': buy_count,
            'sell_actions': sell_count,
            'hold_actions': hold_count,
            'total_actions': total_actions
        }
        
        return report
    
    def print_report(self, report):
        """Print formatted report."""
        print("\n" + "="*70)
        print("RL AGENT PERFORMANCE REPORT")
        print("="*70)
        print(f"\nTimestamp: {report['timestamp']}")
        print(f"\n📊 Portfolio Performance:")
        print(f"   Initial Value:      ${report['initial_value']:,.2f}")
        print(f"   Final Value:        ${report['final_value']:,.2f}")
        print(f"   Total Return:       {report['total_return_pct']:.2f}%")
        print(f"   Cumulative Return:  {report['cumulative_return']:.2f}%")
        
        print(f"\n📈 Risk Metrics:")
        print(f"   Sharpe Ratio:       {report['sharpe_ratio']:.4f}")
        print(f"   Max Drawdown:       {report['max_drawdown']:.2f}%")
        
        print(f"\n🎯 Training Metrics:")
        print(f"   Average Reward:     {report['avg_reward']:.6f}")
        print(f"   Reward Std Dev:     {report['std_reward']:.6f}")
        print(f"   Total Steps:        {report['total_steps']}")
        
        print(f"\n🔄 Action Statistics:")
        print(f"   Buy Actions:        {report['buy_actions']:,} ({report['buy_actions']/max(report['total_actions'],1)*100:.1f}%)")
        print(f"   Sell Actions:       {report['sell_actions']:,} ({report['sell_actions']/max(report['total_actions'],1)*100:.1f}%)")
        print(f"   Hold Actions:        {report['hold_actions']:,} ({report['hold_actions']/max(report['total_actions'],1)*100:.1f}%)")
        print(f"   Total Actions:       {report['total_actions']:,}")
        
        print("="*70 + "\n")


def main():
    """Main evaluation pipeline."""
    print("="*70)
    print("Full RL Evaluation Pipeline with Dynamic Visualization")
    print("="*70)
    
    # Step 1: Load GNN Model
    print("\n[Step 1/4] Loading GNN Model...")
    try:
        gnn_model = load_gnn_model_for_rl()
    except Exception as e:
        print(f"❌ Failed to load GNN model: {e}")
        return
    
    # Step 2: Check if RL agent exists, if not train it
    print("\n[Step 2/4] Checking RL Agent...")
    
    # Get date range for environment
    graph_files = list(DATA_GRAPHS_DIR.glob('graph_t_*.pt'))
    graph_dates = sorted([pd.to_datetime(f.stem.split('_')[-1]) for f in graph_files])
    graph_start = graph_dates[0]
    graph_end = graph_dates[-1]
    date_range_days = (graph_end - graph_start).days
    start_offset_days = int(date_range_days * 0.8)
    START_DATE = graph_start + pd.Timedelta(days=start_offset_days)
    END_DATE = graph_end
    
    def make_env():
        return StockTradingEnv(
            start_date=START_DATE,
            end_date=END_DATE,
            gnn_model=gnn_model,
            device=DEVICE
        )
    
    # Check if agent exists and is compatible
    agent_exists = RL_SAVE_PATH.exists()
    needs_retrain = False
    
    if agent_exists:
        print("✅ Found existing RL agent. Checking compatibility...")
        try:
            # Try to load and check observation space
            test_env = make_env()
            temp_agent = StockTradingAgent(
                gnn_model=gnn_model,
                env_factory=make_env,
                device=DEVICE,
                learning_rate=1e-5,
                tensorboard_log=None,
                policy="MlpPolicy",
                verbose=0
            )
            temp_agent.load(RL_SAVE_PATH)
            # If we get here, check if observation spaces match
            if temp_agent.agent.observation_space.shape != test_env.observation_space.shape:
                print(f"⚠️  Observation space mismatch!")
                print(f"   Saved: {temp_agent.agent.observation_space.shape}")
                print(f"   Current: {test_env.observation_space.shape}")
                print("   Agent needs to be retrained with correct observation space.")
                needs_retrain = True
            else:
                agent = temp_agent
                print("✅ RL agent loaded successfully (compatible)")
        except Exception as e:
            print(f"⚠️  Error loading agent: {e}")
            print("   Agent needs to be retrained.")
            needs_retrain = True
    
    if not agent_exists or needs_retrain:
        if needs_retrain:
            print("\n⚠️  Retraining RL agent with correct observation space...")
        else:
            print("⚠️  RL agent not found. Training new agent...")
        print("   This may take a while...")
        try:
            agent = run_rl_pipeline()
            if agent is None:
                print("❌ RL training failed")
                return
        except Exception as e:
            print(f"❌ RL training error: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Step 3: Run backtest
    print("\n[Step 3/4] Running Full Backtest...")
    graph_files = list(DATA_GRAPHS_DIR.glob('graph_t_*.pt'))
    graph_dates = sorted([pd.to_datetime(f.stem.split('_')[-1]) for f in graph_files])
    graph_start = graph_dates[0]
    graph_end = graph_dates[-1]
    date_range_days = (graph_end - graph_start).days
    start_offset_days = int(date_range_days * 0.8)
    START_DATE = graph_start + pd.Timedelta(days=start_offset_days)
    END_DATE = graph_end
    
    visualizer = RLResultsVisualizer()
    total_steps = visualizer.run_backtest_with_visualization(
        agent, gnn_model, START_DATE, END_DATE
    )
    
    # Step 4: Generate report and plots
    print("\n[Step 4/4] Generating Report and Visualizations...")
    report = visualizer.generate_report(total_steps)
    visualizer.print_report(report)
    
    # Save results
    report_df = pd.DataFrame([report])
    report_path = RESULTS_DIR / f'rl_evaluation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    report_df.to_csv(report_path, index=False)
    print(f"✅ Report saved to: {report_path}")
    
    # Create and save plots
    plot_path = RESULTS_DIR / f'rl_performance_plots_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    visualizer.create_static_plots(save_path=plot_path)
    
    print("\n" + "="*70)
    print("✅ Evaluation Complete!")
    print("="*70)
    print(f"\n📁 Results saved to:")
    print(f"   - Report: {report_path}")
    print(f"   - Plots:  {plot_path}")
    print("\n")


if __name__ == '__main__':
    main()


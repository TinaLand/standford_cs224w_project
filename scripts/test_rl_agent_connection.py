# scripts/test_rl_agent_connection.py
"""
Test script to verify Agent and RL integration.
Tests: GNN model loading, Agent creation, Environment connection, and basic training step.
"""
import torch
import pandas as pd
from pathlib import Path
import sys

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / 'scripts'))
sys.path.append(str(PROJECT_ROOT / 'scripts' / 'components'))

# Import required modules
from phase4_core_training import RoleAwareGraphTransformer
from rl_environment import StockTradingEnv
from rl_agent import StockTradingAgent

# Configuration
MODELS_DIR = PROJECT_ROOT / "models"
DATA_GRAPHS_DIR = PROJECT_ROOT / "data" / "graphs"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CORE_GNN_MODEL_PATH = MODELS_DIR / 'core_transformer_model.pt'

def test_gnn_model_loading():
    """Test 1: Verify GNN model can be loaded and has get_embeddings method."""
    print("\n" + "="*70)
    print("TEST 1: GNN Model Loading")
    print("="*70)
    
    try:
        # Check if model file exists
        if not CORE_GNN_MODEL_PATH.exists():
            print(f"❌ FAIL: GNN model not found at {CORE_GNN_MODEL_PATH}")
            print("   Run phase4_core_training.py first to train the model.")
            return False
        
        # Load sample graph to get input dimension
        graph_files = list(DATA_GRAPHS_DIR.glob('graph_t_*.pt'))
        if not graph_files:
            print(f"❌ FAIL: No graph files found in {DATA_GRAPHS_DIR}")
            return False
        
        sample_graph = torch.load(graph_files[0], weights_only=False)
        INPUT_DIM = sample_graph['stock'].x.shape[1]
        print(f"✅ Sample graph loaded. Input dimension: {INPUT_DIM}")
        
        # Initialize model
        gnn_model = RoleAwareGraphTransformer(
            INPUT_DIM, 256, 2, 2, 4
        ).to(DEVICE)
        
        # Load trained weights
        gnn_model.load_state_dict(
            torch.load(CORE_GNN_MODEL_PATH, map_location=DEVICE, weights_only=False)
        )
        print(f"✅ GNN model weights loaded from {CORE_GNN_MODEL_PATH}")
        
        # Freeze parameters
        for param in gnn_model.parameters():
            param.requires_grad = False
        print("✅ GNN model parameters frozen")
        
        # Test get_embeddings method
        if not hasattr(gnn_model, 'get_embeddings'):
            print("❌ FAIL: get_embeddings() method not found in GNN model")
            return False
        
        # Test get_embeddings with sample graph
        sample_graph_tensor = sample_graph.to(DEVICE)
        with torch.no_grad():
            gnn_model.eval()
            embeddings = gnn_model.get_embeddings(sample_graph_tensor)
        
        print(f"✅ get_embeddings() works. Output shape: {embeddings.shape}")
        print(f"   Expected: [num_stocks, hidden_dim=256]")
        
        return True, gnn_model
        
    except Exception as e:
        print(f"❌ FAIL: Error loading GNN model: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_environment_creation(gnn_model):
    """Test 2: Verify Environment can be created and uses GNN correctly."""
    print("\n" + "="*70)
    print("TEST 2: Environment Creation")
    print("="*70)
    
    try:
        # Get date range from graph files
        graph_files = list(DATA_GRAPHS_DIR.glob('graph_t_*.pt'))
        graph_dates = sorted([pd.to_datetime(f.stem.split('_')[-1]) for f in graph_files])
        
        if len(graph_dates) < 10:
            print(f"❌ FAIL: Not enough graph files. Need at least 10, got {len(graph_dates)}")
            return False
        
        # Use a small date range for testing
        start_date = graph_dates[0]
        end_date = graph_dates[min(9, len(graph_dates)-1)]  # Use first 10 dates
        
        print(f"📅 Test date range: {start_date.date()} to {end_date.date()}")
        
        # Create environment
        env = StockTradingEnv(
            start_date=start_date,
            end_date=end_date,
            gnn_model=gnn_model,
            device=DEVICE
        )
        print(f"✅ Environment created successfully")
        print(f"   - Number of stocks: {env.NUM_STOCKS}")
        print(f"   - Observation space: {env.observation_space.shape}")
        print(f"   - Action space: {env.action_space}")
        
        # Test reset
        obs, info = env.reset()
        print(f"✅ Environment reset successful")
        print(f"   - Observation shape: {obs.shape}")
        print(f"   - Initial portfolio value: ${info.get('portfolio_value', 'N/A')}")
        
        # Test one step
        action = env.action_space.sample()  # Random action for testing
        next_obs, reward, terminated, truncated, info = env.step(action)
        print(f"✅ Environment step successful")
        print(f"   - Next observation shape: {next_obs.shape}")
        print(f"   - Reward: {reward:.6f}")
        print(f"   - Portfolio value: ${info.get('portfolio_value', 'N/A'):.2f}")
        
        return True, env
        
    except Exception as e:
        print(f"❌ FAIL: Error creating environment: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_agent_creation(gnn_model, env):
    """Test 3: Verify Agent can be created and connected to environment."""
    print("\n" + "="*70)
    print("TEST 3: Agent Creation and Connection")
    print("="*70)
    
    try:
        # Create environment factory
        def make_env():
            return StockTradingEnv(
                start_date=env.data_loader['dates'][0],
                end_date=env.data_loader['dates'][-1],
                gnn_model=gnn_model,
                device=DEVICE
            )
        
        # Create agent
        agent = StockTradingAgent(
            gnn_model=gnn_model,
            env_factory=make_env,
            device=DEVICE,
            learning_rate=1e-5,
            tensorboard_log=None,
            policy="MlpPolicy",
            verbose=0
        )
        print(f"✅ Agent created successfully")
        print(f"   - Agent type: {type(agent).__name__}")
        print(f"   - PPO agent: {type(agent.agent).__name__}")
        print(f"   - Is trained: {agent.is_trained}")
        
        # Test predict (before training, should still work)
        test_obs, _ = env.reset()
        action, _ = agent.predict(test_obs, deterministic=True)
        print(f"✅ Agent predict works")
        print(f"   - Action shape: {action.shape}")
        print(f"   - Action sample: {action[:5]}")  # Show first 5 actions
        
        return True, agent
        
    except Exception as e:
        print(f"❌ FAIL: Error creating agent: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_training_step(agent, env):
    """Test 4: Verify a short training step works."""
    print("\n" + "="*70)
    print("TEST 4: Training Step (Short Run)")
    print("="*70)
    
    try:
        # Run a very short training (just 10 steps to verify it works)
        print("Running 10 training steps...")
        training_stats = agent.train(
            total_timesteps=10,
            progress_bar=False
        )
        print(f"✅ Training step completed")
        print(f"   - Training stats: {training_stats}")
        print(f"   - Agent is now trained: {agent.is_trained}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("RL Agent & RL Integration Verification Test")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Project root: {PROJECT_ROOT}")
    
    results = {}
    
    # Test 1: GNN Model Loading
    result, gnn_model = test_gnn_model_loading()
    results['gnn_loading'] = result
    if not result:
        print("\n❌ CRITICAL: GNN model loading failed. Cannot continue.")
        return
    
    # Test 2: Environment Creation
    result, env = test_environment_creation(gnn_model)
    results['env_creation'] = result
    if not result:
        print("\n❌ CRITICAL: Environment creation failed. Cannot continue.")
        return
    
    # Test 3: Agent Creation
    result, agent = test_agent_creation(gnn_model, env)
    results['agent_creation'] = result
    if not result:
        print("\n❌ CRITICAL: Agent creation failed. Cannot continue.")
        return
    
    # Test 4: Training Step
    result = test_training_step(agent, env)
    results['training_step'] = result
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name:20s}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Agent and RL are properly connected.")
    else:
        print("\n⚠️  SOME TESTS FAILED. Please check the errors above.")
    
    print("="*70 + "\n")


if __name__ == '__main__':
    main()


# scripts/check_and_evaluate_final.py
"""
Check if final training is complete and evaluate if ready
"""

import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / 'scripts'))

FINAL_AGENT_PATH = PROJECT_ROOT / "models" / "rl_ppo_agent_model_final" / "ppo_stock_agent_final.zip"


def check_training_status():
    """Check if final training is complete."""
    print("=" * 80)
    print("🔍 Checking Final Training Status")
    print("=" * 80)
    
    if FINAL_AGENT_PATH.exists():
        print(f"✅ Final agent found: {FINAL_AGENT_PATH}")
        print(f"   File size: {FINAL_AGENT_PATH.stat().st_size / 1024 / 1024:.2f} MB")
        print(f"   Modified: {time.ctime(FINAL_AGENT_PATH.stat().st_mtime)}")
        return True
    else:
        print(f"⏳ Final agent not found yet")
        print(f"   Expected path: {FINAL_AGENT_PATH}")
        print(f"   Training may still be in progress...")
        return False


def wait_and_evaluate(max_wait_minutes=30, check_interval=60):
    """
    Wait for training to complete and then evaluate.
    
    Args:
        max_wait_minutes: Maximum time to wait (minutes)
        check_interval: How often to check (seconds)
    """
    print(f"\n⏳ Waiting for training to complete (max {max_wait_minutes} minutes)...")
    print(f"   Checking every {check_interval} seconds")
    
    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60
    
    while time.time() - start_time < max_wait_seconds:
        if check_training_status():
            print("\n✅ Training complete! Evaluating final agent...")
            from evaluate_final_agent import evaluate_final_agent
            results = evaluate_final_agent()
            return results
        
        print(f"   Still waiting... ({int((time.time() - start_time) / 60)} minutes elapsed)")
        time.sleep(check_interval)
    
    print(f"\n⏰ Max wait time reached ({max_wait_minutes} minutes)")
    print("   You can manually run: python scripts/evaluate_final_agent.py")
    return None


if __name__ == '__main__':
    if check_training_status():
        # Training complete, evaluate immediately
        print("\n✅ Training complete! Evaluating final agent...")
        from evaluate_final_agent import evaluate_final_agent
        results = evaluate_final_agent()
    else:
        # Wait for training
        print("\n💡 Options:")
        print("   1. Wait for training to complete (this script will auto-evaluate)")
        print("   2. Run manually later: python scripts/evaluate_final_agent.py")
        
        response = input("\nWait for training? (y/n): ").strip().lower()
        if response == 'y':
            wait_and_evaluate()
        else:
            print("\n💡 To evaluate later, run:")
            print("   python scripts/evaluate_final_agent.py")


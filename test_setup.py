"""
Test Script to Verify Project Setup
"""

import sys
import os

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    try:
        import gymnasium
        print("  ✓ Gymnasium")
    except ImportError as e:
        print(f"  ✗ Gymnasium - {e}")
        return False
    
    try:
        import stable_baselines3
        print("  ✓ Stable-Baselines3")
    except ImportError as e:
        print(f"  ✗ Stable-Baselines3 - {e}")
        return False
    
    try:
        import torch
        print("  ✓ PyTorch")
    except ImportError as e:
        print(f"  ✗ PyTorch - {e}")
        return False
    
    try:
        import numpy
        print("  ✓ NumPy")
    except ImportError as e:
        print(f"  ✗ NumPy - {e}")
        return False
    
    try:
        import matplotlib
        print("  ✓ Matplotlib")
    except ImportError as e:
        print(f"  ✗ Matplotlib - {e}")
        return False
    
    try:
        from panda3d.core import *
        print("  ✓ Panda3D")
    except ImportError as e:
        print(f"  ✗ Panda3D - {e}")
        return False
    
    return True


def test_environment():
    """Test custom environment"""
    print("\nTesting custom environment...")
    
    try:
        from environment.custom_env import CrowdControlEnv
        
        env = CrowdControlEnv()
        obs, info = env.reset()
        
        print(f"  ✓ Environment created")
        print(f"  ✓ Observation space: {env.observation_space.shape}")
        print(f"  ✓ Action space: {env.action_space.n} actions")
        
        # Test a few steps
        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"  ✓ Environment step working")
        print(f"  ✓ Initial crowd: {info['total_crowd']:.1f}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"  ✗ Environment test failed - {e}")
        return False


def test_directory_structure():
    """Test directory structure"""
    print("\nTesting directory structure...")
    
    required_dirs = [
        "environment",
        "training",
        "models",
        "models/dqn",
        "models/ppo",
        "models/a2c",
        "models/reinforce",
        "logs",
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"  ✓ {dir_path}/")
        else:
            print(f"  ✗ {dir_path}/ - Missing!")
            all_exist = False
    
    return all_exist


def test_files():
    """Test required files exist"""
    print("\nTesting required files...")
    
    required_files = [
        "requirements.txt",
        "README.md",
        "main.py",
        "demo_random_agent.py",
        "environment/__init__.py",
        "environment/custom_env.py",
        "environment/rendering.py",
        "training/dqn_training.py",
        "training/ppo_training.py",
        "training/a2c_training.py",
        "training/reinforce_training.py",
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} - Missing!")
            all_exist = False
    
    return all_exist


def main():
    print("="*60)
    print("CROWD CONTROL RL PROJECT - SETUP TEST")
    print("="*60)
    
    # Run all tests
    tests = [
        ("Import Test", test_imports),
        ("Directory Structure Test", test_directory_structure),
        ("Required Files Test", test_files),
        ("Environment Test", test_environment),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        results.append(test_func())
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for (test_name, _), result in zip(tests, results):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    if all(results):
        print("\n✓ ALL TESTS PASSED - Project setup is complete!")
        print("\nYou can now:")
        print("  1. Run random agent: python demo_random_agent.py")
        print("  2. Start training: python training/ppo_training.py --config 0 --timesteps 10000")
        print("="*60)
        return 0
    else:
        print("\n✗ SOME TESTS FAILED - Please fix the issues above")
        print("\nCommon fixes:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Activate virtual environment: .\\venv\\Scripts\\Activate.ps1")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())

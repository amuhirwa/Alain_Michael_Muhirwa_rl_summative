"""
Vectorized Environment Training - FAST Hyperparameter Experiments
==================================================================

Uses parallel environments to speed up training by 2-4x.
Perfect for running multiple hyperparameter configurations quickly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.enhanced_env_fast import EnhancedCrowdControlEnvFast
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
import json
from datetime import datetime

def make_vec_env(difficulty='medium', pattern='rush', adversarial=False, n_envs=4):
    """
    Create vectorized environment with n_envs parallel environments.
    
    Args:
        difficulty: easy/medium/hard
        pattern: rush/steady/evacuation
        adversarial: Enable adversarial scenarios
        n_envs: Number of parallel environments
    
    Returns:
        SubprocVecEnv for true parallelism
    """
    def make_env(rank):
        """Create a single environment with unique seed"""
        def _init():
            env = EnhancedCrowdControlEnvFast(
                crowd_arrival_pattern=pattern,
                adversarial_mode=adversarial,
                difficulty=difficulty
            )
            env.reset(seed=42 + rank)  # Unique seed for each env
            return env
        return _init
    
    # Create list of environment functions
    env_fns = [make_env(i) for i in range(n_envs)]
    
    # Use SubprocVecEnv for true parallelism (separate processes)
    # Use DummyVecEnv for sequential execution (debugging)
    return SubprocVecEnv(env_fns)


def train_ppo_vectorized(timesteps=50000, n_envs=4, difficulty='medium'):
    """
    Train PPO with vectorized environments for 2-4x speedup
    
    Args:
        timesteps: Total timesteps
        n_envs: Number of parallel environments (4-8 recommended)
        difficulty: Environment difficulty
    """
    print("\n" + "="*70)
    print("VECTORIZED PPO TRAINING")
    print("="*70)
    print(f"Parallel Environments: {n_envs}")
    print(f"Total Timesteps: {timesteps:,}")
    print(f"Difficulty: {difficulty}")
    print(f"Expected Speedup: {n_envs}x")
    print("="*70 + "\n")
    
    os.makedirs("models/vec_ppo", exist_ok=True)
    os.makedirs("logs/vec_ppo", exist_ok=True)
    
    # Create vectorized training environment
    print(f"Creating {n_envs} parallel environments...")
    env = make_vec_env(difficulty=difficulty, pattern='rush', n_envs=n_envs)
    
    # Create single evaluation environment (no vectorization for eval)
    eval_env = EnhancedCrowdControlEnvFast(
        crowd_arrival_pattern='rush',
        difficulty=difficulty
    )
    eval_env = Monitor(eval_env, "logs/vec_ppo/eval")
    
    print("✓ Environments created")
    
    # Create PPO model
    # IMPORTANT: Adjust n_steps based on n_envs
    # Total steps per update = n_steps * n_envs
    # Keep this around 2048 total
    n_steps_per_env = max(128, 2048 // n_envs)
    
    print(f"\nModel Configuration:")
    print(f"  n_steps per env: {n_steps_per_env}")
    print(f"  Total steps per update: {n_steps_per_env * n_envs}")
    print(f"  Batch size: 64")
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=n_steps_per_env,  # Adjusted for vectorization
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="logs/vec_ppo"
    )
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/vec_ppo",
        log_path="logs/vec_ppo",
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True
    )
    
    # Train
    print("\nStarting training...")
    start_time = datetime.now()
    
    model.learn(
        total_timesteps=timesteps,
        callback=eval_callback,
        progress_bar=True
    )
    
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Save
    model.save("models/vec_ppo/ppo_final")
    
    # Evaluate
    print("\nEvaluating...")
    eval_rewards = []
    for i in range(10):
        obs, _ = eval_env.reset()
        ep_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            ep_reward += reward
            done = terminated or truncated
        eval_rewards.append(ep_reward)
    
    results = {
        "algorithm": "PPO",
        "vectorized": True,
        "n_envs": n_envs,
        "timesteps": timesteps,
        "difficulty": difficulty,
        "training_time_seconds": training_time,
        "mean_reward": float(np.mean(eval_rewards)),
        "std_reward": float(np.std(eval_rewards)),
        "speedup_factor": n_envs,
        "eval_rewards": [float(r) for r in eval_rewards]
    }
    
    with open("models/vec_ppo/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Training Time: {training_time:.1f}s ({training_time/60:.1f} min)")
    print(f"Speedup: ~{n_envs}x compared to single env")
    print(f"Model saved: models/vec_ppo/ppo_final.zip")
    print(f"{'='*70}\n")
    
    env.close()
    eval_env.close()
    
    return model, results


def train_dqn_vectorized(timesteps=50000, n_envs=4, difficulty='medium'):
    """
    Train DQN with vectorized environments
    
    Note: DQN benefits less from vectorization than PPO because it uses
    experience replay, but still provides 2x speedup from data collection.
    """
    print("\n" + "="*70)
    print("VECTORIZED DQN TRAINING")
    print("="*70)
    print(f"Parallel Environments: {n_envs}")
    print(f"Total Timesteps: {timesteps:,}")
    print(f"Difficulty: {difficulty}")
    print("="*70 + "\n")
    
    os.makedirs("models/vec_dqn", exist_ok=True)
    os.makedirs("logs/vec_dqn", exist_ok=True)
    
    # Create vectorized environment
    env = make_vec_env(difficulty=difficulty, pattern='rush', n_envs=n_envs)
    
    # Eval environment
    eval_env = EnhancedCrowdControlEnvFast(
        crowd_arrival_pattern='rush',
        difficulty=difficulty
    )
    eval_env = Monitor(eval_env, "logs/vec_dqn/eval")
    
    # Create DQN model
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        target_update_interval=1000,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log="logs/vec_dqn"
    )
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/vec_dqn",
        log_path="logs/vec_dqn",
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True
    )
    
    # Train
    print("Starting training...")
    start_time = datetime.now()
    
    model.learn(
        total_timesteps=timesteps,
        callback=eval_callback,
        progress_bar=True
    )
    
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Save
    model.save("models/vec_dqn/dqn_final")
    
    # Evaluate
    print("\nEvaluating...")
    eval_rewards = []
    for i in range(10):
        obs, _ = eval_env.reset()
        ep_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            ep_reward += reward
            done = terminated or truncated
        eval_rewards.append(ep_reward)
    
    results = {
        "algorithm": "DQN",
        "vectorized": True,
        "n_envs": n_envs,
        "timesteps": timesteps,
        "difficulty": difficulty,
        "training_time_seconds": training_time,
        "mean_reward": float(np.mean(eval_rewards)),
        "std_reward": float(np.std(eval_rewards)),
        "eval_rewards": [float(r) for r in eval_rewards]
    }
    
    with open("models/vec_dqn/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Training Time: {training_time:.1f}s ({training_time/60:.1f} min)")
    print(f"Model saved: models/vec_dqn/dqn_final.zip")
    print(f"{'='*70}\n")
    
    env.close()
    eval_env.close()
    
    return model, results


def compare_speedup():
    """
    Compare training time: single env vs vectorized
    """
    print("\n" + "="*70)
    print("SPEEDUP COMPARISON")
    print("="*70)
    print("\nTraining PPO for 10,000 timesteps with different configurations...\n")
    
    results = {}
    
    # Test configurations
    configs = [
        ("Single Env", 1),
        ("2 Envs", 2),
        ("4 Envs", 4),
        ("8 Envs", 8)
    ]
    
    for name, n_envs in configs:
        print(f"Testing: {name}")
        start = datetime.now()
        
        _, res = train_ppo_vectorized(timesteps=10000, n_envs=n_envs, difficulty='easy')
        
        elapsed = (datetime.now() - start).total_seconds()
        results[name] = elapsed
        
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Reward: {res['mean_reward']:.2f}\n")
    
    # Display comparison
    baseline = results["Single Env"]
    print("="*70)
    print("SPEEDUP SUMMARY")
    print("="*70)
    for name, time in results.items():
        speedup = baseline / time
        print(f"{name:15s}: {time:6.1f}s  |  Speedup: {speedup:.2f}x")
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='ppo', choices=['ppo', 'dqn', 'both'])
    parser.add_argument('--timesteps', type=int, default=50000)
    parser.add_argument('--n-envs', type=int, default=4, help='Number of parallel environments')
    parser.add_argument('--difficulty', type=str, default='medium', choices=['easy', 'medium', 'hard'])
    parser.add_argument('--compare', action='store_true', help='Run speedup comparison')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_speedup()
    else:
        if args.algorithm in ['ppo', 'both']:
            train_ppo_vectorized(args.timesteps, args.n_envs, args.difficulty)
        
        if args.algorithm in ['dqn', 'both']:
            train_dqn_vectorized(args.timesteps, args.n_envs, args.difficulty)

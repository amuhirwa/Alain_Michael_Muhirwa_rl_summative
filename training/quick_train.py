"""
Quick Training Script for Report Generation
============================================

Trains PPO and DQN with reduced timesteps for fast results.
Use this to generate results for the report quickly.

For full training, use curriculum_training.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.enhanced_env_fast import EnhancedCrowdControlEnvFast
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import numpy as np
import json
from datetime import datetime

def make_vec_env(difficulty='medium', pattern='rush', adversarial=False, n_envs=4):
    """Create vectorized environment for faster training"""
    def make_env(rank):
        def _init():
            env = EnhancedCrowdControlEnvFast(
                crowd_arrival_pattern=pattern,
                adversarial_mode=adversarial,
                difficulty=difficulty
            )
            env.reset(seed=42 + rank)
            return env
        return _init
    
    env_fns = [make_env(i) for i in range(n_envs)]
    return SubprocVecEnv(env_fns)

def train_ppo_quick(timesteps=50000, difficulty='medium', n_envs=4):
    """Train PPO quickly for report with vectorized environments"""
    print("\n" + "="*70)
    print("QUICK PPO TRAINING (VECTORIZED)")
    print("="*70)
    print(f"Parallel Environments: {n_envs}")
    print(f"Expected Speedup: ~{n_envs}x")
    print("="*70)
    
    # Create directories
    os.makedirs("models/quick_ppo", exist_ok=True)
    os.makedirs("logs/quick_ppo", exist_ok=True)
    
    # Create vectorized environment (2-4x faster!)
    env = make_vec_env(
        difficulty=difficulty,
        pattern='rush',
        adversarial=False,
        n_envs=n_envs
    )
    
    # Create eval environment
    eval_env = EnhancedCrowdControlEnvFast(
        crowd_arrival_pattern='rush',
        adversarial_mode=False,
        difficulty=difficulty
    )
    eval_env = Monitor(eval_env, "logs/quick_ppo/eval")
    
    # Create model (adjust n_steps for vectorization)
    n_steps_per_env = max(128, 2048 // n_envs)
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=n_steps_per_env,  # Adjusted for parallel envs
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="logs/quick_ppo"
    )
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/quick_ppo",
        log_path="logs/quick_ppo",
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="models/quick_ppo",
        name_prefix="ppo_checkpoint"
    )
    
    # Train
    print(f"\nTraining PPO for {timesteps:,} timesteps...")
    start_time = datetime.now()
    
    model.learn(
        total_timesteps=timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Save
    model.save("models/quick_ppo/ppo_final")
    
    # Evaluate
    print("\nEvaluating final model...")
    eval_rewards = []
    eval_lengths = []
    
    for i in range(10):
        obs, _ = eval_env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
        
        eval_rewards.append(episode_reward)
        eval_lengths.append(episode_length)
        print(f"  Episode {i+1}: Reward={episode_reward:.2f}, Length={episode_length}")
    
    results = {
        "algorithm": "PPO",
        "timesteps": timesteps,
        "difficulty": difficulty,
        "training_time_seconds": training_time,
        "mean_reward": float(np.mean(eval_rewards)),
        "std_reward": float(np.std(eval_rewards)),
        "mean_length": float(np.mean(eval_lengths)),
        "eval_rewards": [float(r) for r in eval_rewards]
    }
    
    with open("models/quick_ppo/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"PPO Training Complete!")
    print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Mean Length: {results['mean_length']:.1f}")
    print(f"  Training Time: {training_time:.1f}s")
    print(f"  Model saved: models/quick_ppo/ppo_final.zip")
    print(f"{'='*70}\n")
    
    env.close()
    eval_env.close()
    
    return model, results


def train_dqn_quick(timesteps=50000, difficulty='medium', n_envs=4):
    """Train DQN quickly for report with vectorized environments"""
    print("\n" + "="*70)
    print("QUICK DQN TRAINING (VECTORIZED)")
    print("="*70)
    print(f"Parallel Environments: {n_envs}")
    print(f"Expected Speedup: ~2x (DQN benefits less than PPO)")
    print("="*70)
    
    # Create directories
    os.makedirs("models/quick_dqn", exist_ok=True)
    os.makedirs("logs/quick_dqn", exist_ok=True)
    
    # Create vectorized environment
    env = make_vec_env(
        difficulty=difficulty,
        pattern='rush',
        adversarial=False,
        n_envs=n_envs
    )
    
    # Create eval environment
    eval_env = EnhancedCrowdControlEnvFast(
        crowd_arrival_pattern='rush',
        adversarial_mode=False,
        difficulty=difficulty
    )
    eval_env = Monitor(eval_env, "logs/quick_dqn/eval")
    
    # Create model
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
        tensorboard_log="logs/quick_dqn"
    )
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/quick_dqn",
        log_path="logs/quick_dqn",
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="models/quick_dqn",
        name_prefix="dqn_checkpoint"
    )
    
    # Train
    print(f"\nTraining DQN for {timesteps:,} timesteps...")
    start_time = datetime.now()
    
    model.learn(
        total_timesteps=timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Save
    model.save("models/quick_dqn/dqn_final")
    
    # Evaluate
    print("\nEvaluating final model...")
    eval_rewards = []
    eval_lengths = []
    
    for i in range(10):
        obs, _ = eval_env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
        
        eval_rewards.append(episode_reward)
        eval_lengths.append(episode_length)
        print(f"  Episode {i+1}: Reward={episode_reward:.2f}, Length={episode_length}")
    
    results = {
        "algorithm": "DQN",
        "timesteps": timesteps,
        "difficulty": difficulty,
        "training_time_seconds": training_time,
        "mean_reward": float(np.mean(eval_rewards)),
        "std_reward": float(np.std(eval_rewards)),
        "mean_length": float(np.mean(eval_lengths)),
        "eval_rewards": [float(r) for r in eval_rewards]
    }
    
    with open("models/quick_dqn/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"DQN Training Complete!")
    print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Mean Length: {results['mean_length']:.1f}")
    print(f"  Training Time: {training_time:.1f}s")
    print(f"  Model saved: models/quick_dqn/dqn_final.zip")
    print(f"{'='*70}\n")
    
    env.close()
    eval_env.close()
    
    return model, results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=50000)
    parser.add_argument('--difficulty', type=str, default='medium', choices=['easy', 'medium', 'hard'])
    parser.add_argument('--algorithm', type=str, default='both', choices=['ppo', 'dqn', 'both'])
    parser.add_argument('--n-envs', type=int, default=4, help='Number of parallel environments (2-8 recommended)')
    
    args = parser.parse_args()
    
    if args.algorithm in ['ppo', 'both']:
        train_ppo_quick(args.timesteps, args.difficulty, args.n_envs)
    
    if args.algorithm in ['dqn', 'both']:
        train_dqn_quick(args.timesteps, args.difficulty, args.n_envs)
    
    print("\n" + "="*70)
    print("QUICK TRAINING COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Run: python evaluation/generate_report_plots.py")
    print("  2. Check the 'results' folder for all plots")
    print("  3. Use these plots in your report")
    print("="*70)

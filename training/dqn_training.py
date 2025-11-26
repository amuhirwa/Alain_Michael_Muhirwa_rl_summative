"""
DQN Training Script for Crowd Control Environment
================================================

Deep Q-Network (Value-Based RL Method)
Hyperparameter tuning with 10+ configurations

Key Hyperparameters:
- Learning rate
- Buffer size
- Batch size
- Gamma (discount factor)
- Target update interval
- Exploration parameters
- Network architecture
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.enhanced_env_fast import EnhancedCrowdControlEnvFast
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
import torch


def detect_convergence(eval_rewards, window=3, threshold=0.9):
    """
    Detect convergence point during training.
    
    Convergence = when rolling mean reaches 90% of best reward for 'window' consecutive evals.
    
    Args:
        eval_rewards: List of evaluation rewards at checkpoints
        window: Number of consecutive evaluations to check (default: 3)
        threshold: % of best reward to consider converged (default: 0.9 = 90%)
    
    Returns:
        convergence_step: Index where convergence occurred, or None
    """
    if len(eval_rewards) < window:
        return None
    
    best_reward = max(eval_rewards)
    convergence_value = threshold * best_reward
    
    # Check rolling mean
    for i in range(len(eval_rewards) - window + 1):
        rolling_mean = np.mean(eval_rewards[i:i+window])
        if rolling_mean >= convergence_value:
            return i  # Return index (multiply by eval_freq for timestep)
    
    return None  # Not converged


# Hyperparameter configurations for extensive tuning
HYPERPARAMETER_CONFIGS = [
    {
        "name": "config_1_baseline",
        "learning_rate": 1e-4,           # Conservative LR - prevents Q-value overestimation
        "buffer_size": 50000,            # Moderate buffer - balances diversity and memory
        "batch_size": 32,                # Standard batch - efficient gradient computation
        "gamma": 0.99,                   # Standard discount for episodic tasks
        "target_update_interval": 1000,  # Standard target sync - stabilizes Q-learning
        "exploration_fraction": 0.8,     # Long exploration - discovers strategic actions (barrier placement)
        "exploration_initial_eps": 1.0,  # Full random start - unbiased initial exploration
        "exploration_final_eps": 0.2,    # Maintain 20% exploration - prevents premature convergence
        "learning_starts": 1000,         # Buffer warm-up - ensures diverse initial samples
    },
    {
        "name": "config_2_high_lr",
        "learning_rate": 5e-4,
        "buffer_size": 50000,
        "batch_size": 32,
        "gamma": 0.99,
        "target_update_interval": 1000,
        "exploration_fraction": 0.8,  # INCREASED: Extended exploration
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.25,  # INCREASED: High final exploration
        "learning_starts": 1000,
    },
    {
        "name": "config_3_large_buffer",
        "learning_rate": 1e-4,
        "buffer_size": 100000,
        "batch_size": 64,
        "gamma": 0.99,
        "target_update_interval": 1000,
        "exploration_fraction": 0.85,  # INCREASED: Very long exploration for large buffer
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.2,  # INCREASED: Keep exploring
        "learning_starts": 2000,
    },
    {
        "name": "config_4_high_gamma",
        "learning_rate": 1e-4,
        "buffer_size": 50000,
        "batch_size": 32,
        "gamma": 0.995,
        "target_update_interval": 1000,
        "exploration_fraction": 0.8,  # INCREASED: Extended exploration
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.2,  # INCREASED: Maintain exploration
        "learning_starts": 1000,
    },
    {
        "name": "config_5_fast_target_update",  # WINNER: 1213.5 mean reward!
        "learning_rate": 1e-4,           # Conservative LR - stable Q-value updates
        "buffer_size": 50000,            # Balanced buffer - sufficient diversity without staleness
        "batch_size": 32,                # Standard batch - efficient learning
        "gamma": 0.99,                   # Standard discount - suitable for typical episode lengths
        "target_update_interval": 500,   # 2x FASTER updates - CRITICAL for non-stationary crowd dynamics
        "exploration_fraction": 0.8,     # Extended exploration - finds optimal gate/barrier strategies
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.25,  # INCREASED: High final exploration
        "learning_starts": 1000,
    },
    {
        "name": "config_6_extended_exploration",
        "learning_rate": 1e-4,
        "buffer_size": 50000,
        "batch_size": 32,
        "gamma": 0.99,
        "target_update_interval": 1000,
        "exploration_fraction": 0.6,  # INCREASED - explore longer
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.15,  # INCREASED - keep exploring
        "learning_starts": 1000,
    },
    {
        "name": "config_7_large_batch",
        "learning_rate": 1e-4,
        "buffer_size": 80000,
        "batch_size": 128,
        "gamma": 0.99,
        "target_update_interval": 1000,
        "exploration_fraction": 0.3,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "learning_starts": 2000,
    },
    {
        "name": "config_8_high_final_exploration",
        "learning_rate": 2e-4,
        "buffer_size": 50000,
        "batch_size": 32,
        "gamma": 0.99,
        "target_update_interval": 1000,
        "exploration_fraction": 0.4,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.2,  # INCREASED - keep exploring diverse actions
        "learning_starts": 500,
    },
    {
        "name": "config_9_balanced_exploration",
        "learning_rate": 1e-4,
        "buffer_size": 80000,
        "batch_size": 64,
        "gamma": 0.995,
        "target_update_interval": 1500,
        "exploration_fraction": 0.5,
        "exploration_initial_eps": 1.0,  # INCREASED - start with full exploration
        "exploration_final_eps": 0.15,  # INCREASED - maintain exploration
        "learning_starts": 2000,
    },
    {
        "name": "config_10_balanced",
        "learning_rate": 3e-4,
        "buffer_size": 60000,
        "batch_size": 64,
        "gamma": 0.99,
        "target_update_interval": 800,
        "exploration_fraction": 0.35,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "learning_starts": 1500,
    },
]


def make_env(difficulty='medium', pattern='rush', adversarial=False):
    """Create a single environment instance"""
    def _init():
        env = EnhancedCrowdControlEnvFast(
            crowd_arrival_pattern=pattern,
            adversarial_mode=adversarial,
            difficulty=difficulty
        )
        env = Monitor(env)
        return env
    return _init


def make_vec_env(difficulty='medium', pattern='rush', adversarial=False, n_envs=8):
    """Create vectorized environment with n_envs parallel environments"""
    env_fns = [make_env(difficulty, pattern, adversarial) for _ in range(n_envs)]
    try:
        vec_env = SubprocVecEnv(env_fns)
    except:
        print("⚠️  SubprocVecEnv failed, falling back to DummyVecEnv")
        vec_env = DummyVecEnv(env_fns)
    return vec_env


def train_dqn_configuration(config, total_timesteps=100000, eval_freq=5000, n_envs=1):
    """Train DQN with specific hyperparameter configuration"""
    
    print(f"\n{'='*70}")
    print(f"Training DQN: {config['name']}")
    print(f"{'='*70}")
    print("Hyperparameters:")
    for key, value in config.items():
        if key != 'name':
            print(f"  {key}: {value}")
    print("="*70)
    
    # Create directories
    log_dir = f"logs/dqn/{config['name']}"
    model_dir = f"models/dqn/{config['name']}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # DQN can use vectorized envs but typically benefits less than PPO
    # n_envs=1 is often sufficient for DQN
    if n_envs > 1:
        print(f"Creating {n_envs} parallel environments...")
        env = make_vec_env(difficulty='medium', pattern='rush', adversarial=False, n_envs=n_envs)
    else:
        # Still wrap single env in DummyVecEnv for consistency
        env = make_vec_env(difficulty='medium', pattern='rush', adversarial=False, n_envs=1)
    
    # Create evaluation environment (same type as training)
    eval_env = make_vec_env(difficulty='medium', pattern='rush', adversarial=False, n_envs=1)
    
    # Extract hyperparameters
    hp = {k: v for k, v in config.items() if k != 'name'}
    
    # Create DQN model
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=hp['learning_rate'],
        buffer_size=hp['buffer_size'],
        batch_size=hp['batch_size'],
        gamma=hp['gamma'],
        target_update_interval=hp['target_update_interval'],
        exploration_fraction=hp['exploration_fraction'],
        exploration_initial_eps=hp['exploration_initial_eps'],
        exploration_final_eps=hp['exploration_final_eps'],
        learning_starts=hp['learning_starts'],
        train_freq=4,
        gradient_steps=1,
        verbose=0,
        tensorboard_log=log_dir,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_dir,
        name_prefix="dqn_checkpoint"
    )
    
    callback_list = CallbackList([eval_callback, checkpoint_callback])
    
    # Train the model
    print(f"\nStarting training for {total_timesteps} timesteps...")
    start_time = datetime.now()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        progress_bar=True
    )
    
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Extract evaluation rewards for convergence analysis
    try:
        eval_rewards_history = eval_callback.evaluations_results
        convergence_idx = detect_convergence(eval_rewards_history, window=3, threshold=0.9)
        if convergence_idx is not None:
            convergence_timestep = convergence_idx * eval_freq
            convergence_time = convergence_idx / len(eval_rewards_history) * training_time if len(eval_rewards_history) > 0 else None
        else:
            convergence_timestep = None
            convergence_time = None
    except:
        convergence_timestep = None
        convergence_time = None
        eval_rewards_history = []
    
    # Save final model
    final_model_path = f"{model_dir}/final_model"
    model.save(final_model_path)
    print(f"\nModel saved to: {final_model_path}")
    
    # Evaluate final performance
    print("\nEvaluating final model...")
    eval_rewards = []
    eval_lengths = []
    eval_successes = 0
    
    for i in range(10):
        obs = eval_env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # DQN predict can return just action or (action, state) depending on version
            prediction = model.predict(obs, deterministic=True)
            action = prediction[0] if isinstance(prediction, tuple) else prediction
            obs, reward, done,info = eval_env.step(action)
            episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
            episode_length += 1
            while isinstance(info, tuple) and len(info) == 1:
                info = info[0]            
            # Check success (vectorized env returns list of infos)
            info_dict = info[0] if isinstance(info, list) else info
            # FIXED: Use the actual 'success' flag from the environment
            if done and info_dict.get('success', False):
                eval_successes += 1
        
        eval_rewards.append(episode_reward)
        eval_lengths.append(episode_length)
    
    # Compute statistics
    results = {
        "config_name": config['name'],
        "hyperparameters": hp,
        "training_time_seconds": training_time,
        "total_timesteps": total_timesteps,
        "mean_eval_reward": float(np.mean(eval_rewards)),
        "std_eval_reward": float(np.std(eval_rewards)),
        "mean_episode_length": float(np.mean(eval_lengths)),
        "success_rate": eval_successes / 10.0,
        "eval_rewards": [float(r) for r in eval_rewards],
        "convergence_timestep": convergence_timestep,
        "convergence_time_seconds": convergence_time,
        "training_eval_rewards": [float(np.mean(r)) for r in eval_rewards_history] if eval_rewards_history else [],
    }
    
    # Save results
    results_path = f"{model_dir}/results.json"
    with open(results_path, 'w') as f:
        json.dump(results, indent=2, fp=f)
    
    print(f"\n{'='*70}")
    print("Training Results:")
    print(f"  Mean Reward: {results['mean_eval_reward']:.2f} ± {results['std_eval_reward']:.2f}")
    print(f"  Mean Episode Length: {results['mean_episode_length']:.1f}")
    print(f"  Success Rate: {results['success_rate']*100:.1f}%")
    print(f"  Training Time: {training_time:.1f} seconds")
    if convergence_timestep:
        print(f"  Convergence: Timestep {convergence_timestep:,} ({convergence_time:.1f}s)" if convergence_time else f"  Convergence: Timestep {convergence_timestep:,}")
    else:
        print(f"  Convergence: Not achieved (still improving)")
    print(f"{'='*70}\n")
    
    # Cleanup
    env.close()
    eval_env.close()
    
    return results


def train_all_configurations(timesteps_per_config=100000):
    """Train all hyperparameter configurations"""
    
    print("\n" + "="*70)
    print("DQN HYPERPARAMETER TUNING - CROWD CONTROL ENVIRONMENT")
    print("="*70)
    print(f"Total configurations: {len(HYPERPARAMETER_CONFIGS)}")
    print(f"Timesteps per configuration: {timesteps_per_config}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("="*70)
    
    all_results = []
    
    for i, config in enumerate(HYPERPARAMETER_CONFIGS):
        print(f"\n\nConfiguration {i+1}/{len(HYPERPARAMETER_CONFIGS)}")
        
        try:
            results = train_dqn_configuration(config, total_timesteps=timesteps_per_config)
            all_results.append(results)
        except Exception as e:
            print(f"ERROR training {config['name']}: {e}")
            continue
    
    # Save combined results
    combined_results_path = "models/dqn/all_results.json"
    with open(combined_results_path, 'w') as f:
        json.dump(all_results, indent=2, fp=f)
    
    # Print summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY - ALL CONFIGURATIONS")
    print("="*70)
    
    # Sort by mean reward
    sorted_results = sorted(all_results, key=lambda x: x['mean_eval_reward'], reverse=True)
    
    print("\nRanking by Mean Reward:")
    for i, result in enumerate(sorted_results):
        print(f"{i+1}. {result['config_name']}: "
              f"{result['mean_eval_reward']:.2f} ± {result['std_eval_reward']:.2f} "
              f"(Success: {result['success_rate']*100:.1f}%)")
    
    # Plot comparison
    plot_results_comparison(sorted_results)
    
    print("\n" + "="*70)
    print(f"Best configuration: {sorted_results[0]['config_name']}")
    print(f"Best mean reward: {sorted_results[0]['mean_eval_reward']:.2f}")
    print("="*70)
    
    return sorted_results


def plot_results_comparison(results):
    """Plot comparison of different configurations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('DQN Hyperparameter Tuning Results', fontsize=16)
    
    configs = [r['config_name'] for r in results]
    mean_rewards = [r['mean_eval_reward'] for r in results]
    std_rewards = [r['std_eval_reward'] for r in results]
    success_rates = [r['success_rate'] * 100 for r in results]
    episode_lengths = [r['mean_episode_length'] for r in results]
    
    # Plot 1: Mean Rewards with error bars
    axes[0, 0].barh(range(len(configs)), mean_rewards, xerr=std_rewards, 
                     color='skyblue', alpha=0.7)
    axes[0, 0].set_yticks(range(len(configs)))
    axes[0, 0].set_yticklabels([c.replace('config_', '') for c in configs], fontsize=8)
    axes[0, 0].set_xlabel('Mean Reward')
    axes[0, 0].set_title('Mean Evaluation Reward')
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # Plot 2: Success Rates
    axes[0, 1].barh(range(len(configs)), success_rates, color='lightgreen', alpha=0.7)
    axes[0, 1].set_yticks(range(len(configs)))
    axes[0, 1].set_yticklabels([c.replace('config_', '') for c in configs], fontsize=8)
    axes[0, 1].set_xlabel('Success Rate (%)')
    axes[0, 1].set_title('Episode Success Rate')
    axes[0, 1].grid(axis='x', alpha=0.3)
    
    # Plot 3: Episode Lengths
    axes[1, 0].barh(range(len(configs)), episode_lengths, color='salmon', alpha=0.7)
    axes[1, 0].set_yticks(range(len(configs)))
    axes[1, 0].set_yticklabels([c.replace('config_', '') for c in configs], fontsize=8)
    axes[1, 0].set_xlabel('Mean Episode Length')
    axes[1, 0].set_title('Average Episode Duration')
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # Plot 4: Scatter - Reward vs Success Rate
    axes[1, 1].scatter(mean_rewards, success_rates, s=100, alpha=0.6, c=range(len(configs)), 
                       cmap='viridis')
    for i, config in enumerate(configs):
        axes[1, 1].annotate(config.replace('config_', ''), 
                            (mean_rewards[i], success_rates[i]),
                            fontsize=7, alpha=0.7)
    axes[1, 1].set_xlabel('Mean Reward')
    axes[1, 1].set_ylabel('Success Rate (%)')
    axes[1, 1].set_title('Reward vs Success Rate')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/dqn/hyperparameter_comparison.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved to: models/dqn/hyperparameter_comparison.png")
    plt.close()


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DQN agent for crowd control')
    parser.add_argument('--config', type=int, default=None, 
                       help='Train specific configuration (0-11), or all if not specified')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Total timesteps for training')
    parser.add_argument('--n-envs', type=int, default=1,
                       help='Number of parallel environments (1-2 recommended for DQN)')
    
    args = parser.parse_args()
    
    if args.config is not None:
        # Train specific configuration
        if 0 <= args.config < len(HYPERPARAMETER_CONFIGS):
            config = HYPERPARAMETER_CONFIGS[args.config]
            train_dqn_configuration(config, total_timesteps=args.timesteps, n_envs=args.n_envs)
        else:
            print(f"Error: Configuration index must be between 0 and {len(HYPERPARAMETER_CONFIGS)-1}")
    else:
        # Train all configurations
        print(f"💡 Note: DQN typically doesn't benefit much from vectorization (using n_envs={args.n_envs})")
        train_all_configurations(timesteps_per_config=args.timesteps)


if __name__ == "__main__":
    main()

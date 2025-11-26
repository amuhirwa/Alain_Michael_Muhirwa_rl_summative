"""
PPO Training Script for Crowd Control Environment
=================================================

Proximal Policy Optimization (Policy Gradient Method)
Hyperparameter tuning with 10+ configurations

Key Hyperparameters:
- Learning rate
- Number of steps
- Batch size
- Number of epochs
- GAE lambda
- Clip range
- Value function coefficient
- Entropy coefficient
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.enhanced_env_fast import EnhancedCrowdControlEnvFast
from stable_baselines3 import PPO
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
        convergence_step: Episode/timestep where convergence occurred, or None
    """
    if len(eval_rewards) < window:
        return None
    
    best_reward = max(eval_rewards)
    convergence_value = threshold * best_reward
    
    # Check rolling mean
    for i in range(len(eval_rewards) - window + 1):
        rolling_mean = np.mean(eval_rewards[i:i+window])
        if rolling_mean >= convergence_value:
            return i  # Return index (multiply by eval_freq for episode number)
    
    return None  # Not converged


# Hyperparameter configurations for extensive tuning
HYPERPARAMETER_CONFIGS = [
    {
        "name": "config_1_baseline",
        "learning_rate": 3e-4,  # Standard PPO LR - conservative baseline for comparison
        "n_steps": 2048,        # Standard rollout length - balances bias/variance
        "batch_size": 64,       # Standard mini-batch - computationally efficient
        "n_epochs": 10,         # Standard update epochs - balance sample reuse vs overfitting
        "gamma": 0.99,          # Standard discount - suitable for episodic tasks
        "gae_lambda": 0.95,     # Standard GAE - balance bias/variance in advantage estimation
        "clip_range": 0.2,      # Standard trust region - prevents destructive policy updates
        "vf_coef": 0.5,         # Standard value weight - equal importance to policy
        "ent_coef": 0.01,       # Low entropy - minimal exploration after initial learning
    },
    {
        "name": "config_2_high_lr",
        "learning_rate": 1e-3,  # 3x higher - tests if dense rewards enable aggressive learning
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "vf_coef": 0.5,
        "ent_coef": 0.01,
    },
    {
        "name": "config_3_more_steps",
        "learning_rate": 3e-4,
        "n_steps": 4096,
        "batch_size": 128,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "vf_coef": 0.5,
        "ent_coef": 0.01,
    },
    {
        "name": "config_4_more_epochs",
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 20,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "vf_coef": 0.5,
        "ent_coef": 0.01,
    },
    {
        "name": "config_5_high_gae",
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.98,
        "clip_range": 0.2,
        "vf_coef": 0.5,
        "ent_coef": 0.01,
    },
    {
        "name": "config_6_large_clip",
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.3,
        "vf_coef": 0.5,
        "ent_coef": 0.01,
    },
    {
        "name": "config_7_high_entropy",
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "vf_coef": 0.5,
        "ent_coef": 0.15,  # INCREASED from 0.05 -> 0.15
    },
    {
        "name": "config_8_very_high_entropy",
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "vf_coef": 0.5,
        "ent_coef": 0.25,  # Very high exploration for strategic gates
    },
    {
        "name": "config_9_aggressive",  # WINNER: 1157.7 mean reward
        "learning_rate": 1e-3,    # High LR - exploit dense reward gradients
        "n_steps": 1024,          # Shorter rollout - faster policy updates in dynamic environment
        "batch_size": 32,         # Smaller batch - more frequent gradient updates
        "n_epochs": 5,            # Fewer epochs - prevent overfitting to old experience
        "gamma": 0.98,            # Lower discount - prioritize immediate crowd control over distant outcomes
        "gae_lambda": 0.9,        # Lower GAE - reduce variance in advantage estimates
        "clip_range": 0.3,        # Larger trust region - enables bold policy improvements when signal is clear
        "vf_coef": 0.4,           # Lower value weight - prioritize policy learning over value accuracy
        "ent_coef": 0.2,          # High entropy - strong exploration discovers gate timing strategies
    },
    {
        "name": "config_10_optimized",
        "learning_rate": 3e-4,    # Conservative LR - fine-tuned balance
        "n_steps": 2560,          # Longer rollout - captures full episode dynamics
        "batch_size": 80,         # Moderate batch - balances efficiency and gradient quality
        "n_epochs": 12,           # More epochs - thorough learning from collected experience
        "gamma": 0.99,            # Standard discount - values long-term crowd dispersal
        "gae_lambda": 0.96,       # High GAE - low-bias advantage for policy gradient
        "clip_range": 0.22,       # Slightly larger clip - gentle policy improvements
        "vf_coef": 0.55,          # Higher value weight - accurate baseline reduces variance
        "ent_coef": 0.18,         # High entropy - encourages diverse action exploration
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
    # Use SubprocVecEnv for true parallelism (faster but more CPU)
    # Use DummyVecEnv for sequential execution (safer, less resource intensive)
    try:
        vec_env = SubprocVecEnv(env_fns)
    except:
        print("⚠️  SubprocVecEnv failed, falling back to DummyVecEnv")
        vec_env = DummyVecEnv(env_fns)
    return vec_env


def train_ppo_configuration(config, total_timesteps=200000, eval_freq=10000, n_envs=8):
    """Train PPO with specific hyperparameter configuration"""
    
    print(f"\n{'='*70}")
    print(f"Training PPO: {config['name']}")
    print(f"{'='*70}")
    print("Hyperparameters:")
    for key, value in config.items():
        if key != 'name':
            print(f"  {key}: {value}")
    print("="*70)
    
    # Create directories
    log_dir = f"logs/ppo/{config['name']}"
    model_dir = f"models/ppo/{config['name']}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Create VECTORIZED training environment (faster!)
    print(f"Creating {n_envs} parallel environments...")
    env = make_vec_env(difficulty='medium', pattern='rush', adversarial=False, n_envs=n_envs)
    
    # Create evaluation environment (wrapped in DummyVecEnv to match training env type)
    eval_env = make_vec_env(difficulty='medium', pattern='rush', adversarial=False, n_envs=1)
    
    # Extract hyperparameters
    hp = {k: v for k, v in config.items() if k != 'name'}
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=hp['learning_rate'],
        n_steps=hp['n_steps'],
        batch_size=hp['batch_size'],
        n_epochs=hp['n_epochs'],
        gamma=hp['gamma'],
        gae_lambda=hp['gae_lambda'],
        clip_range=hp['clip_range'],
        vf_coef=hp['vf_coef'],
        ent_coef=hp['ent_coef'],
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
        save_freq=20000,
        save_path=model_dir,
        name_prefix="ppo_checkpoint"
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
    
    # Extract evaluation rewards from callback for convergence analysis
    try:
        eval_rewards_history = eval_callback.evaluations_results
        convergence_idx = detect_convergence(eval_rewards_history, window=3, threshold=0.9)
        if convergence_idx is not None:
            convergence_episode = convergence_idx * (eval_freq // 2048)  # Approximate episodes
            convergence_time = convergence_idx / len(eval_rewards_history) * training_time if len(eval_rewards_history) > 0 else None
        else:
            convergence_episode = None
            convergence_time = None
    except:
        # Fallback if callback data not available
        convergence_episode = None
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
            action, _ = model.predict(obs, deterministic=True)
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
        "convergence_episode": convergence_episode,
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
    if convergence_episode:
        print(f"  Convergence: Episode {convergence_episode} ({convergence_time:.1f}s)" if convergence_time else f"  Convergence: Episode {convergence_episode}")
    else:
        print(f"  Convergence: Not achieved (still improving)")
    print(f"{'='*70}\n")
    
    # Cleanup
    env.close()
    eval_env.close()
    
    return results


def train_all_configurations(timesteps_per_config=200000):
    """Train all hyperparameter configurations"""
    
    print("\n" + "="*70)
    print("PPO HYPERPARAMETER TUNING - CROWD CONTROL ENVIRONMENT")
    print("="*70)
    print(f"Total configurations: {len(HYPERPARAMETER_CONFIGS)}")
    print(f"Timesteps per configuration: {timesteps_per_config}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("="*70)
    
    all_results = []
    
    for i, config in enumerate(HYPERPARAMETER_CONFIGS):
        print(f"\n\nConfiguration {i+1}/{len(HYPERPARAMETER_CONFIGS)}")
        
        try:
            results = train_ppo_configuration(config, total_timesteps=timesteps_per_config)
            all_results.append(results)
        except Exception as e:
            print(f"ERROR training {config['name']}: {e}")
            continue
    
    # Save combined results
    combined_results_path = "models/ppo/all_results.json"
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
    fig.suptitle('PPO Hyperparameter Tuning Results', fontsize=16)
    
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
    plt.savefig('models/ppo/hyperparameter_comparison.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved to: models/ppo/hyperparameter_comparison.png")
    plt.close()


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PPO agent for crowd control')
    parser.add_argument('--config', type=int, default=None, 
                       help='Train specific configuration (0-11), or all if not specified')
    parser.add_argument('--timesteps', type=int, default=150000,
                       help='Total timesteps for training')
    parser.add_argument('--n-envs', type=int, default=4,
                       help='Number of parallel environments (2-8 recommended)')
    
    args = parser.parse_args()
    
    if args.config is not None:
        # Train specific configuration
        if 0 <= args.config < len(HYPERPARAMETER_CONFIGS):
            config = HYPERPARAMETER_CONFIGS[args.config]
            train_ppo_configuration(config, total_timesteps=args.timesteps, n_envs=args.n_envs)
        else:
            print(f"Error: Configuration index must be between 0 and {len(HYPERPARAMETER_CONFIGS)-1}")
    else:
        # Train all configurations
        print(f"⚠️  Training all configs with {args.n_envs} parallel environments per config")
        print(f"    This will use significant CPU resources!")
        train_all_configurations(timesteps_per_config=args.timesteps)


if __name__ == "__main__":
    main()

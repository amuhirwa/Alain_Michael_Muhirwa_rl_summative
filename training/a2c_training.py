"""
A2C Training Script for Crowd Control Environment
=================================================

Advantage Actor-Critic (Policy Gradient Method)
Hyperparameter tuning with 10+ configurations

Key Hyperparameters:
- Learning rate
- Number of steps
- Gamma (discount factor)
- GAE lambda
- Value function coefficient
- Entropy coefficient
- RMS prop epsilon
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.enhanced_env_fast import EnhancedCrowdControlEnvFast
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
import torch


def detect_convergence(eval_rewards, window=3, threshold=0.9):
    """Detect convergence point during training."""
    if len(eval_rewards) < window:
        return None
    best_reward = max(eval_rewards)
    convergence_value = threshold * best_reward
    for i in range(len(eval_rewards) - window + 1):
        rolling_mean = np.mean(eval_rewards[i:i+window])
        if rolling_mean >= convergence_value:
            return i
    return None


# Hyperparameter configurations for extensive tuning
HYPERPARAMETER_CONFIGS = [
    {
        "name": "config_1_baseline",
        "learning_rate": 7e-4,    # Moderate LR - baseline for A2C
        "n_steps": 5,             # Very short rollout - immediate feedback in dense-reward environment
        "gamma": 0.99,            # Standard discount for episodic tasks
        "gae_lambda": 1.0,        # No TD bias - pure Monte Carlo advantage estimation
        "vf_coef": 0.5,           # Balanced value/policy weight
        "ent_coef": 0.3,          # High entropy - encourages exploration
        "rms_prop_eps": 1e-5,     # RMSProp stability parameter
    },
    {
        "name": "config_2_high_lr",  # CHAMPION: 1289.6 mean reward - BEST OVERALL!
        "learning_rate": 1e-3,    # High LR - A2C's synchronous updates handle aggressive gradients
        "n_steps": 5,             # Minimal rollout - rapid policy updates exploit dense reward signal
        "gamma": 0.99,            # Standard discount - sufficient for typical episode durations
        "gae_lambda": 1.0,        # Full Monte Carlo - low variance with short rollouts
        "vf_coef": 0.5,           # Balanced value weight - prevents value function dominance
        "ent_coef": 0.01,         # Minimal entropy - exploitation-focused after discovering strategies
        "rms_prop_eps": 1e-5,     # RMSProp epsilon - prevents gradient scaling instability
    },
    {
        "name": "config_3_more_steps",
        "learning_rate": 7e-4,
        "n_steps": 10,
        "gamma": 0.99,
        "gae_lambda": 1.0,
        "vf_coef": 0.5,
        "ent_coef": 0.01,
        "rms_prop_eps": 1e-5,
    },
    {
        "name": "config_4_high_gae",
        "learning_rate": 7e-4,
        "n_steps": 5,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "vf_coef": 0.5,
        "ent_coef": 0.01,
        "rms_prop_eps": 1e-5,
    },
    {
        "name": "config_5_high_vf",
        "learning_rate": 7e-4,
        "n_steps": 5,
        "gamma": 0.99,
        "gae_lambda": 1.0,
        "vf_coef": 0.8,
        "ent_coef": 0.01,
        "rms_prop_eps": 1e-5,
    },
    {
        "name": "config_6_high_entropy",
        "learning_rate": 7e-4,
        "n_steps": 5,
        "gamma": 0.99,
        "gae_lambda": 1.0,
        "vf_coef": 0.5,
        "ent_coef": 0.05,
        "rms_prop_eps": 1e-5,
    },
    {
        "name": "config_7_balanced",
        "learning_rate": 5e-4,
        "n_steps": 8,
        "gamma": 0.995,
        "gae_lambda": 0.98,
        "vf_coef": 0.6,
        "ent_coef": 0.02,
        "rms_prop_eps": 1e-5,
    },
    {
        "name": "config_8_aggressive",
        "learning_rate": 2e-3,
        "n_steps": 3,
        "gamma": 0.98,
        "gae_lambda": 0.9,
        "vf_coef": 0.4,
        "ent_coef": 0.03,
        "rms_prop_eps": 1e-6,
    },
    {
        "name": "config_9_exploration",
        "learning_rate": 7e-4,
        "n_steps": 5,
        "gamma": 0.99,
        "gae_lambda": 1.0,
        "vf_coef": 0.5,
        "ent_coef": 0.2,  # INCREASED from 0.1 -> 0.2 for better exploration
        "rms_prop_eps": 1e-5,
    },
    {
        "name": "config_10_optimized",
        "learning_rate": 5e-4,
        "n_steps": 15,  # More steps for better credit assignment
        "gamma": 0.99,
        "gae_lambda": 0.97,
        "vf_coef": 0.55,
        "ent_coef": 0.25,  # INCREASED from 0.015 -> 0.25 for strategic actions
        "rms_prop_eps": 1e-5,
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


def train_a2c_configuration(config, total_timesteps=150000, eval_freq=10000, n_envs=8):
    """Train A2C with specific hyperparameter configuration"""
    
    print(f"\n{'='*70}")
    print(f"Training A2C: {config['name']}")
    print(f"{'='*70}")
    print("Hyperparameters:")
    for key, value in config.items():
        if key != 'name':
            print(f"  {key}: {value}")
    print("="*70)
    
    # Create directories
    log_dir = f"logs/a2c/{config['name']}"
    model_dir = f"models/a2c/{config['name']}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # A2C benefits significantly from vectorized environments
    print(f"Creating {n_envs} parallel environments...")
    env = make_vec_env(difficulty='medium', pattern='rush', adversarial=False, n_envs=n_envs)
    
    # Create evaluation environment (same type as training)
    eval_env = make_vec_env(difficulty='medium', pattern='rush', adversarial=False, n_envs=1)
    
    # Extract hyperparameters
    hp = {k: v for k, v in config.items() if k != 'name'}
    
    # Create A2C model
    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=hp['learning_rate'],
        n_steps=hp['n_steps'],
        gamma=hp['gamma'],
        gae_lambda=hp['gae_lambda'],
        vf_coef=hp['vf_coef'],
        ent_coef=hp['ent_coef'],
        rms_prop_eps=hp['rms_prop_eps'],
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
        save_freq=15000,
        save_path=model_dir,
        name_prefix="a2c_checkpoint"
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
    
    # Extract convergence metrics
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
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done,info = eval_env.step(action)
            episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
            episode_length += 1
            while isinstance(info, tuple) and len(info) == 1:
                info = info[0]
            
            # Check success (vectorized env returns list of infos)
            info_dict = info[0] if isinstance(info, list) else info
            if done and info_dict.get('agents', info_dict.get('total_agents', 100)) < 15:
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


def train_all_configurations(timesteps_per_config=150000):
    """Train all hyperparameter configurations"""
    
    print("\n" + "="*70)
    print("A2C HYPERPARAMETER TUNING - CROWD CONTROL ENVIRONMENT")
    print("="*70)
    print(f"Total configurations: {len(HYPERPARAMETER_CONFIGS)}")
    print(f"Timesteps per configuration: {timesteps_per_config}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("="*70)
    
    all_results = []
    
    for i, config in enumerate(HYPERPARAMETER_CONFIGS):
        print(f"\n\nConfiguration {i+1}/{len(HYPERPARAMETER_CONFIGS)}")
        
        try:
            results = train_a2c_configuration(config, total_timesteps=timesteps_per_config)
            all_results.append(results)
        except Exception as e:
            print(f"ERROR training {config['name']}: {e}")
            continue
    
    # Save combined results
    combined_results_path = "models/a2c/all_results.json"
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
    fig.suptitle('A2C Hyperparameter Tuning Results', fontsize=16)
    
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
    plt.savefig('models/a2c/hyperparameter_comparison.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved to: models/a2c/hyperparameter_comparison.png")
    plt.close()


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train A2C agent for crowd control')
    parser.add_argument('--config', type=int, default=None, 
                       help='Train specific configuration (0-11), or all if not specified')
    parser.add_argument('--timesteps', type=int, default=150000,
                       help='Total timesteps for training')
    parser.add_argument('--n-envs', type=int, default=4,
                       help='Number of parallel environments (4-8 recommended for A2C)')
    
    args = parser.parse_args()
    
    if args.config is not None:
        # Train specific configuration
        if 0 <= args.config < len(HYPERPARAMETER_CONFIGS):
            config = HYPERPARAMETER_CONFIGS[args.config]
            train_a2c_configuration(config, total_timesteps=args.timesteps, n_envs=args.n_envs)
        else:
            print(f"Error: Configuration index must be between 0 and {len(HYPERPARAMETER_CONFIGS)-1}")
    else:
        # Train all configurations
        print(f"⚡ A2C benefits greatly from vectorization (using {args.n_envs} parallel environments)")
        train_all_configurations(timesteps_per_config=args.timesteps)


if __name__ == "__main__":
    main()

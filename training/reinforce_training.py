"""
REINFORCE Training Script for Crowd Control Environment
=======================================================

REINFORCE Algorithm (Monte Carlo Policy Gradient)
Custom implementation with hyperparameter tuning

Key Hyperparameters:
- Learning rate
- Gamma (discount factor)
- Entropy coefficient
- Baseline (with/without)
- Network architecture
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.enhanced_env_fast import EnhancedCrowdControlEnvFast
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from collections import deque


class PolicyNetwork(nn.Module):
    """Neural network for policy"""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[128, 128]):
        super(PolicyNetwork, self).__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        layers.append(nn.Softmax(dim=-1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        return self.network(state)


class ValueNetwork(nn.Module):
    """Neural network for baseline (value function)"""
    
    def __init__(self, state_dim, hidden_dims=[128, 128]):
        super(ValueNetwork, self).__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        return self.network(state)


class REINFORCEAgent:
    """REINFORCE agent with optional baseline"""
    
    def __init__(self, state_dim, action_dim, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Hyperparameters
        self.gamma = config['gamma']
        self.lr = config['learning_rate']
        self.ent_coef = config.get('ent_coef', 0.01)
        self.use_baseline = config.get('use_baseline', True)
        
        # Networks
        self.policy_net = PolicyNetwork(
            state_dim, 
            action_dim,
            hidden_dims=config.get('hidden_dims', [128, 128])
        ).to(self.device)
        
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
        if self.use_baseline:
            self.value_net = ValueNetwork(
                state_dim,
                hidden_dims=config.get('hidden_dims', [128, 128])
            ).to(self.device)
            self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.lr)
        
        # Episode storage
        self.reset_episode()
    
    def reset_episode(self):
        """Reset episode storage"""
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.entropies = []
    
    def select_action(self, state):
        """Select action from policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            probs = self.policy_net(state)
        
        dist = Categorical(probs)
        action = dist.sample()
        
        return action.item()
    
    def store_transition(self, state, action, reward):
        """Store transition for training"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get action probability
        probs = self.policy_net(state)
        dist = Categorical(probs)
        log_prob = dist.log_prob(torch.tensor(action).to(self.device))
        entropy = dist.entropy()
        
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.entropies.append(entropy)
        
        if self.use_baseline:
            value = self.value_net(state)
            self.values.append(value)
    
    def train(self):
        """Train on collected episode"""
        # Compute returns
        returns = []
        G = 0
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns).to(self.device)
        
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Compute policy loss
        policy_loss = []
        
        for i in range(len(self.log_probs)):
            if self.use_baseline:
                advantage = returns[i] - self.values[i].squeeze()
            else:
                advantage = returns[i]
            
            # Policy gradient with entropy bonus
            policy_loss.append(-self.log_probs[i] * advantage.detach() - 
                             self.ent_coef * self.entropies[i])
        
        policy_loss = torch.stack(policy_loss).sum()
        
        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.policy_optimizer.step()
        
        # Update value network if using baseline
        if self.use_baseline:
            value_loss = 0.5 * sum((returns[i] - self.values[i].squeeze())**2 
                                    for i in range(len(self.values)))
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.value_optimizer.step()
        
        # Reset episode
        self.reset_episode()
    
    def save(self, path):
        """Save model"""
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_state_dict': self.value_net.state_dict() if self.use_baseline else None,
            'value_optimizer_state_dict': self.value_optimizer.state_dict() if self.use_baseline else None,
        }, path)
    
    def load(self, path):
        """Load model"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        
        if self.use_baseline and checkpoint['value_state_dict'] is not None:
            self.value_net.load_state_dict(checkpoint['value_state_dict'])
            self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])


# Hyperparameter configurations
HYPERPARAMETER_CONFIGS = [
    {
        "name": "config_1_baseline",
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "ent_coef": 0.01,
        "use_baseline": True,
        "hidden_dims": [128, 128],
    },
    {
        "name": "config_2_no_baseline",
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "ent_coef": 0.01,
        "use_baseline": False,
        "hidden_dims": [128, 128],
    },
    {
        "name": "config_3_high_lr",
        "learning_rate": 5e-3,
        "gamma": 0.99,
        "ent_coef": 0.01,
        "use_baseline": True,
        "hidden_dims": [128, 128],
    },
    {
        "name": "config_4_low_lr",
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "ent_coef": 0.01,
        "use_baseline": True,
        "hidden_dims": [128, 128],
    },
    {
        "name": "config_5_high_gamma",
        "learning_rate": 1e-3,
        "gamma": 0.995,
        "ent_coef": 0.01,
        "use_baseline": True,
        "hidden_dims": [128, 128],
    },
    {
        "name": "config_6_high_entropy",
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "ent_coef": 0.05,
        "use_baseline": True,
        "hidden_dims": [128, 128],
    },
    {
        "name": "config_7_large_network",
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "ent_coef": 0.01,
        "use_baseline": True,
        "hidden_dims": [256, 256],
    },
    {
        "name": "config_8_deep_network",
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "ent_coef": 0.01,
        "use_baseline": True,
        "hidden_dims": [128, 128, 128],
    },
    {
        "name": "config_9_aggressive",
        "learning_rate": 5e-3,
        "gamma": 0.98,
        "ent_coef": 0.03,
        "use_baseline": False,
        "hidden_dims": [128, 64],
    },
    {
        "name": "config_10_optimized",
        "learning_rate": 2e-3,
        "gamma": 0.99,
        "ent_coef": 0.015,
        "use_baseline": True,
        "hidden_dims": [256, 128],
    },
]


def train_reinforce_configuration(config, num_episodes=1000, eval_freq=100):
    """Train REINFORCE with specific configuration"""
    
    print(f"\n{'='*70}")
    print(f"Training REINFORCE: {config['name']}")
    print(f"{'='*70}")
    print("Hyperparameters:")
    for key, value in config.items():
        if key != 'name':
            print(f"  {key}: {value}")
    print("="*70)
    
    # Create directories
    log_dir = f"logs/reinforce/{config['name']}"
    model_dir = f"models/reinforce/{config['name']}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Create environment with optimized parameters
    env = EnhancedCrowdControlEnvFast(
        crowd_arrival_pattern='rush',
        adversarial_mode=False,
        difficulty='medium'
    )
    
    # Create agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = REINFORCEAgent(state_dim, action_dim, config)
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    eval_rewards = []
    best_eval_reward = -float('inf')
    
    print(f"\nStarting training for {num_episodes} episodes...")
    start_time = datetime.now()
    
    # Training loop
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        # Collect episode
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            agent.store_transition(state, action, reward)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            done = terminated or truncated
        
        # Train on episode
        agent.train()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Periodic evaluation
        if (episode + 1) % eval_freq == 0:
            eval_reward = evaluate_agent(agent, env, n_episodes=5)
            eval_rewards.append(eval_reward)
            
            avg_reward = np.mean(episode_rewards[-eval_freq:])
            print(f"Episode {episode+1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Eval Reward: {eval_reward:.2f}")
            
            # Save best model
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                agent.save(f"{model_dir}/best_model.pth")
    
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Save final model
    agent.save(f"{model_dir}/final_model.pth")
    
    # Final evaluation
    print("\nEvaluating final model...")
    final_eval_rewards = []
    final_eval_lengths = []
    final_successes = 0
    
    for i in range(10):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
            
            if terminated and info['total_crowd'] < 10:
                final_successes += 1
        
        final_eval_rewards.append(episode_reward)
        final_eval_lengths.append(episode_length)
    
    # Compute statistics
    results = {
        "config_name": config['name'],
        "hyperparameters": config,
        "training_time_seconds": training_time,
        "num_episodes": num_episodes,
        "mean_eval_reward": float(np.mean(final_eval_rewards)),
        "std_eval_reward": float(np.std(final_eval_rewards)),
        "mean_episode_length": float(np.mean(final_eval_lengths)),
        "success_rate": final_successes / 10.0,
        "best_eval_reward": float(best_eval_reward),
        "training_rewards": [float(r) for r in episode_rewards],
        "eval_rewards": [float(r) for r in eval_rewards],
    }
    
    # Save results
    with open(f"{model_dir}/results.json", 'w') as f:
        json.dump(results, indent=2, fp=f)
    
    # Plot training curve
    plot_training_curve(episode_rewards, eval_rewards, eval_freq, model_dir)
    
    print(f"\n{'='*70}")
    print("Training Results:")
    print(f"  Mean Reward: {results['mean_eval_reward']:.2f} ± {results['std_eval_reward']:.2f}")
    print(f"  Best Eval Reward: {results['best_eval_reward']:.2f}")
    print(f"  Success Rate: {results['success_rate']*100:.1f}%")
    print(f"  Training Time: {training_time:.1f} seconds")
    print(f"{'='*70}\n")
    
    env.close()
    
    return results


def evaluate_agent(agent, env, n_episodes=5):
    """Evaluate agent performance"""
    eval_rewards = []
    
    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        eval_rewards.append(episode_reward)
    
    return np.mean(eval_rewards)


def plot_training_curve(episode_rewards, eval_rewards, eval_freq, save_dir):
    """Plot training curve"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot episode rewards
    window = 50
    smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
    ax1.plot(smoothed, alpha=0.8)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards (Smoothed)')
    ax1.grid(alpha=0.3)
    
    # Plot eval rewards
    eval_episodes = [i * eval_freq for i in range(len(eval_rewards))]
    ax2.plot(eval_episodes, eval_rewards, marker='o')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Eval Reward')
    ax2.set_title('Evaluation Rewards')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_curve.png", dpi=300)
    plt.close()


def train_all_configurations(episodes_per_config=1000):
    """Train all configurations"""
    
    print("\n" + "="*70)
    print("REINFORCE HYPERPARAMETER TUNING - CROWD CONTROL ENVIRONMENT")
    print("="*70)
    print(f"Total configurations: {len(HYPERPARAMETER_CONFIGS)}")
    print(f"Episodes per configuration: {episodes_per_config}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("="*70)
    
    all_results = []
    
    for i, config in enumerate(HYPERPARAMETER_CONFIGS):
        print(f"\n\nConfiguration {i+1}/{len(HYPERPARAMETER_CONFIGS)}")
        
        try:
            results = train_reinforce_configuration(config, num_episodes=episodes_per_config)
            all_results.append(results)
        except Exception as e:
            print(f"ERROR training {config['name']}: {e}")
            continue
    
    # Save combined results
    with open("models/reinforce/all_results.json", 'w') as f:
        json.dump(all_results, indent=2, fp=f)
    
    # Print summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY - ALL CONFIGURATIONS")
    print("="*70)
    
    sorted_results = sorted(all_results, key=lambda x: x['mean_eval_reward'], reverse=True)
    
    print("\nRanking by Mean Reward:")
    for i, result in enumerate(sorted_results):
        print(f"{i+1}. {result['config_name']}: "
              f"{result['mean_eval_reward']:.2f} ± {result['std_eval_reward']:.2f} "
              f"(Success: {result['success_rate']*100:.1f}%)")
    
    # Plot comparison
    plot_comparison(sorted_results)
    
    print("\n" + "="*70)
    print(f"Best configuration: {sorted_results[0]['config_name']}")
    print(f"Best mean reward: {sorted_results[0]['mean_eval_reward']:.2f}")
    print("="*70)
    
    return sorted_results


def plot_comparison(results):
    """Plot comparison of configurations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('REINFORCE Hyperparameter Tuning Results', fontsize=16)
    
    configs = [r['config_name'] for r in results]
    mean_rewards = [r['mean_eval_reward'] for r in results]
    std_rewards = [r['std_eval_reward'] for r in results]
    success_rates = [r['success_rate'] * 100 for r in results]
    episode_lengths = [r['mean_episode_length'] for r in results]
    
    # Plots
    axes[0, 0].barh(range(len(configs)), mean_rewards, xerr=std_rewards, alpha=0.7)
    axes[0, 0].set_yticks(range(len(configs)))
    axes[0, 0].set_yticklabels([c.replace('config_', '') for c in configs], fontsize=8)
    axes[0, 0].set_xlabel('Mean Reward')
    axes[0, 0].set_title('Mean Evaluation Reward')
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    axes[0, 1].barh(range(len(configs)), success_rates, color='lightgreen', alpha=0.7)
    axes[0, 1].set_yticks(range(len(configs)))
    axes[0, 1].set_yticklabels([c.replace('config_', '') for c in configs], fontsize=8)
    axes[0, 1].set_xlabel('Success Rate (%)')
    axes[0, 1].set_title('Episode Success Rate')
    axes[0, 1].grid(axis='x', alpha=0.3)
    
    axes[1, 0].barh(range(len(configs)), episode_lengths, color='salmon', alpha=0.7)
    axes[1, 0].set_yticks(range(len(configs)))
    axes[1, 0].set_yticklabels([c.replace('config_', '') for c in configs], fontsize=8)
    axes[1, 0].set_xlabel('Mean Episode Length')
    axes[1, 0].set_title('Average Episode Duration')
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    axes[1, 1].scatter(mean_rewards, success_rates, s=100, alpha=0.6, c=range(len(configs)), cmap='viridis')
    for i, config in enumerate(configs):
        axes[1, 1].annotate(config.replace('config_', ''), (mean_rewards[i], success_rates[i]),
                            fontsize=7, alpha=0.7)
    axes[1, 1].set_xlabel('Mean Reward')
    axes[1, 1].set_ylabel('Success Rate (%)')
    axes[1, 1].set_title('Reward vs Success Rate')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/reinforce/hyperparameter_comparison.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved to: models/reinforce/hyperparameter_comparison.png")
    plt.close()


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train REINFORCE agent for crowd control')
    parser.add_argument('--config', type=int, default=None,
                       help='Train specific configuration (0-11), or all if not specified')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of episodes for training')
    
    args = parser.parse_args()
    
    if args.config is not None:
        if 0 <= args.config < len(HYPERPARAMETER_CONFIGS):
            config = HYPERPARAMETER_CONFIGS[args.config]
            train_reinforce_configuration(config, num_episodes=args.episodes)
        else:
            print(f"Error: Configuration index must be between 0 and {len(HYPERPARAMETER_CONFIGS)-1}")
    else:
        train_all_configurations(episodes_per_config=args.episodes)


if __name__ == "__main__":
    main()

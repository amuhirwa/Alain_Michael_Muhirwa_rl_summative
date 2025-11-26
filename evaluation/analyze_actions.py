"""
Action Distribution Analyzer
============================
Tests a trained model and shows which actions it uses and how often.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.enhanced_env_fast import EnhancedCrowdControlEnvFast
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import argparse

ACTION_NAMES = {
    0: "Move Barrier 0",
    1: "Move Barrier 1", 
    2: "Move Barrier 2",
    3: "Move Barrier 3",
    4: "Toggle Gate 0",
    5: "Toggle Gate 1",
    6: "Toggle Gate 2",
    7: "Flow Up",
    8: "Flow Down",
    9: "Flow Left",
    10: "Emergency (All Gates Open)",
    11: "No-Op"
}

def make_env():
    def _init():
        return EnhancedCrowdControlEnvFast(
            crowd_arrival_pattern="rush",
            difficulty="medium",
            adversarial_mode=False
        )
    return _init

def analyze_actions(model_path, n_episodes=10):
    """Run episodes and collect action statistics"""
    
    # Load model
    print(f"\nLoading model: {model_path}")
    env = DummyVecEnv([make_env()])
    model = A2C.load(model_path, env=env)
    
    # Collect actions
    all_actions = []
    episode_rewards = []
    episode_actions = []  # Actions per episode
    
    print(f"\nRunning {n_episodes} episodes...")
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        ep_actions = []
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            all_actions.append(int(action[0]))
            ep_actions.append(int(action[0]))
            episode_reward += reward[0]
        
        episode_rewards.append(episode_reward)
        episode_actions.append(ep_actions)
        print(f"  Episode {ep+1}: Reward = {episode_reward:.1f}, Actions = {len(ep_actions)}")
    
    env.close()
    
    # Analyze results
    print("\n" + "="*60)
    print("ACTION DISTRIBUTION ANALYSIS")
    print("="*60)
    
    action_counts = Counter(all_actions)
    total_actions = len(all_actions)
    
    print(f"\nTotal actions taken: {total_actions}")
    print(f"Average episode reward: {np.mean(episode_rewards):.1f} ± {np.std(episode_rewards):.1f}")
    print(f"Best episode: {max(episode_rewards):.1f}")
    print(f"Worst episode: {min(episode_rewards):.1f}")
    
    print("\n" + "-"*60)
    print("Action Usage:")
    print("-"*60)
    
    # Sort by frequency
    for action_id in sorted(action_counts.keys(), key=lambda x: action_counts[x], reverse=True):
        count = action_counts[action_id]
        percentage = 100 * count / total_actions
        action_name = ACTION_NAMES[action_id]
        bar = "█" * int(percentage / 2)  # Visual bar
        print(f"{action_id:2d} | {action_name:30s} | {count:5d} ({percentage:5.1f}%) {bar}")
    
    # Check diversity
    print("\n" + "-"*60)
    print("Diversity Metrics:")
    print("-"*60)
    
    unique_actions = len(action_counts)
    action_entropy = -sum((count/total_actions) * np.log(count/total_actions) 
                          for count in action_counts.values())
    
    print(f"Unique actions used: {unique_actions} / 12")
    print(f"Action entropy: {action_entropy:.3f} (max = {np.log(12):.3f})")
    
    if unique_actions < 5:
        print("\n⚠️  WARNING: Low action diversity! Agent is stuck in local optimum.")
    elif action_entropy < 1.5:
        print("\n⚠️  WARNING: Low entropy! Agent heavily favors few actions.")
    else:
        print("\n✓ Good action diversity!")
    
    # Plot distribution
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    actions = sorted(action_counts.keys())
    counts = [action_counts[a] for a in actions]
    colors = ['green' if action_counts[a] > total_actions * 0.05 else 'red' for a in actions]
    plt.bar(actions, counts, color=colors)
    plt.xlabel('Action ID')
    plt.ylabel('Frequency')
    plt.title('Action Distribution')
    plt.xticks(actions)
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(episode_rewards, bins=20, edgecolor='black')
    plt.xlabel('Episode Reward')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('action_distribution.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved: action_distribution.png")
    
    return action_counts, episode_rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=10, help="Number of test episodes")
    args = parser.parse_args()
    
    analyze_actions(args.model, args.episodes)

"""
Main Entry Point for Crowd Control RL Project
==============================================

Run trained models with full visualization and evaluation.

Usage:
    python main.py --algorithm dqn --model models/dqn/best_model
    python main.py --algorithm ppo --config config_1_baseline
    python main.py --compare  # Compare all algorithms
"""

import sys
import os
import argparse
from environment.custom_env import CrowdControlEnv
import time
import numpy as np
import json


def load_sb3_model(algorithm, model_path):
    """Load Stable-Baselines3 model"""
    if algorithm == 'dqn':
        from stable_baselines3 import DQN
        return DQN.load(model_path)
    elif algorithm == 'ppo':
        from stable_baselines3 import PPO
        return PPO.load(model_path)
    elif algorithm == 'a2c':
        from stable_baselines3 import A2C
        return A2C.load(model_path)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def load_reinforce_model(model_path, env):
    """Load REINFORCE model"""
    from training.reinforce_training import REINFORCEAgent
    import torch
    
    # Load config
    model_dir = os.path.dirname(model_path)
    with open(os.path.join(model_dir, 'results.json'), 'r') as f:
        results = json.load(f)
    
    config = results['hyperparameters']
    
    # Create agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = REINFORCEAgent(state_dim, action_dim, config)
    
    # Load weights
    agent.load(model_path)
    
    return agent


def run_trained_agent(algorithm, model_path, num_episodes=5, render=True):
    """Run trained agent and display performance"""
    
    print("\n" + "="*70)
    print(f"RUNNING TRAINED {algorithm.upper()} AGENT")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    print("="*70)
    
    # Create environment
    render_mode = 'human' if render else None
    env = CrowdControlEnv(render_mode=render_mode)
    
    # Load model
    if algorithm in ['dqn', 'ppo', 'a2c']:
        model = load_sb3_model(algorithm, model_path)
        use_sb3 = True
    elif algorithm == 'reinforce':
        model = load_reinforce_model(model_path, env)
        use_sb3 = False
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Run episodes
    all_rewards = []
    all_lengths = []
    all_successes = []
    all_max_densities = []
    
    for episode in range(num_episodes):
        print(f"\n{'='*70}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print("="*70)
        
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        print(f"Initial State:")
        print(f"  Total Crowd: {info['total_crowd']:.1f}")
        print(f"  Max Density: {info['max_density']:.2f}")
        print(f"  Open Gates: {info['open_gates']}/{env.NUM_GATES}")
        
        while not done:
            # Get action from model
            if use_sb3:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = model.select_action(obs)
            
            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
            
            # Render
            if render:
                env.render()
                time.sleep(0.05)  # Slow down for visualization
            
            # Print periodic updates
            if episode_length % 100 == 0:
                print(f"\nStep {episode_length}:")
                print(f"  Total Crowd: {info['total_crowd']:.1f}")
                print(f"  Max Density: {info['max_density']:.2f}")
                print(f"  Cumulative Reward: {episode_reward:.2f}")
        
        # Episode complete
        success = terminated and info['total_crowd'] < 10
        
        print(f"\nEpisode Complete:")
        print(f"  Length: {episode_length} steps")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Final Crowd: {info['total_crowd']:.1f}")
        print(f"  Max Density Reached: {info['max_density_reached']:.2f}")
        print(f"  Overcrowding Events: {info['overcrowding_events']}")
        print(f"  Result: {'SUCCESS' if success else 'FAILURE'}")
        
        all_rewards.append(episode_reward)
        all_lengths.append(episode_length)
        all_successes.append(1 if success else 0)
        all_max_densities.append(info['max_density_reached'])
    
    # Summary statistics
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)
    print(f"Mean Reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"Mean Episode Length: {np.mean(all_lengths):.1f} ± {np.std(all_lengths):.1f}")
    print(f"Success Rate: {np.mean(all_successes)*100:.1f}%")
    print(f"Mean Max Density: {np.mean(all_max_densities):.2f}")
    print("="*70)
    
    if render:
        input("\nPress Enter to exit...")
    
    env.close()
    
    return {
        'mean_reward': float(np.mean(all_rewards)),
        'std_reward': float(np.std(all_rewards)),
        'mean_length': float(np.mean(all_lengths)),
        'success_rate': float(np.mean(all_successes)),
        'mean_max_density': float(np.mean(all_max_densities)),
    }


def compare_algorithms():
    """Compare performance of all trained algorithms"""
    
    print("\n" + "="*70)
    print("COMPARING ALL ALGORITHMS")
    print("="*70)
    
    algorithms = {
        'DQN': 'models/dqn/config_1_baseline/best_model',
        'PPO': 'models/ppo/config_1_baseline/best_model',
        'A2C': 'models/a2c/config_1_baseline/best_model',
        'REINFORCE': 'models/reinforce/config_1_baseline/best_model.pth',
    }
    
    results = {}
    
    for name, model_path in algorithms.items():
        print(f"\nEvaluating {name}...")
        
        # Check if model exists
        if not os.path.exists(model_path + '.zip') and not os.path.exists(model_path):
            print(f"  Model not found: {model_path}")
            continue
        
        try:
            algo = name.lower()
            result = run_trained_agent(algo, model_path, num_episodes=10, render=False)
            results[name] = result
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Print comparison
    print("\n" + "="*70)
    print("ALGORITHM COMPARISON")
    print("="*70)
    print(f"{'Algorithm':<15} {'Mean Reward':<15} {'Success Rate':<15} {'Avg Length':<15}")
    print("-"*70)
    
    for name, result in sorted(results.items(), key=lambda x: x[1]['mean_reward'], reverse=True):
        print(f"{name:<15} {result['mean_reward']:>8.2f} ± {result['std_reward']:<4.2f}  "
              f"{result['success_rate']*100:>6.1f}%          "
              f"{result['mean_length']:>8.1f}")
    
    print("="*70)
    
    # Find best
    best_algo = max(results.items(), key=lambda x: x[1]['mean_reward'])
    print(f"\nBest Algorithm: {best_algo[0]}")
    print(f"Mean Reward: {best_algo[1]['mean_reward']:.2f}")
    print(f"Success Rate: {best_algo[1]['success_rate']*100:.1f}%")
    
    return results


def find_best_model(algorithm):
    """Find the best model for an algorithm"""
    results_path = f"models/{algorithm}/all_results.json"
    
    if not os.path.exists(results_path):
        print(f"No results found for {algorithm}")
        return None
    
    with open(results_path, 'r') as f:
        all_results = json.load(f)
    
    # Find best configuration
    best_config = max(all_results, key=lambda x: x['mean_eval_reward'])
    
    config_name = best_config['config_name']
    
    if algorithm == 'reinforce':
        model_path = f"models/{algorithm}/{config_name}/best_model.pth"
    else:
        model_path = f"models/{algorithm}/{config_name}/best_model"
    
    return model_path, best_config


def main():
    parser = argparse.ArgumentParser(description='Run trained crowd control RL agent')
    
    parser.add_argument('--algorithm', type=str, choices=['dqn', 'ppo', 'a2c', 'reinforce'],
                       help='Algorithm to run')
    parser.add_argument('--model', type=str,
                       help='Path to trained model')
    parser.add_argument('--config', type=str,
                       help='Configuration name (e.g., config_1_baseline)')
    parser.add_argument('--best', action='store_true',
                       help='Run the best performing model for the algorithm')
    parser.add_argument('--compare', action='store_true',
                       help='Compare all algorithms')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to run')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable rendering')
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare all algorithms
        compare_algorithms()
    
    elif args.algorithm:
        # Determine model path
        if args.model:
            model_path = args.model
        elif args.best:
            result = find_best_model(args.algorithm)
            if result is None:
                print(f"Could not find best model for {args.algorithm}")
                return
            model_path, best_config = result
            print(f"\nUsing best configuration: {best_config['config_name']}")
            print(f"Training reward: {best_config['mean_eval_reward']:.2f}")
        elif args.config:
            if args.algorithm == 'reinforce':
                model_path = f"models/{args.algorithm}/{args.config}/best_model.pth"
            else:
                model_path = f"models/{args.algorithm}/{args.config}/best_model"
        else:
            print("Error: Must specify --model, --config, or --best")
            return
        
        # Check if model exists
        if not os.path.exists(model_path + '.zip') and not os.path.exists(model_path):
            print(f"Error: Model not found: {model_path}")
            return
        
        # Run the agent
        run_trained_agent(
            args.algorithm,
            model_path,
            num_episodes=args.episodes,
            render=not args.no_render
        )
    
    else:
        print("Error: Must specify --algorithm or --compare")
        parser.print_help()


if __name__ == "__main__":
    main()

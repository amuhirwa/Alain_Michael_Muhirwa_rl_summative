"""
Generate All Plots for Report
==============================

Creates all visualizations needed for the RL summative report:
1. Cumulative rewards comparison (all algorithms)
2. Training stability (loss curves, entropy)
3. Episodes to converge
4. Generalization testing
5. Performance metrics comparison
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.enhanced_env_fast import EnhancedCrowdControlEnvFast
from stable_baselines3 import PPO, DQN
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import pandas as pd
from scipy.ndimage import uniform_filter1d

# Create results directory
os.makedirs("results", exist_ok=True)

def load_tensorboard_data(log_dir):
    """Load training data from tensorboard logs"""
    from tensorboard.backend.event_processing import event_accumulator
    
    try:
        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()
        
        # Extract rewards
        if 'rollout/ep_rew_mean' in ea.Tags()['scalars']:
            rewards = [(s.step, s.value) for s in ea.Scalars('rollout/ep_rew_mean')]
        else:
            rewards = []
        
        # Extract losses
        if 'train/loss' in ea.Tags()['scalars']:
            losses = [(s.step, s.value) for s in ea.Scalars('train/loss')]
        else:
            losses = []
        
        # Extract entropy (for policy methods)
        if 'train/entropy_loss' in ea.Tags()['scalars']:
            entropy = [(s.step, s.value) for s in ea.Scalars('train/entropy_loss')]
        else:
            entropy = []
        
        return rewards, losses, entropy
    except:
        return [], [], []


def evaluate_model(model, env, n_episodes=20):
    """Evaluate trained model"""
    rewards = []
    lengths = []
    successes = 0
    
    for i in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
        
        rewards.append(episode_reward)
        lengths.append(episode_length)
        
        # Count as success if episode completed without critical overcrowding
        if info.get('overcrowding_events', 0) == 0 and episode_length >= 400:
            successes += 1
    
    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_length': np.mean(lengths),
        'success_rate': successes / n_episodes,
        'rewards': rewards
    }


def test_generalization(model, algorithm_name):
    """Test model on different scenarios"""
    patterns = ['rush', 'steady', 'evacuation']
    difficulties = ['easy', 'medium', 'hard']
    
    results = []
    
    for pattern in patterns:
        for difficulty in difficulties:
            env = EnhancedCrowdControlEnvFast(
                crowd_arrival_pattern=pattern,
                adversarial_mode=False,
                difficulty=difficulty
            )
            
            metrics = evaluate_model(model, env, n_episodes=10)
            results.append({
                'pattern': pattern,
                'difficulty': difficulty,
                'mean_reward': metrics['mean_reward'],
                'success_rate': metrics['success_rate']
            })
            
            env.close()
            print(f"  {algorithm_name} | {pattern:10s} | {difficulty:6s} | Reward: {metrics['mean_reward']:7.2f} | Success: {metrics['success_rate']*100:.1f}%")
    
    return results


def plot_cumulative_rewards():
    """Plot 1: Cumulative rewards comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    algorithms = []
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']
    
    # Try to load PPO data
    if Path("models/quick_ppo/ppo_final.zip").exists():
        try:
            model = PPO.load("models/quick_ppo/ppo_final")
            env = EnhancedCrowdControlEnvFast(difficulty='medium')
            
            # Collect episode rewards
            episode_rewards = []
            for i in range(100):
                obs, _ = env.reset()
                ep_reward = 0
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    ep_reward += reward
                    done = terminated or truncated
                episode_rewards.append(ep_reward)
            
            # Plot
            smoothed = uniform_filter1d(episode_rewards, size=10)
            axes[0].plot(episode_rewards, alpha=0.3, color=colors[0], linewidth=0.5)
            axes[0].plot(smoothed, label='PPO', color=colors[0], linewidth=2)
            algorithms.append('PPO')
            env.close()
        except Exception as e:
            print(f"Could not load PPO: {e}")
    
    # Try to load DQN data
    if Path("models/quick_dqn/dqn_final.zip").exists():
        try:
            model = DQN.load("models/quick_dqn/dqn_final")
            env = EnhancedCrowdControlEnvFast(difficulty='medium')
            
            # Collect episode rewards
            episode_rewards = []
            for i in range(100):
                obs, _ = env.reset()
                ep_reward = 0
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    ep_reward += reward
                    done = terminated or truncated
                episode_rewards.append(ep_reward)
            
            # Plot
            smoothed = uniform_filter1d(episode_rewards, size=10)
            axes[0].plot(episode_rewards, alpha=0.3, color=colors[1], linewidth=0.5)
            axes[0].plot(smoothed, label='DQN', color=colors[1], linewidth=2)
            algorithms.append('DQN')
            env.close()
        except Exception as e:
            print(f"Could not load DQN: {e}")
    
    # Add random baseline
    env = EnhancedCrowdControlEnvFast(difficulty='medium')
    random_rewards = []
    for i in range(100):
        obs, _ = env.reset()
        ep_reward = 0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        random_rewards.append(ep_reward)
    
    smoothed = uniform_filter1d(random_rewards, size=10)
    axes[0].plot(random_rewards, alpha=0.3, color=colors[3], linewidth=0.5)
    axes[0].plot(smoothed, label='Random', color=colors[3], linewidth=2, linestyle='--')
    env.close()
    
    axes[0].set_xlabel('Episode', fontsize=12)
    axes[0].set_ylabel('Cumulative Reward', fontsize=12)
    axes[0].set_title('Episode Rewards (Medium Difficulty)', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Subplot 2: Box plot comparison
    if len(algorithms) > 0:
        data_to_plot = []
        labels = []
        
        for algo, color in zip(algorithms, colors):
            if algo == 'PPO' and Path("models/quick_ppo/results.json").exists():
                with open("models/quick_ppo/results.json", 'r') as f:
                    results = json.load(f)
                    data_to_plot.append(results['eval_rewards'])
                    labels.append('PPO')
            elif algo == 'DQN' and Path("models/quick_dqn/results.json").exists():
                with open("models/quick_dqn/results.json", 'r') as f:
                    results = json.load(f)
                    data_to_plot.append(results['eval_rewards'])
                    labels.append('DQN')
        
        data_to_plot.append(random_rewards[-10:])
        labels.append('Random')
        
        bp = axes[1].boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[1].set_ylabel('Cumulative Reward', fontsize=12)
        axes[1].set_title('Performance Comparison', fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/1_cumulative_rewards.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/1_cumulative_rewards.png")
    plt.close()


def plot_training_metrics():
    """Plot 2: Training metrics (loss, learning curve)"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Load PPO metrics from logs
    ppo_log = Path("logs/quick_ppo/monitor.csv")
    dqn_log = Path("logs/quick_dqn/monitor.csv")
    
    colors = ['#2E86AB', '#A23B72']
    
    # PPO learning curve
    if ppo_log.exists():
        try:
            df = pd.read_csv(ppo_log, skiprows=1)
            if 'r' in df.columns:
                rewards = df['r'].values
                smoothed = uniform_filter1d(rewards, size=min(20, len(rewards)//5))
                axes[0].plot(rewards, alpha=0.2, color=colors[0], linewidth=0.5)
                axes[0].plot(smoothed, label='PPO', color=colors[0], linewidth=2)
        except Exception as e:
            print(f"Could not load PPO logs: {e}")
    
    # DQN learning curve
    if dqn_log.exists():
        try:
            df = pd.read_csv(dqn_log, skiprows=1)
            if 'r' in df.columns:
                rewards = df['r'].values
                smoothed = uniform_filter1d(rewards, size=min(20, len(rewards)//5))
                axes[0].plot(rewards, alpha=0.2, color=colors[1], linewidth=0.5)
                axes[0].plot(smoothed, label='DQN', color=colors[1], linewidth=2)
        except Exception as e:
            print(f"Could not load DQN logs: {e}")
    
    axes[0].set_xlabel('Episode', fontsize=12)
    axes[0].set_ylabel('Episode Reward', fontsize=12)
    axes[0].set_title('Training Progress', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Episode length over time
    axes[1].set_xlabel('Episode', fontsize=12)
    axes[1].set_ylabel('Episode Length (steps)', fontsize=12)
    axes[1].set_title('Episode Duration', fontsize=14, fontweight='bold')
    
    if ppo_log.exists():
        try:
            df = pd.read_csv(ppo_log, skiprows=1)
            if 'l' in df.columns:
                lengths = df['l'].values
                smoothed = uniform_filter1d(lengths, size=min(20, len(lengths)//5))
                axes[1].plot(lengths, alpha=0.2, color=colors[0], linewidth=0.5)
                axes[1].plot(smoothed, label='PPO', color=colors[0], linewidth=2)
        except:
            pass
    
    if dqn_log.exists():
        try:
            df = pd.read_csv(dqn_log, skiprows=1)
            if 'l' in df.columns:
                lengths = df['l'].values
                smoothed = uniform_filter1d(lengths, size=min(20, len(lengths)//5))
                axes[1].plot(lengths, alpha=0.2, color=colors[1], linewidth=0.5)
                axes[1].plot(smoothed, label='DQN', color=colors[1], linewidth=2)
        except:
            pass
    
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/2_training_metrics.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/2_training_metrics.png")
    plt.close()


def plot_convergence():
    """Plot 3: Convergence analysis"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    algorithms = []
    convergence_data = []
    
    # Analyze PPO convergence
    if Path("logs/quick_ppo/monitor.csv").exists():
        try:
            df = pd.read_csv("logs/quick_ppo/monitor.csv", skiprows=1)
            if 'r' in df.columns:
                rewards = df['r'].values
                # Find convergence point (when smoothed reward stabilizes)
                window_size = 20
                smoothed = uniform_filter1d(rewards, size=window_size)
                
                # Convergence = when variance in moving window becomes small
                convergence_threshold = np.std(rewards) * 0.3
                for i in range(len(smoothed) - window_size):
                    window_variance = np.std(smoothed[i:i+window_size])
                    if window_variance < convergence_threshold:
                        convergence_episode = i
                        break
                else:
                    convergence_episode = len(rewards)
                
                algorithms.append('PPO')
                convergence_data.append(convergence_episode)
        except:
            pass
    
    # Analyze DQN convergence
    if Path("logs/quick_dqn/monitor.csv").exists():
        try:
            df = pd.read_csv("logs/quick_dqn/monitor.csv", skiprows=1)
            if 'r' in df.columns:
                rewards = df['r'].values
                window_size = 20
                smoothed = uniform_filter1d(rewards, size=window_size)
                
                convergence_threshold = np.std(rewards) * 0.3
                for i in range(len(smoothed) - window_size):
                    window_variance = np.std(smoothed[i:i+window_size])
                    if window_variance < convergence_threshold:
                        convergence_episode = i
                        break
                else:
                    convergence_episode = len(rewards)
                
                algorithms.append('DQN')
                convergence_data.append(convergence_episode)
        except:
            pass
    
    if len(algorithms) > 0:
        colors = ['#2E86AB', '#A23B72']
        bars = ax.bar(algorithms, convergence_data, color=colors[:len(algorithms)], alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, value in zip(bars, convergence_data):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(value)}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Episodes to Converge', fontsize=12)
        ax.set_title('Convergence Speed Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/3_convergence.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/3_convergence.png")
    plt.close()


def plot_generalization():
    """Plot 4: Generalization testing"""
    print("\nTesting generalization...")
    
    all_results = []
    
    # Test PPO
    if Path("models/quick_ppo/ppo_final.zip").exists():
        try:
            model = PPO.load("models/quick_ppo/ppo_final")
            print("\nPPO Generalization:")
            results = test_generalization(model, "PPO")
            for r in results:
                r['algorithm'] = 'PPO'
            all_results.extend(results)
        except Exception as e:
            print(f"Could not test PPO: {e}")
    
    # Test DQN
    if Path("models/quick_dqn/dqn_final.zip").exists():
        try:
            model = DQN.load("models/quick_dqn/dqn_final")
            print("\nDQN Generalization:")
            results = test_generalization(model, "DQN")
            for r in results:
                r['algorithm'] = 'DQN'
            all_results.extend(results)
        except Exception as e:
            print(f"Could not test DQN: {e}")
    
    if len(all_results) == 0:
        print("No models to test for generalization")
        return
    
    # Create visualization
    df = pd.DataFrame(all_results)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Heatmap by pattern and difficulty
    patterns = ['rush', 'steady', 'evacuation']
    difficulties = ['easy', 'medium', 'hard']
    
    for idx, algo in enumerate(df['algorithm'].unique()):
        algo_data = df[df['algorithm'] == algo]
        
        # Create heatmap data
        heatmap_data = np.zeros((len(patterns), len(difficulties)))
        for i, pattern in enumerate(patterns):
            for j, difficulty in enumerate(difficulties):
                row = algo_data[(algo_data['pattern'] == pattern) & (algo_data['difficulty'] == difficulty)]
                if len(row) > 0:
                    heatmap_data[i, j] = row['mean_reward'].values[0]
        
        im = axes[idx].imshow(heatmap_data, cmap='RdYlGn', aspect='auto')
        axes[idx].set_xticks(range(len(difficulties)))
        axes[idx].set_yticks(range(len(patterns)))
        axes[idx].set_xticklabels(difficulties)
        axes[idx].set_yticklabels(patterns)
        axes[idx].set_xlabel('Difficulty', fontsize=12)
        axes[idx].set_ylabel('Pattern', fontsize=12)
        axes[idx].set_title(f'{algo} Generalization', fontsize=14, fontweight='bold')
        
        # Add text annotations
        for i in range(len(patterns)):
            for j in range(len(difficulties)):
                text = axes[idx].text(j, i, f'{heatmap_data[i, j]:.0f}',
                                     ha="center", va="center", color="black", fontsize=10)
        
        plt.colorbar(im, ax=axes[idx], label='Mean Reward')
    
    plt.tight_layout()
    plt.savefig('results/4_generalization.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/4_generalization.png")
    plt.close()
    
    # Save generalization data
    df.to_csv('results/generalization_data.csv', index=False)
    print("✓ Saved: results/generalization_data.csv")


def plot_performance_summary():
    """Plot 5: Overall performance metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    algorithms = []
    mean_rewards = []
    std_rewards = []
    success_rates = []
    training_times = []
    
    # Load PPO data
    if Path("models/quick_ppo/results.json").exists():
        with open("models/quick_ppo/results.json", 'r') as f:
            ppo_results = json.load(f)
            algorithms.append('PPO')
            mean_rewards.append(ppo_results['mean_reward'])
            std_rewards.append(ppo_results['std_reward'])
            training_times.append(ppo_results['training_time_seconds'] / 60)  # Convert to minutes
            
            # Calculate success rate (episodes with reward > -100)
            successes = sum(1 for r in ppo_results['eval_rewards'] if r > -100)
            success_rates.append(successes / len(ppo_results['eval_rewards']) * 100)
    
    # Load DQN data
    if Path("models/quick_dqn/results.json").exists():
        with open("models/quick_dqn/results.json", 'r') as f:
            dqn_results = json.load(f)
            algorithms.append('DQN')
            mean_rewards.append(dqn_results['mean_reward'])
            std_rewards.append(dqn_results['std_reward'])
            training_times.append(dqn_results['training_time_seconds'] / 60)
            
            successes = sum(1 for r in dqn_results['eval_rewards'] if r > -100)
            success_rates.append(successes / len(dqn_results['eval_rewards']) * 100)
    
    colors = ['#2E86AB', '#A23B72']
    
    # Plot 1: Mean rewards with error bars
    axes[0, 0].bar(algorithms, mean_rewards, yerr=std_rewards, color=colors[:len(algorithms)],
                   alpha=0.7, capsize=10, edgecolor='black')
    axes[0, 0].set_ylabel('Mean Reward', fontsize=12)
    axes[0, 0].set_title('Average Performance', fontsize=14, fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Success rates
    axes[0, 1].bar(algorithms, success_rates, color=colors[:len(algorithms)],
                   alpha=0.7, edgecolor='black')
    axes[0, 1].set_ylabel('Success Rate (%)', fontsize=12)
    axes[0, 1].set_title('Success Rate (Reward > -100)', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylim([0, 100])
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Plot 3: Training times
    axes[1, 0].bar(algorithms, training_times, color=colors[:len(algorithms)],
                   alpha=0.7, edgecolor='black')
    axes[1, 0].set_ylabel('Training Time (minutes)', fontsize=12)
    axes[1, 0].set_title('Training Efficiency', fontsize=14, fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Plot 4: Summary table
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    
    table_data = []
    for i, algo in enumerate(algorithms):
        table_data.append([
            algo,
            f"{mean_rewards[i]:.2f} ± {std_rewards[i]:.2f}",
            f"{success_rates[i]:.1f}%",
            f"{training_times[i]:.1f} min"
        ])
    
    table = axes[1, 1].table(cellText=table_data,
                             colLabels=['Algorithm', 'Mean Reward', 'Success Rate', 'Training Time'],
                             cellLoc='center',
                             loc='center',
                             colColours=['#E0E0E0']*4)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    plt.tight_layout()
    plt.savefig('results/5_performance_summary.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/5_performance_summary.png")
    plt.close()


def generate_all_plots():
    """Generate all plots for the report"""
    print("\n" + "="*70)
    print("GENERATING REPORT PLOTS")
    print("="*70)
    
    print("\n[1/5] Cumulative Rewards...")
    plot_cumulative_rewards()
    
    print("\n[2/5] Training Metrics...")
    plot_training_metrics()
    
    print("\n[3/5] Convergence Analysis...")
    plot_convergence()
    
    print("\n[4/5] Generalization Testing...")
    plot_generalization()
    
    print("\n[5/5] Performance Summary...")
    plot_performance_summary()
    
    print("\n" + "="*70)
    print("ALL PLOTS GENERATED!")
    print("="*70)
    print("\nGenerated files:")
    print("  - results/1_cumulative_rewards.png")
    print("  - results/2_training_metrics.png")
    print("  - results/3_convergence.png")
    print("  - results/4_generalization.png")
    print("  - results/5_performance_summary.png")
    print("  - results/generalization_data.csv")
    print("\nUse these plots in your report!")
    print("="*70)


if __name__ == "__main__":
    generate_all_plots()

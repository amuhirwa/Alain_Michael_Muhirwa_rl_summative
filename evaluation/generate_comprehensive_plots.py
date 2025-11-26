"""
Generate Comprehensive Report Plots
===================================

Publication-quality visualizations for RL summative report.
Analyzes all trained models: PPO, DQN, A2C (base + curriculum).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.enhanced_env_fast import EnhancedCrowdControlEnvFast
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
from pathlib import Path
import pandas as pd
from scipy.ndimage import uniform_filter1d
import seaborn as sns

# Style settings
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

os.makedirs("results", exist_ok=True)

COLORS = {
    'PPO': '#2E86AB',
    'DQN': '#A23B72',
    'A2C': '#F18F01',
    'Random': '#999999',
    'Curriculum': '#06A77D'
}


def find_all_models():
    """Discover all trained models"""
    models = {'PPO': [], 'DQN': [], 'A2C': [], 'Curriculum': []}
    
    # Standard hyperparameter tuning models
    for algo in ['ppo', 'dqn', 'a2c']:
        model_dir = Path(f"models/{algo}")
        if model_dir.exists():
            for config_dir in model_dir.iterdir():
                if config_dir.is_dir():
                    model_path = config_dir / "final_model.zip"
                    if model_path.exists():
                        models[algo.upper()].append({
                            'path': model_path,
                            'name': config_dir.name,
                            'type': 'standard'
                        })
    
    # Curriculum models
    for algo in ['ppo', 'dqn', 'a2c']:
        curr_dir = Path(f"models/curriculum_{algo}")
        if curr_dir.exists():
            final_model = curr_dir / f"{algo.upper()}_curriculum_final.zip"
            if final_model.exists():
                models['Curriculum'].append({
                    'path': final_model,
                    'name': f"{algo.upper()}_Curriculum",
                    'type': 'curriculum',
                    'algorithm': algo.upper()
                })
    
    return models


def evaluate_model_detailed(model_path, algo_type, n_episodes=50):
    """Comprehensive model evaluation"""
    print(f"  Evaluating: {model_path.name}...")
    
    try:
        # Load model
        if algo_type == 'PPO':
            model = PPO.load(model_path)
        elif algo_type == 'DQN':
            model = DQN.load(model_path)
        elif algo_type == 'A2C':
            model = A2C.load(model_path)
        else:
            return None
        
        # Create environment
        def make_env():
            return EnhancedCrowdControlEnvFast(difficulty='medium', crowd_arrival_pattern='rush')
        
        env = DummyVecEnv([make_env])
        
        # Evaluate
        rewards = []
        lengths = []
        successes = 0
        terminal_reasons = {}
        
        for episode in range(n_episodes):
            obs = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                
                episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
                episode_length += 1
                
                # Unwrap info
                while isinstance(info, tuple) and len(info) == 1:
                    info = info[0]
                info_dict = info[0] if isinstance(info, list) else info
            
            rewards.append(episode_reward)
            lengths.append(episode_length)
            
            # Check success
            if info_dict.get('success', False):
                successes += 1
            
            # Track terminal reasons
            reason = info_dict.get('terminal_reason', 'unknown')
            terminal_reasons[reason] = terminal_reasons.get(reason, 0) + 1
        
        env.close()
        
        return {
            'rewards': rewards,
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'lengths': lengths,
            'mean_length': np.mean(lengths),
            'success_rate': successes / n_episodes,
            'terminal_reasons': terminal_reasons
        }
    
    except Exception as e:
        print(f"    Error: {e}")
        return None


def evaluate_random_baseline(n_episodes=50):
    """Evaluate random policy"""
    print("  Evaluating: Random Baseline...")
    
    env = EnhancedCrowdControlEnvFast(difficulty='medium', crowd_arrival_pattern='rush')
    
    rewards = []
    lengths = []
    successes = 0
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
        
        rewards.append(episode_reward)
        lengths.append(episode_length)
        
        if info.get('success', False):
            successes += 1
    
    env.close()
    
    return {
        'rewards': rewards,
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'lengths': lengths,
        'mean_length': np.mean(lengths),
        'success_rate': successes / n_episodes
    }


def plot_1_cumulative_rewards(all_results):
    """Plot 1: Cumulative Rewards - Best Model from Each Algorithm"""
    print("\n[1/6] Generating Cumulative Rewards Plot...")
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Top row: Individual algorithm reward curves
    for idx, (algo, color) in enumerate([('PPO', COLORS['PPO']), 
                                          ('DQN', COLORS['DQN']), 
                                          ('A2C', COLORS['A2C'])]):
        ax = fig.add_subplot(gs[0, idx])
        
        if algo in all_results and len(all_results[algo]) > 0:
            # Get best model
            best_model = max(all_results[algo], key=lambda x: x['result']['mean_reward'])
            rewards = best_model['result']['rewards']
            
            # Plot raw and smoothed
            ax.plot(rewards, alpha=0.3, color=color, linewidth=0.5)
            smoothed = uniform_filter1d(rewards, size=min(10, len(rewards)//3))
            ax.plot(smoothed, color=color, linewidth=2, label=f"{algo} Best")
            
            ax.set_title(f"{algo} - {best_model['name']}", fontweight='bold')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Cumulative Reward')
            ax.legend()
            ax.grid(alpha=0.3)
    
    # Middle row: Comparison plot
    ax_comp = fig.add_subplot(gs[1, :])
    
    for algo in ['PPO', 'DQN', 'A2C']:
        if algo in all_results and len(all_results[algo]) > 0:
            best_model = max(all_results[algo], key=lambda x: x['result']['mean_reward'])
            rewards = best_model['result']['rewards']
            smoothed = uniform_filter1d(rewards, size=min(10, len(rewards)//3))
            ax_comp.plot(smoothed, color=COLORS[algo], linewidth=2.5, label=f"{algo}", alpha=0.8)
    
    # Add random baseline
    if 'Random' in all_results:
        random_rewards = all_results['Random']['rewards']
        smoothed_random = uniform_filter1d(random_rewards, size=min(10, len(random_rewards)//3))
        ax_comp.plot(smoothed_random, color=COLORS['Random'], linewidth=2, 
                    label='Random', linestyle='--', alpha=0.7)
    
    # Add curriculum if available
    if 'Curriculum' in all_results and len(all_results['Curriculum']) > 0:
        best_curr = max(all_results['Curriculum'], key=lambda x: x['result']['mean_reward'])
        curr_rewards = best_curr['result']['rewards']
        smoothed_curr = uniform_filter1d(curr_rewards, size=min(10, len(curr_rewards)//3))
        ax_comp.plot(smoothed_curr, color=COLORS['Curriculum'], linewidth=3, 
                    label=f"Curriculum ({best_curr['algorithm']})", alpha=0.9)
    
    ax_comp.set_xlabel('Episode', fontsize=12)
    ax_comp.set_ylabel('Cumulative Reward (Smoothed)', fontsize=12)
    ax_comp.set_title('Algorithm Comparison - Episode Rewards', fontsize=14, fontweight='bold')
    ax_comp.legend(loc='best', framealpha=0.9)
    ax_comp.grid(alpha=0.3)
    
    # Bottom row: Statistics
    ax_box = fig.add_subplot(gs[2, :2])
    ax_table = fig.add_subplot(gs[2, 2])
    
    # Box plot
    box_data = []
    box_labels = []
    box_colors = []
    
    for algo in ['PPO', 'DQN', 'A2C']:
        if algo in all_results and len(all_results[algo]) > 0:
            best_model = max(all_results[algo], key=lambda x: x['result']['mean_reward'])
            box_data.append(best_model['result']['rewards'])
            box_labels.append(algo)
            box_colors.append(COLORS[algo])
    
    if 'Random' in all_results:
        box_data.append(all_results['Random']['rewards'])
        box_labels.append('Random')
        box_colors.append(COLORS['Random'])
    
    if len(box_data) > 0:
        bp = ax_box.boxplot(box_data, labels=box_labels, patch_artist=True, showfliers=False)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax_box.set_ylabel('Reward Distribution', fontsize=11)
        ax_box.set_title('Performance Variability', fontsize=12, fontweight='bold')
        ax_box.grid(axis='y', alpha=0.3)
    
    # Statistics table
    ax_table.axis('off')
    table_data = []
    
    for algo in ['PPO', 'DQN', 'A2C']:
        if algo in all_results and len(all_results[algo]) > 0:
            best = max(all_results[algo], key=lambda x: x['result']['mean_reward'])
            table_data.append([
                algo,
                f"{best['result']['mean_reward']:.1f}",
                f"{best['result']['std_reward']:.1f}",
                f"{best['result']['success_rate']*100:.1f}%"
            ])
    
    if 'Random' in all_results:
        r = all_results['Random']
        table_data.append([
            'Random',
            f"{r['mean_reward']:.1f}",
            f"{r['std_reward']:.1f}",
            f"{r['success_rate']*100:.1f}%"
        ])
    
    if len(table_data) > 0:
        table = ax_table.table(cellText=table_data,
                              colLabels=['Algorithm', 'Mean', 'Std', 'Success'],
                              cellLoc='center',
                              loc='center',
                              colColours=['#E0E0E0']*4)
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
    
    plt.savefig('results/1_cumulative_rewards.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/1_cumulative_rewards.png")
    plt.close()


def plot_2_training_stability():
    """Plot 2: Training Stability - Loss and Entropy Curves"""
    print("\n[2/6] Generating Training Stability Plot...")
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Training Stability Analysis', fontsize=16, fontweight='bold')
    
    # Load training logs
    for idx, algo in enumerate(['PPO', 'DQN', 'A2C']):
        # Top row: Episode rewards during training
        log_files = list(Path(f"logs/{algo.lower()}").rglob("monitor.csv"))
        
        if log_files:
            # Use best config's log
            best_log = log_files[0]
            try:
                df = pd.read_csv(best_log, skiprows=1)
                if 'r' in df.columns:
                    rewards = df['r'].values
                    smoothed = uniform_filter1d(rewards, size=min(20, len(rewards)//5))
                    
                    axes[0, idx].plot(rewards, alpha=0.2, color=COLORS[algo], linewidth=0.5)
                    axes[0, idx].plot(smoothed, color=COLORS[algo], linewidth=2)
                    axes[0, idx].set_title(f"{algo} Training Curve", fontweight='bold')
                    axes[0, idx].set_xlabel('Episode')
                    axes[0, idx].set_ylabel('Episode Reward')
                    axes[0, idx].grid(alpha=0.3)
            except:
                axes[0, idx].text(0.5, 0.5, f'No {algo} logs', ha='center', va='center',
                                 transform=axes[0, idx].transAxes)
                axes[0, idx].set_title(f"{algo} Training Curve", fontweight='bold')
        else:
            axes[0, idx].text(0.5, 0.5, f'No {algo} logs found', ha='center', va='center',
                             transform=axes[0, idx].transAxes)
            axes[0, idx].set_title(f"{algo} Training Curve", fontweight='bold')
        
        # Bottom row: Episode length (stability indicator)
        if log_files:
            try:
                df = pd.read_csv(log_files[0], skiprows=1)
                if 'l' in df.columns:
                    lengths = df['l'].values
                    smoothed_len = uniform_filter1d(lengths, size=min(20, len(lengths)//5))
                    
                    axes[1, idx].plot(lengths, alpha=0.2, color=COLORS[algo], linewidth=0.5)
                    axes[1, idx].plot(smoothed_len, color=COLORS[algo], linewidth=2)
                    axes[1, idx].set_title(f"{algo} Episode Length", fontweight='bold')
                    axes[1, idx].set_xlabel('Episode')
                    axes[1, idx].set_ylabel('Steps per Episode')
                    axes[1, idx].grid(alpha=0.3)
            except:
                axes[1, idx].text(0.5, 0.5, 'No length data', ha='center', va='center',
                                 transform=axes[1, idx].transAxes)
                axes[1, idx].set_title(f"{algo} Episode Length", fontweight='bold')
        else:
            axes[1, idx].text(0.5, 0.5, 'No data available', ha='center', va='center',
                             transform=axes[1, idx].transAxes)
            axes[1, idx].set_title(f"{algo} Episode Length", fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/2_training_stability.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/2_training_stability.png")
    plt.close()


def plot_3_convergence(all_results):
    """Plot 3: Episodes to Converge"""
    print("\n[3/6] Generating Convergence Analysis Plot...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Analyze convergence from logs
    convergence_data = {}
    
    for algo in ['PPO', 'DQN', 'A2C']:
        log_files = list(Path(f"logs/{algo.lower()}").rglob("monitor.csv"))
        
        if log_files:
            try:
                df = pd.read_csv(log_files[0], skiprows=1)
                if 'r' in df.columns:
                    rewards = df['r'].values
                    
                    # Find convergence point (when variance stabilizes)
                    window_size = 20
                    smoothed = uniform_filter1d(rewards, size=window_size)
                    
                    # Convergence = when moving std drops below threshold
                    threshold = np.std(rewards) * 0.3
                    for i in range(len(smoothed) - window_size):
                        window_std = np.std(smoothed[i:i+window_size])
                        if window_std < threshold:
                            convergence_data[algo] = i
                            break
                    else:
                        convergence_data[algo] = len(rewards)
            except:
                pass
    
    # Plot 1: Convergence episodes
    if convergence_data:
        algos = list(convergence_data.keys())
        episodes = [convergence_data[a] for a in algos]
        colors_list = [COLORS[a] for a in algos]
        
        bars = axes[0].bar(algos, episodes, color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        for bar, value in zip(bars, episodes):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(value)}',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        axes[0].set_ylabel('Episodes to Convergence', fontsize=12)
        axes[0].set_title('Convergence Speed', fontsize=13, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, 'No convergence data available', 
                    ha='center', va='center', transform=axes[0].transAxes)
    
    # Plot 2: Final performance vs convergence speed
    if all_results and convergence_data:
        for algo in ['PPO', 'DQN', 'A2C']:
            if algo in all_results and len(all_results[algo]) > 0 and algo in convergence_data:
                best = max(all_results[algo], key=lambda x: x['result']['mean_reward'])
                axes[1].scatter(convergence_data[algo], best['result']['mean_reward'],
                              s=200, color=COLORS[algo], alpha=0.7, edgecolor='black',
                              linewidth=1.5, label=algo)
                axes[1].annotate(algo, 
                               (convergence_data[algo], best['result']['mean_reward']),
                               xytext=(10, 10), textcoords='offset points',
                               fontsize=10, fontweight='bold')
        
        axes[1].set_xlabel('Episodes to Convergence', fontsize=12)
        axes[1].set_ylabel('Final Mean Reward', fontsize=12)
        axes[1].set_title('Convergence vs Performance', fontsize=13, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/3_convergence.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/3_convergence.png")
    plt.close()


def plot_4_generalization(all_results):
    """Plot 4: Generalization Testing"""
    print("\n[4/6] Generating Generalization Analysis...")
    
    patterns = ['rush', 'steady', 'evacuation']
    difficulties = ['easy', 'medium', 'hard']
    
    # Test best model from each algorithm
    generalization_results = []
    
    for algo in ['PPO', 'DQN', 'A2C']:
        if algo in all_results and len(all_results[algo]) > 0:
            best_model_info = max(all_results[algo], key=lambda x: x['result']['mean_reward'])
            model_path = best_model_info['path']
            
            print(f"  Testing {algo} generalization...")
            
            try:
                # Load model
                if algo == 'PPO':
                    model = PPO.load(model_path)
                elif algo == 'DQN':
                    model = DQN.load(model_path)
                elif algo == 'A2C':
                    model = A2C.load(model_path)
                
                for pattern in patterns:
                    for difficulty in difficulties:
                        # Create test environment
                        def make_env():
                            return EnhancedCrowdControlEnvFast(
                                crowd_arrival_pattern=pattern,
                                adversarial_mode=False,
                                difficulty=difficulty
                            )
                        
                        env = DummyVecEnv([make_env])
                        
                        # Evaluate
                        rewards = []
                        successes = 0
                        
                        for _ in range(10):  # 10 episodes per scenario
                            obs = env.reset()
                            ep_reward = 0
                            done = False
                            
                            while not done:
                                action, _ = model.predict(obs, deterministic=True)
                                obs, reward, done, info = env.step(action)
                                ep_reward += reward[0] if isinstance(reward, np.ndarray) else reward
                                
                                # Unwrap info
                                while isinstance(info, tuple) and len(info) == 1:
                                    info = info[0]
                                info_dict = info[0] if isinstance(info, list) else info
                            
                            rewards.append(ep_reward)
                            if info_dict.get('success', False):
                                successes += 1
                        
                        env.close()
                        
                        generalization_results.append({
                            'algorithm': algo,
                            'pattern': pattern,
                            'difficulty': difficulty,
                            'mean_reward': np.mean(rewards),
                            'success_rate': successes / 10
                        })
                        
                        print(f"    {algo} | {pattern:10s} | {difficulty:6s} | "
                              f"Reward: {np.mean(rewards):7.1f} | Success: {successes*10:.0f}%")
            
            except Exception as e:
                print(f"    Error testing {algo}: {e}")
    
    # Create visualization
    if generalization_results:
        df = pd.DataFrame(generalization_results)
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('Generalization Testing - Performance Across Scenarios', 
                    fontsize=16, fontweight='bold')
        
        for idx, algo in enumerate(['PPO', 'DQN', 'A2C']):
            algo_data = df[df['algorithm'] == algo]
            
            if len(algo_data) > 0:
                # Top row: Reward heatmap
                pivot_reward = algo_data.pivot(index='pattern', columns='difficulty', values='mean_reward')
                im = axes[0, idx].imshow(pivot_reward.values, cmap='RdYlGn', aspect='auto', 
                                        vmin=df['mean_reward'].min(), vmax=df['mean_reward'].max())
                axes[0, idx].set_xticks(range(len(difficulties)))
                axes[0, idx].set_yticks(range(len(patterns)))
                axes[0, idx].set_xticklabels(difficulties)
                axes[0, idx].set_yticklabels(patterns)
                axes[0, idx].set_xlabel('Difficulty')
                axes[0, idx].set_ylabel('Pattern')
                axes[0, idx].set_title(f'{algo} - Mean Reward', fontweight='bold')
                
                for i in range(len(patterns)):
                    for j in range(len(difficulties)):
                        text = axes[0, idx].text(j, i, f'{pivot_reward.values[i, j]:.0f}',
                                                ha="center", va="center", color="black", fontsize=9)
                
                plt.colorbar(im, ax=axes[0, idx])
                
                # Bottom row: Success rate heatmap
                pivot_success = algo_data.pivot(index='pattern', columns='difficulty', values='success_rate')
                im2 = axes[1, idx].imshow(pivot_success.values * 100, cmap='Blues', aspect='auto',
                                         vmin=0, vmax=100)
                axes[1, idx].set_xticks(range(len(difficulties)))
                axes[1, idx].set_yticks(range(len(patterns)))
                axes[1, idx].set_xticklabels(difficulties)
                axes[1, idx].set_yticklabels(patterns)
                axes[1, idx].set_xlabel('Difficulty')
                axes[1, idx].set_ylabel('Pattern')
                axes[1, idx].set_title(f'{algo} - Success Rate (%)', fontweight='bold')
                
                for i in range(len(patterns)):
                    for j in range(len(difficulties)):
                        text = axes[1, idx].text(j, i, f'{pivot_success.values[i, j]*100:.0f}%',
                                                ha="center", va="center", color="black", fontsize=9)
                
                plt.colorbar(im2, ax=axes[1, idx])
        
        plt.tight_layout()
        plt.savefig('results/4_generalization.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: results/4_generalization.png")
        
        # Save data
        df.to_csv('results/generalization_data.csv', index=False)
        print("✓ Saved: results/generalization_data.csv")
        plt.close()
    else:
        print("  No generalization data to plot")


def plot_5_performance_summary(all_results):
    """Plot 5: Comprehensive Performance Summary"""
    print("\n[5/6] Generating Performance Summary...")
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Collect data
    algo_names = []
    mean_rewards = []
    std_rewards = []
    success_rates = []
    mean_lengths = []
    
    for algo in ['PPO', 'DQN', 'A2C']:
        if algo in all_results and len(all_results[algo]) > 0:
            best = max(all_results[algo], key=lambda x: x['result']['mean_reward'])
            algo_names.append(f"{algo}\n({best['name'][:15]})")
            mean_rewards.append(best['result']['mean_reward'])
            std_rewards.append(best['result']['std_reward'])
            success_rates.append(best['result']['success_rate'] * 100)
            mean_lengths.append(best['result']['mean_length'])
    
    # Add random
    if 'Random' in all_results:
        r = all_results['Random']
        algo_names.append('Random')
        mean_rewards.append(r['mean_reward'])
        std_rewards.append(r['std_reward'])
        success_rates.append(r['success_rate'] * 100)
        mean_lengths.append(r['mean_length'])
    
    colors_list = [COLORS['PPO'], COLORS['DQN'], COLORS['A2C'], COLORS['Random']][:len(algo_names)]
    
    # Plot 1: Mean Rewards
    ax1 = fig.add_subplot(gs[0, 0])
    bars1 = ax1.bar(range(len(algo_names)), mean_rewards, yerr=std_rewards,
                   color=colors_list, alpha=0.7, capsize=10, edgecolor='black', linewidth=1.5)
    ax1.set_xticks(range(len(algo_names)))
    ax1.set_xticklabels(algo_names, rotation=15, ha='right', fontsize=9)
    ax1.set_ylabel('Mean Reward', fontsize=11)
    ax1.set_title('Average Performance', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars1, mean_rewards):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.0f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Success Rates
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.bar(range(len(algo_names)), success_rates,
                   color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_xticks(range(len(algo_names)))
    ax2.set_xticklabels(algo_names, rotation=15, ha='right', fontsize=9)
    ax2.set_ylabel('Success Rate (%)', fontsize=11)
    ax2.set_title('Success Rate', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 100])
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars2, success_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Episode Lengths
    ax3 = fig.add_subplot(gs[0, 2])
    bars3 = ax3.bar(range(len(algo_names)), mean_lengths,
                   color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.set_xticks(range(len(algo_names)))
    ax3.set_xticklabels(algo_names, rotation=15, ha='right', fontsize=9)
    ax3.set_ylabel('Mean Episode Length', fontsize=11)
    ax3.set_title('Episode Duration', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars3, mean_lengths):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.0f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Reward-Success Scatter
    ax4 = fig.add_subplot(gs[1, :2])
    for i, (name, reward, success, color) in enumerate(zip(algo_names, mean_rewards, success_rates, colors_list)):
        ax4.scatter(reward, success, s=300, color=color, alpha=0.7, 
                   edgecolor='black', linewidth=2, label=name.split('\n')[0])
        ax4.annotate(name.split('\n')[0], (reward, success),
                    xytext=(10, 10), textcoords='offset points', fontsize=10)
    
    ax4.set_xlabel('Mean Reward', fontsize=12)
    ax4.set_ylabel('Success Rate (%)', fontsize=12)
    ax4.set_title('Performance Trade-off Analysis', fontsize=13, fontweight='bold')
    ax4.legend(loc='best')
    ax4.grid(alpha=0.3)
    
    # Plot 5: Statistics Table
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    table_data = []
    for i, name in enumerate(algo_names):
        table_data.append([
            name.split('\n')[0],
            f"{mean_rewards[i]:.0f}±{std_rewards[i]:.0f}",
            f"{success_rates[i]:.1f}%",
            f"{mean_lengths[i]:.0f}"
        ])
    
    table = ax5.table(cellText=table_data,
                     colLabels=['Algorithm', 'Reward', 'Success', 'Length'],
                     cellLoc='center',
                     loc='center',
                     colColours=['#E0E0E0']*4)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Plot 6: Terminal Reasons Distribution
    ax6 = fig.add_subplot(gs[2, :])
    
    # Collect terminal reasons
    terminal_data = {}
    for algo in ['PPO', 'DQN', 'A2C']:
        if algo in all_results and len(all_results[algo]) > 0:
            best = max(all_results[algo], key=lambda x: x['result']['mean_reward'])
            terminal_data[algo] = best['result'].get('terminal_reasons', {})
    
    if terminal_data:
        # Get all unique reasons
        all_reasons = set()
        for reasons in terminal_data.values():
            all_reasons.update(reasons.keys())
        
        x = np.arange(len(terminal_data))
        width = 0.8 / len(all_reasons) if all_reasons else 0.2
        
        for i, reason in enumerate(sorted(all_reasons)):
            counts = [terminal_data[algo].get(reason, 0) for algo in terminal_data.keys()]
            offset = (i - len(all_reasons)/2) * width
            ax6.bar(x + offset, counts, width, label=reason, alpha=0.7)
        
        ax6.set_xlabel('Algorithm', fontsize=12)
        ax6.set_ylabel('Number of Episodes', fontsize=12)
        ax6.set_title('Episode Termination Reasons', fontsize=13, fontweight='bold')
        ax6.set_xticks(x)
        ax6.set_xticklabels(terminal_data.keys())
        ax6.legend(loc='best', fontsize=9)
        ax6.grid(axis='y', alpha=0.3)
    
    plt.savefig('results/5_performance_summary.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/5_performance_summary.png")
    plt.close()


def plot_6_hyperparameter_comparison(all_results):
    """Plot 6: Hyperparameter Tuning Results"""
    print("\n[6/6] Generating Hyperparameter Comparison...")
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Hyperparameter Tuning Analysis', fontsize=16, fontweight='bold')
    
    for idx, algo in enumerate(['PPO', 'DQN', 'A2C']):
        if algo in all_results and len(all_results[algo]) > 1:
            # Sort by performance
            sorted_models = sorted(all_results[algo], 
                                  key=lambda x: x['result']['mean_reward'],
                                  reverse=True)
            
            names = [m['name'][:20] for m in sorted_models[:10]]  # Top 10
            rewards = [m['result']['mean_reward'] for m in sorted_models[:10]]
            success = [m['result']['success_rate'] * 100 for m in sorted_models[:10]]
            
            # Top configs by reward
            axes[0, idx].barh(range(len(names)), rewards, color=COLORS[algo], alpha=0.7)
            axes[0, idx].set_yticks(range(len(names)))
            axes[0, idx].set_yticklabels(names, fontsize=8)
            axes[0, idx].set_xlabel('Mean Reward')
            axes[0, idx].set_title(f'{algo} - Top Configs (Reward)', fontweight='bold')
            axes[0, idx].grid(axis='x', alpha=0.3)
            
            # Top configs by success
            axes[1, idx].barh(range(len(names)), success, color=COLORS[algo], alpha=0.7)
            axes[1, idx].set_yticks(range(len(names)))
            axes[1, idx].set_yticklabels(names, fontsize=8)
            axes[1, idx].set_xlabel('Success Rate (%)')
            axes[1, idx].set_title(f'{algo} - Top Configs (Success)', fontweight='bold')
            axes[1, idx].grid(axis='x', alpha=0.3)
        else:
            axes[0, idx].text(0.5, 0.5, f'No {algo} configs', ha='center', va='center',
                             transform=axes[0, idx].transAxes)
            axes[1, idx].text(0.5, 0.5, f'No {algo} configs', ha='center', va='center',
                             transform=axes[1, idx].transAxes)
    
    plt.tight_layout()
    plt.savefig('results/6_hyperparameter_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/6_hyperparameter_comparison.png")
    plt.close()


def generate_all_plots():
    """Main function - generate all plots"""
    print("\n" + "="*70)
    print("GENERATING COMPREHENSIVE REPORT PLOTS")
    print("="*70)
    
    # Find all models
    print("\nDiscovering trained models...")
    models = find_all_models()
    
    for algo, model_list in models.items():
        print(f"  {algo}: {len(model_list)} models found")
    
    # Evaluate all models
    print("\nEvaluating models (this may take a while)...")
    all_results = {}
    
    for algo in ['PPO', 'DQN', 'A2C', 'Curriculum']:
        if models[algo]:
            all_results[algo] = []
            for model_info in models[algo]:
                if algo == 'Curriculum':
                    result = evaluate_model_detailed(model_info['path'], 
                                                    model_info['algorithm'], 
                                                    n_episodes=20)
                else:
                    result = evaluate_model_detailed(model_info['path'], algo, n_episodes=20)
                
                if result:
                    all_results[algo].append({
                        'name': model_info['name'],
                        'path': model_info['path'],
                        'result': result,
                        'algorithm': model_info.get('algorithm', algo)
                    })
    
    # Evaluate random baseline
    print("\n  Evaluating Random Baseline...")
    all_results['Random'] = evaluate_random_baseline(n_episodes=50)
    
    # Generate plots
    plot_1_cumulative_rewards(all_results)
    plot_2_training_stability()
    plot_3_convergence(all_results)
    plot_4_generalization(all_results)
    plot_5_performance_summary(all_results)
    plot_6_hyperparameter_comparison(all_results)
    
    print("\n" + "="*70)
    print("ALL PLOTS GENERATED SUCCESSFULLY!")
    print("="*70)
    print("\nGenerated files:")
    print("  1. results/1_cumulative_rewards.png - Episode reward curves")
    print("  2. results/2_training_stability.png - Training curves & stability")
    print("  3. results/3_convergence.png - Convergence analysis")
    print("  4. results/4_generalization.png - Cross-scenario testing")
    print("  5. results/5_performance_summary.png - Comprehensive metrics")
    print("  6. results/6_hyperparameter_comparison.png - Config analysis")
    print("  • results/generalization_data.csv - Raw data")
    print("\nAll plots are publication-quality (300 DPI)")
    print("="*70)


if __name__ == "__main__":
    generate_all_plots()

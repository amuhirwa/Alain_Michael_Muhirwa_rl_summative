"""
Utility script to compare algorithm results and generate comparison visualizations
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def load_all_results():
    """Load results from all algorithms"""
    algorithms = ['dqn', 'ppo', 'a2c', 'reinforce']
    all_data = {}
    
    for algo in algorithms:
        results_file = f"models/{algo}/all_results.json"
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                all_data[algo] = json.load(f)
        else:
            print(f"Warning: No results found for {algo}")
    
    return all_data


def create_master_comparison_plot(all_data):
    """Create comprehensive comparison plot across all algorithms"""
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Extract data for each algorithm
    algo_stats = {}
    for algo_name, results in all_data.items():
        rewards = [r['mean_eval_reward'] for r in results]
        success_rates = [r['success_rate'] * 100 for r in results]
        
        algo_stats[algo_name.upper()] = {
            'best_reward': max(rewards),
            'mean_reward': np.mean(rewards),
            'best_success': max(success_rates),
            'mean_success': np.mean(success_rates),
            'rewards': rewards,
            'success_rates': success_rates,
        }
    
    # Plot 1: Best Reward Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    algos = list(algo_stats.keys())
    best_rewards = [algo_stats[a]['best_reward'] for a in algos]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = ax1.bar(algos, best_rewards, color=colors, alpha=0.7)
    ax1.set_ylabel('Best Mean Reward', fontsize=12)
    ax1.set_title('Best Configuration Reward', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Average Reward Across Configs
    ax2 = fig.add_subplot(gs[0, 1])
    mean_rewards = [algo_stats[a]['mean_reward'] for a in algos]
    bars = ax2.bar(algos, mean_rewards, color=colors, alpha=0.7)
    ax2.set_ylabel('Average Reward', fontsize=12)
    ax2.set_title('Mean Reward Across All Configs', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=10)
    
    # Plot 3: Best Success Rate
    ax3 = fig.add_subplot(gs[0, 2])
    best_success = [algo_stats[a]['best_success'] for a in algos]
    bars = ax3.bar(algos, best_success, color=colors, alpha=0.7)
    ax3.set_ylabel('Success Rate (%)', fontsize=12)
    ax3.set_title('Best Configuration Success Rate', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim([0, 100])
    
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10)
    
    # Plot 4: Reward Distribution (Box Plot)
    ax4 = fig.add_subplot(gs[1, :2])
    reward_data = [algo_stats[a]['rewards'] for a in algos]
    bp = ax4.boxplot(reward_data, labels=algos, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax4.set_ylabel('Reward Distribution', fontsize=12)
    ax4.set_title('Reward Distribution Across All Configurations', fontsize=14, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # Plot 5: Success Rate Distribution
    ax5 = fig.add_subplot(gs[1, 2])
    success_data = [algo_stats[a]['success_rates'] for a in algos]
    bp = ax5.boxplot(success_data, labels=algos, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax5.set_ylabel('Success Rate (%)', fontsize=12)
    ax5.set_title('Success Rate Distribution', fontsize=14, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    
    # Plot 6: Reward vs Success Scatter
    ax6 = fig.add_subplot(gs[2, :])
    
    for i, (algo_name, stats) in enumerate(algo_stats.items()):
        ax6.scatter(stats['rewards'], stats['success_rates'], 
                   label=algo_name, color=colors[i], alpha=0.6, s=100)
    
    ax6.set_xlabel('Mean Reward', fontsize=12)
    ax6.set_ylabel('Success Rate (%)', fontsize=12)
    ax6.set_title('Reward vs Success Rate (All Configurations)', fontsize=14, fontweight='bold')
    ax6.legend(fontsize=11)
    ax6.grid(alpha=0.3)
    
    # Overall title
    fig.suptitle('RL Algorithms Comparison - Crowd Control Environment', 
                fontsize=18, fontweight='bold', y=0.995)
    
    plt.savefig('algorithm_comparison_master.png', dpi=300, bbox_inches='tight')
    print("Master comparison plot saved to: algorithm_comparison_master.png")
    plt.close()


def create_summary_table(all_data):
    """Create summary table of results"""
    
    summary_data = []
    
    for algo_name, results in all_data.items():
        if not results:
            continue
        
        # Find best configuration
        best_config = max(results, key=lambda x: x['mean_eval_reward'])
        
        # Calculate statistics
        all_rewards = [r['mean_eval_reward'] for r in results]
        all_success = [r['success_rate'] for r in results]
        
        summary_data.append({
            'Algorithm': algo_name.upper(),
            'Best Config': best_config['config_name'].replace('config_', ''),
            'Best Reward': f"{best_config['mean_eval_reward']:.2f} ± {best_config['std_eval_reward']:.2f}",
            'Best Success Rate': f"{best_config['success_rate']*100:.1f}%",
            'Avg Reward': f"{np.mean(all_rewards):.2f}",
            'Avg Success Rate': f"{np.mean(all_success)*100:.1f}%",
            'Configs Tested': len(results),
        })
    
    df = pd.DataFrame(summary_data)
    
    # Save to CSV
    df.to_csv('algorithm_comparison_summary.csv', index=False)
    print("\nSummary table saved to: algorithm_comparison_summary.csv")
    
    # Print to console
    print("\n" + "="*100)
    print("ALGORITHM COMPARISON SUMMARY")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)
    
    return df


def create_hyperparameter_analysis(all_data):
    """Analyze impact of specific hyperparameters"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Hyperparameter Impact Analysis', fontsize=16, fontweight='bold')
    
    # For each algorithm, analyze learning rate impact
    for idx, (algo_name, results) in enumerate(all_data.items()):
        if idx >= 4:
            break
        
        ax = axes[idx // 2, idx % 2]
        
        # Extract learning rates and rewards
        lr_rewards = {}
        for result in results:
            hp = result['hyperparameters']
            lr = hp.get('learning_rate', 0)
            reward = result['mean_eval_reward']
            
            if lr not in lr_rewards:
                lr_rewards[lr] = []
            lr_rewards[lr].append(reward)
        
        # Plot
        lrs = sorted(lr_rewards.keys())
        mean_rewards = [np.mean(lr_rewards[lr]) for lr in lrs]
        std_rewards = [np.std(lr_rewards[lr]) for lr in lrs]
        
        ax.errorbar(lrs, mean_rewards, yerr=std_rewards, marker='o', capsize=5, linewidth=2)
        ax.set_xlabel('Learning Rate', fontsize=11)
        ax.set_ylabel('Mean Reward', fontsize=11)
        ax.set_title(f'{algo_name.upper()} - Learning Rate Impact', fontsize=12, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hyperparameter_analysis.png', dpi=300, bbox_inches='tight')
    print("Hyperparameter analysis saved to: hyperparameter_analysis.png")
    plt.close()


def generate_report_data():
    """Generate all comparison data for the report"""
    
    print("\n" + "="*100)
    print("GENERATING COMPREHENSIVE ALGORITHM COMPARISON")
    print("="*100)
    
    # Load all results
    print("\nLoading results from all algorithms...")
    all_data = load_all_results()
    
    if not all_data:
        print("Error: No results found. Please train models first.")
        return
    
    print(f"Loaded results for {len(all_data)} algorithms")
    for algo, results in all_data.items():
        print(f"  - {algo.upper()}: {len(results)} configurations")
    
    # Create visualizations
    print("\nGenerating comparison plots...")
    create_master_comparison_plot(all_data)
    
    print("\nGenerating summary table...")
    create_summary_table(all_data)
    
    print("\nGenerating hyperparameter analysis...")
    create_hyperparameter_analysis(all_data)
    
    print("\n" + "="*100)
    print("COMPARISON GENERATION COMPLETE")
    print("="*100)
    print("\nGenerated files:")
    print("  1. algorithm_comparison_master.png")
    print("  2. algorithm_comparison_summary.csv")
    print("  3. hyperparameter_analysis.png")
    print("\nThese files can be included in your report.")
    print("="*100)


if __name__ == "__main__":
    generate_report_data()

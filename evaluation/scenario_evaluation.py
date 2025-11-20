"""
Scenario-Based Evaluation for Crowd Control Models
==================================================

Test trained models across different crowd scenarios with adversarial testing.

NOVEL CONTRIBUTION: Safety-critical evaluation across scenarios:
- Steady flow (normal operation)
- Rush scenarios (peak times, concert entry)
- Evacuation (emergency situations)

Each scenario tested with and without adversarial events (gate failures,
sudden crowd surges, bottlenecks) to validate robustness and safety.

Metrics:
- Success rate (% episodes without critical overcrowding)
- Average reward
- Max panic level reached
- Throughput (agents exited / time)
- Safety score (based on density and panic)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.enhanced_env import EnhancedCrowdControlEnv
from stable_baselines3 import PPO, DQN, A2C
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model_on_scenario(
    model,
    scenario: str,
    adversarial: bool,
    difficulty: str,
    num_episodes: int = 20
) -> Dict:
    """
    Evaluate model on a specific scenario
    
    NOVEL: Safety-critical testing with adversarial scenarios
    
    Args:
        model: Trained RL model
        scenario: 'steady', 'rush', or 'evacuation'
        adversarial: Enable adversarial events
        difficulty: 'easy', 'medium', or 'hard'
        num_episodes: Number of test episodes
        
    Returns:
        Dictionary with evaluation metrics
    """
    
    print(f"\nEvaluating: {scenario.upper()} | Adversarial: {adversarial} | Difficulty: {difficulty}")
    
    # Create environment
    env = EnhancedCrowdControlEnv(
        crowd_arrival_pattern=scenario,
        adversarial_mode=adversarial,
        difficulty=difficulty
    )
    
    # Metrics storage
    episode_rewards = []
    success_count = 0
    overcrowding_count = 0
    max_panic_levels = []
    avg_panic_levels = []
    throughputs = []
    max_densities = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            done = terminated or truncated
        
        # Record metrics
        episode_rewards.append(episode_reward)
        max_panic_levels.append(info['max_panic'])
        avg_panic_levels.append(info['avg_panic'])
        max_densities.append(info['max_density'])
        
        # Calculate throughput
        if step_count > 0:
            throughput = info['total_exited'] / step_count
            throughputs.append(throughput)
        
        # Check success (no critical overcrowding)
        if info['overcrowding_events'] == 0:
            success_count += 1
        else:
            overcrowding_count += 1
    
    env.close()
    
    # Calculate metrics
    results = {
        'scenario': scenario,
        'adversarial': adversarial,
        'difficulty': difficulty,
        'num_episodes': num_episodes,
        'success_rate': success_count / num_episodes,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_max_panic': np.mean(max_panic_levels),
        'mean_avg_panic': np.mean(avg_panic_levels),
        'mean_throughput': np.mean(throughputs) if throughputs else 0,
        'mean_max_density': np.mean(max_densities),
        'overcrowding_rate': overcrowding_count / num_episodes,
    }
    
    # Safety score (0-100)
    # Based on success rate, low panic, and controlled density
    safety_score = (
        results['success_rate'] * 40 +  # 40 points for no overcrowding
        (1 - results['mean_max_panic']) * 30 +  # 30 points for low panic
        (1 - min(results['mean_max_density'] / 10, 1.0)) * 30  # 30 points for controlled density
    )
    results['safety_score'] = safety_score
    
    return results


def comprehensive_scenario_evaluation(
    model_path: str,
    algorithm: str,
    num_episodes: int = 20
) -> pd.DataFrame:
    """
    Comprehensive evaluation across all scenarios
    
    NOVEL: Systematic safety validation across diverse scenarios
    
    Args:
        model_path: Path to trained model
        algorithm: Algorithm type ('PPO', 'DQN', 'A2C')
        num_episodes: Episodes per scenario
        
    Returns:
        DataFrame with all results
    """
    
    print("="*70)
    print("COMPREHENSIVE SCENARIO EVALUATION")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Algorithm: {algorithm}")
    print(f"Episodes per scenario: {num_episodes}")
    print("="*70)
    
    # Load model
    if algorithm == 'PPO':
        model = PPO.load(model_path)
    elif algorithm == 'DQN':
        model = DQN.load(model_path)
    elif algorithm == 'A2C':
        model = A2C.load(model_path)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Define test scenarios
    scenarios = ['steady', 'rush', 'evacuation']
    adversarial_modes = [False, True]
    difficulties = ['medium']  # Can extend to test all difficulties
    
    results_list = []
    
    for scenario in scenarios:
        for adversarial in adversarial_modes:
            for difficulty in difficulties:
                result = evaluate_model_on_scenario(
                    model, scenario, adversarial, difficulty, num_episodes
                )
                results_list.append(result)
                
                # Print summary
                print(f"\n{scenario.upper()} | Adv: {adversarial} | Diff: {difficulty}")
                print(f"  Success Rate: {result['success_rate']:.1%}")
                print(f"  Safety Score: {result['safety_score']:.1f}/100")
                print(f"  Mean Reward: {result['mean_reward']:.2f}")
                print(f"  Max Panic: {result['mean_max_panic']:.3f}")
                print(f"  Throughput: {result['mean_throughput']:.4f}")
    
    # Create DataFrame
    df = pd.DataFrame(results_list)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    
    return df


def compare_algorithms_across_scenarios(
    model_paths: Dict[str, str],
    num_episodes: int = 20,
    save_dir: str = 'results/scenario_evaluation'
) -> pd.DataFrame:
    """
    Compare multiple algorithms across all scenarios
    
    NOVEL: Multi-algorithm safety comparison
    
    Args:
        model_paths: Dict mapping algorithm name to model path
        num_episodes: Episodes per scenario
        save_dir: Directory to save results
        
    Returns:
        Combined DataFrame with all results
    """
    
    print("\n" + "#"*70)
    print("# MULTI-ALGORITHM SCENARIO COMPARISON")
    print("#"*70 + "\n")
    
    os.makedirs(save_dir, exist_ok=True)
    
    all_results = []
    
    for algorithm, model_path in model_paths.items():
        print(f"\n{'='*70}")
        print(f"Evaluating {algorithm}")
        print('='*70)
        
        try:
            df = comprehensive_scenario_evaluation(model_path, algorithm, num_episodes)
            df['algorithm'] = algorithm
            all_results.append(df)
        except Exception as e:
            print(f"✗ Error evaluating {algorithm}: {e}")
    
    if not all_results:
        print("✗ No results to compare")
        return None
    
    # Combine results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Save results
    csv_path = os.path.join(save_dir, 'scenario_evaluation_results.csv')
    combined_df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved: {csv_path}")
    
    # Generate comparison plots
    _generate_comparison_plots(combined_df, save_dir)
    
    return combined_df


def _generate_comparison_plots(df: pd.DataFrame, save_dir: str):
    """Generate visualization plots for scenario evaluation"""
    
    print("\nGenerating comparison plots...")
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # 1. Safety Score Comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='scenario', y='safety_score', hue='algorithm')
    plt.title('Safety Score by Scenario and Algorithm')
    plt.ylabel('Safety Score (0-100)')
    plt.xlabel('Scenario')
    plt.legend(title='Algorithm')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'safety_score_comparison.png'), dpi=300)
    plt.close()
    
    # 2. Success Rate Comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='scenario', y='success_rate', hue='algorithm')
    plt.title('Success Rate by Scenario and Algorithm')
    plt.ylabel('Success Rate')
    plt.xlabel('Scenario')
    plt.legend(title='Algorithm')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'success_rate_comparison.png'), dpi=300)
    plt.close()
    
    # 3. Panic Level Comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='scenario', y='mean_max_panic', hue='algorithm')
    plt.title('Max Panic Level by Scenario and Algorithm')
    plt.ylabel('Max Panic Level')
    plt.xlabel('Scenario')
    plt.legend(title='Algorithm')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'panic_level_comparison.png'), dpi=300)
    plt.close()
    
    # 4. Adversarial Impact
    plt.figure(figsize=(12, 6))
    adv_df = df.groupby(['algorithm', 'adversarial'])['safety_score'].mean().reset_index()
    sns.barplot(data=adv_df, x='algorithm', y='safety_score', hue='adversarial')
    plt.title('Safety Score: Normal vs Adversarial Scenarios')
    plt.ylabel('Average Safety Score')
    plt.xlabel('Algorithm')
    plt.legend(title='Adversarial', labels=['Normal', 'Adversarial'])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'adversarial_impact.png'), dpi=300)
    plt.close()
    
    print(f"✓ Plots saved in {save_dir}/")


def print_evaluation_summary(df: pd.DataFrame):
    """Print human-readable summary of evaluation results"""
    
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    # Overall best algorithm
    overall_best = df.groupby('algorithm')['safety_score'].mean().idxmax()
    print(f"\n🏆 Overall Best Algorithm: {overall_best}")
    
    # Best by scenario
    print("\n📊 Best Algorithm by Scenario:")
    for scenario in df['scenario'].unique():
        scenario_df = df[df['scenario'] == scenario]
        best_algo = scenario_df.groupby('algorithm')['safety_score'].mean().idxmax()
        best_score = scenario_df.groupby('algorithm')['safety_score'].mean().max()
        print(f"  {scenario.upper():12} → {best_algo} (Safety: {best_score:.1f}/100)")
    
    # Adversarial robustness
    print("\n🛡️ Adversarial Robustness:")
    for algo in df['algorithm'].unique():
        algo_df = df[df['algorithm'] == algo]
        normal_score = algo_df[~algo_df['adversarial']]['safety_score'].mean()
        adv_score = algo_df[algo_df['adversarial']]['safety_score'].mean()
        robustness = (adv_score / normal_score * 100) if normal_score > 0 else 0
        print(f"  {algo:5} → Normal: {normal_score:.1f} | Adversarial: {adv_score:.1f} | Robustness: {robustness:.1f}%")
    
    # Safety critical scenarios
    print("\n⚠️ Safety-Critical Performance (Evacuation + Adversarial):")
    critical_df = df[(df['scenario'] == 'evacuation') & (df['adversarial'] == True)]
    for algo in critical_df['algorithm'].unique():
        algo_critical = critical_df[critical_df['algorithm'] == algo]
        success_rate = algo_critical['success_rate'].mean()
        panic_level = algo_critical['mean_max_panic'].mean()
        print(f"  {algo:5} → Success: {success_rate:.1%} | Max Panic: {panic_level:.3f}")
    
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Scenario-Based Evaluation")
    parser.add_argument('--models', nargs='+', required=True,
                       help='Paths to model files (format: algo:path)')
    parser.add_argument('--episodes', type=int, default=20,
                       help='Number of episodes per scenario')
    parser.add_argument('--output', type=str, default='results/scenario_evaluation',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Parse model paths
    model_paths = {}
    for model_spec in args.models:
        algo, path = model_spec.split(':')
        model_paths[algo] = path
    
    # Run evaluation
    df = compare_algorithms_across_scenarios(
        model_paths,
        num_episodes=args.episodes,
        save_dir=args.output
    )
    
    if df is not None:
        print_evaluation_summary(df)

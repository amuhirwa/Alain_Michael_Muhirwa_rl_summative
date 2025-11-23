"""
Reward Structure Analysis Tool
==============================

Diagnoses potential issues with the crowd control environment:
- Reward distribution analysis
- Initial state stability
- Random vs optimal policy comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.enhanced_env_fast import EnhancedCrowdControlEnvFast


class RewardAnalyzer:
    """Analyzes reward structure and environment dynamics"""
    
    def __init__(self, pattern="rush", difficulty="medium"):
        self.pattern = pattern
        self.difficulty = difficulty
        self.env = EnhancedCrowdControlEnvFast(
            render_mode=None,
            crowd_arrival_pattern=pattern,
            adversarial_mode=False,
            difficulty=difficulty
        )
    
    def test_initial_spawning(self, num_trials=50):
        """Test if initial spawning causes immediate overcrowding"""
        print("="*60)
        print("INITIAL SPAWNING STABILITY TEST")
        print("="*60)
        
        initial_densities = []
        immediate_failures = 0
        
        for trial in range(num_trials):
            obs, info = self.env.reset()
            initial_densities.append(info['max_density'])
            
            if info['max_density'] > self.env.CRITICAL_DENSITY * 0.8:
                immediate_failures += 1
                if immediate_failures <= 3:  # Print first 3
                    print(f"Trial {trial+1}: DANGER - Initial density {info['max_density']:.2f}")
        
        avg_initial = np.mean(initial_densities)
        max_initial = np.max(initial_densities)
        
        print(f"\nResults over {num_trials} trials:")
        print(f"  Average initial density: {avg_initial:.2f}")
        print(f"  Maximum initial density: {max_initial:.2f}")
        print(f"  Critical threshold: {self.env.CRITICAL_DENSITY:.2f}")
        print(f"  Danger threshold (80%): {self.env.CRITICAL_DENSITY * 0.8:.2f}")
        print(f"  Immediate danger cases: {immediate_failures}/{num_trials}")
        
        if immediate_failures > num_trials * 0.1:
            print(f"\n⚠️  WARNING: {immediate_failures/num_trials*100:.1f}% of resets are dangerous!")
            print("  RECOMMENDATION: Reduce initial spawn count or increase jitter")
        else:
            print(f"\n✓ Initial spawning is stable")
        
        return initial_densities
    
    def analyze_reward_components(self, num_episodes=10, max_steps=500):
        """Break down reward components to identify balance issues"""
        print("\n" + "="*60)
        print("REWARD COMPONENT ANALYSIS")
        print("="*60)
        
        components = defaultdict(list)
        total_rewards = []
        
        for ep in range(num_episodes):
            obs, info = self.env.reset()
            episode_components = defaultdict(float)
            episode_reward = 0
            
            for step in range(max_steps):
                # Take random action
                action = self.env.action_space.sample()
                
                # Track reward components manually
                before_reward = episode_reward
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                
                # Approximate component breakdown (would need env modification for exact)
                if reward > 50:  # Likely terminal success bonus
                    episode_components['terminal_bonus'] += reward
                elif reward < -50:  # Likely terminal failure penalty
                    episode_components['terminal_penalty'] += reward
                else:
                    episode_components['step_reward'] += reward
                
                if terminated or truncated:
                    break
            
            total_rewards.append(episode_reward)
            for key, val in episode_components.items():
                components[key].append(val)
        
        # Print statistics
        print(f"\nResults over {num_episodes} episodes with random policy:")
        print(f"\n  Total Reward Statistics:")
        print(f"    Mean: {np.mean(total_rewards):.2f}")
        print(f"    Std:  {np.std(total_rewards):.2f}")
        print(f"    Min:  {np.min(total_rewards):.2f}")
        print(f"    Max:  {np.max(total_rewards):.2f}")
        
        print(f"\n  Component Breakdown:")
        for component, values in components.items():
            print(f"    {component}:")
            print(f"      Mean: {np.mean(values):.2f}")
            print(f"      Sum:  {np.sum(values):.2f}")
        
        # Check for issues
        mean_reward = np.mean(total_rewards)
        if mean_reward > 100:
            print(f"\n⚠️  WARNING: Mean reward ({mean_reward:.2f}) very high for random policy!")
            print("  RECOMMENDATION: Reduce positive reward scaling")
        elif mean_reward > 0:
            print(f"\n⚠️  CONCERN: Random policy achieves positive reward ({mean_reward:.2f})")
            print("  RECOMMENDATION: Increase difficulty or adjust reward balance")
        else:
            print(f"\n✓ Random policy achieves negative reward (good!)")
        
        return total_rewards, components
    
    def compare_policies(self, num_episodes=20, max_steps=500):
        """Compare random vs no-op policy to detect reward exploitation"""
        print("\n" + "="*60)
        print("POLICY COMPARISON TEST")
        print("="*60)
        
        random_rewards = []
        noop_rewards = []
        
        # Random policy
        print("\nTesting random policy...")
        for _ in range(num_episodes):
            obs, info = self.env.reset()
            reward_sum = 0
            for step in range(max_steps):
                action = self.env.action_space.sample()
                obs, reward, terminated, truncated, info = self.env.step(action)
                reward_sum += reward
                if terminated or truncated:
                    break
            random_rewards.append(reward_sum)
        
        # No-op policy (action 10 if it exists, else 0)
        print("Testing no-op policy...")
        noop_action = 10 if self.env.action_space.n > 10 else 0
        for _ in range(num_episodes):
            obs, info = self.env.reset()
            reward_sum = 0
            for step in range(max_steps):
                obs, reward, terminated, truncated, info = self.env.step(noop_action)
                reward_sum += reward
                if terminated or truncated:
                    break
            noop_rewards.append(reward_sum)
        
        print(f"\nRandom Policy:")
        print(f"  Mean: {np.mean(random_rewards):.2f} ± {np.std(random_rewards):.2f}")
        print(f"  Range: [{np.min(random_rewards):.2f}, {np.max(random_rewards):.2f}]")
        
        print(f"\nNo-Op Policy:")
        print(f"  Mean: {np.mean(noop_rewards):.2f} ± {np.std(noop_rewards):.2f}")
        print(f"  Range: [{np.min(noop_rewards):.2f}, {np.max(noop_rewards):.2f}]")
        
        # Analysis
        if np.mean(random_rewards) > 0 and np.mean(noop_rewards) > 0:
            print(f"\n⚠️  CRITICAL: Both policies achieve positive rewards!")
            print("  Environment may be too easy or rewards too generous")
        elif abs(np.mean(random_rewards) - np.mean(noop_rewards)) < 20:
            print(f"\n⚠️  WARNING: Random and no-op policies perform similarly")
            print("  Actions may have insufficient impact")
        else:
            print(f"\n✓ Policies show meaningful difference")
        
        return random_rewards, noop_rewards
    
    def analyze_density_evolution(self, num_episodes=5):
        """Track how density evolves over time"""
        print("\n" + "="*60)
        print("DENSITY EVOLUTION ANALYSIS")
        print("="*60)
        
        for ep in range(num_episodes):
            obs, info = self.env.reset()
            
            densities = [info['max_density']]
            overcrowding_step = None
            
            for step in range(500):
                action = self.env.action_space.sample()
                obs, reward, terminated, truncated, info = self.env.step(action)
                densities.append(info['max_density'])
                
                if info['max_density'] > self.env.CRITICAL_DENSITY and overcrowding_step is None:
                    overcrowding_step = step
                
                if terminated or truncated:
                    break
            
            max_density = np.max(densities)
            avg_density = np.mean(densities)
            
            print(f"\nEpisode {ep+1}:")
            print(f"  Duration: {len(densities)} steps")
            print(f"  Average density: {avg_density:.2f}")
            print(f"  Maximum density: {max_density:.2f}")
            if overcrowding_step:
                print(f"  Overcrowding at step: {overcrowding_step}")
            else:
                print(f"  No overcrowding")
    
    def run_full_diagnostic(self):
        """Run all diagnostic tests"""
        print("\n" + "="*60)
        print("ENHANCED CROWD CONTROL ENVIRONMENT DIAGNOSTIC")
        print(f"Pattern: {self.pattern} | Difficulty: {self.difficulty}")
        print("="*60)
        
        # Test 1: Initial spawning
        self.test_initial_spawning(num_trials=50)
        
        # Test 2: Reward components
        self.analyze_reward_components(num_episodes=10)
        
        # Test 3: Policy comparison
        self.compare_policies(num_episodes=20)
        
        # Test 4: Density evolution
        self.analyze_density_evolution(num_episodes=5)
        
        print("\n" + "="*60)
        print("DIAGNOSTIC COMPLETE")
        print("="*60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze reward structure")
    parser.add_argument("--pattern", default="rush", choices=["rush", "steady", "evacuation"])
    parser.add_argument("--difficulty", default="medium", choices=["easy", "medium", "hard"])
    args = parser.parse_args()
    
    analyzer = RewardAnalyzer(pattern=args.pattern, difficulty=args.difficulty)
    analyzer.run_full_diagnostic()


if __name__ == "__main__":
    main()
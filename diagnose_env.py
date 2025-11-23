"""Diagnostic script to identify environment issues"""
import sys
sys.path.append('.')
from environment.enhanced_env_fast import EnhancedCrowdControlEnvFast
import numpy as np

print("=" * 70)
print("ENVIRONMENT DIAGNOSTICS")
print("=" * 70)

env = EnhancedCrowdControlEnvFast(
    crowd_arrival_pattern='rush',
    difficulty='hard',
    adversarial_mode=False
)

obs, info = env.reset(seed=42)

# Track key metrics
step_rewards = []
density_rewards = []
safety_rewards = []
efficiency_rewards = []
infra_costs = []
max_densities = []
avg_panics = []
exits_per_step = []

print("\n1. TESTING REWARD COMPONENTS (first 100 steps)")
print("-" * 70)

for step in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Get individual components
    density_r = env._calculate_density_reward()
    safety_r = env._calculate_safety_reward()
    efficiency_r = env._calculate_efficiency_reward()
    infra_c = env._calculate_infrastructure_cost()
    
    step_rewards.append(reward)
    density_rewards.append(density_r)
    safety_rewards.append(safety_r)
    efficiency_rewards.append(efficiency_r)
    infra_costs.append(infra_c)
    max_densities.append(info['max_density'])
    avg_panics.append(info['avg_panic'])
    
    if step > 0:
        exits = info['exited'] - exits_per_step[-1] if exits_per_step else info['exited']
        exits_per_step.append(exits)
    
    if terminated or truncated:
        break

# Analysis
print(f"\nStep Rewards:")
print(f"  Mean: {np.mean(step_rewards):.2f}")
print(f"  Std:  {np.std(step_rewards):.2f}")
print(f"  Min:  {np.min(step_rewards):.2f}")
print(f"  Max:  {np.max(step_rewards):.2f}")
print(f"  Range: {np.max(step_rewards) - np.min(step_rewards):.2f}")

print(f"\nDensity Rewards:")
print(f"  Mean: {np.mean(density_rewards):.2f} (should be negative for crowding)")
print(f"  Range: [{np.min(density_rewards):.2f}, {np.max(density_rewards):.2f}]")

print(f"\nSafety Rewards (Panic):")
print(f"  Mean: {np.mean(safety_rewards):.2f}")
print(f"  Range: [{np.min(safety_rewards):.2f}, {np.max(safety_rewards):.2f}]")
print(f"  Max Panic Ever: {np.max(avg_panics):.3f}")
print(f"  ⚠️  WARNING: Panic is {np.max(avg_panics):.3f} - should be higher!" if np.max(avg_panics) < 0.1 else "")

print(f"\nEfficiency Rewards:")
print(f"  Mean: {np.mean(efficiency_rewards):.2f}")
print(f"  Range: [{np.min(efficiency_rewards):.2f}, {np.max(efficiency_rewards):.2f}]")

print(f"\nInfrastructure Costs:")
print(f"  Mean: {np.mean(infra_costs):.2f}")
print(f"  Range: [{np.min(infra_costs):.2f}, {np.max(infra_costs):.2f}]")

print(f"\nDensity:")
print(f"  Mean: {np.mean(max_densities):.2f}")
print(f"  Max:  {np.max(max_densities):.2f}")
print(f"  Critical Threshold: {env.CRITICAL_DENSITY}")
print(f"  Target: {env.TARGET_DENSITY}")

print(f"\nAgent Dynamics:")
print(f"  Final Agents: {info['agents']}")
print(f"  Total Spawned: {info['spawned']}")
print(f"  Total Exited: {info['exited']}")
print(f"  Exit Rate: {info['exited'] / step:.2f} per step")

# Test panic triggering
print("\n" + "=" * 70)
print("2. TESTING PANIC SYSTEM")
print("-" * 70)

# Force high density by closing all gates
env2 = EnhancedCrowdControlEnvFast(
    crowd_arrival_pattern='evacuation',  # Spawn everyone at once
    difficulty='hard',
    adversarial_mode=False
)
obs, info = env2.reset(seed=42)

# Close all gates to force crowding
for i in range(env2.NUM_GATES):
    env2.gates[i][2] = 0.0

panic_levels = []
densities = []

for step in range(50):
    # Do nothing - let crowd build up
    action = 11  # No-op or minimal action
    obs, reward, terminated, truncated, info = env2.step(action)
    panic_levels.append(info['avg_panic'])
    densities.append(info['max_density'])
    
    if terminated:
        print(f"  ⚠️  Episode terminated at step {step} (overcrowding)")
        break

print(f"\nWith closed gates:")
print(f"  Max Density Reached: {np.max(densities):.2f}")
print(f"  Max Panic Reached: {np.max(panic_levels):.3f}")
print(f"  Panic Trigger Density: {env2.PANIC_TRIGGER_DENSITY}")

if np.max(panic_levels) < 0.1:
    print(f"  🚨 PROBLEM: Panic never triggered despite density {np.max(densities):.2f}!")
    print(f"     Check _update_panic() logic")
else:
    print(f"  ✓ Panic system working")

# Test reward scale consistency
print("\n" + "=" * 70)
print("3. REWARD SCALE CONSISTENCY ACROSS DIFFICULTIES")
print("-" * 70)

for diff in ['easy', 'medium', 'hard']:
    env_test = EnhancedCrowdControlEnvFast(
        crowd_arrival_pattern='steady',
        difficulty=diff,
        adversarial_mode=False
    )
    obs, info = env_test.reset(seed=42)
    
    rewards = []
    for step in range(50):
        action = env_test.action_space.sample()
        obs, reward, terminated, truncated, info = env_test.step(action)
        rewards.append(reward)
        if terminated or truncated:
            break
    
    print(f"\n{diff.upper()}:")
    print(f"  Mean: {np.mean(rewards):.2f}, Std: {np.std(rewards):.2f}")
    print(f"  Range: [{np.min(rewards):.2f}, {np.max(rewards):.2f}]")
    print(f"  Max Agents: {env_test.MAX_AGENTS}")

print("\n" + "=" * 70)
print("4. RECOMMENDATIONS")
print("=" * 70)

issues_found = []

if np.max(avg_panics) < 0.1:
    issues_found.append("❌ Panic system not triggering")

if np.max(max_densities) < env.CRITICAL_DENSITY * 0.8:
    issues_found.append("⚠️  Environment might be too easy (never reaches critical density)")

if np.mean(step_rewards) > 0:
    issues_found.append("⚠️  Average reward is positive (agent might not learn urgency)")

if np.std(step_rewards) > 10:
    issues_found.append("❌ Reward variance too high for PPO")

if len(exits_per_step) > 0 and np.mean(exits_per_step) < 0.5:
    issues_found.append("⚠️  Exit rate very low (only ~0.3-0.5 agents per step)")

if len(issues_found) == 0:
    print("\n✓ No major issues detected!")
else:
    print("\nIssues found:")
    for issue in issues_found:
        print(f"  {issue}")

print("\n" + "=" * 70)
env.close()
env2.close()

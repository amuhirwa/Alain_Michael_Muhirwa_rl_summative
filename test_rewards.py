import sys
sys.path.append('.')
from environment.enhanced_env import EnhancedCrowdControlEnv
import numpy as np

# Test reward distribution across difficulties
for difficulty in ['easy', 'medium', 'hard']:
    print(f'\n=== {difficulty.upper()} Difficulty ===')
    env = EnhancedCrowdControlEnv(
        crowd_arrival_pattern='rush',
        difficulty=difficulty,
        adversarial_mode=False
    )
    
    obs, info = env.reset(seed=42)
    rewards = []
    density_rewards = []
    safety_rewards = []
    efficiency_rewards = []
    
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Get individual reward components
        density_r = env._calculate_density_reward()
        safety_r = env._calculate_safety_reward()
        efficiency_r = env._calculate_efficiency_reward()
        infra_cost = env._calculate_infrastructure_cost()
        
        rewards.append(reward)
        density_rewards.append(density_r)
        safety_rewards.append(safety_r)
        efficiency_rewards.append(efficiency_r)
        
        if terminated or truncated:
            break
    
    total_agents = info.get('total_agents', 0)
    total_exited = info.get('total_exited', 0)
    
    print(f'Total Reward: mean={np.mean(rewards):.2f}, std={np.std(rewards):.2f}, min={np.min(rewards):.2f}, max={np.max(rewards):.2f}')
    print(f'Density:      mean={np.mean(density_rewards):.2f}, std={np.std(density_rewards):.2f}, min={np.min(density_rewards):.2f}, max={np.max(density_rewards):.2f}')
    print(f'Safety:       mean={np.mean(safety_rewards):.2f}, std={np.std(safety_rewards):.2f}, min={np.min(safety_rewards):.2f}, max={np.max(safety_rewards):.2f}')
    print(f'Efficiency:   mean={np.mean(efficiency_rewards):.2f}, std={np.std(efficiency_rewards):.2f}, min={np.min(efficiency_rewards):.2f}, max={np.max(efficiency_rewards):.2f}')
    print(f'Agents: current={total_agents}, exited={total_exited}')
    
    env.close()

print('\n=== ANALYSIS ===')
print('Reward scale analysis complete')

import sys
sys.path.append('.')
from environment.enhanced_env import EnhancedCrowdControlEnv
import numpy as np

print("Testing for terminal rewards...")
env = EnhancedCrowdControlEnv(difficulty='medium', crowd_arrival_pattern='steady')
obs, info = env.reset(seed=42)

for step in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    if reward > 15 or reward < -15:
        print(f'Step {step}: Reward={reward:.2f}, Agents={info["total_agents"]}, Exited={info["total_exited"]}, Terminated={terminated}')
    
    if terminated or truncated:
        print(f'Episode ended at step {step}: terminated={terminated}, truncated={truncated}')
        break

env.close()

"""Debug panic system specifically"""
import sys
sys.path.append('.')
from environment.enhanced_env_fast import EnhancedCrowdControlEnvFast
import numpy as np

env = EnhancedCrowdControlEnvFast(
    crowd_arrival_pattern='evacuation',
    difficulty='hard',
    adversarial_mode=False
)

obs, info = env.reset(seed=42)

# Close all gates
for i in range(env.NUM_GATES):
    env.gates[i][2] = 0.0

print("Testing panic with closed gates...")
print(f"Initial agents: {env.num_agents}")
print(f"Panic trigger density: {env.PANIC_TRIGGER_DENSITY}")
print(f"Panic increase rate: {env.PANIC_INCREASE_RATE}")
print()

for step in range(10):
    action = 11  # Do nothing
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Manual check
    max_density = np.max(env.grid_density)
    avg_panic = np.mean(env.panic[env.alive])
    max_panic = np.max(env.panic[env.alive])
    
    # Find cells with high density
    high_density_cells = np.where(env.grid_density > env.PANIC_TRIGGER_DENSITY)
    num_high_density = len(high_density_cells[0])
    
    print(f"Step {step+1}:")
    print(f"  Agents: {env.num_agents}")
    print(f"  Max density: {max_density:.1f}")
    print(f"  Cells > {env.PANIC_TRIGGER_DENSITY}: {num_high_density}")
    print(f"  Avg panic: {avg_panic:.3f}")
    print(f"  Max panic: {max_panic:.3f}")
    
    if terminated:
        print(f"\n⚠️  Terminated at step {step+1}!")
        break
    
    if step == 4:
        # Sample a few agents in high density areas
        print("\n  Sampling agents in high-density cells:")
        alive_idx = np.flatnonzero(env.alive)[:5]
        for i in alive_idx:
            xi, yi = int(env.x[i]), int(env.y[i])
            local_d = env.grid_density[yi, xi]
            print(f"    Agent {i}: pos=({env.x[i]:.1f},{env.y[i]:.1f}) cell=({xi},{yi}) density={local_d:.1f} panic={env.panic[i]:.3f}")

env.close()

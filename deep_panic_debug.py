"""Deep dive into panic calculation"""
import sys
sys.path.append('.')
from environment.enhanced_env_fast import EnhancedCrowdControlEnvFast
import numpy as np

env = EnhancedCrowdControlEnvFast(
    crowd_arrival_pattern='steady',  # Changed from evacuation
    difficulty='medium',  # Changed from hard
    adversarial_mode=False
)

obs, info = env.reset(seed=42)

# Close 2 of 3 gates to force some crowding (not instant failure)
env.gates[1][2] = 0.0  # Close left gate
env.gates[2][2] = 0.0  # Close right gate
# Leave top gate open

print("Stepping through with closed gates...")
print(f"Panic trigger: {env.PANIC_TRIGGER_DENSITY}")
print(f"Critical density: {env.CRITICAL_DENSITY}\n")

for step in range(8):
    action = 11
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated:
        print(f"\n⚠️  Terminated at step {step+1} (overcrowding)")
        break
    
    # Analyze grid
    max_density = np.max(env.grid_density)
    cells_above_trigger = np.sum(env.grid_density > env.PANIC_TRIGGER_DENSITY)
    cells_above_target = np.sum(env.grid_density > env.TARGET_DENSITY)
    
    print(f"Step {step+1}: Agents={env.num_agents:3d}, MaxDensity={max_density:.1f}, " +
          f"Cells>{env.PANIC_TRIGGER_DENSITY}={cells_above_trigger}, " +
          f"AvgPanic={np.mean(env.panic[env.alive]):.3f}, MaxPanic={np.max(env.panic[env.alive]):.3f}")
    
    if step == 3:
        # Deep dive into a high-density cell
        print("\n  === DEEP DIVE at Step 4 ===")
        high_cells = np.where(env.grid_density >= env.TARGET_DENSITY)
        
        if len(high_cells[0]) > 0:
            # Pick first high-density cell
            cy, cx = high_cells[0][0], high_cells[1][0]
            cell_density = env.grid_density[cy, cx]
            
            print(f"  High-density cell: ({cx},{cy}) has {cell_density:.1f} agents")
            
            # Find agents in this cell
            agents_in_cell = []
            alive_idx = np.flatnonzero(env.alive)
            for i in alive_idx:
                if int(env.x[i]) == cx and int(env.y[i]) == cy:
                    agents_in_cell.append(i)
            
            print(f"  Found {len(agents_in_cell)} agents in this cell")
            
            if len(agents_in_cell) > 0:
                # Calculate 3x3 neighborhood density for first agent
                i = agents_in_cell[0]
                neighborhood_density = 0.0
                cell_count = 0
                print(f"\n  Agent {i} at ({env.x[i]:.2f}, {env.y[i]:.2f})")
                print(f"  3x3 Neighborhood densities:")
                for oy in range(-1, 2):
                    row = ""
                    for ox in range(-1, 2):
                        ny, nx = cy + oy, cx + ox
                        if 0 <= ny < env.GRID_HEIGHT and 0 <= nx < env.GRID_WIDTH:
                            d = env.grid_density[ny, nx]
                            neighborhood_density += d
                            cell_count += 1
                            row += f"{d:4.1f} "
                        else:
                            row += " OOB "
                    print(f"    {row}")
                
                avg_neighborhood = neighborhood_density / cell_count if cell_count > 0 else 0
                print(f"  Average neighborhood density: {avg_neighborhood:.2f}")
                print(f"  Trigger threshold: {env.PANIC_TRIGGER_DENSITY}")
                print(f"  Would trigger panic: {avg_neighborhood > env.PANIC_TRIGGER_DENSITY}")
                print(f"  Agent's current panic: {env.panic[i]:.3f}")

env.close()

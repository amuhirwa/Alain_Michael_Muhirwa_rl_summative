"""
Enhanced Demo - Simplified Version with Proper Panda3D Integration
===================================================================

This version integrates the simulation properly with Panda3D's task system
to avoid freezing issues.
"""

import sys
import os

from environment.enhanced_env_fast import EnhancedCrowdControlEnvFast
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.enhanced_env import EnhancedCrowdControlEnv
from environment.enhanced_rendering import EnhancedCrowdRenderer
from direct.showbase.ShowBase import ShowBase
import argparse


class CrowdControlDemo(ShowBase):
    """Demo that integrates RL environment with Panda3D properly"""
    
    def __init__(self, pattern='rush', adversarial=False, difficulty='medium', max_steps=500):
        # Don't call ShowBase.__init__ yet - renderer will do it
        
        print("="*70)
        print("ENHANCED CROWD CONTROL - INDIVIDUAL AGENT SIMULATION")
        print("="*70)
        print("\nNOVEL FEATURES DEMONSTRATED:")
        print("  ✓ Individual agents with Social Force Model physics")
        print("  ✓ Panic propagation (blue=calm, orange=stressed, red=panic)")
        print("  ✓ Temporal crowd arrival patterns")
        print("  ✓ Infrastructure constraints (gate delays, barrier cooldowns)")
        if adversarial:
            print("  ✓ Adversarial safety scenarios (gate failures, crowd surges)")
        print("\nSCENARIO CONFIGURATION:")
        print(f"  Pattern: {pattern.upper()}")
        print(f"  Difficulty: {difficulty.upper()}")
        print(f"  Adversarial: {'ENABLED' if adversarial else 'DISABLED'}")
        print("\nCONTROLS:")
        print("  [H] - Toggle heat map visualization")
        print("  [R] - Toggle camera auto-rotation")
        print("  [Arrow Keys] - Move camera")
        print("  [ESC] - Exit")
        print("="*70)
        
        # Create environment WITHOUT rendering
        self.env = EnhancedCrowdControlEnvFast(
            render_mode=None,  # We'll handle rendering separately
            crowd_arrival_pattern=pattern,
            adversarial_mode=adversarial,
            difficulty=difficulty
        )
        
        # Reset environment
        self.obs, self.info = self.env.reset()
        
        print(f"\nStarting simulation with {self.info['total_agents']} initial agents...")
        print(f"Pattern: {self.info['pattern']} | Difficulty: {self.info['difficulty']}")
        
        # Create renderer (this calls ShowBase.__init__)
        self.renderer = EnhancedCrowdRenderer(self.env)
        
        # Simulation state
        self.step_count = 0
        self.max_steps = max_steps
        self.episode_reward = 0
        self.done = False
        self.max_panic_seen = 0
        self.max_density_seen = 0
        
        # Add simulation task to Panda3D's task manager
        self.renderer.taskMgr.add(self.simulation_step, "simulationStep")
        
        # Initial render
        self.update_visualization()
    
    def simulation_step(self, task):
        """Called every frame by Panda3D's task manager"""
        if self.done:
            return task.done
        
        # Take a simulation step
        action = self.env.action_space.sample()  # Random action
        self.obs, reward, terminated, truncated, self.info = self.env.step(action)
        
        self.episode_reward += reward
        self.step_count += 1
        self.done = terminated or truncated or (self.step_count >= self.max_steps)
        
        # Track statistics
        self.max_panic_seen = max(self.max_panic_seen, self.info['max_panic'])
        self.max_density_seen = max(self.max_density_seen, self.info['max_density'])
        
        # Update visualization
        self.update_visualization()
        
        # Print progress every 50 steps
        if self.step_count % 50 == 0:
            print(f"\nStep {self.step_count}:")
            print(f"  Agents: {self.info['total_agents']} | Exited: {self.info['total_exited']}")
            print(f"  Max Density: {self.info['max_density']:.2f} | Avg Panic: {self.info['avg_panic']:.3f}")
            print(f"  Open Gates: {self.info['open_gates']}/{self.env.NUM_GATES}")
            print(f"  Reward: {self.episode_reward:.2f}")
        
        if self.done:
            self.print_final_stats()
            return task.done
        
        return task.cont
    
    def update_visualization(self):
        """Update the 3D visualization"""
        self.renderer.update_scene(
            agents=self.env.agents,
            gates=[tuple(g) for g in self.env.gates],          # list of (x,y,is_open,capacity)
            barriers=[tuple(b) for b in self.env.barriers],    # list of (x,y)
            info=self.info
        )

    
    def print_final_stats(self):
        """Print final statistics"""
        print("\n" + "="*70)
        print("SIMULATION COMPLETE")
        print("="*70)
        print(f"Total Steps: {self.step_count}")
        print(f"Total Reward: {self.episode_reward:.2f}")
        print(f"Total Spawned: {self.info['total_spawned']}")
        print(f"Total Exited: {self.info['total_exited']}")
        print(f"Final Agents: {self.info['total_agents']}")
        print(f"Max Density Reached: {self.max_density_seen:.2f}")
        print(f"Max Panic Reached: {self.max_panic_seen:.3f}")
        print(f"Overcrowding Events: {self.info['overcrowding_events']}")
        print("="*70)
        
        if self.info['overcrowding_events'] == 0:
            print("✓ SUCCESS: No overcrowding events!")
        else:
            print(f"✗ FAILURE: {self.info['overcrowding_events']} overcrowding event(s)")
        
        print("\nWindow will stay open. Press ESC or close window to exit.")


def main():
    parser = argparse.ArgumentParser(description="Enhanced Crowd Control Demo")
    parser.add_argument('--pattern', type=str, default='rush',
                       choices=['steady', 'rush', 'evacuation'],
                       help='Crowd arrival pattern')
    parser.add_argument('--adversarial', action='store_true',
                       help='Enable adversarial scenarios')
    parser.add_argument('--difficulty', type=str, default='medium',
                       choices=['easy', 'medium', 'hard'],
                       help='Difficulty level')
    parser.add_argument('--steps', type=int, default=500,
                       help='Number of steps to run')
    
    args = parser.parse_args()
    
    try:
        demo = CrowdControlDemo(
            pattern=args.pattern,
            adversarial=args.adversarial,
            difficulty=args.difficulty,
            max_steps=args.steps
        )
        demo.renderer.run()  # Start Panda3D's main loop
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Environment closed")


if __name__ == "__main__":
    main()

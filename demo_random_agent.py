"""
Random Agent Demonstration for Crowd Control Environment
========================================================

This script demonstrates the environment visualization with a random agent.
No training is involved - this is purely for demonstrating the GUI capabilities.

Usage:
    python demo_random_agent.py
"""

import sys
import os

from full_script import EnhancedCrowdControlEnv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.custom_env import CrowdControlEnv
import time


def main():
    print("="*60)
    print("CROWD CONTROL ENVIRONMENT - RANDOM AGENT DEMONSTRATION")
    print("="*60)
    print("\nThis demonstrates the environment with random actions.")
    print("No trained model is being used.\n")
    print("Controls:")
    print("  [H] - Toggle heat map visualization")
    print("  [R] - Toggle camera auto-rotation")
    print("  [Arrow Keys] - Move camera")
    print("  [ESC] - Exit")
    print("="*60)
    
    # Create environment with rendering
    env = EnhancedCrowdControlEnv(render_mode='human')
    
    # Reset environment
    obs, info = env.reset()
    
    print(f"\nStarting simulation...")
    print(f"Initial crowd size: {info['total_crowd']:.1f}")
    print(f"Initial max density: {info['max_density']:.2f}")
    
    # Run random agent
    episode_reward = 0
    step = 0
    done = False
    
    try:
        while not done:
            # Render the environment
            env.render()
            
            # Take random action
            action = env.action_space.sample()
            
            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            done = terminated or truncated
            
            # Print periodic updates
            if step % 50 == 0:
                print(f"\nStep {step}:")
                print(f"  Total Crowd: {info['total_crowd']:.1f}")
                print(f"  Max Density: {info['max_density']:.2f}")
                print(f"  Open Gates: {info['open_gates']}/{env.NUM_GATES}")
                print(f"  Cumulative Reward: {episode_reward:.2f}")
            
            # Small delay for visualization
            time.sleep(0.1)
        
        # Final statistics
        print("\n" + "="*60)
        print("SIMULATION COMPLETED")
        print("="*60)
        print(f"Total Steps: {step}")
        print(f"Final Crowd Size: {info['total_crowd']:.1f}")
        print(f"Max Density Reached: {info['max_density_reached']:.2f}")
        print(f"Overcrowding Events: {info['overcrowding_events']}")
        print(f"Total Reward: {episode_reward:.2f}")
        
        if terminated and info['total_crowd'] < 10:
            print("\nResult: SUCCESS - Crowd successfully dispersed!")
        elif terminated:
            print("\nResult: FAILURE - Critical overcrowding occurred!")
        else:
            print("\nResult: TIME LIMIT - Simulation ended at max steps")
        
        print("="*60)
        
        # Keep window open
        input("\nPress Enter to exit...")
        
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
    
    finally:
        env.close()


if __name__ == "__main__":
    main()

"""
Enhanced Crowd Control Environment Demo
========================================

Demonstrates the crowd control environment with 3D visualization.
Shows individual agents with panic-based coloring and interactive controls.

Usage:
    python demo.py [--pattern PATTERN] [--difficulty DIFFICULTY] [--adversarial]

Patterns: rush, steady, evacuation
Difficulty: easy, medium, hard
"""

import argparse
import numpy as np
import time
from typing import Optional

# Import environment and renderer
# Assuming the files are named:
# - enhanced_crowd_control_env.py (contains EnhancedCrowdControlEnvFast)
# - enhanced_rendering.py (contains EnhancedCrowdRenderer)
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.enhanced_env_fast import EnhancedCrowdControlEnvFast

from environment.enhanced_rendering import EnhancedCrowdRenderer

# Import RL libraries for model loading
try:
    from stable_baselines3 import PPO, DQN, A2C, SAC
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: stable-baselines3 not available. Model loading disabled.")


def load_sb3_model(model_path: str, env):
    """Load a Stable-Baselines3 model"""
    if not SB3_AVAILABLE:
        raise ImportError("stable-baselines3 not installed")
    
    # Try to detect algorithm from path
    if 'ppo' in model_path.lower():
        return PPO.load(model_path, env=env)
    elif 'dqn' in model_path.lower():
        return DQN.load(model_path, env=env)
    elif 'a2c' in model_path.lower():
        return A2C.load(model_path, env=env)
    elif 'sac' in model_path.lower():
        return SAC.load(model_path, env=env)
    else:
        # Try PPO as default
        try:
            return PPO.load(model_path, env=env)
        except:
            try:
                return DQN.load(model_path, env=env)
            except:
                raise ValueError(f"Could not load model from {model_path}")


class CrowdControlDemo:
    """Demo runner (now supports RL model policy)"""
    
    def __init__(self, 
                 pattern="rush",
                 difficulty="medium",
                 adversarial=False,
                 use_renderer=True,
                 model_path: Optional[str] = None):

        self.pattern = pattern
        self.difficulty = difficulty
        self.adversarial = adversarial
        self.use_renderer = use_renderer

        print(f"Creating environment:")
        print(f"  Pattern: {pattern}")
        print(f"  Difficulty: {difficulty}")
        print(f"  Adversarial: {adversarial}")
        print()

        self.env = EnhancedCrowdControlEnvFast(
            render_mode="human" if use_renderer else None,
            crowd_arrival_pattern=pattern,
            adversarial_mode=adversarial,
            difficulty=difficulty
        )

        # Renderer
        self.renderer = None
        if use_renderer:
            try:
                self.renderer = EnhancedCrowdRenderer(self.env)
                print("3D Renderer initialized successfully.\n")
            except Exception as e:
                print(f"Renderer init failed: {e}")
                self.use_renderer = False

        # Load RL model if provided
        self.model = None
        if model_path:
            print(f"Loading model from: {model_path}")
            self.model = load_sb3_model(model_path, self.env)
            print("Model loaded!\n")

        self.episode_count = 0
        self.best_score = float('-inf')

    # ========================================================================
    # NEW: RL Model Policy
    # ========================================================================
    def run_model_policy(self, num_episodes=1, max_steps=None):

        if self.model is None:
            raise RuntimeError("No model loaded! Use --model-path <file>")

        print(f"Running {num_episodes} episodes with model policy...")
        print("=" * 60)

        for episode in range(num_episodes):
            self.episode_count += 1
            obs, info = self.env.reset()

            episode_reward = 0
            step = 0
            done = False

            print(f"\n[Episode {self.episode_count}]")

            while not done:

                # Model chooses action
                action, _ = self.model.predict(obs, deterministic=True)

                # Step environment
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated
                step += 1

                # Renderer update
                if self.renderer:
                    self.renderer.update_scene(
                        agents=self.env.agents,
                        gates=self.env.gates,
                        barriers=self.env.barriers,
                        info=info,
                    )
                    self.renderer.taskMgr.step()
                    time.sleep(0.05)  # INCREASED: Was 0.01 - slower visualization (50ms per frame)

                if step % 50 == 0:
                    self._print_status(step, info, episode_reward, action)

                if max_steps and step >= max_steps:
                    done = True

            self._print_episode_summary(step, episode_reward, info)
            self.best_score = max(self.best_score, episode_reward)

        print("\nDemo complete! Best score:", self.best_score)
    
    def run_random_policy(self, num_episodes: int = 3, max_steps: Optional[int] = None):
        """
        Run episodes with a random policy
        
        Args:
            num_episodes: Number of episodes to run
            max_steps: Maximum steps per episode (None = use env default)
        """
        print(f"Running {num_episodes} episodes with random policy...")
        print("=" * 60)
        
        for episode in range(num_episodes):
            self.episode_count += 1
            obs, info = self.env.reset()
            
            episode_reward = 0
            step = 0
            done = False
            
            print(f"\n[Episode {self.episode_count}]")
            
            while not done:
                # Random action
                action = self.env.action_space.sample()
                
                # Step environment
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                step += 1
                done = terminated or truncated
                
                # Update renderer
                if self.renderer is not None:
                    self.renderer.update_scene(
                        agents=self.env.agents,
                        gates=self.env.gates,
                        barriers=self.env.barriers,
                        info=info
                    )
                    
                    # Process Panda3D events
                    self.renderer.taskMgr.step()
                
                # Print progress every 50 steps
                if step % 50 == 0:
                    self._print_status(step, info, episode_reward, action)
                
                # Respect max_steps if provided
                if max_steps and step >= max_steps:
                    done = True
                
                # Small delay for visualization
                if self.renderer is not None:
                    time.sleep(0.05)
            
            # Episode summary
            self._print_episode_summary(step, episode_reward, info)
            
            if episode_reward > self.best_score:
                self.best_score = episode_reward
        
        print("\n" + "=" * 60)
        print(f"Demo complete! Best score: {self.best_score:.2f}")
    
    def run_heuristic_policy(self, num_episodes: int = 3, max_steps: Optional[int] = None):
        """
        Run episodes with a simple heuristic policy
        
        Heuristic strategy:
        - Open all gates when density is high
        - Close gates when crowd is manageable
        - Move barriers away from high-density areas
        """
        print(f"Running {num_episodes} episodes with heuristic policy...")
        print("=" * 60)
        
        for episode in range(num_episodes):
            self.episode_count += 1
            obs, info = self.env.reset()
            
            episode_reward = 0
            step = 0
            done = False
            
            print(f"\n[Episode {self.episode_count}]")
            
            while not done:
                # Heuristic decision making
                max_density = info.get('max_density', 0)
                avg_panic = info.get('avg_panic', 0)
                num_agents = info.get('agents', 0)
                
                # Decision logic
                if max_density > self.env.CRITICAL_DENSITY * 0.7 or avg_panic > 0.6:
                    # Emergency: open all gates
                    action = 11
                elif max_density > self.env.TARGET_DENSITY * 1.5:
                    # High density: open a gate or move barrier
                    if np.random.random() < 0.7:
                        action = np.random.randint(4, 7)  # Toggle gate
                    else:
                        action = np.random.randint(0, 4)  # Move barrier
                elif num_agents < 30:
                    # Low crowd: maintain current state
                    action = 10  # No-op (or minimal intervention)
                else:
                    # Normal operation: occasional adjustments
                    action = np.random.choice([10, np.random.randint(0, 4)])
                
                # Step environment
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                step += 1
                done = terminated or truncated
                
                # Update renderer
                if self.renderer is not None:
                    self.renderer.update_scene(
                        agents=self.env.agents,
                        gates=self.env.gates,
                        barriers=self.env.barriers,
                        info=info
                    )
                    self.renderer.taskMgr.step()
                
                # Print progress
                if step % 50 == 0:
                    self._print_status(step, info, episode_reward, action)
                
                if max_steps and step >= max_steps:
                    done = True
                
                if self.renderer is not None:
                    time.sleep(0.05)
            
            self._print_episode_summary(step, episode_reward, info)
            
            if episode_reward > self.best_score:
                self.best_score = episode_reward
        
        print("\n" + "=" * 60)
        print(f"Demo complete! Best score: {self.best_score:.2f}")
    
    def _print_status(self, step: int, info: dict, reward: float, action):
        """Print current status"""
        print(f"  Step {step:3d} | "
              f"Agents: {info.get('agents', 0):3d} | "
              f"Exited: {info.get('exited', 0):3d} | "
              f"Density: {info.get('max_density', 0):.2f} | "
              f"Panic: {info.get('avg_panic', 0):.2f} | "
              f"Reward: {reward:7.2f} | "
              f"Action: {action:7.2f}")
    
    def _print_episode_summary(self, steps: int, reward: float, info: dict):
        """Print episode summary"""
        print(f"\n  Episode Summary:")
        print(f"    Steps: {steps}")
        print(f"    Total Reward: {reward:.2f}")
        print(f"    Agents Spawned: {info.get('spawned', 0)}")
        print(f"    Agents Exited: {info.get('exited', 0)}")
        print(f"    Final Density: {info.get('max_density', 0):.2f}")
        print(f"    Overcrowding Events: {info.get('overcrowding_events', 0)}")
        
        # Termination reason
        if hasattr(self.env, '_terminal_reason'):
            print(f"    Termination: {self.env._terminal_reason if self.env._terminal_reason else 'Survived'}")
    
    def close(self):
        """Clean up resources"""
        if self.renderer is not None:
            self.renderer.close()
        self.env.close()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Enhanced Crowd Control Environment Demo"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="rush",
        choices=["rush", "steady", "evacuation"],
        help="Crowd arrival pattern"
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="medium",
        choices=["easy", "medium", "hard"],
        help="Difficulty level"
    )
    parser.add_argument(
        "--adversarial",
        action="store_true",
        help="Enable adversarial scenarios"
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="random",
        choices=["random", "heuristic"],
        help="Policy to use for demo"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to run"
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable 3D rendering"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model (.zip file)"
    )
    
    args = parser.parse_args()
    
    # Create demo
    demo = CrowdControlDemo(
        pattern=args.pattern,
        difficulty=args.difficulty,
        adversarial=args.adversarial,
        use_renderer=not args.no_render,
        model_path=args.model
    )
    
    try:
        # Run demo with selected policy
        if args.model:
            # Use trained model
            demo.run_model_policy(
                num_episodes=args.episodes,
                max_steps=args.max_steps
            )
        elif args.policy == "random":
            demo.run_random_policy(
                num_episodes=args.episodes,
                max_steps=args.max_steps
            )
        else:
            demo.run_heuristic_policy(
                num_episodes=args.episodes,
                max_steps=args.max_steps
            )
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    finally:
        demo.close()


if __name__ == "__main__":
    main()
"""
Curriculum Learning for Enhanced Crowd Control
==============================================

Progressive difficulty training with scenario diversity.

NOVEL CONTRIBUTION: Trains models through increasing difficulty stages,
starting with simple steady flows and progressing to complex rush scenarios
and adversarial safety-critical situations.

Training Stages:
1. Easy: 100 agents, steady pattern, no adversarial
2. Medium: 150 agents, rush pattern, no adversarial  
3. Hard: 200 agents, rush pattern, adversarial enabled

This curriculum approach allows the agent to learn basic crowd management
before facing complex emergency scenarios.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.enhanced_env_fast import EnhancedCrowdControlEnvFast

from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import numpy as np
from typing import Dict, Tuple
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

def make_vec_env(difficulty='easy', pattern='steady', adversarial=False, n_envs=4):
    """
    Create a vectorized environment with n_envs parallel environments.
    """
    def make_env():
        def _init():
            env = create_env_with_curriculum(difficulty, pattern, adversarial)
            return Monitor(env)
        return _init

    env_fns = [make_env() for _ in range(n_envs)]
    # Use SubprocVecEnv for true parallelism or DummyVecEnv for simple vectorization
    vec_env = SubprocVecEnv(env_fns)
    return vec_env


def create_env_with_curriculum(difficulty='easy', pattern='steady', adversarial=False):
    """
    Create environment with specific difficulty settings
    
    NOVEL: Parameterized difficulty for curriculum learning
    """
    env = EnhancedCrowdControlEnvFast(
        crowd_arrival_pattern=pattern,
        adversarial_mode=adversarial,
        difficulty=difficulty
    )
    return env


def train_with_curriculum_ppo(
    total_timesteps=300000,
    algorithm='PPO',
    learning_rate=3e-4,
    save_dir='models/curriculum'
):
    """
    Train PPO agent with curriculum learning
    
    NOVEL CONTRIBUTION: Three-stage training with progressive difficulty
    
    Args:
        total_timesteps: Total training steps (divided across stages)
        algorithm: RL algorithm to use
        learning_rate: Learning rate
        save_dir: Directory to save models
    """
    
    print("="*70)
    print("CURRICULUM LEARNING FOR CROWD CONTROL")
    print("="*70)
    print(f"\nAlgorithm: {algorithm}")
    print(f"Total Steps: {total_timesteps:,}")
    print(f"Learning Rate: {learning_rate}")
    print("\nTraining Stages:")
    print("  1. Easy: 100 agents, steady flow")
    print("  2. Medium: 150 agents, rush scenario")
    print("  3. Hard: 200 agents, rush + adversarial")
    print("="*70)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Give more time to early stages where agents learn basics
    stages = [
        # {
        #     'name': 'easy',
        #     'difficulty': 'easy',
        #     'pattern': 'steady',
        #     'adversarial': False,
        #     'timesteps': int(total_timesteps * 0.4),
        #     'description': 'Basic crowd management - learn to avoid overcrowding'
        # },
        {
            'name': 'medium',
            'difficulty': 'medium',
            'pattern': 'rush',
            'adversarial': False,
            'timesteps': int(total_timesteps * 0.7),
            'description': 'Rush hour scenarios - handle crowd surges'
        },
        {
            'name': 'hard',
            'difficulty': 'hard',
            'pattern': 'rush',
            'adversarial': True,
            'timesteps': int(total_timesteps * 0.3),
            'description': 'Adversarial safety testing - handle emergencies'
        }
    ]
    
    model = None
    
    for stage_idx, stage in enumerate(stages):
        print(f"\n{'='*70}")
        print(f"STAGE {stage_idx + 1}/3: {stage['name'].upper()}")
        print(f"Description: {stage['description']}")
        print(f"Timesteps: {stage['timesteps']:,}")
        print(f"Pattern: {stage['pattern']} | Adversarial: {stage['adversarial']}")
        print('='*70)
        
        # Create environment for this stage
        env = make_vec_env(
            difficulty=stage['difficulty'],
            pattern=stage['pattern'],
            adversarial=stage['adversarial'],
            n_envs=8  # for example, 8 parallel environments
        )
        
        # Create or update model
        if model is None:
            # First stage - create new model
            print(f"\nCreating new {algorithm} model...")
            if algorithm == 'PPO':
                # BEST CONFIG: config_9_aggressive (1157.7 mean reward)
                model = PPO(
                    "MlpPolicy",
                    env,
                    learning_rate=0.001,  # 3x higher than baseline - enables faster learning in dense-reward environment
                    n_steps=1024,         # Balanced rollout - captures episode structure without excessive memory
                    batch_size=128,       # 2x larger - improves gradient stability across diverse scenarios
                    n_epochs=15,          # 50% more updates - maximizes sample efficiency from experience buffer
                    gamma=0.995,          # Higher discount - better credit assignment for 500-step episodes
                    gae_lambda=0.98,      # Near-full advantage - reduces bias in long-horizon tasks
                    clip_range=0.3,       # Larger trust region - allows bolder policy updates when gradient is clear
                    vf_coef=0.6,          # Increased value weight - improves baseline for advantage estimation
                    ent_coef=0.18,        # 18x baseline - strong exploration to discover barrier/gate strategies
                    verbose=1,            # Show progress
                    tensorboard_log=f"logs/tensorboard/curriculum_{algorithm}"
                )
            elif algorithm == 'DQN':
                # BEST CONFIG: config_5_fast_target_update (1213.5 mean reward)
                model = DQN(
                    "MlpPolicy",
                    env,
                    learning_rate=0.001,           # 3x higher - Q-network adapts quickly to non-stationary crowd dynamics
                    buffer_size=100000,            # Large replay - captures diverse crowd configurations for robust learning
                    learning_starts=5000,          # Early start - begins learning after seeing sufficient state diversity
                    batch_size=64,                 # 2x larger - reduces mini-batch variance in Q-value estimation
                    tau=1.0,                       # Hard update - prevents target network staleness in changing environment
                    gamma=0.995,                   # High discount - values long-term crowd dispersal over immediate rewards
                    target_update_interval=500,    # 2x faster updates - CRITICAL for tracking rapidly evolving optimal policy
                    exploration_fraction=0.5,      # Extended exploration - finds strategic barrier positions and gate timings
                    exploration_initial_eps=1.0,
                    exploration_final_eps=0.05,    # Maintains 5% exploration - prevents convergence to local optima
                    verbose=1,
                    tensorboard_log=f"logs/tensorboard/curriculum_{algorithm}"
                )
            elif algorithm == 'A2C':
                # BEST CONFIG: config_2_high_lr (1289.6 mean reward - WINNER!)
                model = A2C(
                    "MlpPolicy",
                    env,
                    learning_rate=0.001,  # High LR essential - A2C's synchronous updates handle aggressive gradients well
                    n_steps=5,            # Very short rollout - provides immediate feedback in dense-reward environment
                    gamma=0.99,           # Standard discount - sufficient for episodes that rarely reach 500 steps
                    gae_lambda=1.0,       # No TD bias - pure Monte Carlo advantage = lower variance with short rollouts
                    vf_coef=0.5,          # Balanced value/policy - prevents value function from dominating gradient
                    ent_coef=0.01,        # Minimal entropy - exploitation-focused after exploration phase discovers strategies
                    rms_prop_eps=1e-05,   # RMSProp stability - prevents division by zero in gradient scaling
                    verbose=1,            # Show progress
                    tensorboard_log=f"logs/tensorboard/curriculum_{algorithm}"
                )
        else:
            # Continue training with new environment
            print(f"\nContinuing training on {stage['name']} stage...")
            model.set_env(env)
        
        # Setup callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=f"{save_dir}/{algorithm}_{stage['name']}",
            name_prefix=f"{algorithm}_curriculum"
        )
        
        # Train for this stage
        print(f"\nStarting training for {stage['timesteps']:,} steps...")
        model.learn(
            total_timesteps=stage['timesteps'],
            callback=checkpoint_callback,
            progress_bar=True,
            reset_num_timesteps=False  # Continue timestep count
        )
        
        # Save stage model
        stage_model_path = f"{save_dir}/{algorithm}_curriculum_{stage['name']}.zip"
        model.save(stage_model_path)
        print(f"✓ Stage {stage_idx + 1} complete. Model saved: {stage_model_path}")
        
        env.close()
    
    # Save final model
    final_model_path = f"{save_dir}/{algorithm}_curriculum_final.zip"
    model.save(final_model_path)
    
    print("\n" + "="*70)
    print("CURRICULUM TRAINING COMPLETE")
    print("="*70)
    print(f"Final model saved: {final_model_path}")
    print("\nStage models:")
    for stage in stages:
        print(f"  - {algorithm}_curriculum_{stage['name']}.zip")
    print("="*70)
    
    return model, final_model_path


def train_curriculum_all_algorithms(total_timesteps=300000):
    """
    Train all algorithms with curriculum learning
    
    NOVEL: Systematic comparison across algorithms with curriculum
    """
    algorithms = ['PPO', 'DQN', 'A2C']
    results = {}
    
    for algo in algorithms:
        print(f"\n\n{'#'*70}")
        print(f"# TRAINING {algo} WITH CURRICULUM LEARNING")
        print(f"{'#'*70}\n")
        
        try:
            model, model_path = train_with_curriculum_ppo(
                total_timesteps=total_timesteps,
                algorithm=algo,
                save_dir=f'models/curriculum_{algo.lower()}'
            )
            results[algo] = {'success': True, 'model_path': model_path}
        except Exception as e:
            print(f"\n✗ Error training {algo}: {e}")
            results[algo] = {'success': False, 'error': str(e)}
    
    # Print summary
    print("\n" + "="*70)
    print("CURRICULUM TRAINING SUMMARY")
    print("="*70)
    for algo, result in results.items():
        if result['success']:
            print(f"✓ {algo}: Successfully trained")
            print(f"  Model: {result['model_path']}")
        else:
            print(f"✗ {algo}: Failed - {result.get('error', 'Unknown error')}")
    print("="*70)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Curriculum learning for crowd control')
    parser.add_argument('--algorithm', type=str, default='PPO', 
                       choices=['PPO', 'DQN', 'A2C', 'all'],
                       help='Algorithm to train (default: PPO)')
    parser.add_argument('--timesteps', type=int, default=750000,
                       help='Total timesteps (default: 750k - 5x original training for demo quality)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001 - proven winner across all algorithms)')
    
    args = parser.parse_args()
    
    print(f"\n🎓 Starting Curriculum Training")
    print(f"   Algorithm: {args.algorithm}")
    print(f"   Total Timesteps: {args.timesteps:,}")
    print(f"   Breakdown: Easy={int(args.timesteps*0.4):,}, Medium={int(args.timesteps*0.4):,}, Hard={int(args.timesteps*0.2):,}")
    print(f"   Learning Rate: {args.learning_rate}\n")
    
    if args.algorithm == 'all':
        train_curriculum_all_algorithms(total_timesteps=args.timesteps)
    else:
        train_with_curriculum_ppo(
            total_timesteps=args.timesteps,
            algorithm=args.algorithm,
            learning_rate=args.learning_rate,
            save_dir=f'models/curriculum_{args.algorithm.lower()}'
        )

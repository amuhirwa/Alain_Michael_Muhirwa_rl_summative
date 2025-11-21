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

from environment.custom_env import CrowdControlEnv
from environment.enhanced_env import EnhancedCrowdControlEnv
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
    env = CrowdControlEnv(
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
    
    # Define curriculum stages
    stages = [
        {
            'name': 'easy',
            'difficulty': 'easy',
            'pattern': 'steady',
            'adversarial': False,
            'timesteps': total_timesteps // 3,
            'description': 'Basic crowd management'
        },
        {
            'name': 'medium',
            'difficulty': 'medium',
            'pattern': 'rush',
            'adversarial': False,
            'timesteps': total_timesteps // 3,
            'description': 'Rush hour scenarios'
        },
        {
            'name': 'hard',
            'difficulty': 'hard',
            'pattern': 'rush',
            'adversarial': True,
            'timesteps': total_timesteps // 3,
            'description': 'Adversarial safety testing'
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
                model = PPO(
                    "MlpPolicy",
                    env,
                    learning_rate=learning_rate,
                    n_steps=256,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    verbose=1,
                    tensorboard_log=f"logs/tensorboard/curriculum_{algorithm}"
                )
            elif algorithm == 'DQN':
                model = DQN(
                    "MlpPolicy",
                    env,
                    learning_rate=learning_rate,
                    buffer_size=100000,
                    learning_starts=10000,
                    batch_size=32,
                    tau=1.0,
                    gamma=0.99,
                    target_update_interval=1000,
                    verbose=1,
                    tensorboard_log=f"logs/tensorboard/curriculum_{algorithm}"
                )
            elif algorithm == 'A2C':
                model = A2C(
                    "MlpPolicy",
                    env,
                    learning_rate=learning_rate,
                    n_steps=5,
                    gamma=0.99,
                    gae_lambda=0.95,
                    verbose=1,
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
    train_curriculum_all_algorithms(total_timesteps=5000000)

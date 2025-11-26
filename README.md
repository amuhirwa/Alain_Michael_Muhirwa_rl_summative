# 🎯 Crowd Control Reinforcement Learning System

<div align="center">

**Intelligent Crowd Management and Safety Optimization using Deep Reinforcement Learning**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-2.2.1-green.svg)](https://stable-baselines3.readthedocs.io/)
[![Panda3D](https://img.shields.io/badge/Panda3D-1.10.14-orange.svg)](https://www.panda3d.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29.1-red.svg)](https://gymnasium.farama.org/)

**Author:** Alain Michael Muhirwa  
**Course:** Reinforcement Learning Summative Assignment

</div>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Environment Details](#-environment-details)
- [Implemented Algorithms](#-implemented-algorithms)
- [Results & Performance](#-results--performance)
- [3D Visualization](#-3d-visualization)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [References](#-references)

---

## 🌟 Project Overview

This project implements a sophisticated **Crowd Control System** using Reinforcement Learning to manage crowd flow in venues, prevent overcrowding at gates/exits, and ensure public safety. The system trains and compares **four different RL algorithms** (DQN, PPO, A2C, REINFORCE) across **40+ hyperparameter configurations** to find the optimal crowd management strategy.

### 🎯 Problem Statement

In crowded venues (stadiums, concerts, public events), dangerous overcrowding can occur at gates and exits, leading to safety hazards and potential stampedes. This RL system acts as an intelligent crowd control manager that:

- **Monitors** real-time crowd density across a 15×15 grid venue
- **Controls** 4 movable barriers to redirect crowd flow
- **Manages** 3 exit gates to optimize throughput
- **Prevents** dangerous overcrowding (density > 3.5 people/cell)
- **Mitigates** panic propagation through strategic interventions
- **Ensures** safe and efficient crowd evacuation

### 🏆 Key Results

| Algorithm  | Best Config       | Mean Reward | Convergence  | Ranking |
| ---------- | ----------------- | ----------- | ------------ | ------- |
| **A2C** ⭐ | config_2_high_lr  | **1289.6**  | 39,000 steps | 🥇 1st  |
| PPO        | config_2_high_lr  | 1010.4      | 65,536 steps | 🥈 2nd  |
| REINFORCE  | config_1_baseline | 879.1       | ~48 episodes | 🥉 3rd  |
| DQN        | config_1_baseline | 703.6       | 52,000 steps | 4th     |

---

## 🏗️ Environment Details

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CROWD CONTROL AGENT                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌───────────┐ │
│  │    DQN     │  │    PPO     │  │    A2C     │  │ REINFORCE │ │
│  │  (Value)   │  │  (Policy)  │  │  (Actor-   │  │ (Monte    │ │
│  │            │  │  Gradient) │  │   Critic)  │  │  Carlo)   │ │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └─────┬─────┘ │
│        └───────────────┴───────────────┴───────────────┘        │
│                                │                                 │
│                    ┌───────────▼───────────┐                    │
│                    │   ACTION SELECTION    │                    │
│                    │   (25 Discrete)       │                    │
│                    └───────────┬───────────┘                    │
└────────────────────────────────┼────────────────────────────────┘
                                 │
           ┌─────────────────────▼─────────────────────┐
           │      CROWD CONTROL ENVIRONMENT            │
           │  ┌─────────────────────────────────────┐  │
           │  │  • 15×15 Grid (225 cells)           │  │
           │  │  • Social Force Model Physics       │  │
           │  │  • JIT-Accelerated (Numba)          │  │
           │  │  • Panic Propagation Dynamics       │  │
           │  │  • 4 Movable Barriers               │  │
           │  │  • 3 Controllable Gates             │  │
           │  │  • 5 Spawn Entrances                │  │
           │  │  • 80-160 Max Agents (difficulty)   │  │
           │  └─────────────────────────────────────┘  │
           └─────────────────────┬─────────────────────┘
                                 │
           ┌─────────────────────▼─────────────────────┐
           │        PANDA3D 3D VISUALIZATION           │
           │  • Real-time Humanoid Agent Rendering     │
           │  • Panic-based Color Coding               │
           │  • Density Heat Maps                      │
           │  • Interactive Camera Controls            │
           │  • Performance HUD Overlay                │
           └───────────────────────────────────────────┘
```

### Observation Space (918 dimensions)

The agent observes a normalized feature vector containing:

| Component         | Dimensions  | Description                                          |
| ----------------- | ----------- | ---------------------------------------------------- |
| Grid Density      | 15×15 = 225 | Crowd count per cell, normalized by MAX_CROWD (10.0) |
| Panic Grid        | 15×15 = 225 | Panic level per cell, range [0, 1]                   |
| Velocity X        | 15×15 = 225 | Crowd movement X-vectors, normalized                 |
| Velocity Y        | 15×15 = 225 | Crowd movement Y-vectors, normalized                 |
| Gate States       | 3×2 = 6     | Position (x, y) and open/closed status               |
| Barrier Positions | 4×3 = 12    | Position (x, y) and cooldown status                  |
| Global Statistics | 3           | Avg density, timestep ratio, agents ratio            |

**Total:** 918 features (normalized to [0, 1])

### Action Space (25 Discrete Actions)

| Action ID | Category           | Description                                    |
| --------- | ------------------ | ---------------------------------------------- |
| 0-15      | Barrier Movement   | 4 barriers × 4 directions (up/down/left/right) |
| 16-18     | Gate Toggle        | Open/close 3 exit gates independently          |
| 19-22     | Flow Direction     | Nudge crowd flow (up/down/left/right)          |
| 23        | Emergency Response | Open all gates simultaneously                  |
| 24        | No-op              | Maintain current configuration                 |

### Reward Structure

The reward function balances multiple objectives with the following components:

```
Total Reward = Density_Reward × 2.0
             + Efficiency_Reward × 2.5
             + Safety_Reward × 0.5
             - Infrastructure_Cost
             + Survival_Bonus
             + Milestone_Bonuses
             + Success_Reward (terminal)
```

| Component              | Description                                                       | Range              |
| ---------------------- | ----------------------------------------------------------------- | ------------------ |
| **Throughput**         | +2.0 per person exiting + flow efficiency bonus                   | +0.5 to +2.5       |
| **Density Management** | Reward for staying below target (1.5), penalties for overcrowding | -5.0 to +5.0       |
| **Panic Regulation**   | Penalty for high panic, bonus for calm crowds                     | -0.2 to +0.05      |
| **Survival Bonus**     | +5.0 per safe timestep (encourages longevity)                     | +5.0/step          |
| **Milestone Bonuses**  | Rewards at steps 100/200/300/400                                  | +50/+100/+150/+200 |
| **Success Reward**     | Large terminal reward for successful evacuation                   | +400 to +1000      |
| **Failure Penalty**    | Critical density exceeded (>3.5)                                  | -10 to -20         |

### Environment Parameters

| Parameter             | Value    | Description                      |
| --------------------- | -------- | -------------------------------- |
| Grid Size             | 15×15    | Venue dimensions                 |
| Max Steps             | 500      | Episode timeout                  |
| Critical Density      | 3.5      | Dangerous overcrowding threshold |
| Target Density        | 1.5      | Optimal comfort zone             |
| Panic Trigger         | 2.0      | Density that triggers panic      |
| Gate Transition Delay | 10 steps | Cooldown for gate operations     |
| Barrier Move Cost     | 5 steps  | Cooldown for barrier movement    |

### Difficulty Levels

| Difficulty | Max Agents | Rush Peak Time | Description                       |
| ---------- | ---------- | -------------- | --------------------------------- |
| Easy       | 80         | 0.4            | Manageable crowd, gradual arrival |
| Medium     | 120        | 0.3            | Moderate challenge, faster peak   |
| Hard       | 160        | 0.25           | Dense crowds, rapid influx        |

### Crowd Arrival Patterns

- **Rush**: Gaussian peak at rush_peak_time (concerts, events)
- **Steady**: Constant flow until 80% of episode
- **Evacuation**: Massive initial spawn with high panic

---

## 🤖 Implemented Algorithms

### 1. DQN (Deep Q-Network) - Value-Based

**Architecture:** 3-layer fully connected network [256, 256] with Dueling DQN architecture

**Key Features:**

- Experience Replay (50k-100k buffer)
- Target Network (soft updates every 1000 steps)
- ε-greedy exploration (1.0 → 0.2-0.25 over 80% training)

**Best Configuration (config_1_baseline):**

```python
learning_rate = 1e-4
buffer_size = 50000
batch_size = 32
gamma = 0.99
exploration_fraction = 0.8
exploration_final_eps = 0.2
```

**Result:** 703.6 mean reward

---

### 2. PPO (Proximal Policy Optimization) - Policy Gradient

**Architecture:** Shared feature extractor [256, 256] with separate policy/value heads

**Key Features:**

- Clipped Surrogate Objective (trust region: 0.2-0.3)
- Generalized Advantage Estimation (GAE λ=0.95)
- Mini-batch updates (10 epochs × 64 batch)

**Best Configuration (config_2_high_lr):**

```python
learning_rate = 1e-3      # Aggressive - enabled by dense rewards
n_steps = 2048
batch_size = 64
n_epochs = 10
gae_lambda = 0.95
clip_range = 0.2
ent_coef = 0.01
```

**Result:** 1010.4 mean reward

---

### 3. A2C (Advantage Actor-Critic) - Policy Gradient ⭐ BEST

**Architecture:** Shared backbone [256, 256] with synchronous updates (4 parallel envs)

**Key Features:**

- Synchronous updates with 5-step TD returns
- RMSprop optimizer for adaptive learning rates
- Fast policy updates for dense reward exploitation

**Best Configuration (config_2_high_lr) - CHAMPION:**

```python
learning_rate = 1e-3      # High LR - A2C handles aggressive gradients
n_steps = 5               # Very short rollouts - immediate feedback
gamma = 0.99
gae_lambda = 1.0          # Full Monte Carlo advantage
vf_coef = 0.5
ent_coef = 0.01           # Minimal entropy - exploitation focused
```

**Result:** 1289.6 mean reward (HIGHEST)

**Why A2C Won:**

- Short rollouts (5 steps) enabled rapid policy iteration
- Dense reward signal exploited effectively with high learning rate
- Synchronous updates produced stable gradient flows

---

### 4. REINFORCE - Monte Carlo Policy Gradient

**Architecture:** Custom 2-layer policy network [128, 128] with optional baseline

**Key Features:**

- Monte Carlo returns (full episode rollouts)
- Baseline variance reduction (value network)
- Entropy regularization (0.01-0.1)

**Best Configuration (config_1_baseline):**

```python
learning_rate = 1e-3
gamma = 0.99
entropy_coef = 0.01
use_baseline = True
hidden_dims = [128, 128]
```

**Result:** 879.1 mean reward

---

## 📊 Results & Performance

### Training Comparison

| Metric              | DQN           | PPO      | A2C         | REINFORCE           |
| ------------------- | ------------- | -------- | ----------- | ------------------- |
| Best Mean Reward    | 703.6         | 1010.4   | **1289.6**  | 879.1               |
| Convergence (steps) | 52,000        | 65,536   | **39,000**  | ~48 episodes        |
| Training Stability  | Moderate      | High     | **Highest** | Low (high variance) |
| Sample Efficiency   | Good (replay) | Moderate | Good        | Low                 |

### Hyperparameter Configurations Tested

Each algorithm was trained with **10 configurations** for comprehensive tuning:

- **DQN:** Learning rates, buffer sizes, exploration strategies, batch sizes
- **PPO:** N-steps, clip ranges, entropy coefficients, GAE lambda
- **A2C:** N-steps, entropy coefficients, value function weights
- **REINFORCE:** Baselines, network architectures, entropy regularization

**Total Training Runs:** 40+ configurations

### Key Findings

1. **On-Policy Methods Dominate:** A2C and PPO outperformed DQN due to dense reward signals benefiting immediate policy updates

2. **High Learning Rates Work:** 1e-3 learning rate succeeded for both A2C and PPO, contradicting typical conservative RL tuning

3. **Short Rollouts Win:** A2C's 5-step rollouts enabled fastest convergence (39,000 steps)

4. **Exploration Matters:** DQN with extended exploration (ε=0.2 final) outperformed aggressive decay

5. **Generalization Gap:** All algorithms struggled with evacuation scenarios (panic-dominated), indicating training distribution mismatch

### Generalization Performance

Models tested on unseen scenarios (3 patterns × 3 difficulties):

| Pattern    | Easy          | Medium        | Hard          |
| ---------- | ------------- | ------------- | ------------- |
| Rush       | A2C: 2739.4   | A2C: 703.7    | PPO: 100.4    |
| Steady     | DQN: 2000+    | All: ~600     | All: <100     |
| Evacuation | All: Negative | All: Negative | All: Negative |

---

## 🎨 3D Visualization

### Panda3D Rendering Features

The system features **high-quality 3D visualization** using Panda3D with GLTF humanoid models:

#### Visual Elements

1. **Crowd Agents (Humanoid 3D Models)**

   - 🔵 **Blue:** Calm agents (panic < 0.3)
   - 🟡 **Yellow:** Stressed agents (0.3 ≤ panic < 0.7)
   - 🔴 **Red:** Panicking agents (panic ≥ 0.7)

2. **Infrastructure**

   - 🟨 **Yellow Blocks:** Movable barriers
   - 🟩 **Green Squares:** Open gates
   - 🟥 **Red Squares:** Closed gates
   - ⬜ **Translucent Gray:** Venue walls

3. **Environment**

   - 15×15 grid floor
   - Ambient + directional + point lighting
   - Real-time density heat map overlay

4. **HUD Display**
   - Current timestep
   - Agent count (alive/exited/total)
   - Maximum density
   - Panic levels
   - Gate status
   - Cumulative reward

#### Interactive Controls

| Key                  | Action                        |
| -------------------- | ----------------------------- |
| `H`                  | Toggle heat map visualization |
| `R`                  | Toggle camera auto-rotation   |
| `Arrow Keys`         | Move camera horizontally      |
| `Q/E` or `PgUp/PgDn` | Move camera vertically        |
| `ESC`                | Exit simulation               |

---

## 🚀 Installation

### Prerequisites

- Python 3.11+
- Windows/Linux/Mac
- (Optional) CUDA-capable GPU for faster training

### Quick Start with uv (Recommended)

```bash
# Clone repository
git clone https://github.com/amuhirwa/Alain_Michael_Muhirwa_rl_summative.git
cd Alain_Michael_Muhirwa_rl_summative

# Install with uv (fastest)
uv sync

# Verify installation
uv run python -c "import gymnasium; import panda3d; print('✅ Setup successful!')"
```

### Alternative: pip installation

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
gymnasium==0.29.1
stable-baselines3==2.2.1
torch==2.1.0
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
panda3d==1.10.14
tensorboard==2.15.1
numba  # JIT acceleration
```

---

## 💻 Usage

### 1. Run Best Trained Model (A2C Champion)

```bash
# Run with 3D visualization
uv run python main.py --model models/a2c/config_2_high_lr/best_model.zip --episodes 3

# Run for longer episodes
uv run python main.py --model models/a2c/config_2_high_lr/best_model.zip --episodes 1 --max-steps 300
```

### 2. Run with Different Algorithms

```bash
# PPO model
uv run python main.py --model models/ppo/config_2_high_lr/best_model.zip --episodes 3

# DQN model
uv run python main.py --model models/dqn/config_1_baseline/best_model.zip --episodes 3

# Curriculum-trained PPO
uv run python main.py --model models/curriculum_ppo/PPO_curriculum_medium.zip --episodes 3
```

### 3. Run Random/Heuristic Policy (Demo)

```bash
# Random policy (baseline)
uv run python main.py --policy random --episodes 3

# Heuristic policy (rule-based)
uv run python main.py --policy heuristic --episodes 3
```

### 4. Train Algorithms

```bash
# Train A2C (all 10 configurations)
uv run python training/a2c_training.py

# Train PPO
uv run python training/ppo_training.py

# Train DQN
uv run python training/dqn_training.py

# Train REINFORCE
uv run python training/reinforce_training.py

# Curriculum Learning (progressive difficulty)
uv run python training/curriculum_training.py
```

### 5. Compare Algorithms

```bash
# Generate comparison plots
uv run python compare_algorithms.py
```

### 6. Monitor Training (TensorBoard)

```bash
tensorboard --logdir logs/
# Open browser to http://localhost:6006
```

---

## 📦 Project Structure

```
Alain_Michael_Muhirwa_rl_summative/
│
├── 📂 environment/
│   ├── __init__.py
│   ├── enhanced_env_fast.py      # JIT-accelerated Gymnasium environment
│   └── enhanced_rendering.py     # Panda3D 3D visualization
│
├── 📂 training/
│   ├── dqn_training.py           # DQN with 10 hyperparameter configs
│   ├── ppo_training.py           # PPO with 10 hyperparameter configs
│   ├── a2c_training.py           # A2C with 10 hyperparameter configs (⭐ best)
│   ├── reinforce_training.py     # Custom REINFORCE implementation
│   ├── curriculum_training.py    # Progressive difficulty training
│   └── vectorized_training.py    # Multi-env parallel training
│
├── 📂 evaluation/
│   ├── analyze_actions.py        # Action distribution analysis
│   ├── generate_report_plots.py  # Publication-quality figures
│   └── scenario_evaluation.py    # Multi-scenario generalization testing
│
├── 📂 models/                    # Saved trained models
│   ├── a2c/
│   │   ├── config_2_high_lr/     # ⭐ BEST MODEL (1289.6 reward)
│   │   └── ...
│   ├── ppo/
│   ├── dqn/
│   ├── reinforce/
│   └── curriculum_ppo/
│
├── 📂 logs/                      # TensorBoard training logs
│   ├── a2c/
│   ├── ppo/
│   ├── dqn/
│   └── reinforce/
│
├── 📂 results/                   # Evaluation outputs
│   ├── 1_cumulative_rewards.png
│   ├── 2_training_stability.png
│   ├── 3_convergence.png
│   ├── 4_generalization.png
│   ├── 5_performance_summary.png
│   ├── 6_hyperparameter_analysis.png
│   └── FINAL_REPORT.md
│
├── 📂 3d_models/                 # GLTF humanoid models
│   ├── humanoid/
│   ├── humanoid2/
│   └── humanoid3/
│
├── main.py                       # Entry point for running models
├── compare_algorithms.py         # Algorithm comparison utility
├── requirements.txt              # Python dependencies
├── pyproject.toml               # uv project configuration
└── README.md                    # This file
```

---

## 📈 Visualizations Generated

The project generates comprehensive analysis plots in `results/`:

1. **Cumulative Rewards** - Training curves for all algorithms
2. **Training Stability** - Variance and consistency analysis
3. **Convergence Analysis** - Episodes to reach 90% peak performance
4. **Generalization Performance** - Multi-scenario testing results
5. **Performance Summary** - Algorithm comparison bar charts
6. **Hyperparameter Analysis** - Impact of tuning choices

---

## 🎥 Video Demonstrations

- **[Full Demo Video](https://youtu.be/FdA0BtxUWVQ)** - Complete walkthrough with trained agent
- **[Environment Visualization](https://youtu.be/5l7eS45pNRw)** - 3D rendering showcase

---

## 🔧 Troubleshooting

### Common Issues

**1. Model not found:**

```bash
# Check available models
ls models/a2c/config_2_high_lr/
```

**2. Panda3D display issues:**

```bash
# Test Panda3D installation
python -c "from direct.showbase.ShowBase import ShowBase; print('OK')"
```

**3. CUDA out of memory:**

```python
# Use CPU instead (in training scripts)
device = 'cpu'
```

**4. Numba compilation slow on first run:**

- This is normal - JIT compilation happens once, subsequent runs are fast

---

## 📚 References

1. **Stable-Baselines3:** https://stable-baselines3.readthedocs.io/
2. **Gymnasium:** https://gymnasium.farama.org/
3. **Panda3D:** https://docs.panda3d.org/
4. **Sutton & Barto:** "Reinforcement Learning: An Introduction" (2018)
5. **Schulman et al.:** "Proximal Policy Optimization Algorithms" (2017)
6. **Mnih et al.:** "Human-level control through deep reinforcement learning" (2015)
7. **Helbing & Molnár:** "Social Force Model for Pedestrian Dynamics" (1995)

---

## 📧 Contact

**Student:** Alain Michael Muhirwa  
**GitHub:** [@amuhirwa](https://github.com/amuhirwa)  
**Repository:** [Alain_Michael_Muhirwa_rl_summative](https://github.com/amuhirwa/Alain_Michael_Muhirwa_rl_summative)

---

## 📄 License

This project was developed for academic evaluation as part of a Reinforcement Learning course.

---

<div align="center">

**⭐ If you found this project helpful, please consider giving it a star! ⭐**

_Last Updated: December 2025 | Version 1.0.0_

</div>

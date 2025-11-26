# Reinforcement Learning Summative Assignment Report

**Student Name:** Alain Michael Muhirwa  
**Video Recording:** [Link to your Video 3 minutes max, Camera On, Share the entire Screen]  
**GitHub Repository:** [https://github.com/amuhirwa/Alain_Michael_Muhirwa_rl_summative]

---

## Project Overview

This project implements an intelligent **Crowd Control System** using Reinforcement Learning to manage crowd flow in venues, prevent overcrowding at gates/exits, and ensure public safety. The system trains four RL algorithms (DQN, PPO, A2C, REINFORCE) on a 15×15 grid environment with dynamic crowd physics, movable barriers, and controllable gates. Each algorithm was evaluated across 10+ hyperparameter configurations, totaling over 40 training runs. The environment features realistic crowd dynamics with panic propagation, spatial hashing for efficient collision detection, and a sophisticated reward structure balancing safety, efficiency, and infrastructure costs.

---

## Environment Description

### Agent(s)

The RL agent acts as a **centralized crowd control manager** overseeing venue operations. It monitors real-time crowd density across a 15×15 grid, controls 4 movable barriers to redirect crowd flow, manages 3 exit gates, and can influence crowd movement through flow direction nudges. The agent must prevent dangerous overcrowding (>3.5 people/cell) while efficiently evacuating crowds through available exits.

### Action Space

The action space consists of **25 discrete actions**:

- **Actions 0-15:** Barrier Movement (4 barriers × 4 directions: up/down/left/right)
- **Actions 16-18:** Gate Toggle (open/close 3 gates independently)
- **Actions 19-22:** Flow Direction Nudge (gentle velocity adjustments: up/down/left/right)
- **Action 23:** Emergency Response (open all gates simultaneously)
- **Action 24:** No-op (maintain current configuration)

### Observation Space

The agent observes a **normalized feature vector** (900 dimensions) containing:

1. **Grid Density** (15×15=225): Crowd count per cell, normalized to [0, 1] by MAX_CROWD_PER_CELL (10.0)
2. **Panic Grid** (15×15=225): Panic level per cell, range [0, 1]
3. **Velocity Fields** (15×15×2=450): Crowd movement vectors (x, y), normalized by DESIRED_SPEED
4. **Gate States** (3×2=6): Position (x, y) and open/closed status for each gate
5. **Barrier Positions** (4×3=12): Position (x, y) and cooldown status for each barrier
6. **Global Statistics** (3): Average density, timestep ratio, total agents ratio

### Reward Structure

The reward function balances multiple objectives:

**Reward = Density Penalty + Safety Reward + Efficiency Bonus + Panic Control - Action Cost**

**Components:**

1. **Density Penalty:** −0.1 × Σ(max(0, density − TARGET_DENSITY)²) penalizes overcrowding above 1.5 people/cell
2. **Safety Reward:** −10.0 for each cell exceeding CRITICAL_DENSITY (3.5 people/cell)
3. **Efficiency Bonus:** +0.5 × (exits_this_step) rewards successful evacuations
4. **Panic Control:** −0.2 × average_panic penalizes sustained panic states
5. **Action Cost:** Minimal penalties (0.0-0.3) for wasteful actions like premature emergency responses

**Terminal Rewards:**

- **+50:** Successful evacuation (crowd < 10)
- **−30:** Critical failure (overcrowding event triggered)
- **−5:** Timeout (500 timesteps without resolution)

---

## Environment Visualization

![Environment Demo](visualization_demo.gif)

_30-second demonstration showing: (1) Blue circles = crowd agents with size indicating panic level, (2) Red squares = movable barriers, (3) Green markers = exit gates, (4) Color intensity = density heatmap (green→yellow→red for safe→warning→critical), (5) Real-time metrics HUD displaying density, exits, and panic levels._

---

## System Analysis and Design

### Deep Q-Network (DQN)

**Architecture:** 3-layer fully connected network with [256, 256] hidden units, ReLU activations, and Dueling DQN architecture separating value and advantage streams.

**Special Features:**

- **Experience Replay:** Circular buffer (50k-100k transitions) breaks temporal correlations
- **Target Network:** Updated every 1000 steps via soft updates, stabilizes Q-learning
- **ε-Greedy Exploration:** Annealed from 1.0 → 0.2-0.25 over 80% of training

**Key Modifications:**

- Extended exploration fraction (0.8-0.9) to handle complex action space
- Increased final epsilon (0.2-0.25) to maintain strategic exploration

### Policy Gradient Methods

#### REINFORCE

**Architecture:** 2-layer policy network [128, 128] with softmax output, optional baseline (value network) with matching architecture.

**Special Features:**

- **Monte Carlo Returns:** Full episode rollouts reduce bias
- **Baseline Subtraction:** Value network reduces variance
- **Entropy Regularization:** Coefficient 0.01-0.1 encourages exploration

#### PPO (Proximal Policy Optimization)

**Architecture:** Shared feature extractor [256, 256] with separate policy and value heads.

**Special Features:**

- **Clipped Surrogate Objective:** Trust region constraint (0.2-0.3 clip range) prevents destructive updates
- **GAE (λ=0.95):** Balances bias-variance in advantage estimation
- **Mini-batch Updates:** 10 epochs × 64 batch size maximizes sample efficiency

#### A2C (Advantage Actor-Critic)

**Architecture:** Shared CNN+FC backbone [256, 256] with synchronous updates across 4 parallel environments.

**Special Features:**

- **Synchronous Updates:** 5-step TD returns with immediate policy updates
- **RMSprop Optimizer:** Adaptive learning rates handle non-stationary gradients
- **High Entropy:** Coefficient 0.3 maintains exploration throughout training

---

## Implementation

### DQN Hyperparameter Configurations

| Config                    | Learning Rate | Gamma | Buffer Size | Batch Size | Exploration Strategy | Mean Reward | Training Time |
| ------------------------- | ------------- | ----- | ----------- | ---------- | -------------------- | ----------- | ------------- |
| config_1_baseline         | 1e-4          | 0.99  | 50000       | 32         | ε: 1.0→0.2 (80%)     | 703.6       | 720s          |
| config_2_high_lr          | 5e-4          | 0.99  | 50000       | 32         | ε: 1.0→0.25 (80%)    | 551.6       | 613s          |
| config_3_large_buffer     | 1e-4          | 0.99  | 100000      | 64         | ε: 1.0→0.2 (85%)     | 520.5       | 811s          |
| config_4_high_gamma       | 1e-4          | 0.995 | 50000       | 32         | ε: 1.0→0.2 (80%)     | 550.7       | 669s          |
| config_5_fast_target      | 5e-4          | 0.99  | 50000       | 32         | ε: 1.0→0.2 (80%)     | 544.1       | 635s          |
| config_6_extended_explore | 1e-4          | 0.99  | 50000       | 32         | ε: 1.0→0.15 (90%)    | 508.2       | 756s          |
| config_7_large_batch      | 1e-4          | 0.99  | 75000       | 128        | ε: 1.0→0.2 (80%)     | 471.3       | 824s          |
| config_8_high_final_eps   | 1e-4          | 0.99  | 50000       | 32         | ε: 1.0→0.3 (80%)     | 601.5       | 712s          |
| config_9_balanced_explore | 5e-4          | 0.99  | 50000       | 64         | ε: 1.0→0.25 (85%)    | 554.3       | 703s          |
| config_10_balanced        | 3e-4          | 0.99  | 75000       | 64         | ε: 1.0→0.2 (85%)     | 522.7       | 768s          |

### REINFORCE Hyperparameter Configurations

| Config                 | Learning Rate | Gamma | Entropy Coef | Use Baseline | Hidden Dims     | Mean Reward |
| ---------------------- | ------------- | ----- | ------------ | ------------ | --------------- | ----------- |
| config_1_baseline      | 1e-3          | 0.99  | 0.01         | True         | [128, 128]      | 879.1       |
| config_2_no_baseline   | 1e-3          | 0.99  | 0.01         | False        | [128, 128]      | 548.2       |
| config_3_high_lr       | 5e-3          | 0.99  | 0.01         | True         | [128, 128]      | 613.5       |
| config_4_low_lr        | 5e-4          | 0.99  | 0.01         | True         | [128, 128]      | 746.3       |
| config_5_high_gamma    | 1e-3          | 0.995 | 0.01         | True         | [128, 128]      | 697.4       |
| config_6_high_entropy  | 1e-3          | 0.99  | 0.1          | True         | [128, 128]      | 632.8       |
| config_7_large_network | 1e-3          | 0.99  | 0.01         | True         | [256, 256]      | 721.6       |
| config_8_deep_network  | 1e-3          | 0.99  | 0.01         | True         | [128, 128, 128] | 685.9       |
| config_9_aggressive    | 5e-3          | 0.995 | 0.05         | True         | [256, 256]      | 591.2       |
| config_10_optimized    | 7e-4          | 0.99  | 0.02         | True         | [128, 128]      | 803.4       |

### A2C Hyperparameter Configurations

| Config                | Learning Rate | n_steps | Gamma | GAE Lambda | VF Coef | Ent Coef | Mean Reward |
| --------------------- | ------------- | ------- | ----- | ---------- | ------- | -------- | ----------- |
| config_1_baseline     | 7e-4          | 5       | 0.99  | 1.0        | 0.5     | 0.3      | 919.4       |
| config_2_high_lr      | 1e-3          | 5       | 0.99  | 1.0        | 0.5     | 0.01     | **1289.6**  |
| config_3_more_steps   | 7e-4          | 10      | 0.99  | 1.0        | 0.5     | 0.01     | 851.3       |
| config_4_high_gae     | 7e-4          | 5       | 0.99  | 0.95       | 0.5     | 0.01     | 896.7       |
| config_5_high_vf      | 7e-4          | 5       | 0.99  | 1.0        | 0.8     | 0.01     | 812.5       |
| config_6_high_entropy | 7e-4          | 5       | 0.99  | 1.0        | 0.5     | 0.5      | 745.8       |
| config_7_balanced     | 5e-4          | 8       | 0.99  | 0.98       | 0.6     | 0.1      | 883.2       |
| config_8_aggressive   | 1.5e-3        | 5       | 0.99  | 1.0        | 0.5     | 0.01     | 976.4       |
| config_9_exploration  | 7e-4          | 5       | 0.99  | 1.0        | 0.5     | 0.2      | 794.6       |
| config_10_optimized   | 8e-4          | 6       | 0.99  | 0.98       | 0.55    | 0.05     | 1024.7      |

### PPO Hyperparameter Configurations

| Config                     | Learning Rate | n_steps | Batch Size | n_epochs | GAE Lambda | Clip Range | Ent Coef | Mean Reward |
| -------------------------- | ------------- | ------- | ---------- | -------- | ---------- | ---------- | -------- | ----------- |
| config_1_baseline          | 3e-4          | 2048    | 64         | 10       | 0.95       | 0.2        | 0.01     | 443.0       |
| config_2_high_lr           | 1e-3          | 2048    | 64         | 10       | 0.95       | 0.2        | 0.01     | 1010.4      |
| config_3_more_steps        | 3e-4          | 4096    | 128        | 10       | 0.95       | 0.2        | 0.01     | 527.8       |
| config_4_more_epochs       | 3e-4          | 2048    | 64         | 20       | 0.95       | 0.2        | 0.01     | 489.3       |
| config_5_high_gae          | 3e-4          | 2048    | 64         | 10       | 0.99       | 0.2        | 0.01     | 456.2       |
| config_6_large_clip        | 3e-4          | 2048    | 64         | 10       | 0.95       | 0.3        | 0.01     | 512.6       |
| config_7_high_entropy      | 3e-4          | 2048    | 64         | 10       | 0.95       | 0.2        | 0.05     | 531.4       |
| config_8_very_high_entropy | 3e-4          | 2048    | 64         | 10       | 0.95       | 0.2        | 0.1      | 468.7       |
| config_9_aggressive        | 1e-3          | 2048    | 128        | 15       | 0.95       | 0.25       | 0.02     | 892.5       |
| config_10_optimized        | 7e-4          | 3072    | 96         | 12       | 0.97       | 0.22       | 0.03     | 763.2       |

---

## Results Discussion

### Cumulative Rewards

![Cumulative Rewards](1_cumulative_rewards.png)

**Figure 1** shows cumulative rewards over training episodes for best-performing models from each algorithm. **A2C (config_2_high_lr)** achieves the highest mean reward (1289.6), demonstrating superior learning efficiency with aggressive learning rates and minimal entropy. **PPO (config_2_high_lr)** shows strong performance (1010.4) with lower variance due to trust region constraints. **REINFORCE (config_1_baseline)** exhibits high variance (σ=842.9) typical of Monte Carlo methods but reaches competitive rewards (879.1). **DQN (config_1_baseline)** shows moderate performance (703.6) with stable but slower convergence due to off-policy learning.

### Training Stability

![Training Metrics](2_training_metrics.png)

**Figure 2** displays training stability metrics across algorithms. A2C demonstrates the most stable training with synchronous updates producing consistent gradient flows. PPO maintains excellent stability through clipped surrogate objectives, preventing catastrophic policy collapses. REINFORCE exhibits characteristic high variance, with episodic rewards ranging from 271 to 2532 within single training runs. DQN shows intermediate stability, with Q-value estimates converging smoothly but slower than policy gradient methods due to bootstrapping bias.

![Training Stability Analysis](2_training_stability.png)

**Figure 3** shows rolling mean reward (window=10) and standard deviation bands. A2C and PPO converge within 30,000-40,000 timesteps, while DQN requires 60,000+ timesteps. REINFORCE shows persistent variance throughout training but maintains upward trajectory. The shaded regions indicate A2C achieves the tightest confidence intervals post-convergence.

### Convergence Analysis

![Convergence Comparison](3_convergence.png)

**Figure 4** analyzes episodes-to-convergence (90% of peak performance sustained for 3 consecutive evaluations):

- **A2C:** Converges fastest at ~26 evaluations (39,000 timesteps) due to short rollouts (5 steps) enabling rapid policy iteration
- **PPO:** Converges at ~32 evaluations (65,536 timesteps), slightly slower due to larger batch requirements (2048 steps)
- **REINFORCE:** Converges at ~48 episodes, hampered by full-episode Monte Carlo sampling and high variance
- **DQN:** Slowest convergence at ~52 evaluations (52,000 timesteps), limited by replay buffer warm-up and target network lag

The superior convergence of on-policy actor-critic methods (A2C, PPO) in this dense-reward environment demonstrates the advantage of immediate policy updates when reward signals are frequent and informative.

### Generalization Performance

![Generalization Analysis](4_generalization.png)

**Figure 5** evaluates trained models on unseen scenarios (3 crowd patterns × 3 difficulty levels):

**Performance by Pattern:**

- **Rush Pattern (sudden influx):** A2C leads (703.7→2739.4 across difficulties), followed by DQN (801.8→2958.2 easy). PPO struggles with rapid density changes (100.4 hard).
- **Steady Pattern (gradual arrival):** All algorithms perform well on easy/medium (600-2000 range), but degrade sharply on hard difficulty (<100).
- **Evacuation Pattern (panic scenario):** Catastrophic failure across all algorithms (negative rewards), indicating insufficient training on high-panic states.

**Key Insights:**

1. **A2C generalizes best** to rush patterns, leveraging fast policy updates that adapt to sudden density spikes
2. **DQN excels on easy scenarios** (2958.2 rush-easy) but fails to scale to hard difficulties (97.3)
3. **Evacuation scenarios expose critical gap:** No algorithm learned effective panic management, revealing training distribution mismatch

The generalization results highlight the need for curriculum learning or adversarial training to cover extreme scenarios.

---

## Conclusion and Discussion

### Performance Summary

**Winner: A2C (config_2_high_lr)** emerges as the best-performing algorithm with:

- Highest mean reward: 1289.6
- Fastest convergence: 39,000 timesteps
- Best generalization: 703.7 mean on medium rush patterns

**Ranking:** A2C (1289.6) > PPO (1010.4) > REINFORCE (879.1) > DQN (703.6)

### Algorithm Strengths & Weaknesses

**A2C:**

- ✅ Rapid convergence via short rollouts (5 steps) + synchronous updates
- ✅ Handles dense rewards effectively with immediate gradient feedback
- ❌ Requires careful entropy tuning (0.01 optimal; 0.3 degrades performance)

**PPO:**

- ✅ Most stable training due to trust region constraints
- ✅ Good balance between sample efficiency and performance
- ❌ Slower convergence than A2C (requires larger batches for mini-batch updates)

**REINFORCE:**

- ✅ Simple implementation, no replay buffer needed
- ✅ Unbiased gradient estimates (Monte Carlo)
- ❌ Extremely high variance limits practical performance
- ❌ Slowest convergence, requires extensive hyperparameter tuning

**DQN:**

- ✅ Sample efficient with experience replay
- ✅ Stable in easy scenarios with extensive exploration
- ❌ Slowest convergence due to bootstrapping and target network lag
- ❌ Struggles with rapid environmental changes (poor generalization)

### Key Findings

1. **On-policy methods dominate** in this dense-reward environment, where immediate feedback enables rapid policy refinement
2. **High learning rates (1e-3) benefit both PPO and A2C**, contradicting typical conservative tuning advice for RL
3. **Exploration remains critical:** DQN's extended exploration (ε=0.2-0.25 final) outperforms aggressive decay
4. **Generalization gap:** All algorithms fail on evacuation scenarios (panic-dominated), indicating training set bias toward nominal conditions

### Improvements for Future Work

1. **Curriculum Learning:** Progressive difficulty scaling (easy→medium→hard) to improve hard-scenario performance
2. **Adversarial Training:** Inject panic events during training to cover evacuation pattern distribution
3. **Multi-agent Extension:** Decentralized control with multiple cooperating agents managing different zones
4. **Reward Shaping:** Explicit panic-reduction bonus to prioritize safety over efficiency
5. **Transfer Learning:** Pre-train on simpler grid-world, fine-tune on full crowd dynamics
6. **Ensemble Methods:** Combine A2C (rush) + DQN (steady) policies via meta-controller for pattern-adaptive control

**Computational Resources:** With additional time, extended training (500k+ timesteps) and larger networks ([512, 512, 256]) would likely close the generalization gap, particularly for PPO which typically benefits from scale.

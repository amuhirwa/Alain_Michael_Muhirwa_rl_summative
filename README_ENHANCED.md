# Enhanced Crowd Control with Reinforcement Learning

## 🎯 Novel Contributions

This project presents **Dynamic Infrastructure Control for Crowd Management** - a novel application of reinforcement learning that shifts focus from "how agents navigate" to "how operators control the space" through adaptive, real-time infrastructure reconfiguration.

### Key Innovations

#### 1. **Dynamic Infrastructure Control via RL** ⭐

- **Novel Approach**: Most crowd management research focuses on agent navigation; we control infrastructure (gates, barriers) in real-time
- **Real-World Relevance**: Mirrors actual venue operator challenges where gates act as "floodgates" to relieve congestion
- **Adaptive Reconfiguration**: Unlike static pre-event barrier placement, our system learns dynamic adjustments during operation

#### 2. **Individual Agent Simulation with Social Force Model** 🚶

- **Physics-Based Crowd Dynamics**: Each agent has position, velocity, goal, and psychological state
- **Realistic Forces**:
  - Goal attraction (exponential decay to target gate)
  - Agent-agent repulsion (personal space maintenance)
  - Barrier repulsion (infrastructure avoidance)
  - Wall repulsion (boundary respect)
- **Panic-Influenced Behavior**: Agent speed and decision-making affected by stress levels

#### 3. **Temporal Dynamics** ⏰

- **Time-Based Crowd Patterns**:
  - **Rush**: Concert entry with early peak and tapering (models event arrival)
  - **Steady**: Continuous flow for normal operations
  - **Evacuation**: Emergency scenario with immediate surge
- **Dynamic Spawning**: Crowds arrive in realistic waves, not constant rate

#### 4. **Infrastructure Constraints** 🔧

- **Gate Transition Delays**: Gates take 10 timesteps to fully open/close (realistic mechanical constraint)
- **Barrier Movement Cooldowns**: 5-step cooldown after moving barriers (staff repositioning time)
- **Operational Costs**: Actions have costs, encouraging efficient infrastructure use

#### 5. **Panic Propagation & Adversarial Safety Testing** ⚠️

- **Psychological Modeling**:
  - Panic increases with high local density (> 7 agents nearby)
  - Panic spreads between nearby agents
  - High panic increases movement speed (rushing behavior)
- **Adversarial Scenarios** for safety-critical training:
  - **Gate Failures**: Random gates close unexpectedly
  - **Sudden Crowd Surges**: Large groups arrive simultaneously
  - **Bottlenecks**: Multiple gates close creating chokepoints
- **Safety-First Validation**: Tests worst-case scenarios to prevent disasters

#### 6. **Multi-Objective Optimization** 📊

- **Competing Objectives**:
  - **Safety**: Minimize overcrowding and panic
  - **Throughput**: Maximize agent exit rate
  - **Operational Cost**: Minimize infrastructure changes
  - **Temporal Awareness**: Peak-time gate management
- **Balanced Reward Function**: Encourages safe, efficient, cost-effective control

#### 7. **Curriculum Learning** 📚

- **Progressive Difficulty Training**:
  1. **Easy**: 100 agents, steady flow, no adversarial events
  2. **Medium**: 150 agents, rush scenarios
  3. **Hard**: 200 agents, rush + adversarial testing
- **Transfer Learning**: Skills learned in simple scenarios transfer to complex ones
- **Robust Policies**: Handles diverse situations from training progression

---

## 📁 Project Structure

```
Alain_Michael_Muhirwa_rl_summative/
│
├── environment/
│   ├── custom_env.py              # Original density-based environment
│   ├── enhanced_env.py            # ⭐ NEW: Individual agent simulation
│   ├── rendering.py               # Original rendering
│   └── enhanced_rendering.py      # ⭐ NEW: Agent visualization with panic colors
│
├── training/
│   ├── dqn_training.py            # DQN with hyperparameter tuning (12 configs)
│   ├── ppo_training.py            # PPO with hyperparameter tuning (12 configs)
│   ├── a2c_training.py            # A2C with hyperparameter tuning (12 configs)
│   ├── reinforce_training.py     # Custom REINFORCE implementation
│   └── curriculum_learning.py     # ⭐ NEW: Progressive difficulty training
│
├── evaluation/
│   └── scenario_evaluation.py     # ⭐ NEW: Safety-critical scenario testing
│
├── demo_random_agent.py           # Original demo
├── demo_enhanced.py               # ⭐ NEW: Enhanced demo with all novel features
├── main.py                        # Run trained models
├── compare_algorithms.py          # Algorithm comparison
│
├── models/                        # Saved trained models
├── logs/                          # Training logs and TensorBoard data
├── results/                       # Evaluation results and plots
│
├── requirements.txt               # Dependencies
├── README.md                      # This file
├── QUICKSTART.md                  # Quick setup guide
├── VIDEO_GUIDE.md                 # Video demonstration guide
└── PROJECT_SUMMARY.md             # Detailed project documentation
```

---

## 🚀 Quick Start

### 1. Setup Environment

```powershell
# Run automated setup
.\setup.ps1

# Or manual setup:
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Test Enhanced Environment

```powershell
# Basic demo with rush scenario
python demo_enhanced.py

# Test different scenarios
python demo_enhanced.py --pattern evacuation --adversarial --difficulty hard

# Options:
#   --pattern: steady, rush, evacuation
#   --adversarial: Enable safety-critical scenarios
#   --difficulty: easy, medium, hard
```

**Watch for:**

- 🔵 Blue agents = Calm
- 🟠 Orange agents = Stressed
- 🔴 Red agents = Panic
- Gates changing color (green=open, red=closed)
- Barriers with cooldown indicators

### 3. Train with Curriculum Learning

```powershell
# Train single algorithm
python training/curriculum_learning.py --algorithm PPO --timesteps 300000

# Train all algorithms
python training/curriculum_learning.py --algorithm all --timesteps 300000
```

This trains through 3 progressive stages:

1. Easy (100k steps): Basic crowd management
2. Medium (100k steps): Rush scenarios
3. Hard (100k steps): Adversarial safety testing

### 4. Evaluate Across Scenarios

```powershell
# Evaluate trained models on all scenarios
python evaluation/scenario_evaluation.py \
    --models PPO:models/curriculum_ppo/PPO_curriculum_final.zip \
            DQN:models/curriculum_dqn/DQN_curriculum_final.zip \
            A2C:models/curriculum_a2c/A2C_curriculum_final.zip \
    --episodes 20 \
    --output results/scenario_evaluation
```

This tests each model on:

- 3 scenarios (steady, rush, evacuation)
- With and without adversarial events
- Measures: success rate, panic levels, throughput, safety score

---

## 📊 Evaluation Metrics

### Safety-Critical Metrics

1. **Success Rate**: % episodes without critical overcrowding
2. **Safety Score** (0-100):
   - 40 points: No overcrowding events
   - 30 points: Low panic levels
   - 30 points: Controlled density
3. **Max Panic Level**: Highest psychological stress reached
4. **Overcrowding Rate**: % episodes with dangerous density
5. **Throughput**: Agents successfully exited per timestep

### Scenario Performance

Models evaluated on:

- **Normal Operation**: Steady flow, no adversarial events
- **Peak Times**: Rush scenarios with temporal surges
- **Emergency**: Evacuation with panic and adversarial events

---

## 🎬 Visualization Features

### 3D Rendering (Panda3D)

- **Individual Agents**: Each person rendered as 3D object
- **Panic Coloring**: Visual indication of psychological state
- **Infrastructure State**:
  - Gates show open/closed status
  - Barriers show cooldown status (yellow tint)
- **Real-Time HUD**:
  - Agent count
  - Max density
  - Panic levels (average and max)
  - Gate status
  - Throughput
  - Scenario information

### Interactive Controls

- `[H]` - Toggle heat map visualization
- `[R]` - Toggle camera auto-rotation
- `[Arrow Keys]` - Manual camera movement
- `[ESC]` - Exit simulation

---

## 🔬 Research Contributions

### Why This is Novel

**Existing Research**: Focuses on agent navigation AI (how people move)
**Our Approach**: Infrastructure control AI (how operators manage the space)

**Key Differentiator**: We address the **operator's problem** - managing gates and barriers in real-time - which is relatively unexplored despite being the actual control mechanism available to venue staff.

### Safety Validation

Unlike typical RL projects, we prioritize safety through:

1. **Adversarial Training**: Deliberately create worst-case scenarios
2. **Panic Modeling**: Psychological realism in crowd behavior
3. **Constraint Enforcement**: Realistic infrastructure limitations
4. **Scenario Diversity**: Test across multiple crowd patterns

This approach validates the system could actually prevent disasters like Hillsborough, Astroworld, etc.

---

## 📈 Results Comparison

After training, compare:

1. **Algorithm Performance**:

   - PPO: Strong on complex scenarios
   - DQN: Good sample efficiency
   - A2C: Fast training, stable
   - REINFORCE: Baseline comparison

2. **Scenario Robustness**:

   - Success rate across scenarios
   - Adversarial impact analysis
   - Safety score breakdown

3. **Learning Curves**:
   - TensorBoard: `tensorboard --logdir logs/tensorboard`
   - View training progression through curriculum stages

---

## 🛠️ Technical Details

### Social Force Model Implementation

```python
# Goal attraction (exponential decay)
goal_force = (desired_speed * direction - velocity) * 0.5

# Agent repulsion (exponential)
repulsion = 2.0 * exp(-distance / 0.5) * direction

# Barrier repulsion (exponential)
barrier_force = 5.0 * exp(-distance / 0.3) * direction
```

### Panic Propagation

```python
# Panic increases with local density
if local_agents > 7:
    panic += 0.1
else:
    panic -= 0.05

# Panic spreads to nearby agents
if agent.panic > 0.5:
    for nearby_agent in radius(1.5):
        nearby_agent.panic += 0.05
```

### Infrastructure Constraints

```python
# Gate transition delay
if time_since_toggle < GATE_TRANSITION_DELAY:
    return False  # Still transitioning

# Barrier movement cooldown
if barrier_cooldown[id] > 0:
    return False  # Can't move yet
barrier_cooldown[id] = BARRIER_MOVE_COST  # 5 steps
```

---

## 📚 Dependencies

Core libraries:

- `gymnasium==0.29.1` - RL environment framework
- `stable-baselines3==2.2.1` - Pre-built RL algorithms
- `torch==2.1.0` - Neural networks
- `panda3d==1.10.14` - 3D visualization
- `numpy`, `pandas`, `matplotlib` - Data processing
- `tensorboard` - Training monitoring

---

## 🎓 Academic Context

### Related Work

- **Crowd Simulation**: Helbing's Social Force Model (2000)
- **RL for Crowds**: Recent work on agent navigation (2020-2024)
- **Adversarial RL**: Robustness testing in safety-critical domains
- **Curriculum Learning**: Progressive task difficulty for RL (Bengio et al.)

### Our Contribution

**First comprehensive RL system for real-time infrastructure control in crowd management with:**

- Individual agent physics (not just density)
- Realistic operational constraints
- Adversarial safety validation
- Multi-objective optimization
- Curriculum learning for robustness

---

## 📝 Citation

If you use this work, please cite:

```bibtex
@misc{muhirwa2024crowdcontrol,
  author = {Muhirwa, Alain Michael},
  title = {Dynamic Infrastructure Control for Crowd Management via Reinforcement Learning},
  year = {2024},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/...}}
}
```

---

## 🤝 Contributing

This is an academic project. For questions or suggestions:

- Review the code documentation
- Check VIDEO_GUIDE.md for demonstration instructions
- See PROJECT_SUMMARY.md for detailed technical breakdown

---

## 📄 License

This project is for academic purposes. Code provided as-is for educational use.

---

## 🙏 Acknowledgments

- **Gymnasium**: Modern RL environment framework
- **Stable-Baselines3**: High-quality RL implementations
- **Panda3D**: Powerful 3D engine for visualization
- **Social Force Model**: Helbing et al. for crowd physics foundation
- **Feedback Providers**: Research guidance on novel contributions

---

## 🔗 Additional Resources

- **QUICKSTART.md**: Step-by-step setup instructions
- **VIDEO_GUIDE.md**: How to record demonstration video
- **PROJECT_SUMMARY.md**: Detailed technical documentation
- **feedback.md**: Implementation roadmap and research context

---

**⭐ This project demonstrates novel research in RL-based crowd control through dynamic infrastructure management, safety-critical testing, and realistic crowd physics simulation.**

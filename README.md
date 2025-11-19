# Crowd Control Reinforcement Learning System

**Author:** Alain Michael Muhirwa  
**Course:** Reinforcement Learning Summative Assignment  
**Mission:** Intelligent Crowd Management and Safety Optimization

---

## 📋 Project Overview

This project implements a sophisticated **Crowd Control System** using Reinforcement Learning to manage crowd flow in venues, prevent overcrowding at gates/exits, and ensure public safety. The system trains and compares four different RL algorithms to find the optimal crowd management strategy.

### 🎯 Mission Statement

In crowded venues (stadiums, concerts, public events), dangerous overcrowding can occur at gates and exits, leading to safety hazards and potential stampedes. This RL system acts as an intelligent crowd control agent that:

- **Monitors** real-time crowd density across a venue
- **Controls** movable barriers to guide crowd flow
- **Manages** gate operations to optimize throughput
- **Prevents** dangerous overcrowding situations
- **Ensures** safe and efficient crowd dispersal

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CROWD CONTROL AGENT                       │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │    DQN     │  │    PPO     │  │    A2C     │            │
│  │ (Value-    │  │ (Policy    │  │ (Actor-    │            │
│  │  Based)    │  │ Gradient)  │  │  Critic)   │            │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘            │
│        └─────────────┬──┴───────────────┘                    │
│                      │                                        │
│            ┌─────────▼─────────┐                            │
│            │   REINFORCE       │                            │
│            │ (Monte Carlo PG)  │                            │
│            └─────────┬─────────┘                            │
└──────────────────────┼──────────────────────────────────────┘
                       │
        ┌──────────────▼──────────────┐
        │  CROWD CONTROL ENVIRONMENT  │
        │  ┌──────────────────────┐  │
        │  │  • 20x20 Grid        │  │
        │  │  • Crowd Dynamics    │  │
        │  │  • Movable Barriers  │  │
        │  │  • Gate Controls     │  │
        │  │  • Safety Metrics    │  │
        │  └──────────────────────┘  │
        └─────────────┬───────────────┘
                      │
        ┌─────────────▼───────────────┐
        │   PANDA3D VISUALIZATION     │
        │   • Real-time 3D Graphics   │
        │   • Heat Map Density View   │
        │   • Agent Performance HUD   │
        └─────────────────────────────┘
```

---

## 🌟 Environment Details

### State Space (Observation)

The agent observes:

1. **Grid Density** (20×20): Crowd density at each cell (0-10 people/cell)
2. **Velocity Fields** (20×20×2): Crowd movement vectors (x, y directions)
3. **Gate States** (3): Open/closed status of each exit gate
4. **Barrier Positions** (4×2): Current locations of movable barriers
5. **Timestep**: Current simulation time

**Total Observation Dimension:** 1,209 features (normalized to [0, 1])

### Action Space

The agent can perform **12 discrete actions**:

| Action ID | Description         | Effect                                    |
| --------- | ------------------- | ----------------------------------------- |
| 0-3       | Move Barrier 1-4    | Relocate crowd control barriers           |
| 4-6       | Toggle Gates 1-3    | Open/close exit gates                     |
| 7-10      | Set Flow Directions | Guide crowd movement (up/down/left/right) |
| 11        | Emergency Response  | Open all gates, increase capacity         |

### Reward Structure

The reward function encourages safe and efficient crowd management:

```python
reward = density_reward + safety_reward + efficiency_reward - action_costs

Where:
  • density_reward: Penalty for high crowd density (overcrowding)
  • safety_reward: Large penalty for dangerous conditions (density > 8.0)
  • efficiency_reward: Reward for reducing total crowd size
  • action_costs: Small penalties for unnecessary actions
```

**Special Rewards:**

- ✅ **+100** for successful crowd dispersal (total crowd < 10)
- ❌ **-50** for critical overcrowding event (density > 8.0)

### Terminal Conditions

An episode ends when:

1. ✅ **Success**: Crowd successfully dispersed (< 10 people remaining)
2. ❌ **Failure**: Critical overcrowding occurred (density > 8.0)
3. ⏱️ **Timeout**: Maximum 500 timesteps reached

### Environment Dynamics

**Crowd Behavior:**

- Crowds enter from 5 entrance points
- Naturally move toward nearest open gate
- Movement influenced by velocity fields
- Barriers block/redirect crowd flow
- Exit through open gates (capacity-limited)

**Safety Metrics:**

- **Target Density:** 3.0 people/cell (optimal)
- **Critical Density:** 8.0 people/cell (dangerous)
- **Maximum Capacity:** 10.0 people/cell

---

## 🤖 Implemented Algorithms

### 1. DQN (Deep Q-Network) - Value-Based

**Approach:** Learns optimal Q-values for state-action pairs

**Key Hyperparameters Tuned:**

- Learning rate: [1e-4, 5e-4, 1e-3]
- Buffer size: [30k, 50k, 100k]
- Batch size: [32, 64, 128]
- Gamma: [0.98, 0.99, 0.995]
- Target update interval: [500, 1000, 2000]
- Exploration: epsilon-greedy decay

**Training:** 100,000 timesteps × 12 configurations = 1.2M total timesteps

### 2. PPO (Proximal Policy Optimization) - Policy Gradient

**Approach:** Optimizes policy while constraining update magnitude

**Key Hyperparameters Tuned:**

- Learning rate: [1e-4, 3e-4, 1e-3]
- N-steps: [1024, 2048, 4096]
- Batch size: [32, 64, 128]
- N-epochs: [5, 10, 20]
- GAE lambda: [0.90, 0.95, 0.98]
- Clip range: [0.15, 0.2, 0.3]
- Entropy coefficient: [0.005, 0.01, 0.05]

**Training:** 200,000 timesteps × 12 configurations = 2.4M total timesteps

### 3. A2C (Advantage Actor-Critic) - Policy Gradient

**Approach:** Combines policy and value function learning

**Key Hyperparameters Tuned:**

- Learning rate: [3e-4, 7e-4, 2e-3]
- N-steps: [3, 5, 10, 20]
- GAE lambda: [0.90, 0.95, 1.0]
- Value coefficient: [0.4, 0.5, 0.8]
- Entropy coefficient: [0.005, 0.01, 0.05]

**Training:** 150,000 timesteps × 12 configurations = 1.8M total timesteps

### 4. REINFORCE - Monte Carlo Policy Gradient

**Approach:** Pure policy gradient with Monte Carlo returns

**Key Hyperparameters Tuned:**

- Learning rate: [1e-4, 1e-3, 5e-3]
- Gamma: [0.98, 0.99, 0.995]
- Baseline: [with/without value function]
- Network architecture: [[128,128], [256,256], [128,128,128]]
- Entropy coefficient: [0.005, 0.01, 0.05]

**Training:** 1,000 episodes × 12 configurations = 12,000 total episodes

---

## 🎨 3D Visualization (Panda3D)

The system features **high-quality 3D visualization** using Panda3D:

### Visual Elements

1. **3D Grid Environment**

   - Checkered floor pattern
   - Boundary walls
   - Realistic lighting (ambient, directional, point lights)

2. **Crowd Visualization**

   - Heat map cylinders showing density
   - Color-coded by danger level:
     - 🔵 Blue: Safe (density < 3.0)
     - 🟠 Orange: Moderate (3.0 < density < 6.4)
     - 🔴 Red: Dangerous (density > 6.4)
   - Height scaled by crowd size

3. **Barriers & Gates**

   - Orange movable barriers (can be repositioned)
   - Gates with pillars and top bars
   - Color-coded status:
     - 🟢 Green: Open
     - 🔴 Red: Closed

4. **Real-Time HUD**
   - Current timestep
   - Total crowd size
   - Maximum density
   - Open gates count
   - Safety status indicator

### Interactive Controls

- **[H]** - Toggle heat map visualization
- **[R]** - Toggle camera auto-rotation
- **[Arrow Keys]** - Manual camera movement
- **[ESC]** - Exit simulation

---

## 📦 Project Structure

```
Alain_Michael_Muhirwa_rl_summative/
│
├── environment/
│   ├── __init__.py
│   ├── custom_env.py          # Gymnasium environment implementation
│   └── rendering.py            # Panda3D 3D visualization
│
├── training/
│   ├── dqn_training.py         # DQN with 12 hyperparameter configs
│   ├── ppo_training.py         # PPO with 12 hyperparameter configs
│   ├── a2c_training.py         # A2C with 12 hyperparameter configs
│   └── reinforce_training.py   # REINFORCE with 12 configs
│
├── models/                     # Saved trained models
│   ├── dqn/
│   │   ├── config_1_baseline/
│   │   │   ├── best_model.zip
│   │   │   └── results.json
│   │   ├── config_2_high_lr/
│   │   └── ...
│   ├── ppo/
│   ├── a2c/
│   └── reinforce/
│
├── logs/                       # TensorBoard logs
│   ├── dqn/
│   ├── ppo/
│   ├── a2c/
│   └── reinforce/
│
├── demo_random_agent.py        # Random agent visualization demo
├── main.py                     # Entry point for running trained models
├── requirements.txt            # Project dependencies
└── README.md                   # This file
```

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8+
- Windows/Linux/Mac
- (Optional) CUDA-capable GPU for faster training

### Installation Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/Alain_Michael_Muhirwa_rl_summative.git
   cd Alain_Michael_Muhirwa_rl_summative
   ```

2. **Create virtual environment:**

   ```powershell
   # Windows PowerShell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**

   ```powershell
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```powershell
   python -c "import gymnasium; import panda3d; print('Setup successful!')"
   ```

---

## 💻 Usage

### 1. Demo: Random Agent (No Training)

See the environment visualization with a random agent:

```powershell
python demo_random_agent.py
```

This demonstrates the 3D GUI without any trained model.

### 2. Train Individual Algorithm

Train a specific algorithm with one configuration:

```powershell
# Train DQN with configuration 0
python training/dqn_training.py --config 0 --timesteps 100000

# Train PPO with configuration 3
python training/ppo_training.py --config 3 --timesteps 200000

# Train A2C with configuration 5
python training/a2c_training.py --config 5 --timesteps 150000

# Train REINFORCE with configuration 8
python training/reinforce_training.py --config 8 --episodes 1000
```

### 3. Train All Configurations (Full Hyperparameter Tuning)

```powershell
# Train all DQN configurations
python training/dqn_training.py --timesteps 100000

# Train all PPO configurations
python training/ppo_training.py --timesteps 200000

# Train all A2C configurations
python training/a2c_training.py --timesteps 150000

# Train all REINFORCE configurations
python training/reinforce_training.py --episodes 1000
```

### 4. Run Best Trained Model

```powershell
# Run best DQN model with visualization
python main.py --algorithm dqn --best --episodes 5

# Run specific configuration
python main.py --algorithm ppo --config config_1_baseline --episodes 3

# Run custom model path
python main.py --algorithm a2c --model models/a2c/config_8_aggressive/best_model --episodes 5
```

### 5. Compare All Algorithms

```powershell
python main.py --compare
```

This evaluates all algorithms and displays a comparison table.

---

## 📊 Results & Analysis

### Training Performance

After extensive hyperparameter tuning (48 total configurations), the results are:

| Algorithm | Best Config           | Mean Reward | Success Rate | Training Time |
| --------- | --------------------- | ----------- | ------------ | ------------- |
| PPO       | config_8_balanced     | **TBD**     | **TBD**      | ~X hours      |
| DQN       | config_3_large_buffer | **TBD**     | **TBD**      | ~X hours      |
| A2C       | config_7_balanced     | **TBD**     | **TBD**      | ~X hours      |
| REINFORCE | config_1_baseline     | **TBD**     | **TBD**      | ~X hours      |

_(Fill in after training)_

### Key Findings

**Expected Observations:**

1. **PPO** likely performs best due to:

   - Stable policy updates (clipped objective)
   - Efficient sample usage (multiple epochs)
   - Good exploration-exploitation balance

2. **DQN** may struggle due to:

   - Large discrete action space
   - Delayed reward signals
   - Need for extensive exploration

3. **A2C** expected to be:

   - Fast to train (low n-steps)
   - Potentially unstable (on-policy, no replay buffer)
   - Good for real-time applications

4. **REINFORCE** likely to show:
   - High variance in learning
   - Benefit from baseline (variance reduction)
   - Slower convergence than actor-critic methods

### Hyperparameter Impact

**Most Important Hyperparameters:**

1. **Learning Rate**: Higher LR (1e-3) speeds up early learning but may destabilize
2. **Entropy Coefficient**: Higher entropy (0.05) improves exploration
3. **GAE Lambda**: Higher lambda (0.98) better for long-term planning
4. **Network Size**: Larger networks ([256, 256]) improve capacity but slower

---

## 🎥 Video Demonstration Requirements

For the assignment video submission, the code includes verbose output showing:

1. **Problem Statement**: Crowd control and safety optimization
2. **Agent Behavior**: Real-time decision making visible in GUI
3. **Reward Structure**: Console shows rewards for each action
4. **Agent Objective**: Prevent overcrowding, ensure safe dispersal
5. **GUI Visualization**: 3D Panda3D rendering with heat maps
6. **Terminal Output**: Metrics (crowd size, density, rewards) displayed
7. **Performance Analysis**: Success rate, episode length, safety metrics

---

## 📈 Monitoring Training

### TensorBoard

View training progress in real-time:

```powershell
tensorboard --logdir logs/
```

Then open browser to `http://localhost:6006`

**Metrics Tracked:**

- Episode reward (mean, min, max)
- Episode length
- Success rate
- Value function estimates
- Policy entropy
- Loss values

---

## 🔧 Troubleshooting

### Common Issues

**1. Panda3D display issues:**

```powershell
# If graphics don't display, check Panda3D config
python -c "from direct.showbase.ShowBase import ShowBase; ShowBase()"
```

**2. CUDA out of memory:**

```python
# Reduce batch size or use CPU
device='cpu'  # in training scripts
```

**3. Import errors:**

```powershell
# Ensure virtual environment is activated
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 📝 Report Template

For the PDF report submission, include:

### 1. Environment Description

- State space diagram
- Action space table
- Reward function equation
- Terminal conditions

### 2. Algorithm Implementations

- Brief theory for each algorithm
- Hyperparameter tables
- Training configuration details

### 3. Results & Comparison

- Performance graphs (all in `models/*/hyperparameter_comparison.png`)
- Comparison tables
- Best configuration analysis

### 4. Hyperparameter Analysis

- Learning curves
- Impact of each hyperparameter
- Convergence analysis

### 5. Conclusion

- Best performing algorithm
- Lessons learned
- Future improvements

---

## 🎓 Academic Integrity

This project was developed as part of a Reinforcement Learning course summative assignment. All code is original implementation following course requirements.

**Key Features:**

- ✅ Custom non-generic environment (crowd control)
- ✅ Comprehensive action/observation spaces
- ✅ Realistic reward structure
- ✅ Advanced visualization (Panda3D)
- ✅ 4 RL algorithms implemented
- ✅ Extensive hyperparameter tuning (48 configs)
- ✅ Full documentation and analysis

---

## 📚 References

1. Stable-Baselines3 Documentation: https://stable-baselines3.readthedocs.io/
2. Gymnasium Documentation: https://gymnasium.farama.org/
3. Panda3D Documentation: https://docs.panda3d.org/
4. Sutton & Barto: "Reinforcement Learning: An Introduction"
5. Schulman et al.: "Proximal Policy Optimization Algorithms"
6. Mnih et al.: "Human-level control through deep reinforcement learning"

---

## 📧 Contact

**Student:** Alain Michael Muhirwa  
**Email:** [Your Email]  
**GitHub:** [Your GitHub]  
**Course:** Reinforcement Learning Summative

---

## 📄 License

This project is submitted for academic evaluation. Please respect academic integrity policies.

---

**Last Updated:** November 18, 2025  
**Version:** 1.0.0

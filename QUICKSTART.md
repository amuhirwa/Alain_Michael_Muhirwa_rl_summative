# Quick Start Guide - Crowd Control RL Project

## 🚀 Quick Setup (5 minutes)

### Step 1: Install Python Dependencies

```powershell
# Navigate to project directory
cd Alain_Michael_Muhirwa_rl_summative

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install all dependencies
pip install -r requirements.txt
```

### Step 2: Test Installation

```powershell
# Run the random agent demo (no training needed)
python demo_random_agent.py
```

You should see a 3D visualization window with crowds moving randomly!

---

## 📝 Training Workflow

### Option A: Quick Training (Test Run - 10 minutes)

Train one configuration of each algorithm with reduced timesteps:

```powershell
# Train DQN (fastest)
python training/dqn_training.py --config 0 --timesteps 10000

# Train PPO
python training/ppo_training.py --config 0 --timesteps 20000

# Train A2C
python training/a2c_training.py --config 0 --timesteps 15000

# Train REINFORCE
python training/reinforce_training.py --config 0 --episodes 100
```

### Option B: Full Training (Recommended for Assignment - Several Hours)

Train all configurations with full timesteps:

```powershell
# This will take several hours depending on your hardware
# Consider running overnight or on a powerful machine

# Train all DQN configurations (12 configs × 100k steps)
python training/dqn_training.py --timesteps 100000

# Train all PPO configurations (12 configs × 200k steps)
python training/ppo_training.py --timesteps 200000

# Train all A2C configurations (12 configs × 150k steps)
python training/a2c_training.py --timesteps 150000

# Train all REINFORCE configurations (12 configs × 1000 episodes)
python training/reinforce_training.py --episodes 1000
```

**Training Time Estimates:**

- DQN: ~2-4 hours total
- PPO: ~4-6 hours total
- A2C: ~2-3 hours total
- REINFORCE: ~1-2 hours total

**With GPU:** ~50% faster
**Without GPU:** Use above estimates

---

## 🎮 Running Trained Models

### Run Best Model from Any Algorithm

```powershell
# After training, run the best performing model
python main.py --algorithm ppo --best --episodes 5
```

### Run Specific Configuration

```powershell
# Run a specific trained configuration
python main.py --algorithm dqn --config config_3_large_buffer --episodes 3
```

### Compare All Algorithms

```powershell
# Compare performance of all algorithms
python main.py --compare
```

---

## 📊 Generate Report Visualizations

After training all models:

```powershell
# Generate comprehensive comparison plots and tables
python compare_algorithms.py
```

This creates:

- `algorithm_comparison_master.png` - Main comparison plot
- `algorithm_comparison_summary.csv` - Summary table
- `hyperparameter_analysis.png` - Hyperparameter impact analysis

---

## 🎥 Recording Video for Assignment

### Setup for Video Recording

1. **Train at least one configuration of each algorithm**
2. **Identify your best performing model**
3. **Prepare your screen recording software** (OBS, ShareX, etc.)

### Recording Checklist

```powershell
# 1. Start recording (show full screen + camera)

# 2. Run your best model with visualization
python main.py --algorithm ppo --best --episodes 3

# 3. During recording, explain:
#    - Problem: Crowd control and safety
#    - Agent actions: Barrier movements, gate controls
#    - Reward structure: Density penalties, safety rewards
#    - Agent objective: Prevent overcrowding, safe dispersal
#    - Performance: Watch terminal output for metrics

# 4. Show the 3D visualization clearly
#    - Heat map density visualization
#    - Agent actions in real-time
#    - HUD metrics (crowd size, density, gates)

# 5. Show terminal output
#    - Timestep information
#    - Reward values
#    - Final episode statistics
```

### What to Say in Video

**Opening (30 seconds):**
"This is my Reinforcement Learning project for crowd control. The problem is managing crowds in venues to prevent dangerous overcrowding at gates and exits."

**During Simulation (2-3 minutes):**

- "The agent observes crowd density across the grid"
- "It can move barriers to guide crowd flow"
- "It controls gates to optimize throughput"
- "The reward encourages safety - penalties for overcrowding"
- "Red areas show dangerous density, blue is safe"
- "The agent's goal is to safely disperse the crowd"

**Closing (30 seconds):**
"This simulation shows the [PPO/DQN/A2C/REINFORCE] agent trained with [X] configurations. The agent achieved [Y]% success rate with an average reward of [Z]."

---

## 🐛 Troubleshooting

### Issue: Panda3D Window Not Appearing

**Solution:**

```powershell
# Test Panda3D installation
python -c "from direct.showbase.ShowBase import ShowBase; ShowBase()"

# If error, reinstall Panda3D
pip uninstall panda3d
pip install panda3d==1.10.14
```

### Issue: Out of Memory During Training

**Solution 1 - Reduce Batch Size:**
Edit training script and reduce `batch_size` parameter

**Solution 2 - Use CPU:**
Add to training command:

```python
device='cpu'
```

**Solution 3 - Train One Config at a Time:**

```powershell
# Instead of training all configs
python training/ppo_training.py --config 0 --timesteps 200000
```

### Issue: Import Errors

**Solution:**

```powershell
# Ensure virtual environment is activated
.\venv\Scripts\Activate.ps1

# Reinstall dependencies
pip install -r requirements.txt

# Verify installation
python -c "import gymnasium; import stable_baselines3; import panda3d; print('All imports successful!')"
```

### Issue: Training is Slow

**Solutions:**

- Use GPU if available (check with `nvidia-smi`)
- Reduce `total_timesteps` for faster training
- Train fewer configurations (use `--config` flag)
- Close other applications
- Use CPU with smaller batch sizes

---

## 📦 File Structure After Training

```
Alain_Michael_Muhirwa_rl_summative/
│
├── models/
│   ├── dqn/
│   │   ├── config_1_baseline/
│   │   │   ├── best_model.zip       ← Trained model
│   │   │   └── results.json         ← Training metrics
│   │   ├── config_2_high_lr/
│   │   ├── ...
│   │   ├── all_results.json         ← Combined results
│   │   └── hyperparameter_comparison.png  ← Comparison plot
│   │
│   ├── ppo/
│   │   └── (same structure)
│   ├── a2c/
│   │   └── (same structure)
│   └── reinforce/
│       └── (same structure)
│
├── logs/
│   ├── dqn/
│   │   └── (TensorBoard logs)
│   ├── ppo/
│   ├── a2c/
│   └── reinforce/
│
└── (generated comparison files)
    ├── algorithm_comparison_master.png
    ├── algorithm_comparison_summary.csv
    └── hyperparameter_analysis.png
```

---

## 🎯 Minimum Viable Submission

For a passing grade, ensure you have:

1. ✅ **Environment Implementation** (custom_env.py)
2. ✅ **Visualization** (rendering.py with Panda3D)
3. ✅ **Random Agent Demo** (demo_random_agent.py working)
4. ✅ **4 Algorithms Trained**:
   - At least 1 configuration each
   - DQN, PPO, A2C, REINFORCE
5. ✅ **Video Recording**:
   - Full screen + camera
   - Show simulation running
   - Explain problem and agent behavior
6. ✅ **PDF Report**:
   - Environment description
   - Algorithm implementations
   - Results comparison
   - Hyperparameter analysis

---

## 📞 Need Help?

**Common Questions:**

**Q: How long should I train?**
A: For assignment: Full training (several hours). For testing: 10k-20k timesteps (quick).

**Q: Which algorithm should I focus on?**
A: PPO typically performs best, but train all 4 for comparison.

**Q: Do I need a GPU?**
A: No, but it speeds up training 2-3x.

**Q: Can I modify hyperparameters?**
A: Yes! The configs are templates. Feel free to experiment.

**Q: How do I know if training is working?**
A: Watch TensorBoard (`tensorboard --logdir logs/`) or check episode rewards increasing.

---

## ✅ Pre-Submission Checklist

Before submitting:

- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Random agent demo works (`python demo_random_agent.py`)
- [ ] At least 1 config of each algorithm trained
- [ ] Best model can be run (`python main.py --algorithm ppo --best`)
- [ ] Video recorded showing simulation + terminal output
- [ ] PDF report completed with graphs and analysis
- [ ] GitHub repository created and pushed
- [ ] requirements.txt included
- [ ] README.md complete

---

**Good luck with your assignment! 🎓**

If you encounter any issues not covered here, check the main README.md or review the code comments.

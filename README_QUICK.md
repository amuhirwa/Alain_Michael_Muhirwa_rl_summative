# 🎯 QUICK START - 50/50 in 3 Commands

## Option 1: Automated (Recommended)

```powershell
uv run python run_complete_workflow.py
```

⏱️ 40-70 minutes (trains both algorithms, generates all plots)

## Option 2: Manual Steps

```powershell
# Step 1: Train (30-60 min)
uv run python training/quick_train.py --timesteps 50000 --algorithm both

# Step 2: Generate plots (5-10 min)
uv run python evaluation/generate_report_plots.py

# Step 3: Record demo video (start recording first!)
uv run python demo.py --model models/quick_ppo/ppo_final.zip --difficulty medium --pattern rush
```

## Then: Write Report

- Open `COMPLETE_GUIDE.md`
- Copy template sections
- Replace [YOUR DATA] with numbers from `models/quick_ppo/results.json`
- Include all 5 plots from `results/` folder

---

## 📊 What You Get

### After Training:

- ✅ `models/quick_ppo/ppo_final.zip` - PPO model
- ✅ `models/quick_dqn/dqn_final.zip` - DQN model
- ✅ `models/quick_*/results.json` - Performance metrics

### After Plot Generation:

- ✅ `results/1_cumulative_rewards.png`
- ✅ `results/2_training_metrics.png`
- ✅ `results/3_convergence.png`
- ✅ `results/4_generalization.png`
- ✅ `results/5_performance_summary.png`

---

## 🎥 Video Checklist

- [ ] Screen recording started (Win+G or OBS)
- [ ] Camera visible ✅
- [ ] Full screen shared ✅
- [ ] Run demo with trained model
- [ ] Explain what's happening
- [ ] Show for 1-2 minutes
- [ ] Upload and get link

---

## 📝 Report Sections (All in COMPLETE_GUIDE.md)

1. ✅ Project Overview - COPY-PASTE READY
2. ✅ Environment Description - ALL DETAILS PROVIDED
3. ✅ Reward Structure - MATHEMATICAL FORMULATION
4. ✅ Implementation Details - COPY-PASTE READY
5. ✅ Hyperparameter Tables - TEMPLATE PROVIDED
6. ✅ Plot Descriptions - ALL 5 WRITTEN
7. ✅ Conclusion - EXAMPLE PROVIDED

---

## ⚡ Time Breakdown

- ⏱️ Training: 30-60 min
- ⏱️ Plots: 5-10 min
- ⏱️ Video: 5 min
- ⏱️ Report: 2-3 hours
- **⏱️ TOTAL: 3-4 hours**

---

## 🎯 Expected Score: 50/50

| Criterion      | Score    |
| -------------- | -------- |
| Environment    | 10/10 ✅ |
| Training       | 10/10 ✅ |
| Visualization  | 10/10 ✅ |
| Implementation | 10/10 ✅ |
| Discussion     | 10/10 ✅ |

---

## 🆘 If Issues

See `COMPLETE_GUIDE.md` Section: "🆘 Troubleshooting"

---

**NOW GO GET THAT 50/50! 🚀**

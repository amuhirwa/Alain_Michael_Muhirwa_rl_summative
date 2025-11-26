# 🎥 VIDEO RECORDING - QUICK REFERENCE CARD

## ✅ REQUIREMENTS CHECKLIST

```
TECHNICAL SETUP:
  [ ] Share ENTIRE SCREEN (not just window)
  [ ] Camera ON (face visible throughout)
  [ ] Two terminals visible: GUI + Verbose output
  [ ] Audio clear (test microphone)

CONTENT REQUIRED:
  [ ] State the problem briefly
  [ ] State agent behavior
  [ ] Explain reward structure briefly
  [ ] State objective of the agent
  [ ] Run simulation (GUI + Terminal verbose)
  [ ] Explain agent performance during simulation
```

---

## 🎬 EXACT SCRIPT (3 Minutes)

### [0:00-0:30] Problem Statement

```
"Hi, I'm Alain Michael Muhirwa. This is my reinforcement learning
crowd control system.

THE PROBLEM: In crowded venues like stadiums and concerts, dangerous
overcrowding occurs at exits and gates, leading to stampedes and safety
hazards. This system manages crowd flow intelligently to prevent
overcrowding while efficiently evacuating people."
```

**Camera ON, Full Screen Share** ✅

---

### [0:30-1:00] Agent Behavior

```
"THE AGENT acts as a centralized crowd control manager. It observes
crowd density across a 15×15 grid and can perform 25 different actions:

- Move 4 barriers to redirect crowd flow
- Open or close 3 exit gates
- Influence crowd movement direction
- Trigger emergency protocols

The agent learns from experience which actions prevent dangerous
situations while maintaining efficient evacuation."
```

---

### [1:00-1:30] Reward Structure & Objective

```
"THE REWARD STRUCTURE balances multiple objectives:

1. DENSITY PENALTY: Negative reward for exceeding 1.5 people per cell
2. SAFETY PENALTY: Large negative (-10) for critical density over 3.5
3. EFFICIENCY BONUS: Positive reward (+0.5) for successful evacuations
4. PANIC CONTROL: Penalty for high panic levels

THE OBJECTIVE: Keep density below 3.5 people per cell while efficiently
evacuating crowds. The agent must balance safety and efficiency."
```

---

### [1:30-2:30] Run Simulation (GUI + Terminal)

**SETUP BEFORE RECORDING:**

```powershell
# Split screen: GUI left, Terminal right

# Terminal command:
python main.py --visualize --model models/a2c/config_2_high_lr/best_model.zip --verbose
```

**Say while running:**

```
"Now running my BEST-PERFORMING AGENT - A2C with optimized hyperparameters.
This achieved 1289.6 mean reward, highest of all algorithms tested.

[Point to GUI]
- Blue circles: crowd agents (size = panic level)
- Red squares: barriers the agent is moving
- Green markers: exit gates
- Heatmap: green=safe, yellow=warning, red=critical

[Point to Terminal]
The verbose output shows:
- Timesteps and actions taken
- Crowd density levels in real-time
- Rewards being accumulated
- Safety metrics tracked

Watch as the agent proactively moves barriers before density gets critical."
```

**CRITICAL: Let simulation run 30-60 seconds with both GUI and Terminal visible**

---

### [2:30-3:00] Agent Performance Explanation

```
"Looking at the AGENT PERFORMANCE:

WHAT IT LEARNED:
- Proactively repositions barriers BEFORE density becomes critical
- Keeps gates open in high-traffic areas
- Redirects flow away from building congestion
- Prioritizes safety over speed

METRICS:
- 1289.6 mean reward (best of all algorithms)
- Converged in 39,000 timesteps (fastest)
- Trained with A2C: learning rate 1e-3, 5-step rollouts

The terminal shows positive rewards - the agent successfully manages
crowd density while evacuating people safely. This demonstrates that
reinforcement learning can learn complex crowd control strategies
balancing multiple competing objectives."
```

---

## 🖥️ SCREEN LAYOUT

```
┌─────────────────────────────────────────────────────────────────┐
│  [Your Video Camera - Top Right Corner]                        │
├──────────────────────────┬──────────────────────────────────────┤
│                          │                                      │
│   VISUALIZATION GUI      │   TERMINAL (VERBOSE OUTPUT)         │
│   (Left Half)            │   (Right Half)                      │
│                          │                                      │
│   - Blue circles (agents)│   $ python main.py --visualize...   │
│   - Red squares (barriers│   Timestep: 1                        │
│   - Green gates          │   Action: Move Barrier 0 Up         │
│   - Density heatmap      │   Density: 2.3 (Safe)              │
│   - Panic colors         │   Reward: +2.45                     │
│   - HUD metrics          │   Agents: 45/120                    │
│                          │   Exits: 8                          │
│                          │   Timestep: 2...                    │
│                          │                                      │
└──────────────────────────┴──────────────────────────────────────┘
```

**BOTH MUST BE VISIBLE SIMULTANEOUSLY!**

---

## ⚙️ SETUP INSTRUCTIONS

### 1. Prepare Terminals (BEFORE Recording)

```powershell
# Terminal 1: Position on RIGHT half of screen
cd c:\Users\mbric\Documents\Sook\Alain_Michael_Muhirwa_rl_summative
# DON'T RUN YET - wait for recording to start
```

### 2. Position Camera

- Top right corner overlay (OBS Studio)
- OR bottom right corner
- Face must be visible throughout
- Don't block critical GUI/terminal areas

### 3. Test Run (BEFORE Recording)

```powershell
# Make sure this works:
python main.py --visualize --model models/a2c/config_2_high_lr/best_model.zip --verbose

# Expected: GUI window opens AND terminal shows verbose logs
# If GUI doesn't open, try:
python main.py --visualize --model models/ppo/config_2_high_lr/best_model.zip --verbose
```

### 4. Screen Recording Setup

**Option A: OBS Studio (Recommended)**

```
1. Add Source → Display Capture (captures entire screen)
2. Add Source → Video Capture Device (webcam)
3. Position webcam in corner (right-click → Transform → Scale to 15%)
4. Audio: Enable Desktop Audio + Microphone
5. Start Recording
```

**Option B: Windows Game Bar**

```
1. Press Win + G
2. Settings → "Record my microphone" ON
3. Settings → "Record game only" OFF (need full screen!)
4. Click Record button
```

---

## 📝 SPEAKING NOTES

### Key Numbers to Mention:

- **1289.6** - mean reward (best model)
- **39,000** timesteps to convergence (fastest)
- **25** discrete actions available
- **15×15** grid environment
- **3.5** people/cell - critical density threshold
- **1.5** people/cell - target density

### Tone & Delivery:

- ✅ Confident and clear
- ✅ Point at screen when explaining
- ✅ Show enthusiasm about results
- ✅ Professional but personable
- ❌ Don't rush (speak at moderate pace)
- ❌ Don't monotone (vary your tone)

---

## 🚨 COMMON MISTAKES TO AVOID

```
❌ Window-only capture          → ✅ Entire screen
❌ Camera off                   → ✅ Camera on throughout
❌ Only showing GUI             → ✅ Show GUI + Terminal verbose
❌ Not explaining what agent does → ✅ Clearly state agent behavior
❌ Vague reward explanation     → ✅ Specific reward components
❌ Not stating objective        → ✅ Clearly state objective
❌ Silent simulation            → ✅ Explain performance during run
❌ Terminal output hidden       → ✅ Both GUI and Terminal visible
```

---

## ⏱️ TIMING BREAKDOWN

```
0:00-0:30  [30s]  Problem statement (what's being solved)
0:30-1:00  [30s]  Agent behavior (what agent can do)
1:00-1:30  [30s]  Reward structure + objective
1:30-2:30  [60s]  Run simulation (BOTH GUI + Terminal)
2:30-3:00  [30s]  Explain agent performance
────────────────────────────────────────────────────
Total:     3:00   All requirements covered ✅
```

---

## ✅ PRE-RECORDING CHECKLIST

**5 Minutes Before Recording:**

- [ ] Close unnecessary apps (Discord, Spotify, etc.)
- [ ] Clear desktop (optional but professional)
- [ ] Position camera (test it's visible)
- [ ] Test microphone (say "testing 1, 2, 3")
- [ ] Open terminal in project directory
- [ ] Test visualization runs: `python main.py --visualize --model models/a2c/config_2_high_lr/best_model.zip --verbose`
- [ ] Arrange screen: space for GUI left, terminal right
- [ ] Set recording software to capture entire screen
- [ ] Webcam overlay positioned (not blocking content)
- [ ] Good lighting on face
- [ ] Read script one more time

**Immediately Before Recording:**

- [ ] Close test visualization
- [ ] Terminal ready (in project directory)
- [ ] Camera on and visible
- [ ] Microphone working
- [ ] Recording software ready
- [ ] Take a deep breath
- [ ] Click Record
- [ ] Start speaking with confidence!

---

## 🎯 SUCCESS CRITERIA

Your video must show:

- ✅ Entire screen shared (not just window)
- ✅ Camera showing your face throughout
- ✅ Problem stated clearly
- ✅ Agent behavior explained
- ✅ Reward structure described
- ✅ Agent objective stated
- ✅ Simulation running with GUI visible
- ✅ Terminal verbose output visible simultaneously
- ✅ Agent performance explained during/after simulation
- ✅ Duration: 2:45-3:00 minutes
- ✅ Audio clear and professional

---

## 🚀 WHEN YOU'RE READY

1. **Practice once** (without recording)
2. **Check all items** in Pre-Recording Checklist
3. **Start recording**
4. **Follow the script** ([0:00] through [3:00])
5. **Stop recording**
6. **Watch it back** (audio/video good?)
7. **Upload to YouTube** (Unlisted)
8. **Add link to FINAL_REPORT.md** line 3
9. **Submit with confidence!**

---

## 💡 PRO TIP

**If nervous, remember:**

- This is YOUR work - you know it best
- 48 hours of effort behind this project
- The hard part (implementation) is done
- Just showing what you built
- You've got this! 🚀

**If you make a mistake:**

- Pause, take a breath
- Start that section again
- Edit it out later
- OR just keep going (minor mistakes are fine!)

---

## 📞 EMERGENCY BACKUP

**If visualization won't run during recording:**

```
Option 1: Use different model
python main.py --visualize --model models/ppo/config_2_high_lr/best_model.zip --verbose

Option 2: Show plots instead
"Due to a technical issue, let me show the training results instead..."
[Show cumulative_rewards.png and explain]

Still meets most requirements, gets 9/10 instead of 10/10
```

---

## 🎬 YOU'RE READY!

Everything is prepared:

- ✅ Script written
- ✅ Best model identified (A2C config_2_high_lr)
- ✅ Commands ready
- ✅ Screen layout planned
- ✅ Requirements understood

**Next step: Press Record and follow this guide!**

**Expected result: 10/10 on performance video requirement**

**Good luck! You've built something impressive - now show it! 🌟**

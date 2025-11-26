# Video Recording Guide - Step by Step

## 🎥 Setup (5 minutes)

### 1. Prepare Your Workspace

- [ ] Close unnecessary applications (Discord, Spotify, social media)
- [ ] Clean up desktop (optional but professional)
- [ ] Test webcam position (can you see your face?)
- [ ] Test microphone (say "testing 1, 2, 3" and listen back)
- [ ] Adjust lighting (face a window or lamp)

### 2. Open Required Windows

```powershell
# Terminal 1: Run visualization
cd c:\Users\mbric\Documents\Sook\Alain_Michael_Muhirwa_rl_summative

# Terminal 2: Ready for commands (optional)
# Keep VS Code open with training scripts visible
# Open results folder in File Explorer
```

### 3. Open Recording Software

**Option A: OBS Studio (Recommended)**

- Download: https://obsproject.com/
- Settings → Video → Base Resolution: 1920×1080
- Settings → Video → FPS: 30
- Add Source → Display Capture (captures everything)
- Add Source → Video Capture Device (webcam)
- Position webcam in corner (right-click → Transform → Scale)

**Option B: Windows Game Bar (Built-in)**

- Press `Win + G`
- Click Settings → Capture → "Record my microphone" ON
- "Record game only" OFF (need full screen)

---

## 📹 Recording Script (3 minutes)

### [0:00-0:30] Introduction & Problem Statement

**Say:**

```
"Hi, I'm Alain Michael Muhirwa. This is my reinforcement learning
crowd control system for the RL summative assignment.

THE PROBLEM: In crowded venues like stadiums and concerts, dangerous
overcrowding can occur at exits and gates, leading to stampedes and
safety hazards. This system needs to manage crowd flow intelligently
to prevent overcrowding while efficiently evacuating people."
```

**Do:**

- [ ] Wave or gesture naturally
- [ ] Camera ON - ensure face visible
- [ ] Share ENTIRE SCREEN (not just window)
- [ ] Speak clearly about the problem

---

### [0:30-1:00] Agent Behavior & Objective

**Say:**

```
"THE AGENT acts as a centralized crowd control manager. It observes
crowd density across a 15×15 grid and can perform 25 different actions:
- Move 4 barriers to redirect crowd flow
- Open or close 3 exit gates
- Influence crowd movement direction
- Trigger emergency protocols

THE OBJECTIVE: The agent must prevent dangerous overcrowding - keeping
density below 3.5 people per cell - while efficiently evacuating crowds
through available exits. It needs to balance safety and efficiency."
```

**Do:**

- [ ] Speak confidently about agent capabilities
- [ ] Explain the 25 actions briefly
- [ ] State the objective clearly

---

### [1:00-1:30] Reward Structure

**Say:**

```
"THE REWARD STRUCTURE balances multiple objectives:

1. DENSITY PENALTY: Negative reward for cells exceeding 1.5 people -
   the agent is punished for letting areas get crowded.

2. SAFETY REWARD: Large negative penalty (-10) for any cell exceeding
   critical density of 3.5 people - this prevents dangerous situations.

3. EFFICIENCY BONUS: Positive reward (+0.5) for each person successfully
   evacuated through gates - encourages throughput.

4. PANIC CONTROL: Penalty for high panic levels - the agent must keep
   crowds calm.

The agent learns to balance all these factors to maximize cumulative reward."
```

**Do:**

- [ ] Speak clearly about each reward component
- [ ] Emphasize safety vs efficiency tradeoff

---

### [1:30-2:30] Run Best Agent Simulation

**BEFORE STARTING RECORDING - PREPARE THIS:**

```powershell
# Open TWO terminals side by side:
# Terminal 1: Run with verbose output
cd c:\Users\mbric\Documents\Sook\Alain_Michael_Muhirwa_rl_summative
python main.py --visualize --model models/a2c/config_2_high_lr/best_model.zip --verbose

# This will show BOTH:
# - GUI window (visualization)
# - Terminal output (verbose logs)
```

**Say while simulation runs:**

```
"Now let me run my BEST-PERFORMING AGENT - A2C with optimized hyperparameters.
This achieved 1289.6 mean reward, the highest of all algorithms tested.

Watch the GUI on the left:
- Blue circles are crowd agents - size shows panic level
- Red squares are barriers the agent is moving
- Green markers are exit gates
- The heatmap shows density: green is safe, yellow is warning, red is critical

And the TERMINAL on the right shows verbose output:
- Current timestep and actions taken
- Crowd density levels
- Rewards being accumulated
- Safety metrics"
```

**Do:**

- [ ] Arrange screen: GUI left, Terminal right (both visible)
- [ ] Point at GUI elements as you explain them
- [ ] Point at terminal output as data streams
- [ ] Let it run for 30-45 seconds minimum
- [ ] Highlight when agent takes actions (barriers move, gates toggle)

---

### [0:20-1:15] Implementation Overview

**Say:**

```
"I implemented and compared four reinforcement learning algorithms.
Let me show you the code structure."
```

**Do:**

- [ ] Open VS Code (or File Explorer)
- [ ] Navigate through folders:
  - `environment/` - "This is the custom environment"
  - `training/` - "Training scripts for all 4 algorithms"
  - `models/` - "Over 40 trained model configurations"

**Show training script:**

```powershell
# Open A2C training script
code training/a2c_training.py
# Scroll to hyperparameter configs (around line 46)
```

**Say while showing code:**

```
"Each algorithm has 10+ hyperparameter configurations.
Here's A2C config 2 which won with these parameters:
- Learning rate: 1e-3 - quite aggressive
- Very short rollouts: just 5 steps for fast updates
- Minimal entropy: 0.01 for exploitation after learning
- Full Monte Carlo advantage estimation with GAE lambda 1.0

I tested learning rates from 1e-4 to 1.5e-3,
entropy from 0.01 to 0.5,
and different rollout lengths."
```

---

### [2:30-3:00] Agent Performance Explanation

**Say while simulation continues to run:**

```
"Looking at the AGENT PERFORMANCE in this simulation:

WHAT THE AGENT LEARNED:
- It learned to proactively move barriers BEFORE density gets critical
- It keeps gates open in high-traffic areas
- When it sees density building up, it redirects flow away from that area
- It prioritizes safety over speed - better to evacuate slowly than cause panic

PERFORMANCE METRICS:
- This agent achieved 1289.6 mean reward across evaluations
- It converged in only 39,000 timesteps - the fastest of all algorithms
- Trained with A2C using learning rate 1e-3, very short rollouts of 5 steps,
  and minimal entropy for exploitation

You can see in the terminal the rewards are positive and the agent is
successfully managing crowd density while evacuating people safely.

This demonstrates that reinforcement learning can effectively learn
complex crowd control strategies that balance multiple competing objectives."
```

**Do:**

- [ ] Point at specific behaviors in the GUI
- [ ] Reference terminal output showing positive rewards
- [ ] Speak confidently about what was learned
- [ ] Conclude with impact statement

---

### [2:00-2:35] Exploration vs Exploitation

**Say:**

```
"The exploration-exploitation balance was critical for success."
```

**Do:**

- [ ] Open `2_training_stability.png` or training script

**Say:**

```
"For DQN, I used epsilon-greedy exploration.
Starting at 100% random actions, decaying to 20% by the end.
That final 20% exploration was important - configurations that
dropped to 10% performed worse because they stopped discovering
strategic barrier placements.

For A2C and PPO, I used entropy regularization.
Surprisingly, very low entropy worked best - just 0.01.
High entropy like 0.3 actually degraded performance by 30%
because the agent kept taking random actions after it had already
learned good policies.

This differs from typical RL advice to maintain high exploration,
but it makes sense here - once you learn which barriers prevent
overcrowding, you want to exploit that knowledge consistently."
```

---

### [2:35-2:55] Weaknesses & Improvements

**Say:**

```
"Now for the critical analysis. The main weakness?"
```

**Do:**

- [ ] Open `4_generalization.png`

**Say while showing:**

```
"All algorithms failed catastrophically on evacuation scenarios.
You can see negative rewards across the board.
The agents never learned to handle high-panic situations.

This happened because my training distribution was biased toward
normal crowd conditions. The environment mostly generated gradual
arrivals and occasional density spikes, but almost never full-scale
panic evacuations.

For future work, I would use curriculum learning:
Start with easy scenarios, gradually introduce rush patterns,
then add evacuation situations.

I'd also add explicit panic-reduction bonuses to the reward function
to make safety the top priority.

And potentially use adversarial training where a second agent
tries to create dangerous situations."
```

---

### [2:55-3:00] Conclusion

**Say:**

```
"In summary: A2C won due to fast policy updates that exploited
the dense reward signal. But the generalization gap shows that
real-world deployment would need more diverse training scenarios.

Thank you for watching! Full code and report are on GitHub."
```

**Do:**

- [ ] Wave or thumbs up
- [ ] Stop recording

---

## ✂️ After Recording (10 minutes)

### 1. Review the Video

- [ ] Watch entire video
- [ ] Check audio levels (can you hear clearly?)
- [ ] Verify screen is visible (not blurry)
- [ ] Confirm webcam shows your face
- [ ] Check video length (should be 2:45-3:00)

### 2. Edit (Optional)

If you stumbled or need to fix something:

- **Windows:** Use built-in Photos app (trim ends)
- **Free tools:** DaVinci Resolve, Shotcut, OpenShot
- **Easy fix:** Re-record just that section, splice in

### 3. Export Settings

- Format: MP4
- Resolution: 1920×1080 (minimum 1280×720)
- Frame rate: 30 FPS
- Bitrate: 5-10 Mbps (balances quality/file size)

### 4. Upload to YouTube

```
1. Go to: https://studio.youtube.com/
2. Click "CREATE" → "Upload videos"
3. Select your video file
4. Title: "RL Crowd Control - Alain Michael Muhirwa"
5. Description: "Reinforcement Learning Summative Assignment -
   Crowd Control System comparing DQN, PPO, A2C, and REINFORCE"
6. Visibility: "Unlisted" (not Private, not Public)
7. Click "SAVE"
8. Wait for processing (1-5 minutes)
9. Copy shareable link
```

### 5. Update Report

```powershell
# Open report
code results/FINAL_REPORT.md

# Update line 3 with YouTube link:
**Video Recording:** [https://youtu.be/YOUR_VIDEO_ID]
```

---

## 🎬 30-Second Visualization Demo

### Quick GIF Recording

**Tools:**

- **ScreenToGif** (Windows): https://www.screentogif.com/
- **LICEcap** (Windows/Mac): https://www.cockos.com/licecap/
- **Peek** (Linux): https://github.com/phw/peek

**Steps:**

```powershell
1. Open ScreenToGif → Recorder
2. Position frame over visualization window only (not entire screen)
3. Click Record (or press F7)
4. Run: python main.py --visualize --model models/a2c/config_2_high_lr/best_model.zip
5. Let run for exactly 30 seconds
6. Stop recording (F8)
7. Edit:
   - Trim to best 30 seconds
   - Add text overlay (optional): "A2C Managing Crowd Density"
   - Reduce to 15 FPS if file size > 10MB
8. Save As → results/visualization_demo.gif
```

**What to capture in 30 seconds:**

- [0:00-0:10] Initial crowd state, barriers visible
- [0:10-0:20] Agent moves barrier, density changes color
- [0:20-0:30] Gate operations, agents exiting

---

## 📋 Pre-Recording Checklist

### Environment

- [ ] Visualization runs smoothly (no crashes)
- [ ] FPS is acceptable (>20 FPS)
- [ ] Colors are visible (density heatmap clear)
- [ ] Window size is appropriate (not too small)

### Audio/Video

- [ ] Microphone works (test recording)
- [ ] No background noise (close windows, turn off fans)
- [ ] Webcam focused (not blurry)
- [ ] Good lighting on your face

### Materials Ready

- [ ] All plots generated (`results/*.png`)
- [ ] VS Code open with training scripts
- [ ] File Explorer in results folder
- [ ] Browser ready (if showing GitHub)

### Recording Software

- [ ] OBS/Game Bar configured
- [ ] Test recording done (5-second test)
- [ ] Storage space available (>500MB free)
- [ ] Screen capture set to correct display

---

## 🚨 Common Issues & Fixes

### Issue: "Visualization window appears then crashes"

**Fix:**

```powershell
# Try without Panda3D
python main.py --visualize --model models/a2c/config_2_high_lr/best_model.zip --renderer pygame

# Or reduce render quality
python main.py --visualize --model models/a2c/config_2_high_lr/best_model.zip --render-fps 10
```

### Issue: "Video file too large (>100MB)"

**Fix:**

- Re-export with lower bitrate (3 Mbps)
- Reduce resolution to 1280×720
- Use YouTube/Drive (no file size limit)

### Issue: "Webcam not showing in recording"

**Fix OBS:**

- Add Source → Video Capture Device
- Select your webcam from dropdown
- Right-click → Transform → Fit to Screen (then scale down)

**Fix Game Bar:**

- Settings → Capture → "Record my camera" ON

### Issue: "Audio is too quiet"

**Fix:**

- Move microphone closer (6-12 inches from mouth)
- Windows Settings → Sound → Input → Volume = 80-100%
- OBS: Audio Mixer → Desktop Audio = -6dB, Mic = +3dB

### Issue: "Screen too cluttered"

**Fix:**

- Close taskbar icons (right-click → Exit)
- Windows + D (show desktop)
- Hide desktop icons (right-click desktop → View → Show desktop icons OFF)

---

## 💡 Pro Tips

### Make It Engaging

1. **Vary your tone:** Don't monotone - show excitement!
2. **Use hand gestures:** Point at screen, emphasize with hands
3. **Smile:** Especially in intro and conclusion
4. **Pause briefly:** Between sections for easy editing later

### Technical Quality

1. **Speak clearly:** Enunciate, don't rush
2. **Look at camera:** Briefly, then back to screen
3. **Stay in frame:** Don't lean out of webcam view
4. **Stable camera:** Don't move laptop/webcam during recording

### Content Quality

1. **Use specific numbers:** "1289.6 reward" not "high reward"
2. **Show confidence:** You built something impressive!
3. **Be critical:** Rubric rewards identifying weaknesses
4. **Tell a story:** Intro → Demo → Results → Analysis → Conclusion

---

## ⏱️ Time Estimates

| Task                            | Time Required   |
| ------------------------------- | --------------- |
| Setup (software install, test)  | 10 minutes      |
| Practice run-through            | 5 minutes       |
| Actual recording (with retakes) | 20 minutes      |
| Review & minor edits            | 5 minutes       |
| YouTube upload & processing     | 5 minutes       |
| Update report with link         | 2 minutes       |
| **Total**                       | **~45 minutes** |

---

## 🎯 Success Criteria

Your video should:

- ✅ Be 2:45-3:00 minutes long
- ✅ Show entire screen (not just window)
- ✅ Include webcam with your face visible
- ✅ Have clear audio (no excessive noise)
- ✅ Demonstrate trained agent running
- ✅ Show at least 2 performance plots
- ✅ Discuss specific metrics (numbers!)
- ✅ Explain exploration vs exploitation
- ✅ Identify weaknesses with solutions
- ✅ Have professional presentation (confident, clear)

---

## 📞 Last-Minute Help

### If You're Running Out of Time

**Minimum viable video (90 seconds):**

1. [0:00-0:15] Intro + name
2. [0:15-0:45] Show visualization running
3. [0:45-1:15] Show ONE plot, state best number
4. [1:15-1:30] Name ONE weakness + solution

This gets you 8-9/10 instead of 10/10, but better than nothing!

### If Visualization Won't Run

**Backup plan:**

1. Show static plots only (open in browser)
2. Explain what WOULD happen in visualization
3. Apologize and explain technical difficulty
4. Focus more on metrics and analysis

You'll lose 1-2 points on visualization but can still get 8/10.

### If Recording Software Fails

**Emergency backup:**

- Use phone camera to record laptop screen
- Not ideal, but acceptable
- Make sure screen is visible and audio clear

---

**You've got this! The hard work is done - now just showcase it! 🚀**

**Remember:** Confidence is key. You built a sophisticated RL system with 40+ trained models, publication-quality plots, and novel insights. Be proud and show it!

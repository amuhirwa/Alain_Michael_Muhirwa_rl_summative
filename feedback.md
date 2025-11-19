Looking at your code, here are **concrete, actionable steps** to implement the novel contributions I suggested:

## 1. ENVIRONMENT CHANGES (Most Critical)

### A. Add Individual Agents (Replace Pure Density)

**Current problem:** You only track density, not individual people with goals.

```python
# Add to CrowdControlEnv.__init__()
self.agents = []  # List of individual agents
self.MAX_AGENTS = 200

# Replace grid_density initialization with:
class Agent:
    def __init__(self, x, y, goal_gate, panic_level=0.0):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.goal_gate = goal_gate
        self.panic_level = panic_level  # 0-1, affects behavior

# In reset():
self.agents = []
for entrance in entrance_points:
    for _ in range(np.random.randint(30, 50)):
        agent = Agent(
            x=entrance[1] + np.random.uniform(-1, 1),
            y=entrance[0] + np.random.uniform(-1, 1),
            goal_gate=np.random.randint(0, self.NUM_GATES),
            panic_level=0.0
        )
        self.agents.append(agent)
```

### B. Implement Social Force Model

**Add this method to simulate realistic crowd behavior:**

```python
def _update_agent_forces(self, agent):
    """Social Force Model for realistic crowd dynamics"""
    # Goal attraction force
    target_gate = self.gates[agent.goal_gate]
    dx = target_gate[0] - agent.x
    dy = target_gate[1] - agent.y
    dist = np.sqrt(dx**2 + dy**2) + 0.01

    desired_speed = 1.0 * (1 + agent.panic_level)  # Panic increases speed
    goal_force_x = (desired_speed * dx / dist - agent.vx) * 0.5
    goal_force_y = (desired_speed * dy / dist - agent.vy) * 0.5

    # Repulsion from other agents
    repulsion_x, repulsion_y = 0.0, 0.0
    for other in self.agents:
        if other is agent:
            continue
        odx = agent.x - other.x
        ody = agent.y - other.y
        odist = np.sqrt(odx**2 + ody**2) + 0.01

        if odist < 2.0:  # Within personal space
            strength = 2.0 * np.exp(-odist / 0.5)
            repulsion_x += strength * odx / odist
            repulsion_y += strength * ody / odist

    # Repulsion from barriers
    barrier_force_x, barrier_force_y = 0.0, 0.0
    for bx, by in self.barriers:
        bdx = agent.x - bx
        bdy = agent.y - by
        bdist = np.sqrt(bdx**2 + bdy**2) + 0.01
        if bdist < 1.5:
            strength = 5.0 * np.exp(-bdist / 0.3)
            barrier_force_x += strength * bdx / bdist
            barrier_force_y += strength * bdy / bdist

    # Total force
    total_fx = goal_force_x + repulsion_x + barrier_force_x
    total_fy = goal_force_y + repulsion_y + barrier_force_y

    return total_fx, total_fy
```

### C. Add Temporal Dynamics (Crowd Waves)

**Replace constant crowd spawn with time-based patterns:**

```python
def __init__(self, ...):
    # Add these
    self.crowd_arrival_pattern = 'rush'  # 'rush', 'steady', 'evacuation'
    self.rush_peak_time = 0.3  # Peak at 30% of episode

def _spawn_new_arrivals(self):
    """Time-based crowd arrival patterns"""
    progress = self.timestep / self.MAX_STEPS

    if self.crowd_arrival_pattern == 'rush':
        # Concert/event entry - peak early, taper off
        spawn_rate = 5.0 * np.exp(-((progress - self.rush_peak_time) ** 2) / 0.05)
    elif self.crowd_arrival_pattern == 'steady':
        spawn_rate = 2.0 if progress < 0.8 else 0.0
    elif self.crowd_arrival_pattern == 'evacuation':
        # Emergency - everyone arrives at once
        spawn_rate = 10.0 if progress < 0.1 else 0.0

    num_new = int(spawn_rate)
    for _ in range(num_new):
        if len(self.agents) < self.MAX_AGENTS:
            entrance = self.np_random.choice(len(entrance_points))
            ex, ey = entrance_points[entrance]
            self.agents.append(Agent(
                x=ex + self.np_random.uniform(-1, 1),
                y=ey + self.np_random.uniform(-1, 1),
                goal_gate=self.np_random.integers(0, self.NUM_GATES),
                panic_level=0.3 if self.crowd_arrival_pattern == 'evacuation' else 0.0
            ))
```

### D. Add Realistic Infrastructure Constraints

**Make barriers/gates have costs and delays:**

```python
def __init__(self, ...):
    self.gate_open_times = [0, 0, 0]  # Timesteps when gates were opened
    self.gate_transition_delay = 10  # Takes 10 steps to fully open/close
    self.barrier_move_cooldown = [0, 0, 0, 0]  # Cooldown per barrier
    self.BARRIER_MOVE_COST = 5  # Time units to move a barrier

def _move_barrier(self, barrier_id: int, direction: int):
    # Check cooldown
    if self.barrier_move_cooldown[barrier_id] > 0:
        return  # Can't move yet

    # Original movement code...

    # Set cooldown
    self.barrier_move_cooldown[barrier_id] = self.BARRIER_MOVE_COST

def _toggle_gate(self, gate_id: int):
    # Check if gate is in transition
    time_since_toggle = self.timestep - self.gate_open_times[gate_id]
    if time_since_toggle < self.gate_transition_delay:
        return  # Gate still transitioning

    # Toggle gate
    self.gates[gate_id][2] = 1.0 - self.gates[gate_id][2]
    self.gate_open_times[gate_id] = self.timestep

# Update cooldowns each step
def step(self, action):
    # ... existing code ...

    # Decrement cooldowns
    for i in range(len(self.barrier_move_cooldown)):
        if self.barrier_move_cooldown[i] > 0:
            self.barrier_move_cooldown[i] -= 1
```

### E. Add Panic/Adversarial Behavior

**Create worst-case scenarios:**

```python
def __init__(self, ...):
    self.adversarial_mode = False  # Enable for harder training
    self.panic_trigger_threshold = 7.0  # Density that triggers panic

def _update_panic_levels(self):
    """Panic spreads based on local density"""
    for agent in self.agents:
        # Calculate local density
        local_density = 0
        for other in self.agents:
            dist = np.sqrt((agent.x - other.x)**2 + (agent.y - other.y)**2)
            if dist < 2.0:
                local_density += 1

        # Panic increases with density
        if local_density > self.panic_trigger_threshold:
            agent.panic_level = min(1.0, agent.panic_level + 0.1)
        else:
            agent.panic_level = max(0.0, agent.panic_level - 0.05)

        # Panic spreads to nearby agents
        if agent.panic_level > 0.5:
            for other in self.agents:
                dist = np.sqrt((agent.x - other.x)**2 + (agent.y - other.y)**2)
                if dist < 1.5:
                    other.panic_level = min(1.0, other.panic_level + 0.05)

def _adversarial_scenario(self):
    """Create worst-case scenarios for training"""
    if not self.adversarial_mode:
        return

    if self.np_random.random() < 0.1:  # 10% chance each step
        scenario = self.np_random.choice(['gate_failure', 'sudden_rush', 'bottleneck'])

        if scenario == 'gate_failure':
            # Random gate closes unexpectedly
            gate_id = self.np_random.integers(0, self.NUM_GATES)
            self.gates[gate_id][2] = 0.0

        elif scenario == 'sudden_rush':
            # Large crowd surge
            for _ in range(20):
                if len(self.agents) < self.MAX_AGENTS:
                    entrance = self.np_random.choice(len(entrance_points))
                    ex, ey = entrance_points[entrance]
                    self.agents.append(Agent(ex, ey,
                                           self.np_random.integers(0, self.NUM_GATES),
                                           panic_level=0.5))

        elif scenario == 'bottleneck':
            # Close multiple gates temporarily
            for i in range(2):
                self.gates[i][2] = 0.0
```

---

## 2. OBSERVATION SPACE CHANGES

**Update observation to include new features:**

```python
def _get_observation(self) -> np.ndarray:
    # Convert agents to density grid (for backward compatibility)
    self.grid_density = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH))
    panic_grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH))

    for agent in self.agents:
        x, y = int(agent.x), int(agent.y)
        if 0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT:
            self.grid_density[y, x] += 1.0
            panic_grid[y, x] = max(panic_grid[y, x], agent.panic_level)

    # Normalize
    norm_density = np.clip(self.grid_density / self.MAX_CROWD_PER_CELL, 0, 1)

    obs = np.concatenate([
        norm_density.flatten(),
        panic_grid.flatten(),  # NEW: Panic levels
        self.velocity_x.flatten(),
        self.velocity_y.flatten(),
    ])

    # Gate states + transition status
    gate_states = []
    for i, gate in enumerate(self.gates):
        gate_states.append(gate[2])  # Open/closed
        # NEW: Gate transition progress
        time_since_toggle = self.timestep - self.gate_open_times[i]
        transition_progress = min(1.0, time_since_toggle / self.gate_transition_delay)
        gate_states.append(transition_progress)

    obs = np.concatenate([obs, np.array(gate_states, dtype=np.float32)])

    # Barrier positions + cooldowns
    barrier_data = []
    for i, (bx, by) in enumerate(self.barriers):
        barrier_data.extend([bx / self.GRID_WIDTH, by / self.GRID_HEIGHT])
        # NEW: Cooldown status
        cooldown_status = self.barrier_move_cooldown[i] / self.BARRIER_MOVE_COST
        barrier_data.append(cooldown_status)

    obs = np.concatenate([obs, np.array(barrier_data, dtype=np.float32)])

    # NEW: Scenario info
    scenario_info = [
        self.timestep / self.MAX_STEPS,
        len(self.agents) / self.MAX_AGENTS,
        np.mean([a.panic_level for a in self.agents]) if self.agents else 0.0,
    ]

    obs = np.concatenate([obs, np.array(scenario_info, dtype=np.float32)])

    return obs.astype(np.float32)
```

**Update observation space size:**

```python
def __init__(self, ...):
    obs_size = (
        self.GRID_WIDTH * self.GRID_HEIGHT * 4 +  # density + panic + 2 velocities
        self.NUM_GATES * 2 +  # state + transition
        self.NUM_BARRIERS * 3 +  # x, y, cooldown
        3  # timestep, agent count, avg panic
    )
    self.observation_space = spaces.Box(
        low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
    )
```

---

## 3. REWARD FUNCTION CHANGES

**Add multi-objective rewards:**

```python
def _calculate_safety_reward(self) -> float:
    """Enhanced safety with panic consideration"""
    reward = 0.0

    # Density-based danger
    dangerous_cells = np.sum(self.grid_density > (self.CRITICAL_DENSITY * 0.8))
    reward -= 2.0 * dangerous_cells

    # NEW: Panic penalty
    if self.agents:
        avg_panic = np.mean([a.panic_level for a in self.agents])
        reward -= 5.0 * avg_panic

        # Severe penalty if any agent has extreme panic
        max_panic = max(a.panic_level for a in self.agents)
        if max_panic > 0.8:
            reward -= 10.0

    return reward

def _calculate_infrastructure_cost(self) -> float:
    """NEW: Cost for using infrastructure"""
    cost = 0.0

    # Penalize frequent barrier moves
    active_cooldowns = sum(1 for c in self.barrier_move_cooldown if c > 0)
    cost -= 0.5 * active_cooldowns

    # Penalize having too many gates closed during peak times
    progress = self.timestep / self.MAX_STEPS
    if 0.2 < progress < 0.6:  # Peak arrival time
        closed_gates = sum(1 for g in self.gates if g[2] < 0.5)
        cost -= 1.0 * closed_gates

    return cost

# Update step() to use all rewards:
def step(self, action):
    # ... existing code ...

    density_reward = self._calculate_density_reward()
    safety_reward = self._calculate_safety_reward()
    efficiency_reward = self._calculate_efficiency_reward()
    infrastructure_cost = self._calculate_infrastructure_cost()  # NEW

    reward = density_reward + safety_reward + efficiency_reward + infrastructure_cost
```

---

## 4. RENDERING CHANGES

**Visualize individual agents instead of just density:**

```python
def __init__(self, env):
    # ... existing code ...
    self.agent_nodes = []  # Individual agent representations

def _initialize_scene_objects(self):
    # ... existing code ...

    # Create agent node pool
    for i in range(200):  # Max agents
        agent = self._create_box(0, 0, 0.3, 0.3, 0.3, 0.6)
        agent.setColor(0.2, 0.5, 0.8, 0.9)
        agent.reparentTo(self.render)
        agent.hide()
        self.agent_nodes.append(agent)

def update_scene(self, agents: List, gates: List, barriers: List, info: dict):
    """Update with individual agents"""

    # Update agent positions
    for i, agent in enumerate(agents):
        if i < len(self.agent_nodes):
            node = self.agent_nodes[i]
            node.show()
            node.setPos(agent.x, agent.y, 0.3)

            # Color based on panic level
            panic = agent.panic_level
            if panic > 0.7:
                node.setColor(1, 0, 0, 0.9)  # Red = panic
            elif panic > 0.3:
                node.setColor(1, 0.5, 0, 0.9)  # Orange = moderate stress
            else:
                node.setColor(0.2, 0.5, 0.8, 0.9)  # Blue = calm

    # Hide unused agent nodes
    for i in range(len(agents), len(self.agent_nodes)):
        self.agent_nodes[i].hide()

    # ... rest of existing update code ...
```

**Add panic level visualization:**

```python
def _setup_hud(self):
    # ... existing code ...

    # NEW: Panic indicator
    self.panic_text = OnscreenText(
        text="Panic Level: 0.0",
        pos=(-1.3, 0.3),
        scale=0.05,
        fg=(1, 1, 1, 1),
        align=TextNode.ALeft
    )
    self.hud_texts.append(self.panic_text)

def _update_hud(self, info: dict):
    # ... existing code ...

    # Update panic indicator
    panic = info.get('avg_panic', 0.0)
    self.panic_text.setText(f"Panic Level: {panic:.2f}")
    if panic > 0.7:
        self.panic_text.setFg((1, 0, 0, 1))
    elif panic > 0.3:
        self.panic_text.setFg((1, 0.5, 0, 1))
    else:
        self.panic_text.setFg((0, 1, 0, 1))
```

---

## 5. TRAINING CHANGES

**Add curriculum learning and scenario diversity:**

```python
# Add to train_ppo_configuration():

def create_env_with_curriculum(difficulty='easy'):
    """Create environment with different difficulty levels"""
    env = CrowdControlEnv()

    if difficulty == 'easy':
        env.MAX_AGENTS = 100
        env.crowd_arrival_pattern = 'steady'
        env.adversarial_mode = False
    elif difficulty == 'medium':
        env.MAX_AGENTS = 150
        env.crowd_arrival_pattern = 'rush'
        env.adversarial_mode = False
    elif difficulty == 'hard':
        env.MAX_AGENTS = 200
        env.crowd_arrival_pattern = 'rush'
        env.adversarial_mode = True

    return env

# Update training loop:
def train_with_curriculum(config, total_timesteps=200000):
    """Train with increasing difficulty"""

    stages = [
        ('easy', total_timesteps // 3),
        ('medium', total_timesteps // 3),
        ('hard', total_timesteps // 3),
    ]

    model = None

    for difficulty, timesteps in stages:
        print(f"\n--- Training Stage: {difficulty} ({timesteps} steps) ---")

        env = create_env_with_curriculum(difficulty)
        env = Monitor(env, f"logs/ppo/{config['name']}/{difficulty}")

        if model is None:
            # Create new model
            model = PPO("MlpPolicy", env, **config)
        else:
            # Continue training
            model.set_env(env)

        model.learn(total_timesteps=timesteps, progress_bar=True)

        env.close()

    return model
```

**Add scenario testing:**

```python
def evaluate_scenarios(model_path):
    """Test trained model on different scenarios"""
    model = PPO.load(model_path)

    scenarios = ['steady', 'rush', 'evacuation']
    results = {}

    for scenario in scenarios:
        env = CrowdControlEnv()
        env.crowd_arrival_pattern = scenario
        env.adversarial_mode = True

        rewards = []
        successes = 0

        for _ in range(20):
            obs, _ = env.reset()
            done = False
            episode_reward = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated

            rewards.append(episode_reward)
            if info['total_crowd'] < 10:
                successes += 1

        results[scenario] = {
            'mean_reward': np.mean(rewards),
            'success_rate': successes / 20,
            'max_panic': info.get('max_panic', 0),
        }

        env.close()

    return results
```

---

## 6. PRIORITY ORDER

**Implement in this order:**

1. **Week 1:** Individual agents + Social Force Model (sections 1A, 1B)
2. **Week 2:** Infrastructure constraints + temporal dynamics (sections 1C, 1D)
3. **Week 3:** Panic behavior + observation updates (sections 1E, 2)
4. **Week 4:** Enhanced rewards + rendering (sections 3, 4)
5. **Week 5:** Curriculum training + scenario testing (section 5)

This gives you a **publishable, novel system** focusing on infrastructure control with realistic crowd dynamics and adversarial safety testing.

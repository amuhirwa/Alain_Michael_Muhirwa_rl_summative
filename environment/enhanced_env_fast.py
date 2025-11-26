import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numba import njit
from typing import Optional, Tuple, Dict

# -----------------------------------------------------------
# JIT ACCELERATED PHYSICS
# -----------------------------------------------------------
@njit
def update_physics(
    x, y, vx, vy, goal, panic, gate_x, gate_y, gate_open,
    barrier_x, barrier_y, cell_heads, cell_next, cell_size,
    GRID_W, GRID_H, R_PERSONAL, R_BARRIER, DESIRED_SPEED
):
    N = x.shape[0]
    for i in range(N):
        gx = gate_x[goal[i]]
        gy = gate_y[goal[i]]
        dx = gx - x[i]
        dy = gy - y[i]
        dist = (dx*dx + dy*dy) ** 0.5 + 1e-6
        desired = DESIRED_SPEED * (1.0 + panic[i])
        fx = (desired * dx / dist - vx[i]) * 0.5
        fy = (desired * dy / dist - vy[i]) * 0.5

        cx, cy = int(x[i]), int(y[i])
        for oy in range(-1, 2):
            ny = cy + oy
            if ny < 0 or ny >= GRID_H:
                continue
            for ox in range(-1, 2):
                nx = cx + ox
                if nx < 0 or nx >= GRID_W:
                    continue
                head = cell_heads[ny, nx]
                idx = head
                while idx != -1:
                    if idx != i:
                        dx2 = x[i] - x[idx]
                        dy2 = y[i] - y[idx]
                        d2 = (dx2 * dx2 + dy2 * dy2) ** 0.5 + 1e-6
                        if d2 < R_PERSONAL:
                            s = 2.0 * np.exp(-d2 / 0.5)
                            fx += s * dx2 / d2
                            fy += s * dy2 / d2
                    idx = cell_next[idx]

        for b in range(barrier_x.shape[0]):
            dx2 = x[i] - barrier_x[b]
            dy2 = y[i] - barrier_y[b]
            d2 = (dx2*dx2 + dy2*dy2) ** 0.5 + 1e-6
            if d2 < R_BARRIER:
                s = 5.0 * np.exp(-d2 / 0.3)
                fx += s * dx2 / d2
                fy += s * dy2 / d2

        if x[i] < 1.0:
            fx += 2.0 * (1.0 - x[i])
        if x[i] > GRID_W - 1.0:
            fx -= 2.0 * (x[i] - GRID_W + 1.0)
        if y[i] < 1.0:
            fy += 2.0 * (1.0 - y[i])
        if y[i] > GRID_H - 1.0:
            fy -= 2.0 * (y[i] - GRID_H + 1.0)

        vx[i] = (vx[i] + fx * 0.1) * 0.8
        vy[i] = (vy[i] + fy * 0.1) * 0.8
        speed = (vx[i]*vx[i] + vy[i]*vy[i]) ** 0.5
        max_speed = 1.5 * (1.0 + panic[i] * 0.5)
        if speed > max_speed:
            s = max_speed / speed
            vx[i] *= s
            vy[i] *= s

        x[i] += vx[i] * 0.1
        y[i] += vy[i] * 0.1
        if x[i] < 0.5: x[i] = 0.5
        if x[i] > GRID_W - 0.5: x[i] = GRID_W - 0.5
        if y[i] < 0.5: y[i] = 0.5
        if y[i] > GRID_H - 0.5: y[i] = GRID_H - 0.5
    return


@njit
def update_panic_jit(x, y, panic, alive, grid_density, panic_grid,
                     N, GRID_W, GRID_H, PANIC_TRIGGER, PANIC_INC, PANIC_DEC):
    """
    O(N) panic update using pre-computed grid density.
    Panic spreads via the panic_grid rather than pairwise checks.
    
    FIXED: Check 3x3 neighborhood density, not just single cell.
    """
    for i in range(N):
        if not alive[i]:
            continue
        
        xi = int(x[i])
        yi = int(y[i])
        if xi < 0 or xi >= GRID_W or yi < 0 or yi >= GRID_H:
            continue
        
        # Calculate LOCAL density (MAX in 3x3 neighborhood, not average!)
        # Using max instead of average prevents dilution by empty cells
        local_density = 0.0
        for oy in range(-1, 2):
            ny = yi + oy
            if ny < 0 or ny >= GRID_H:
                continue
            for ox in range(-1, 2):
                nx = xi + ox
                if nx < 0 or nx >= GRID_W:
                    continue
                if grid_density[ny, nx] > local_density:
                    local_density = grid_density[ny, nx]
        
        # BALANCED panic dynamics
        if local_density > PANIC_TRIGGER:
            # Panic increases, but NOT 3x multiplier!
            intensity = min(2.0, (local_density - PANIC_TRIGGER) / PANIC_TRIGGER)
            panic[i] = min(1.0, panic[i] + PANIC_INC * (1.0 + intensity))  # Removed 3.0x
        else:
            # FASTER panic decrease when safe (agents calm down naturally)
            # The safer it is, the faster panic reduces
            safety_factor = (PANIC_TRIGGER - local_density) / PANIC_TRIGGER
            reduction_rate = PANIC_DEC * (2.0 + safety_factor)  # 2-3x base rate
            panic[i] = max(0.0, panic[i] - reduction_rate)
        
        # Panic spread (balanced - reduced spread rate)
        for oy in range(-1, 2):
            ny = yi + oy
            if ny < 0 or ny >= GRID_H:
                continue
            for ox in range(-1, 2):
                nx = xi + ox
                if nx < 0 or nx >= GRID_W:
                    continue
                if panic_grid[ny, nx] > 0.5:
                    # Nearby high panic spreads (reduced from 0.08)
                    panic[i] = min(1.0, panic[i] + 0.04)
    return


# -----------------------------------------------------------
# MAIN ENVIRONMENT
# -----------------------------------------------------------
class EnhancedCrowdControlEnvFast(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    # === UPDATED GRID SIZE FOR BETTER DENSITY ===
    GRID_WIDTH = 15  # Reduced from 20 -> more realistic venue
    GRID_HEIGHT = 15  # Reduced from 20 -> higher natural density
    MAX_STEPS = 500
    MAX_CROWD_PER_CELL = 10.0
    
    # === UPDATED REALISTIC DENSITY THRESHOLDS ===
    CRITICAL_DENSITY = 3.5  # FIXED: More achievable danger threshold (was 5.0)
    TARGET_DENSITY = 1.5    # FIXED: Tighter comfort zone (was 2.0)
    
    NUM_GATES = 3
    NUM_BARRIERS = 4
    PERSONAL_RADIUS = 1.8  # Slightly reduced for tighter venue
    BARRIER_RADIUS = 1.5
    DESIRED_SPEED = 0.5  # REDUCED: Was 1.0 - slower agent movement for better visibility
    
    # === UPDATED PANIC THRESHOLDS (VISIBLE FOR DEMO) ===
    PANIC_TRIGGER_DENSITY = 2.0  # LOWERED: Was 3.0 - panic starts earlier for better visualization
    PANIC_SPREAD_RADIUS = 1.5
    PANIC_INCREASE_RATE = 0.15   # INCREASED: Was 0.1 - faster panic growth for visible color changes
    PANIC_DECREASE_RATE = 0.08   # REDUCED: Was 0.15 - panic lingers longer for dramatic effect
    
    # Infrastructure constraints
    GATE_TRANSITION_DELAY = 10
    BARRIER_MOVE_COST = 5

    def __init__(self, render_mode=None, crowd_arrival_pattern="rush",
                 adversarial_mode=False, difficulty="medium"):
        super().__init__()
        self.render_mode = render_mode
        self.crowd_arrival_pattern = crowd_arrival_pattern
        self.adversarial_mode = adversarial_mode
        self.difficulty = difficulty
        self.overcrowding_events = 0


        # Agent limits scaled for 15x15 grid (was 20x20)
        # Grid area: 15x15=225 cells (was 20x20=400)
        # Ratio: 225/400 = 0.5625
        if difficulty == "easy":
            self.MAX_AGENTS = 80   # Was 100 -> scaled by ~0.8
            self.rush_peak_time = 0.4
        elif difficulty == "medium":
            self.MAX_AGENTS = 120  # Was 150 -> scaled by 0.8
            self.rush_peak_time = 0.3
        else:
            self.MAX_AGENTS = 160  # Was 200 -> scaled by 0.8
            self.rush_peak_time = 0.25

        # Action space: 
        # 0-15: Barrier movement (4 barriers × 4 directions)
        # 16-18: Toggle gates (3 gates)
        # 19-22: Flow direction nudge (4 directions: up, down, left, right)
        # 23: Emergency (open all gates)
        # 24: No-op
        self.action_space = spaces.Discrete(25)
        obs_size = (
            self.GRID_WIDTH * self.GRID_HEIGHT * 4
            + self.NUM_GATES * 2
            + self.NUM_BARRIERS * 3
            + 3
        )
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        # Agent arrays
        N = self.MAX_AGENTS
        self.x = np.zeros(N, dtype=np.float32)
        self.y = np.zeros(N, dtype=np.float32)
        self.vx = np.zeros(N, dtype=np.float32)
        self.vy = np.zeros(N, dtype=np.float32)
        self.goal = np.zeros(N, dtype=np.int32)
        self.panic = np.zeros(N, dtype=np.float32)
        self.alive = np.zeros(N, dtype=np.bool_)
        self.num_agents = 0

        # Gates: [x, y, is_open, capacity] - UPDATED positions for 15x15 grid
        self.gates = np.array([
            [7, 0, 1.0, 5.0],     # Top center (was 10, 0)
            [0, 7, 1.0, 5.0],     # Left center (was 0, 10)
            [14, 7, 1.0, 5.0],    # Right center (was 19, 10)
        ], dtype=np.float32)
        
        # Barriers - UPDATED positions for 15x15 grid
        self.barriers = np.array([
            [4, 7],   # Left-center (was 5, 10)
            [11, 7],  # Right-center (was 15, 10)
            [7, 4],   # Top-center (was 10, 5)
            [7, 11],  # Bottom-center (was 10, 15)
        ], dtype=np.float32)

        # Grids
        self.grid_density = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.float32)
        self.panic_grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.float32)
        self.vel_x_grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.float32)
        self.vel_y_grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.float32)

        # Spatial hashing
        self.cell_heads = np.full((self.GRID_HEIGHT, self.GRID_WIDTH), -1, dtype=np.int32)
        self.cell_next = np.full(N, -1, dtype=np.int32)

        # Infrastructure state
        self.gate_open_times = np.zeros(self.NUM_GATES, dtype=np.int32)
        self.barrier_move_cooldown = np.zeros(self.NUM_BARRIERS, dtype=np.int32)

        # Entrances - UPDATED positions for 15x15 grid
        self.entrances = [
            (2, 2),    # Bottom-left
            (2, 12),   # Top-left (was 2, 17)
            (12, 2),   # Bottom-right (was 17, 2)
            (12, 12),  # Top-right (was 17, 17)
            (7, 2)     # Bottom-center (was 10, 2)
        ]
        self.timestep = 0
        self.total_spawned = 0
        self.total_exited = 0
        self._last_exit_count = 0
        
        # Action tracking for analysis
        self.action_counts = np.zeros(25, dtype=np.int32)
        
        # Reward component tracking
        self._last_density_reward = 0.0
        self._last_safety_reward = 0.0
        self._last_efficiency_reward = 0.0
        self._last_infrastructure_cost = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.timestep = 0
        self.total_spawned = 0
        self.total_exited = 0
        self._last_exit_count = 0
        self.alive[:] = False
        self.num_agents = 0
        self.gate_open_times[:] = 0
        self.barrier_move_cooldown[:] = 0
        self.gates[:, 2] = 1.0  # All gates open
        
        # Reset action tracking
        self.action_counts.fill(0)
        
        # Reset reward component tracking
        self._last_density_reward = 0.0
        self._last_safety_reward = 0.0
        self._last_efficiency_reward = 0.0
        self._last_infrastructure_cost = 0.0
        
        # Reset terminal reason tracking
        self._terminal_reason = None
        
        self._spawn_initial()
        return self._get_obs(), self._get_info()

    def _spawn_agent(self, x0, y0, init_panic=0.0):
        if self.num_agents >= self.MAX_AGENTS:
            return
        idx = np.flatnonzero(~self.alive)
        if len(idx) == 0:
            return
        i = idx[0]
        self.x[i] = x0
        self.y[i] = y0
        self.vx[i] = 0
        self.vy[i] = 0
        self.goal[i] = self.np_random.integers(0, self.NUM_GATES)
        self.panic[i] = init_panic
        self.alive[i] = True
        self.num_agents += 1
        self.total_spawned += 1

    def _spawn_initial(self):
        """Spawn initial crowd - balanced for 15x15 grid"""
        # REDUCED initial spawn to prevent immediate overcrowding (CRITICAL_DENSITY now 3.5)
        # With 5 entrances, we need ~25-40 agents total initially to stay safe
        if self.difficulty == 'easy':
            agents_per_entrance = (3, 6)   # 15-30 total agents
        elif self.difficulty == 'medium':
            agents_per_entrance = (5, 8)   # 25-40 total agents
        else:  # hard
            agents_per_entrance = (7, 10)  # 35-50 total agents
            
        for ex, ey in self.entrances:
            n = self.np_random.integers(*agents_per_entrance)
            for _ in range(n):
                self._spawn_agent(
                    ex + self.np_random.uniform(-2.0, 2.0),  # Wider spread to reduce local density
                    ey + self.np_random.uniform(-2.0, 2.0),
                    init_panic=0.0
                )

    def _spawn_arrivals(self):
        """Spawn new arrivals - scaled for 15x15 grid with CRITICAL_DENSITY=3.5"""
        p = self.timestep / self.MAX_STEPS
        if self.crowd_arrival_pattern == "rush":
            rate = 3.0 * np.exp(-((p - self.rush_peak_time) ** 2) / 0.05)  # Reduced: 4.5 -> 3.0
        elif self.crowd_arrival_pattern == "steady":
            rate = 1.2 if p < 0.8 else 0  # Reduced: 1.5 -> 1.2
        elif self.crowd_arrival_pattern == "evacuation":
            rate = 6 if p < 0.1 else 0  # Reduced: 8 -> 6
        else:
            rate = 1.2  # Reduced: 1.5 -> 1.2
        for _ in range(int(rate)):
            ex, ey = self.entrances[self.np_random.integers(len(self.entrances))]
            self._spawn_agent(
                ex + self.np_random.uniform(-1.5, 1.5),  # Wider spread
                ey + self.np_random.uniform(-1.5, 1.5),
                init_panic=0.3 if self.crowd_arrival_pattern == "evacuation" else 0.0
            )

    def _update_spatial_grid(self):
        self.cell_heads.fill(-1)
        self.cell_next.fill(-1)
        for i in range(self.MAX_AGENTS):
            if not self.alive[i]:
                continue
            cx = int(self.x[i])
            cy = int(self.y[i])
            if cx < 0 or cx >= self.GRID_WIDTH or cy < 0 or cy >= self.GRID_HEIGHT:
                continue
            head = self.cell_heads[cy, cx]
            self.cell_next[i] = head
            self.cell_heads[cy, cx] = i

    def _update_physics(self):
        self._update_spatial_grid()
        if self.num_agents == 0:
            return
        update_physics(
            self.x, self.y, self.vx, self.vy, self.goal, self.panic,
            self.gates[:, 0], self.gates[:, 1], self.gates[:, 2],
            self.barriers[:, 0], self.barriers[:, 1],
            self.cell_heads, self.cell_next, 1,
            self.GRID_WIDTH, self.GRID_HEIGHT,
            self.PERSONAL_RADIUS, self.BARRIER_RADIUS, self.DESIRED_SPEED
        )

    def _update_panic(self):
        """Controllable panic: increases with density, decreases otherwise. O(N) version."""
        update_panic_jit(
            self.x, self.y, self.panic, self.alive,
            self.grid_density, self.panic_grid,
            self.MAX_AGENTS, self.GRID_WIDTH, self.GRID_HEIGHT,
            self.PANIC_TRIGGER_DENSITY, self.PANIC_INCREASE_RATE,
            self.PANIC_DECREASE_RATE
        )

    def _process_exits(self):
        alive_idx = np.flatnonzero(self.alive)
        for i in alive_idx:
            for g in range(self.NUM_GATES):
                if self.gates[g][2] < 0.5:
                    continue
                gx, gy = self.gates[g][:2]
                dx = self.x[i] - gx
                dy = self.y[i] - gy
                if dx*dx + dy*dy < 2.0:
                    self.alive[i] = False
                    self.num_agents -= 1
                    self.total_exited += 1
                    break

    def _move_barrier(self, barrier_id: int, direction: int) -> bool:
        if barrier_id >= self.NUM_BARRIERS:
            return False
        if self.barrier_move_cooldown[barrier_id] > 0:
            return False
        bx, by = self.barriers[barrier_id]
        if direction == 0 and by > 1: by -= 1
        elif direction == 1 and by < self.GRID_HEIGHT - 2: by += 1
        elif direction == 2 and bx > 1: bx -= 1
        elif direction == 3 and bx < self.GRID_WIDTH - 2: bx += 1
        self.barriers[barrier_id] = [bx, by]
        self.barrier_move_cooldown[barrier_id] = self.BARRIER_MOVE_COST
        return True

    def _toggle_gate(self, gate_id: int) -> bool:
        if gate_id >= self.NUM_GATES:
            return False
        time_since = self.timestep - self.gate_open_times[gate_id]
        if time_since < self.GATE_TRANSITION_DELAY:
            return False
        self.gates[gate_id][2] = 1.0 - self.gates[gate_id][2]
        self.gate_open_times[gate_id] = self.timestep
        return True

    def _update_cooldowns(self):
        self.barrier_move_cooldown = np.maximum(0, self.barrier_move_cooldown - 1)

    def step(self, action):
        self.timestep += 1
        action_cost = 0.0
        action_bonus = 0.0
        
        # Track action usage for analysis
        self.action_counts[action] += 1

        # === BARRIER MOVEMENT (Actions 0-15) ===
        # 4 barriers × 4 directions = 16 actions
        if action < 16:
            barrier_id = action // 4  # 0-3
            direction = action % 4     # 0=up, 1=down, 2=left, 3=right
            if self._move_barrier(barrier_id, direction):
                # No base cost - we want agent to use barriers strategically
                action_cost = 0.0
                
        # === GATE TOGGLE (Actions 16-18) ===
        elif action < 19:
            gate_id = action - 16
            if self._toggle_gate(gate_id):
                action_cost = 0.0  # No cost for gate management
                
        # === FLOW DIRECTION NUDGE (Actions 19-22) ===
        elif action < 23:
            # 19=up, 20=down, 21=left, 22=right
            directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            dy, dx = directions[action - 19]
            alive_idx = np.flatnonzero(self.alive)
            if len(alive_idx) > 0:
                # Gentle nudge to influence crowd flow
                self.vx[alive_idx] += dx * 0.25
                self.vy[alive_idx] += dy * 0.25
            action_cost = 0.0  # Cheap soft intervention
            
        # === EMERGENCY: OPEN ALL GATES (Action 23) ===
        elif action == 23:
            closed_gates = np.sum(self.gates[:, 2] < 0.5)
            max_density = np.max(self.grid_density)
            
            # Only reward if there's actually an emergency
            if closed_gates > 0 and max_density > self.CRITICAL_DENSITY * 0.7:
                self.gates[:, 2] = 1.0
                action_bonus = 1.0  # Good emergency response!
            elif closed_gates > 0:
                # Opening gates but not critical yet
                self.gates[:, 2] = 1.0
                action_cost = 0.2  # Slightly premature
            else:
                # Wasteful - all gates already open
                action_cost = 0.3
                
        # === NO-OP (Action 24) ===
        else:  # action == 24
            # No penalty - sometimes stability is the right choice
            action_cost = 0.0

        self._update_cooldowns()
        self._spawn_arrivals()
        self._update_physics()
        self._rebuild_grids()      # Build grids FIRST
        self._update_panic()       # Then use grids for O(N) panic
        self._process_exits()
        
        if self.adversarial_mode:
            self._adversarial_scenario()

        # === IMPROVED REWARD STRUCTURE ===
        reward = self._compute_reward() - action_cost + action_bonus
        
        # CRITICAL FIX: Survival bonus - reward staying alive!
        # Increased to balance the density penalties
        reward += 5.0  # INCREASED: Base survival reward per step (was 2.0)
        
        # Bigger milestone bonuses for sustained control
        if self.timestep == 100:
            reward += 50.0  # Made it to 100 steps (was 10)
        elif self.timestep == 200:
            reward += 100.0  # Made it to 200 steps (was 20)
        elif self.timestep == 300:
            reward += 150.0  # Made it to 300 steps (was 30)
        elif self.timestep == 400:
            reward += 200.0  # Made it to 400 steps (was 40)

        terminated = False
        truncated = self.timestep >= self.MAX_STEPS
        max_density = np.max(self.grid_density)

        # === TERMINAL STATE HANDLING ===
        # Critical overcrowding: SMALL penalty (learning signal, not punishment!)
        if max_density > self.CRITICAL_DENSITY:
            # MINIMAL penalty - just end the episode early
            overcrowding_severity = (max_density - self.CRITICAL_DENSITY) / self.CRITICAL_DENSITY
            penalty = 10.0 + overcrowding_severity * 10.0  # 10-20 range (was 30-50)
            reward -= penalty
            terminated = True
            # Track for analysis
            self._terminal_reason = "critical_overcrowding"
            self.overcrowding_events += 1

        # Success: Processed most of the crowd safely
        # CRITICAL FIX: Success must be MORE rewarding than just surviving!
        success_achieved = False
        success_reward = 0.0
        
        if self.timestep > 200 and self.total_spawned > 0:
            throughput_ratio = self.total_exited / self.total_spawned
            occupancy_ratio = self.num_agents / self.MAX_AGENTS
            
            # SUCCESS PATH 1: Reasonable throughput (35%+ processed)
            # Random achieves 40-44%, so this is reachable!
            if throughput_ratio > 0.35:
                success_achieved = True
                base_reward = 500.0  # INCREASED: was 200
                # Scale excellence bonus from 35% to 70% (full success)
                excellence_bonus = 500.0 * min(1.0, (throughput_ratio - 0.35) / 0.35)  # was 300
                
                # CRITICAL: Compensate for lost survival rewards due to early termination
                # If agent succeeds at step 300, they lose 200 steps × 5.0 = 1000 reward
                remaining_steps = self.MAX_STEPS - self.timestep
                survival_compensation = remaining_steps * 5.0
                
                success_reward = base_reward + excellence_bonus + survival_compensation
                self._terminal_reason = "success_high_throughput"
            
            # SUCCESS PATH 2: Moderate remaining crowd (<50% occupancy = <60 agents for medium)
            # This is easier than 20% but still requires good control
            elif occupancy_ratio < 0.50:
                success_achieved = True
                base_reward = 400.0  # INCREASED: was 150
                # Scale excellence bonus from 50% down to 0%
                excellence_bonus = 400.0 * (0.50 - occupancy_ratio) / 0.50  # was 150
                
                # Survival compensation
                remaining_steps = self.MAX_STEPS - self.timestep
                survival_compensation = remaining_steps * 5.0
                
                success_reward = base_reward + excellence_bonus + survival_compensation
                self._terminal_reason = "success_low_crowd"
        
        if success_achieved:
            reward += success_reward
            terminated = True

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _adversarial_scenario(self):
        if self.np_random.random() < 0.05:
            scenario = self.np_random.choice(['gate_failure', 'sudden_rush', 'bottleneck'])
            if scenario == 'gate_failure':
                gate_id = self.np_random.integers(self.NUM_GATES)
                self.gates[gate_id][2] = 0.0
            elif scenario == 'sudden_rush':
                for _ in range(15):
                    ex, ey = self.entrances[self.np_random.integers(len(self.entrances))]
                    self._spawn_agent(
                        ex + self.np_random.uniform(-3, 3),
                        ey + self.np_random.uniform(-3, 3),
                        init_panic=0.5
                    )
            elif scenario == 'bottleneck':
                for i in range(min(2, self.NUM_GATES)):
                    self.gates[i][2] = 0.0

    def _rebuild_grids(self):
        self.grid_density.fill(0)
        self.panic_grid.fill(0)
        self.vel_x_grid.fill(0)
        self.vel_y_grid.fill(0)
        alive = np.flatnonzero(self.alive)
        for i in alive:
            xi = int(np.clip(self.x[i], 0, self.GRID_WIDTH - 1))
            yi = int(np.clip(self.y[i], 0, self.GRID_HEIGHT - 1))
            self.grid_density[yi, xi] += 1
            self.panic_grid[yi, xi] = max(self.panic_grid[yi, xi], self.panic[i])
            self.vel_x_grid[yi, xi] = self.vx[i]
            self.vel_y_grid[yi, xi] = self.vy[i]

    def _compute_reward(self) -> float:
        """
        FIXED: More positive reward for good behavior, clearer gradients
        - Density control (PRIMARY): Strong rewards for safety, progressive penalties for danger
        - Throughput (SECONDARY): Substantial rewards for exits
        - Safety/Panic (TERTIARY): Bonus for calm crowds
        - Infrastructure: Encourage strategic use
        """
        reward = 0.0
        
        # Calculate components and store for tracking
        density_reward = self._calculate_density_reward()
        efficiency_reward = self._calculate_efficiency_reward()
        safety_reward = self._calculate_safety_reward()
        infrastructure_cost = self._calculate_infrastructure_cost()
        
        # Store raw values for info tracking
        self._last_density_reward = density_reward
        self._last_efficiency_reward = efficiency_reward
        self._last_safety_reward = safety_reward
        self._last_infrastructure_cost = infrastructure_cost
        
        # BALANCED weighting: rewards AND penalties
        # Primary objective: Density control (weight: 2.0x - increased positive signal)
        reward += density_reward * 2.0
        
        # Secondary objective: Throughput (weight: 2.5x - MAJOR increase for exits)
        reward += efficiency_reward * 2.5
        
        # Tertiary objective: Safety/Panic (weight: 0.5x - modest bonus)
        reward += safety_reward * 0.5
        
        # Infrastructure: minimal costs
        reward -= infrastructure_cost
        
        # SURVIVAL BONUS: Reward staying alive (prevent "fail fast" strategy)
        # Every step without catastrophic failure is valuable!
        survival_bonus = 0.5  # +0.5 per step = +250 for full episode
        reward += survival_bonus
        
        # MILESTONE BONUSES: Reward reaching key thresholds
        if self.timestep == 100:
            reward += 10.0  # Made it to 100 steps!
        elif self.timestep == 200:
            reward += 20.0  # Made it to 200 steps!
        elif self.timestep == 300:
            reward += 30.0  # Made it to 300 steps!
        elif self.timestep == 400:
            reward += 40.0  # Made it to 400 steps!
        
        return reward

    def _calculate_density_reward(self) -> float:
        """
        FIXED: Strong positive rewards for safety, progressive penalties for danger
        Goal: Make "staying safe" more rewarding than current -3k average
        """
        reward = 0.0
        total_cells = self.GRID_WIDTH * self.GRID_HEIGHT
        max_density = np.max(self.grid_density)
        
        # 1. STRONG positive reward for safe density (INCREASED)
        if max_density < self.TARGET_DENSITY:
            # Big reward for excellent safety
            safety_margin = (self.TARGET_DENSITY - max_density) / self.TARGET_DENSITY
            reward += 3.0 + safety_margin * 2.0  # 3.0-5.0 range (was 1.5)
        elif max_density < self.CRITICAL_DENSITY * 0.5:
            # Moderate reward for acceptable density
            reward += 1.5  # (was 0.5)
        
        # 2. Base penalty for exceeding target (reduced - focus on positives)
        avg_above = np.sum(np.maximum(0, self.grid_density - self.TARGET_DENSITY)) / total_cells
        reward -= avg_above * 0.3  # Reduced from 0.5
        
        # 3. Progressive penalties for danger (GENTLER curve - was overwhelming survival bonus!)
        if max_density > self.TARGET_DENSITY:
            # Normalize: 0.0 at TARGET_DENSITY, 1.0 at CRITICAL_DENSITY
            danger_ratio = (max_density - self.TARGET_DENSITY) / (self.CRITICAL_DENSITY - self.TARGET_DENSITY)
            danger_ratio = np.clip(danger_ratio, 0.0, 1.0)
            
            # MUCH GENTLER penalty: Linear with slight quadratic boost
            # At 50%: -1.25, at 75%: -2.8, at 95%: -4.5
            # This lets survival bonus (+2.0) still have impact!
            exponential_penalty = danger_ratio * (1.0 + danger_ratio * 4.0)
            reward -= exponential_penalty
            
        return reward

    def _calculate_safety_reward(self) -> float:
        """
        REBALANCED: Panic as early warning indicator, not primary objective.
        Density control is the main goal; panic signals when density becomes dangerous.
        """
        reward = 0.0
        if self.num_agents > 0:
            alive_panic = self.panic[self.alive]
            avg_panic = np.mean(alive_panic)
            max_density = np.max(self.grid_density)
            
            # Small linear penalty for panic (it's a symptom, not the root cause)
            reward -= avg_panic * 0.1  # Reduced from 0.2
            
            # BIG penalty when panic + high density (actual danger zone)
            if avg_panic > 0.5 and max_density > self.TARGET_DENSITY:
                danger_score = avg_panic * (max_density / self.CRITICAL_DENSITY)
                reward -= danger_score * 2.0  # Compound penalty for danger
            
            # Modest reward for calm crowds (balanced - not excessive)
            if avg_panic < 0.2:
                reward += 0.8  # Reduced from 3.0 - panic isn't the primary goal
            elif avg_panic < 0.4:
                reward += 0.3  # Reduced from 1.5
        else:
            reward += 0.2  # Small reward for empty venue (down from 0.5)
        return reward

    def _calculate_efficiency_reward(self) -> float:
        """
        FIXED: Much stronger rewards for throughput - this should be PRIMARY goal
        Goal: Agent learns "process crowd quickly AND safely"
        """
        reward = 0.0
        exits_this_step = self.total_exited - self._last_exit_count
        self._last_exit_count = self.total_exited
        
        # BIG reward for exits (this is what we want!)
        reward += exits_this_step * 2.0  # DOUBLED: was 1.2, no cap
        
        # Reward FLOW efficiency (exits per occupied capacity)
        if self.num_agents > 50:
            occupancy_ratio = self.num_agents / self.MAX_AGENTS
            if exits_this_step > 0:
                # Flow rate: how efficiently are we processing the crowd?
                flow_efficiency = exits_this_step / occupancy_ratio
                reward += flow_efficiency * 0.5  # Increased from 0.3
            else:
                # Stagnation penalty (high occupancy but no movement)
                reward -= 1.5  # Increased from 0.8 - stagnation is bad!
        
        # Strong reward for dispersal progress (every step counts)
        if self.num_agents > 0:
            # Progress reward: fewer agents = better
            progress = 1.0 - (self.num_agents / self.MAX_AGENTS)
            reward += progress * 1.0  # Smooth gradient from 0 to +1
        
        return reward

    def _calculate_infrastructure_cost(self) -> float:
        """
        REFINED: Encourage strategic infrastructure use.
        - Barrier movement has minimal cost (let agent experiment)
        - Gate balance rewarded (prevent single-gate bottlenecks)
        - Long-term gate closure penalized (reduces throughput)
        """
        cost = 0.0
        
        # Minimal cost for barrier activity (down from 0.1)
        active_cooldowns = np.sum(self.barrier_move_cooldown > 0)
        cost += 0.05 * active_cooldowns
        
        # Reward balanced gate usage (prevent all crowd at one exit)
        if self.num_agents > 20:
            gate_crowds = []
            for g in range(self.NUM_GATES):
                gx, gy = int(self.gates[g, 0]), int(self.gates[g, 1])
                # Count agents in 5x5 area around gate
                local_count = 0
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        nx, ny = gx + dx, gy + dy
                        if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                            local_count += self.grid_density[ny, nx]
                gate_crowds.append(local_count)
            
            # Reward balanced distribution (prevents single-exit crush)
            if len(gate_crowds) > 1 and max(gate_crowds) > 0:
                balance = 1.0 - (np.std(gate_crowds) / (np.mean(gate_crowds) + 1e-6))
                cost -= 0.8 * np.clip(balance, 0, 1)  # Reduced from 1.0
        
        # Penalty if gate closed too long (limits throughput)
        for g in range(self.NUM_GATES):
            if self.gates[g, 2] < 0.5:  # Gate is closed
                time_closed = self.timestep - self.gate_open_times[g]
                if time_closed > 50:
                    cost += 0.4 * (time_closed - 50) / 10  # Reduced from 0.5
        
        return cost

    def _get_obs(self):
        obs = []
        obs.append((self.grid_density / self.MAX_CROWD_PER_CELL).flatten())
        obs.append(self.panic_grid.flatten())
        obs.append(self.vel_x_grid.flatten())
        obs.append(self.vel_y_grid.flatten())
        
        for i in range(self.NUM_GATES):
            time_since = self.timestep - self.gate_open_times[i]
            transition = min(1.0, time_since / self.GATE_TRANSITION_DELAY)
            obs.append(np.array([self.gates[i][2], transition]))
        
        for i in range(self.NUM_BARRIERS):
            bx, by = self.barriers[i]
            cooldown = self.barrier_move_cooldown[i] / self.BARRIER_MOVE_COST
            obs.append(np.array([bx / self.GRID_WIDTH, by / self.GRID_HEIGHT, cooldown]))
        
        avg_panic = np.mean(self.panic[self.alive]) if self.num_agents > 0 else 0.0
        obs.append(np.array([
            self.timestep / self.MAX_STEPS,
            self.num_agents / self.MAX_AGENTS,
            avg_panic
        ]))
        return np.concatenate(obs).astype(np.float32)

    def _get_info(self):
        avg_panic = np.mean(self.panic[self.alive]) if self.num_agents > 0 else 0.0
        max_panic = np.max(self.panic[self.alive]) if self.num_agents > 0 else 0.0
        
        # Calculate action diversity (how many different actions have been used)
        action_diversity = np.count_nonzero(self.action_counts) / 25  # 25 total actions
        
        info = dict(
            timestep=self.timestep,
            agents=self.num_agents,
            exited=self.total_exited,
            spawned=self.total_spawned,
            max_density=np.max(self.grid_density),
            avg_panic=avg_panic,
            max_panic=max_panic,
            pattern=self.crowd_arrival_pattern,
            difficulty=self.difficulty,
            open_gates=int(np.sum(self.gates[:, 2] > 0.5)),
            
            # Action tracking
            action_diversity=action_diversity,
            action_counts=self.action_counts.copy(),  # Copy for safety
            
            # Reward component tracking (raw values before weighting)
            reward_components={
                'density': self._last_density_reward,
                'efficiency': self._last_efficiency_reward,
                'safety': self._last_safety_reward,
                'infrastructure_cost': self._last_infrastructure_cost,
                # Weighted values
                'density_weighted': self._last_density_reward * 1.5,
                'efficiency_weighted': self._last_efficiency_reward * 1.0,
                'safety_weighted': self._last_safety_reward * 0.3,
            },
            
            # Backwards-compatible keys expected by the renderer/demo:
            total_agents=self.num_agents,
            total_exited=self.total_exited,
            total_spawned=self.total_spawned,
            overcrowding_events=self.overcrowding_events,
            
            # Success/failure tracking (CRITICAL for evaluation!)
            success=self._terminal_reason.startswith('success') if self._terminal_reason else False,
            terminal_reason=self._terminal_reason if self._terminal_reason else 'in_progress',
            throughput_ratio=self.total_exited / self.total_spawned if self.total_spawned > 0 else 0.0,
        )
        return info
    

    @property
    def agents(self):
        """Return a list of lightweight objects the renderer can consume.

        Each agent has attributes: x, y, panic_level
        Only alive agents are returned, in arbitrary order.
        """
        from types import SimpleNamespace
        alive_idx = np.flatnonzero(self.alive)
        agents = []
        for i in alive_idx:
            agents.append(SimpleNamespace(
                x=float(self.x[i]),
                y=float(self.y[i]),
                panic_level=float(self.panic[i])
            ))
        return agents


    def render(self):
        pass

    def close(self):
        pass

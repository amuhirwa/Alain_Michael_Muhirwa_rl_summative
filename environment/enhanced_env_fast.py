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
        
        # Panic increases with high density, decreases otherwise
        if local_density > PANIC_TRIGGER:
            # MUCH faster panic increase for dangerous situations
            intensity = min(3.0, (local_density - PANIC_TRIGGER) / PANIC_TRIGGER)
            panic[i] = min(1.0, panic[i] + PANIC_INC * 3.0 * (1.0 + intensity))
        else:
            panic[i] = max(0.0, panic[i] - PANIC_DEC)
        
        # Panic spread: check neighboring cells for high panic
        for oy in range(-1, 2):
            ny = yi + oy
            if ny < 0 or ny >= GRID_H:
                continue
            for ox in range(-1, 2):
                nx = xi + ox
                if nx < 0 or nx >= GRID_W:
                    continue
                if panic_grid[ny, nx] > 0.5:
                    # Nearby high panic spreads faster
                    panic[i] = min(1.0, panic[i] + 0.08)
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
    CRITICAL_DENSITY = 5.0  # Reduced from 8.0 -> realistic danger level (5 people/m²)
    TARGET_DENSITY = 2.0    # Reduced from 3.0 -> comfortable spacing
    
    NUM_GATES = 3
    NUM_BARRIERS = 4
    PERSONAL_RADIUS = 1.8  # Slightly reduced for tighter venue
    BARRIER_RADIUS = 1.5
    DESIRED_SPEED = 1.0
    
    # === UPDATED PANIC THRESHOLDS ===
    PANIC_TRIGGER_DENSITY = 4.0  # Reduced from 7.0 -> triggers earlier
    PANIC_SPREAD_RADIUS = 1.5
    PANIC_INCREASE_RATE = 0.15   # Increased from 0.1 -> panic grows faster
    PANIC_DECREASE_RATE = 0.05
    
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

        self.action_space = spaces.Discrete(12)
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
        self.goal[i] = np.random.randint(0, self.NUM_GATES)
        self.panic[i] = init_panic
        self.alive[i] = True
        self.num_agents += 1
        self.total_spawned += 1

    def _spawn_initial(self):
        """Spawn initial crowd - balanced for 15x15 grid"""
        # Scale initial spawn by difficulty to prevent instant failure
        if self.difficulty == 'easy':
            agents_per_entrance = (8, 12)   # Was (10, 18)
        elif self.difficulty == 'medium':
            agents_per_entrance = (10, 15)  # Was (12, 20)
        else:  # hard
            agents_per_entrance = (12, 18)  # Was (15, 22)
            
        for ex, ey in self.entrances:
            n = np.random.randint(*agents_per_entrance)
            for _ in range(n):
                self._spawn_agent(
                    ex + np.random.uniform(-1.5, 1.5),  # Slightly tighter spread for smaller grid
                    ey + np.random.uniform(-1.5, 1.5),
                    init_panic=0.0
                )

    def _spawn_arrivals(self):
        """Spawn new arrivals - scaled for 15x15 grid"""
        p = self.timestep / self.MAX_STEPS
        if self.crowd_arrival_pattern == "rush":
            rate = 4.5 * np.exp(-((p - self.rush_peak_time) ** 2) / 0.05)  # Was 6
        elif self.crowd_arrival_pattern == "steady":
            rate = 1.5 if p < 0.8 else 0  # Was 2
        elif self.crowd_arrival_pattern == "evacuation":
            rate = 8 if p < 0.1 else 0  # Was 10
        else:
            rate = 1.5  # Was 2
        for _ in range(int(rate)):
            ex, ey = self.entrances[np.random.randint(len(self.entrances))]
            self._spawn_agent(
                ex + np.random.uniform(-1, 1),
                ey + np.random.uniform(-1, 1),
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

        # Execute actions with constraints
        if action < 4:
            direction = np.random.randint(4)
            if self._move_barrier(action, direction):
                action_cost = 0.1
        elif action < 7:
            if self._toggle_gate(action - 4):
                action_cost = 0.05
        elif action < 11:
            # Flow direction nudge
            directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            dy, dx = directions[action - 7]
            alive_idx = np.flatnonzero(self.alive)
            self.vx[alive_idx] += dx * 0.05
            self.vy[alive_idx] += dy * 0.05
            action_cost = 0.02
        else:
            # Emergency: open all gates
            self.gates[:, 2] = 1.0
            action_cost = 1.0

        self._update_cooldowns()
        self._spawn_arrivals()
        self._update_physics()
        self._rebuild_grids()      # Build grids FIRST
        self._update_panic()       # Then use grids for O(N) panic
        self._process_exits()
        
        if self.adversarial_mode:
            self._adversarial_scenario()

        # === IMPROVED REWARD STRUCTURE ===
        reward = self._compute_reward() - action_cost

        terminated = False
        truncated = self.timestep >= self.MAX_STEPS
        max_density = np.max(self.grid_density)

        # === TERMINAL STATE HANDLING ===
        # Critical overcrowding: catastrophic failure
        if max_density > self.CRITICAL_DENSITY:
            # Massive penalty that dwarfs any possible cumulative reward
            # Agent should learn this is NEVER acceptable
            reward -= 100.0
            terminated = True
            # Track for analysis
            self._terminal_reason = "critical_overcrowding"
            self.overcrowding_events += 1

        # Success: crowd dispersed safely
        if self.num_agents < 10 and self.timestep > 100:
            reward += 20.0
            terminated = True
            self._terminal_reason = "success"

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _adversarial_scenario(self):
        if np.random.random() < 0.05:
            scenario = np.random.choice(['gate_failure', 'sudden_rush', 'bottleneck'])
            if scenario == 'gate_failure':
                gate_id = np.random.randint(self.NUM_GATES)
                self.gates[gate_id][2] = 0.0
            elif scenario == 'sudden_rush':
                for _ in range(15):
                    ex, ey = self.entrances[np.random.randint(len(self.entrances))]
                    self._spawn_agent(
                        ex + np.random.uniform(-3, 3),
                        ey + np.random.uniform(-3, 3),
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
        Balanced multi-objective reward from Enhanced version:
        - Density reward
        - Safety reward (panic-aware)
        - Efficiency reward (per-step exits)
        - Infrastructure cost
        """
        reward = 0.0
        reward += self._calculate_density_reward()
        reward += self._calculate_safety_reward()
        reward += self._calculate_efficiency_reward()
        reward -= self._calculate_infrastructure_cost()
        return reward

    def _calculate_density_reward(self) -> float:
        reward = 0.0
        total_cells = self.GRID_WIDTH * self.GRID_HEIGHT
        max_density = np.max(self.grid_density)
        
        # Base penalty for exceeding target
        avg_above = np.sum(np.maximum(0, self.grid_density - self.TARGET_DENSITY)) / total_cells
        reward -= avg_above * 0.5
        
        # === GRADUATED DANGER ZONE PENALTIES ===
        # These create a "force field" pushing agent away from terminal state
        
        # Warning zone: 60-80% of critical
        if max_density > self.CRITICAL_DENSITY * 0.6:
            warning_severity = (max_density - self.CRITICAL_DENSITY * 0.6) / (self.CRITICAL_DENSITY * 0.2)
            reward -= 2.0 * np.clip(warning_severity, 0, 1)
        
        # Danger zone: 80-95% of critical  
        if max_density > self.CRITICAL_DENSITY * 0.8:
            danger_severity = (max_density - self.CRITICAL_DENSITY * 0.8) / (self.CRITICAL_DENSITY * 0.15)
            reward -= 5.0 * np.clip(danger_severity, 0, 1)
        
        # Critical zone: 95%+ of critical - MASSIVE penalty before terminal
        if max_density > self.CRITICAL_DENSITY * 0.95:
            critical_severity = (max_density - self.CRITICAL_DENSITY * 0.95) / (self.CRITICAL_DENSITY * 0.05)
            reward -= 15.0 * np.clip(critical_severity, 0, 1)
        
        # Reward for staying safe
        if max_density < self.TARGET_DENSITY:
            reward += 1.5
        elif max_density < self.CRITICAL_DENSITY * 0.5:
            reward += 0.5
            
        return reward

    def _calculate_safety_reward(self) -> float:
        reward = 0.0
        if self.num_agents > 0:
            alive_panic = self.panic[self.alive]
            avg_panic = np.mean(alive_panic)
            max_panic = np.max(alive_panic)
            
            # Base panic penalty
            reward -= np.clip(avg_panic * 2.0, 0, 2.0)
            
            # === GRADUATED PANIC WARNINGS ===
            # High average panic is dangerous - crowds become unpredictable
            if avg_panic > 0.5:
                reward -= 2.0 * (avg_panic - 0.5) / 0.5  # Up to -2 more
            
            # Extreme panic anywhere is a warning sign
            if max_panic > 0.7:
                reward -= 3.0 * (max_panic - 0.7) / 0.3  # Up to -3 more
            
            # Reward for calm crowds (positive shaping)
            if avg_panic < 0.2:
                reward += 1.5
            elif avg_panic < 0.4:
                reward += 0.5
        else:
            reward += 1.0
        return reward

    def _calculate_efficiency_reward(self) -> float:
        reward = 0.0
        exits_this_step = self.total_exited - self._last_exit_count
        self._last_exit_count = self.total_exited
        reward += min(exits_this_step * 0.5, 3.0)
        
        if self.num_agents > self.MAX_AGENTS * 0.7:
            reward -= 1.0
        
        if self.num_agents / self.MAX_AGENTS < 0.5:
            reward += 0.5
        return reward

    def _calculate_infrastructure_cost(self) -> float:
        cost = 0.0
        active_cooldowns = np.sum(self.barrier_move_cooldown > 0)
        cost += 0.1 * active_cooldowns
        
        progress = self.timestep / self.MAX_STEPS
        if 0.2 < progress < 0.6 and not self.adversarial_mode:
            closed_gates = np.sum(self.gates[:, 2] < 0.5)
            cost += 2.0 * closed_gates
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
            # Backwards-compatible keys expected by the renderer/demo:
            total_agents=self.num_agents,
            total_exited=self.total_exited,
            total_spawned=self.total_spawned,
            overcrowding_events=self.overcrowding_events
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
    
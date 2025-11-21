"""
Enhanced Crowd Control Environment for Reinforcement Learning
==============================================================

NOVEL CONTRIBUTIONS:
1. Dynamic Infrastructure Control via RL - Real-time adaptive reconfiguration
2. Individual Agent Simulation with Social Force Model - Realistic crowd physics
3. Temporal Dynamics - Time-based crowd arrival patterns (rush, steady, evacuation)
4. Infrastructure Constraints - Gates have transition delays, barriers have movement costs
5. Panic Propagation & Adversarial Scenarios - Safety-critical testing
6. Multi-Objective Optimization - Throughput vs Safety vs Operational Cost

This shifts focus from "how agents navigate" to "how operators control the space"
with realistic constraints and safety-first validation.

Mission: Manage crowd flow in a venue to prevent overcrowding through dynamic
infrastructure control (gates, barriers) under realistic constraints.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Optional, List
import math
from dataclasses import dataclass


@dataclass
class Agent:
    """Individual agent with position, velocity, goal, and psychological state"""
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    goal_gate: int = 0
    panic_level: float = 0.0  # 0-1, affects behavior
    id: int = 0


class EnhancedCrowdControlEnv(gym.Env):
    """
    Enhanced Environment for Crowd Control using Individual Agent Simulation
    
    Key Novel Features:
    - Individual agents with Social Force Model physics
    - Temporal crowd arrival patterns (rush, steady, evacuation)
    - Infrastructure constraints (gate delays, barrier cooldowns)
    - Panic propagation and adversarial scenarios
    - Multi-objective rewards (safety, throughput, cost)
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}
    
    # Environment constants
    GRID_WIDTH = 10
    GRID_HEIGHT = 10
    MAX_CROWD_PER_CELL = 10.0
    CRITICAL_DENSITY = 8.0
    TARGET_DENSITY = 3.0
    MAX_STEPS = 500
    
    # Agent constants
    MAX_AGENTS = 50
    AGENT_DESIRED_SPEED = 1.0
    AGENT_PERSONAL_SPACE = 2.0  # Distance for repulsion
    
    # Gate and barrier constants
    NUM_GATES = 3
    NUM_BARRIERS = 4
    GATE_TRANSITION_DELAY = 10  # Steps to fully open/close
    BARRIER_MOVE_COST = 5  # Steps cooldown after moving
    
    # Panic constants
    PANIC_TRIGGER_DENSITY = 7.0
    PANIC_SPREAD_RADIUS = 1.5
    PANIC_INCREASE_RATE = 0.1
    PANIC_DECREASE_RATE = 0.05
    
    def __init__(self, 
                 render_mode: Optional[str] = None,
                 crowd_arrival_pattern: str = 'rush',  # 'rush', 'steady', 'evacuation'
                 adversarial_mode: bool = False,
                 difficulty: str = 'medium'):  # 'easy', 'medium', 'hard'
        super().__init__()
        
        self.render_mode = render_mode
        self.renderer = None
        
        # Scenario configuration
        self.crowd_arrival_pattern = crowd_arrival_pattern
        self.adversarial_mode = adversarial_mode
        self.difficulty = difficulty
        
        # Set difficulty parameters
        if difficulty == 'easy':
            self.MAX_AGENTS = 100
            self.adversarial_mode = False
            self.rush_peak_time = 0.4
        elif difficulty == 'medium':
            self.MAX_AGENTS = 150
            self.rush_peak_time = 0.3
        elif difficulty == 'hard':
            self.MAX_AGENTS = 200
            self.rush_peak_time = 0.25
        
        # Define action space (same as before for compatibility)
        self.action_space = spaces.Discrete(12)
        
        # Enhanced observation space
        # [grid_density (400), panic_grid (400), velocity_x (400), velocity_y (400),
        #  gate_states (3), gate_transition_progress (3), 
        #  barrier_positions (4x2), barrier_cooldowns (4),
        #  timestep_progress (1), agent_count_ratio (1), avg_panic (1)]
        obs_size = (self.GRID_WIDTH * self.GRID_HEIGHT * 4 +  # density + panic + 2 velocities
                   self.NUM_GATES * 2 +  # state + transition
                   self.NUM_BARRIERS * 3 +  # x, y, cooldown
                   3)  # scenario info
        
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        
        # Initialize state variables
        self.agents: List[Agent] = []
        self.next_agent_id = 0
        self.grid_density = None
        self.panic_grid = None
        self.velocity_x = None
        self.velocity_y = None
        self.gates = None
        self.barriers = None
        self.timestep = 0
        
        # Infrastructure state
        self.gate_open_times = [0, 0, 0]  # Last time each gate was toggled
        self.barrier_move_cooldown = [0, 0, 0, 0]  # Cooldown for each barrier
        
        # Statistics tracking
        self.total_agents_spawned = 0
        self.total_agents_exited = 0
        self.max_density_reached = 0
        self.max_panic_reached = 0
        self.overcrowding_events = 0
        self.episode_rewards = []
        
        # Entrance points for crowd spawning
        self.entrance_points = [
            (2, 2), (2, 17), (17, 2), (17, 17), (10, 2)
        ]
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        # Reset timestep and stats
        self.timestep = 0
        self.episode_rewards = []
        self.max_density_reached = 0
        self.max_panic_reached = 0
        self.overcrowding_events = 0
        self.total_agents_spawned = 0
        self.total_agents_exited = 0
        self.next_agent_id = 0
        self._last_exit_count = 0  # For tracking exits per step
        
        # Reset infrastructure state
        self.gate_open_times = [0, 0, 0]
        self.barrier_move_cooldown = [0, 0, 0, 0]
        
        # Initialize grids
        self.grid_density = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.float32)
        self.panic_grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.float32)
        self.velocity_x = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.float32)
        self.velocity_y = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.float32)
        
        # Initialize gates (positioned at exits)
        # Gate: [x, y, is_open (1=open, 0=closed), capacity]
        self.gates = [
            [10, 0, 1.0, 5.0],   # Top center gate (open)
            [0, 10, 1.0, 5.0],   # Left center gate (open)
            [19, 10, 1.0, 5.0],  # Right center gate (open)
        ]
        
        # Initialize barriers (movable crowd control barriers)
        self.barriers = [
            [5, 10],
            [15, 10],
            [10, 5],
            [10, 15],
        ]
        
        # Initialize agents
        self.agents = []
        self._spawn_initial_crowd()
        
        return self._get_observation(), self._get_info()
    
    def _spawn_initial_crowd(self):
        """Spawn initial crowd at entrances"""
        for entrance in self.entrance_points:
            ex, ey = entrance
            num_agents = self.np_random.integers(20, 35)
            
            for _ in range(num_agents):
                if len(self.agents) < self.MAX_AGENTS:
                    agent = Agent(
                        x=ex + self.np_random.uniform(-1.5, 1.5),
                        y=ey + self.np_random.uniform(-1.5, 1.5),
                        vx=0.0,
                        vy=0.0,
                        goal_gate=self.np_random.integers(0, self.NUM_GATES),
                        panic_level=0.0,
                        id=self.next_agent_id
                    )
                    self.next_agent_id += 1
                    self.total_agents_spawned += 1
                    self.agents.append(agent)
    
    def _spawn_new_arrivals(self):
        """Time-based crowd arrival patterns - NOVEL CONTRIBUTION"""
        progress = self.timestep / self.MAX_STEPS
        
        if self.crowd_arrival_pattern == 'rush':
            # Concert/event entry - peak early, taper off
            spawn_rate = 5.0 * np.exp(-((progress - self.rush_peak_time) ** 2) / 0.05)
        elif self.crowd_arrival_pattern == 'steady':
            # Steady flow until late in episode
            spawn_rate = 2.0 if progress < 0.8 else 0.0
        elif self.crowd_arrival_pattern == 'evacuation':
            # Emergency - everyone arrives at once early
            spawn_rate = 10.0 if progress < 0.1 else 0.0
        else:
            spawn_rate = 2.0
        
        num_new = int(spawn_rate)
        
        for _ in range(num_new):
            if len(self.agents) < self.MAX_AGENTS:
                entrance_idx = self.np_random.integers(0, len(self.entrance_points))
                ex, ey = self.entrance_points[entrance_idx]
                
                agent = Agent(
                    x=ex + self.np_random.uniform(-1.0, 1.0),
                    y=ey + self.np_random.uniform(-1.0, 1.0),
                    vx=0.0,
                    vy=0.0,
                    goal_gate=self.np_random.integers(0, self.NUM_GATES),
                    panic_level=0.3 if self.crowd_arrival_pattern == 'evacuation' else 0.0,
                    id=self.next_agent_id
                )
                self.next_agent_id += 1
                self.total_agents_spawned += 1
                self.agents.append(agent)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.timestep += 1
        reward = 0.0
        action_cost = 0.0
        
        # Execute action with infrastructure constraints
        if action < 4:  # Move barrier
            barrier_id = action
            direction = self.np_random.integers(0, 4)
            moved = self._move_barrier(barrier_id, direction)
            if moved:
                action_cost = 0.1  # Reduced cost for moving barrier
            
        elif action < 7:  # Toggle gate
            gate_id = action - 4
            toggled = self._toggle_gate(gate_id)
            if toggled:
                action_cost = 0.05  # Reduced cost for gate action
            
        elif action < 11:  # Set flow direction
            flow_id = action - 7
            self._set_flow_direction(flow_id)
            action_cost = 0.02  # Very light cost
            
        else:  # Emergency call
            action_cost = 1.0  # Moderate cost for emergency (reduced from 2.0)
            self._emergency_response()
        
        # Update infrastructure cooldowns
        self._update_cooldowns()
        
        # Spawn new arrivals based on temporal pattern
        self._spawn_new_arrivals()
        
        # Update agent physics with Social Force Model
        self._update_agent_forces()
        
        # Update panic levels
        self._update_panic_levels()
        
        # Process agent exits through gates
        self._process_agent_exits()
        
        # Run adversarial scenarios if enabled
        if self.adversarial_mode:
            self._adversarial_scenario()
        
        # Update density and velocity grids from agents
        self._update_grids_from_agents()
        
        # Calculate multi-objective rewards
        density_reward = self._calculate_density_reward()
        safety_reward = self._calculate_safety_reward()
        efficiency_reward = self._calculate_efficiency_reward()
        infrastructure_cost = self._calculate_infrastructure_cost()
        
        reward = density_reward + safety_reward + efficiency_reward - infrastructure_cost - action_cost
        
        # Check terminal conditions
        terminated = False
        truncated = False
        
        # Check for critical overcrowding
        max_density = np.max(self.grid_density)
        self.max_density_reached = max(self.max_density_reached, max_density)
        
        if max_density > self.CRITICAL_DENSITY:
            self.overcrowding_events += 1
            reward -= 20.0  # Reduced from 50.0 to match step reward scale
            terminated = True
        
        # Check for successful dispersal
        if len(self.agents) < 10 and self.timestep > 100:
            reward += 30.0  # Reduced from 100.0 to match step reward scale
            terminated = True
        
        # Check for max steps
        if self.timestep >= self.MAX_STEPS:
            truncated = True
        
        # Track max panic
        if self.agents:
            self.max_panic_reached = max(self.max_panic_reached, 
                                        max(a.panic_level for a in self.agents))
        
        self.episode_rewards.append(reward)
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()
    
    def _update_agent_forces(self):
        """Social Force Model for realistic crowd dynamics - NOVEL CONTRIBUTION"""
        for agent in self.agents:
            # Goal attraction force
            target_gate = self.gates[agent.goal_gate]
            gx, gy = target_gate[0], target_gate[1]
            dx = gx - agent.x
            dy = gy - agent.y
            dist_to_goal = np.sqrt(dx**2 + dy**2) + 0.01
            
            # Desired speed increases with panic
            desired_speed = self.AGENT_DESIRED_SPEED * (1.0 + agent.panic_level)
            
            # Goal force (exponential decay)
            goal_force_x = (desired_speed * dx / dist_to_goal - agent.vx) * 0.5
            goal_force_y = (desired_speed * dy / dist_to_goal - agent.vy) * 0.5
            
            # Repulsion from other agents
            repulsion_x, repulsion_y = 0.0, 0.0
            for other in self.agents:
                if other.id == agent.id:
                    continue
                
                odx = agent.x - other.x
                ody = agent.y - other.y
                odist = np.sqrt(odx**2 + ody**2) + 0.01
                
                if odist < self.AGENT_PERSONAL_SPACE:
                    # Exponential repulsion
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
            
            # Wall repulsion
            wall_force_x, wall_force_y = 0.0, 0.0
            if agent.x < 1.0:
                wall_force_x = 2.0 * (1.0 - agent.x)
            elif agent.x > self.GRID_WIDTH - 1.0:
                wall_force_x = -2.0 * (agent.x - self.GRID_WIDTH + 1.0)
            
            if agent.y < 1.0:
                wall_force_y = 2.0 * (1.0 - agent.y)
            elif agent.y > self.GRID_HEIGHT - 1.0:
                wall_force_y = -2.0 * (agent.y - self.GRID_HEIGHT + 1.0)
            
            # Total force
            total_fx = goal_force_x + repulsion_x + barrier_force_x + wall_force_x
            total_fy = goal_force_y + repulsion_y + barrier_force_y + wall_force_y
            
            # Update velocity
            agent.vx += total_fx * 0.1
            agent.vy += total_fy * 0.1
            
            # Apply damping and limit speed
            agent.vx *= 0.8
            agent.vy *= 0.8
            max_speed = 1.5 * (1.0 + agent.panic_level * 0.5)
            speed = np.sqrt(agent.vx**2 + agent.vy**2)
            if speed > max_speed:
                agent.vx = (agent.vx / speed) * max_speed
                agent.vy = (agent.vy / speed) * max_speed
            
            # Update position
            agent.x += agent.vx * 0.1
            agent.y += agent.vy * 0.1
            
            # Clamp to grid
            agent.x = np.clip(agent.x, 0.5, self.GRID_WIDTH - 0.5)
            agent.y = np.clip(agent.y, 0.5, self.GRID_HEIGHT - 0.5)
    
    def _update_panic_levels(self):
        """Panic spreads based on local density - NOVEL CONTRIBUTION"""
        for agent in self.agents:
            # Calculate local density
            local_count = 0
            for other in self.agents:
                dist = np.sqrt((agent.x - other.x)**2 + (agent.y - other.y)**2)
                if dist < 2.0:
                    local_count += 1
            
            # Panic increases with high local density
            if local_count > self.PANIC_TRIGGER_DENSITY:
                agent.panic_level = min(1.0, agent.panic_level + self.PANIC_INCREASE_RATE)
            else:
                agent.panic_level = max(0.0, agent.panic_level - self.PANIC_DECREASE_RATE)
            
            # Panic spreads to nearby agents
            if agent.panic_level > 0.5:
                for other in self.agents:
                    if other.id == agent.id:
                        continue
                    dist = np.sqrt((agent.x - other.x)**2 + (agent.y - other.y)**2)
                    if dist < self.PANIC_SPREAD_RADIUS:
                        other.panic_level = min(1.0, other.panic_level + 0.05)
    
    def _adversarial_scenario(self):
        """Create worst-case scenarios for safety testing - NOVEL CONTRIBUTION"""
        if self.np_random.random() < 0.05:  # 5% chance each step
            scenario = self.np_random.choice(['gate_failure', 'sudden_rush', 'bottleneck'])
            
            if scenario == 'gate_failure':
                # Random gate closes unexpectedly
                gate_id = self.np_random.integers(0, self.NUM_GATES)
                self.gates[gate_id][2] = 0.0
                
            elif scenario == 'sudden_rush':
                # Large crowd surge
                for _ in range(15):
                    if len(self.agents) < self.MAX_AGENTS:
                        entrance_idx = self.np_random.integers(0, len(self.entrance_points))
                        ex, ey = self.entrance_points[entrance_idx]
                        agent = Agent(
                            x=ex + self.np_random.uniform(-1.0, 1.0),
                            y=ey + self.np_random.uniform(-1.0, 1.0),
                            vx=0.0, vy=0.0,
                            goal_gate=self.np_random.integers(0, self.NUM_GATES),
                            panic_level=0.5,
                            id=self.next_agent_id
                        )
                        self.next_agent_id += 1
                        self.total_agents_spawned += 1
                        self.agents.append(agent)
                        
            elif scenario == 'bottleneck':
                # Temporarily close multiple gates
                num_to_close = min(2, self.NUM_GATES)
                for i in range(num_to_close):
                    self.gates[i][2] = 0.0
    
    def _move_barrier(self, barrier_id: int, direction: int) -> bool:
        """Move barrier with cooldown constraint - NOVEL CONTRIBUTION"""
        if barrier_id >= len(self.barriers):
            return False
        
        # Check cooldown
        if self.barrier_move_cooldown[barrier_id] > 0:
            return False  # Can't move yet
        
        x, y = self.barriers[barrier_id]
        
        if direction == 0:  # Up
            y = max(1, y - 1)
        elif direction == 1:  # Down
            y = min(self.GRID_HEIGHT - 2, y + 1)
        elif direction == 2:  # Left
            x = max(1, x - 1)
        elif direction == 3:  # Right
            x = min(self.GRID_WIDTH - 2, x + 1)
        
        self.barriers[barrier_id] = [x, y]
        self.barrier_move_cooldown[barrier_id] = self.BARRIER_MOVE_COST
        return True
    
    def _toggle_gate(self, gate_id: int) -> bool:
        """Toggle gate with transition delay - NOVEL CONTRIBUTION"""
        if gate_id >= len(self.gates):
            return False
        
        # Check if gate is in transition
        time_since_toggle = self.timestep - self.gate_open_times[gate_id]
        if time_since_toggle < self.GATE_TRANSITION_DELAY:
            return False  # Gate still transitioning
        
        # Toggle gate
        self.gates[gate_id][2] = 1.0 - self.gates[gate_id][2]
        self.gate_open_times[gate_id] = self.timestep
        return True
    
    def _set_flow_direction(self, flow_id: int):
        """Set crowd flow direction (guide crowds)"""
        # This influences agent goals slightly
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        dy, dx = directions[flow_id]
        
        # Nudge agents in this general direction
        for agent in self.agents:
            agent.vx += dx * 0.05
            agent.vy += dy * 0.05
    
    def _emergency_response(self):
        """Emergency response - open all gates"""
        for i, gate in enumerate(self.gates):
            gate[2] = 1.0
            gate[3] = 8.0
            self.gate_open_times[i] = self.timestep
    
    def _update_cooldowns(self):
        """Update infrastructure cooldowns"""
        for i in range(len(self.barrier_move_cooldown)):
            if self.barrier_move_cooldown[i] > 0:
                self.barrier_move_cooldown[i] -= 1
    
    def _process_agent_exits(self):
        """Remove agents that exit through open gates"""
        agents_to_remove = []
        
        for agent in self.agents:
            for gate in self.gates:
                gx, gy, is_open, capacity = gate
                
                if is_open > 0.5:  # Gate is open
                    dist = np.sqrt((agent.x - gx)**2 + (agent.y - gy)**2)
                    if dist < 1.5:
                        # Agent exits
                        agents_to_remove.append(agent)
                        self.total_agents_exited += 1
                        break
        
        for agent in agents_to_remove:
            self.agents.remove(agent)
    
    def _update_grids_from_agents(self):
        """Convert individual agents to density/panic grids for observation"""
        self.grid_density = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.float32)
        self.panic_grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.float32)
        self.velocity_x = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.float32)
        self.velocity_y = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.float32)
        
        for agent in self.agents:
            x, y = int(agent.x), int(agent.y)
            if 0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT:
                self.grid_density[y, x] += 1.0
                self.panic_grid[y, x] = max(self.panic_grid[y, x], agent.panic_level)
                self.velocity_x[y, x] = agent.vx
                self.velocity_y[y, x] = agent.vy
    
    def _calculate_density_reward(self) -> float:
        """Reward for balanced density - NORMALIZED to prevent scaling issues"""
        reward = 0.0
        
        # Calculate average density (normalized by grid size)
        total_cells = self.GRID_WIDTH * self.GRID_HEIGHT
        avg_above_target = np.sum(np.maximum(0, self.grid_density - self.TARGET_DENSITY)) / total_cells
        reward -= avg_above_target * 0.5  # Scaled to ~[-2, 0]
        
        # Heavy penalty only for truly dangerous density
        critical_cells = np.sum(self.grid_density > self.CRITICAL_DENSITY)
        if critical_cells > 0:
            reward -= 3.0  # Fixed penalty, not scaled by count
        
        # Small reward for low average density
        avg_density = np.mean(self.grid_density)
        if avg_density < self.TARGET_DENSITY:
            reward += 1.0
        
        return reward
    
    def _calculate_safety_reward(self) -> float:
        """Enhanced safety with panic consideration - NOVEL CONTRIBUTION (NORMALIZED)"""
        reward = 0.0
        
        # Panic penalty (clamped to prevent explosion)
        if self.agents:
            avg_panic = np.mean([a.panic_level for a in self.agents])
            reward -= np.clip(avg_panic * 3.0, 0, 3.0)  # Max penalty: -3.0
            
            # Severe penalty for extreme panic
            max_panic = max(a.panic_level for a in self.agents)
            if max_panic > 0.8:
                reward -= 2.0  # Fixed penalty
            
            # Reward for calm crowds
            if avg_panic < 0.2:
                reward += 1.0
        else:
            reward += 0.5  # Small bonus for empty venue
        
        return reward
    
    def _calculate_efficiency_reward(self) -> float:
        """Reward for throughput - NORMALIZED to prevent explosion"""
        reward = 0.0
        
        # Reward for agents successfully exiting (normalized per step, not cumulative)
        # Track exits THIS step, not cumulative rate
        if not hasattr(self, '_last_exit_count'):
            self._last_exit_count = 0
        
        exits_this_step = self.total_agents_exited - self._last_exit_count
        self._last_exit_count = self.total_agents_exited
        reward += min(exits_this_step * 0.5, 3.0)  # Capped at +3.0 even if many exit
        
        # Penalize having too many agents waiting
        if len(self.agents) > self.MAX_AGENTS * 0.7:
            reward -= 1.0
        
        # Small reward for reducing crowd size
        agent_ratio = len(self.agents) / self.MAX_AGENTS
        if agent_ratio < 0.5:
            reward += 0.5  # Bonus for keeping venue clear
        
        return reward
    
    def _calculate_infrastructure_cost(self) -> float:
        """Cost for using infrastructure - NOVEL CONTRIBUTION"""
        cost = 0.0
        
        # Light penalty for active cooldowns (reduced from 0.3)
        active_cooldowns = sum(1 for c in self.barrier_move_cooldown if c > 0)
        cost += 0.1 * active_cooldowns
        
        # Penalize having too many gates closed during peak times
        progress = self.timestep / self.MAX_STEPS
        if 0.2 < progress < 0.6:  # Peak arrival time
            closed_gates = sum(1 for g in self.gates if g[2] < 0.5)
            cost += 2.0 * closed_gates  # Increased - gates should be open during peak!
        
        return cost
    
    def _get_observation(self) -> np.ndarray:
        """Enhanced observation with panic, transitions, cooldowns - NOVEL CONTRIBUTION"""
        # Normalize density
        norm_density = np.clip(self.grid_density / self.MAX_CROWD_PER_CELL, 0, 1)
        
        # Build observation
        obs = np.concatenate([
            norm_density.flatten(),
            self.panic_grid.flatten(),
            self.velocity_x.flatten(),
            self.velocity_y.flatten(),
        ])
        
        # Gate states + transition status
        gate_states = []
        for i, gate in enumerate(self.gates):
            gate_states.append(gate[2])  # Open/closed
            # Gate transition progress
            time_since_toggle = self.timestep - self.gate_open_times[i]
            transition_progress = min(1.0, time_since_toggle / self.GATE_TRANSITION_DELAY)
            gate_states.append(transition_progress)
        
        obs = np.concatenate([obs, np.array(gate_states, dtype=np.float32)])
        
        # Barrier positions + cooldowns
        barrier_data = []
        for i, (bx, by) in enumerate(self.barriers):
            barrier_data.extend([bx / self.GRID_WIDTH, by / self.GRID_HEIGHT])
            # Cooldown status
            cooldown_status = self.barrier_move_cooldown[i] / self.BARRIER_MOVE_COST
            barrier_data.append(cooldown_status)
        
        obs = np.concatenate([obs, np.array(barrier_data, dtype=np.float32)])
        
        # Scenario info
        scenario_info = [
            self.timestep / self.MAX_STEPS,
            len(self.agents) / self.MAX_AGENTS,
            np.mean([a.panic_level for a in self.agents]) if self.agents else 0.0,
        ]
        
        obs = np.concatenate([obs, np.array(scenario_info, dtype=np.float32)])
        
        return obs.astype(np.float32)
    
    def _get_info(self) -> Dict:
        """Return environment info"""
        avg_panic = np.mean([a.panic_level for a in self.agents]) if self.agents else 0.0
        max_panic = max([a.panic_level for a in self.agents]) if self.agents else 0.0
        
        return {
            'timestep': self.timestep,
            'total_agents': len(self.agents),
            'total_spawned': self.total_agents_spawned,
            'total_exited': self.total_agents_exited,
            'max_density': np.max(self.grid_density),
            'avg_panic': avg_panic,
            'max_panic': max_panic,
            'overcrowding_events': self.overcrowding_events,
            'open_gates': sum(1 for g in self.gates if g[2] > 0.5),
            'pattern': self.crowd_arrival_pattern,
            'difficulty': self.difficulty,
        }
    
    def render(self):
        """Render the environment"""
        if self.render_mode is None:
            return
        
        if self.render_mode == 'human':
            if self.renderer is None:
                from environment.enhanced_rendering import EnhancedCrowdRenderer
                self.renderer = EnhancedCrowdRenderer(self)
            
            self.renderer.update_scene(
                agents=self.agents,
                gates=self.gates,
                barriers=self.barriers,
                info=self._get_info()
            )
        
        return None
    
    def close(self):
        """Clean up resources"""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

"""
Curriculum Learning for Enhanced Crowd Control
==============================================

Progressive difficulty training with scenario diversity.

NOVEL CONTRIBUTION: Trains models through increasing difficulty stages,
starting with simple steady flows and progressing to complex rush scenarios
and adversarial safety-critical situations.

Training Stages:
1. Easy: 100 agents, steady pattern, no adversarial
2. Medium: 150 agents, rush pattern, no adversarial  
3. Hard: 200 agents, rush pattern, adversarial enabled

This curriculum approach allows the agent to learn basic crowd management
before facing complex emergency scenarios.
"""

import sys
import os
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import numpy as np
from typing import Dict, Tuple
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

def make_vec_env(difficulty='easy', pattern='steady', adversarial=False, n_envs=4):
    """
    Create a vectorized environment with n_envs parallel environments.
    """
    def make_env():
        def _init():
            env = create_env_with_curriculum(difficulty, pattern, adversarial)
            return Monitor(env)
        return _init

    env_fns = [make_env() for _ in range(n_envs)]
    # Use SubprocVecEnv for true parallelism or DummyVecEnv for simple vectorization
    vec_env = SubprocVecEnv(env_fns)
    return vec_env


def create_env_with_curriculum(difficulty='easy', pattern='steady', adversarial=False):
    """
    Create environment with specific difficulty settings
    
    NOVEL: Parameterized difficulty for curriculum learning
    """
    env = EnhancedCrowdControlEnv(
        crowd_arrival_pattern=pattern,
        adversarial_mode=adversarial,
        difficulty=difficulty
    )
    return env


def train_with_curriculum_ppo(
    total_timesteps=300000,
    algorithm='PPO',
    learning_rate=3e-4,
    save_dir='models/curriculum'
):
    """
    Train PPO agent with curriculum learning
    
    NOVEL CONTRIBUTION: Three-stage training with progressive difficulty
    
    Args:
        total_timesteps: Total training steps (divided across stages)
        algorithm: RL algorithm to use
        learning_rate: Learning rate
        save_dir: Directory to save models
    """
    
    print("="*70)
    print("CURRICULUM LEARNING FOR CROWD CONTROL")
    print("="*70)
    print(f"\nAlgorithm: {algorithm}")
    print(f"Total Steps: {total_timesteps:,}")
    print(f"Learning Rate: {learning_rate}")
    print("\nTraining Stages:")
    print("  1. Easy: 100 agents, steady flow")
    print("  2. Medium: 150 agents, rush scenario")
    print("  3. Hard: 200 agents, rush + adversarial")
    print("="*70)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Define curriculum stages
    stages = [
        {
            'name': 'easy',
            'difficulty': 'easy',
            'pattern': 'steady',
            'adversarial': False,
            'timesteps': total_timesteps // 3,
            'description': 'Basic crowd management'
        },
        {
            'name': 'medium',
            'difficulty': 'medium',
            'pattern': 'rush',
            'adversarial': False,
            'timesteps': total_timesteps // 3,
            'description': 'Rush hour scenarios'
        },
        {
            'name': 'hard',
            'difficulty': 'hard',
            'pattern': 'rush',
            'adversarial': True,
            'timesteps': total_timesteps // 3,
            'description': 'Adversarial safety testing'
        }
    ]
    
    model = None
    
    for stage_idx, stage in enumerate(stages):
        print(f"\n{'='*70}")
        print(f"STAGE {stage_idx + 1}/3: {stage['name'].upper()}")
        print(f"Description: {stage['description']}")
        print(f"Timesteps: {stage['timesteps']:,}")
        print(f"Pattern: {stage['pattern']} | Adversarial: {stage['adversarial']}")
        print('='*70)
        
        # Create environment for this stage
        env = make_vec_env(
            difficulty=stage['difficulty'],
            pattern=stage['pattern'],
            adversarial=stage['adversarial'],
            n_envs=8  # for example, 8 parallel environments
        )
        
        # Create or update model
        if model is None:
            # First stage - create new model
            print(f"\nCreating new {algorithm} model...")
            if algorithm == 'PPO':
                model = PPO(
                    "MlpPolicy",
                    env,
                    learning_rate=learning_rate,
                    n_steps=256,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    verbose=1,
                    tensorboard_log=f"logs/tensorboard/curriculum_{algorithm}"
                )
            elif algorithm == 'DQN':
                model = DQN(
                    "MlpPolicy",
                    env,
                    learning_rate=learning_rate,
                    buffer_size=100000,
                    learning_starts=10000,
                    batch_size=32,
                    tau=1.0,
                    gamma=0.99,
                    target_update_interval=1000,
                    verbose=1,
                    tensorboard_log=f"logs/tensorboard/curriculum_{algorithm}"
                )
            elif algorithm == 'A2C':
                model = A2C(
                    "MlpPolicy",
                    env,
                    learning_rate=learning_rate,
                    n_steps=5,
                    gamma=0.99,
                    gae_lambda=0.95,
                    verbose=1,
                    tensorboard_log=f"logs/tensorboard/curriculum_{algorithm}"
                )
        else:
            # Continue training with new environment
            print(f"\nContinuing training on {stage['name']} stage...")
            model.set_env(env)
        
        # Setup callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=f"{save_dir}/{algorithm}_{stage['name']}",
            name_prefix=f"{algorithm}_curriculum"
        )
        
        # Train for this stage
        print(f"\nStarting training for {stage['timesteps']:,} steps...")
        model.learn(
            total_timesteps=stage['timesteps'],
            callback=checkpoint_callback,
            progress_bar=True,
            reset_num_timesteps=False  # Continue timestep count
        )
        
        # Save stage model
        stage_model_path = f"{save_dir}/{algorithm}_curriculum_{stage['name']}.zip"
        model.save(stage_model_path)
        print(f"✓ Stage {stage_idx + 1} complete. Model saved: {stage_model_path}")
        
        env.close()
    
    # Save final model
    final_model_path = f"{save_dir}/{algorithm}_curriculum_final.zip"
    model.save(final_model_path)
    
    print("\n" + "="*70)
    print("CURRICULUM TRAINING COMPLETE")
    print("="*70)
    print(f"Final model saved: {final_model_path}")
    print("\nStage models:")
    for stage in stages:
        print(f"  - {algorithm}_curriculum_{stage['name']}.zip")
    print("="*70)
    
    return model, final_model_path


def train_curriculum_all_algorithms(total_timesteps=300000):
    """
    Train all algorithms with curriculum learning
    
    NOVEL: Systematic comparison across algorithms with curriculum
    """
    algorithms = ['PPO', 'DQN', 'A2C']
    results = {}
    
    for algo in algorithms:
        print(f"\n\n{'#'*70}")
        print(f"# TRAINING {algo} WITH CURRICULUM LEARNING")
        print(f"{'#'*70}\n")
        
        try:
            model, model_path = train_with_curriculum_ppo(
                total_timesteps=total_timesteps,
                algorithm=algo,
                save_dir=f'models/curriculum_{algo.lower()}'
            )
            results[algo] = {'success': True, 'model_path': model_path}
        except Exception as e:
            print(f"\n✗ Error training {algo}: {e}")
            results[algo] = {'success': False, 'error': str(e)}
    
    # Print summary
    print("\n" + "="*70)
    print("CURRICULUM TRAINING SUMMARY")
    print("="*70)
    for algo, result in results.items():
        if result['success']:
            print(f"✓ {algo}: Successfully trained")
            print(f"  Model: {result['model_path']}")
        else:
            print(f"✗ {algo}: Failed - {result.get('error', 'Unknown error')}")
    print("="*70)
    
    return results


if __name__ == "__main__":
    train_curriculum_all_algorithms(total_timesteps=5000000)

"""
Crowd Control Environment for Reinforcement Learning
=====================================================

Mission: Manage crowd flow in a venue to prevent overcrowding at gates/exits
The agent controls barrier placements and crowd flow directions to ensure safety.

State Space:
- Grid-based representation (20x20)
- Crowd density at each cell
- Crowd velocity vectors
- Gate positions and capacities
- Barrier positions
- Emergency indicators

Action Space:
- Move barriers (4 directions per barrier)
- Open/close gates
- Issue crowd flow directions (guide signs)
- Call for backup (emergency response)

Rewards:
- Negative reward for high crowd density (overcrowding risk)
- Positive reward for balanced crowd distribution
- Large negative reward for stampede conditions
- Positive reward for successful crowd dispersal
- Small penalty for unnecessary actions

Terminal Conditions:
- Maximum timesteps reached
- Critical overcrowding event (failure)
- All crowds safely dispersed (success)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Optional
import math


class CrowdControlEnv(gym.Env):
    """Custom Environment for Crowd Control using Gymnasium API"""
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}
    
    # Environment constants
    GRID_WIDTH = 20
    GRID_HEIGHT = 20
    MAX_CROWD_PER_CELL = 10.0
    CRITICAL_DENSITY = 8.0  # Dangerous crowd density
    TARGET_DENSITY = 3.0    # Optimal crowd density
    MAX_STEPS = 500
    
    # Gate and barrier constants
    NUM_GATES = 3
    NUM_BARRIERS = 4
    
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        self.render_mode = render_mode
        self.renderer = None
        
        # Define action space
        # Actions: [barrier_id (4), direction (5: up/down/left/right/stay), 
        #           gate_id (3), gate_action (2: open/close), 
        #           flow_direction (4), emergency (2)]
        # Simplified: 0-3: move barriers, 4-6: toggle gates, 7-10: set flow directions, 11: emergency
        self.action_space = spaces.Discrete(12)
        
        # Define observation space
        # [grid_density (20x20), velocity_x (20x20), velocity_y (20x20),
        #  gate_states (3), barrier_positions (4x2), timestep (1)]
        obs_size = (self.GRID_WIDTH * self.GRID_HEIGHT * 3 +  # density + velocity
                   self.NUM_GATES +                            # gate states
                   self.NUM_BARRIERS * 2 +                     # barrier positions
                   1)                                          # timestep
        
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        
        # Initialize state variables
        self.grid_density = None
        self.velocity_x = None
        self.velocity_y = None
        self.gates = None
        self.barriers = None
        self.timestep = 0
        self.total_crowd = 0
        self.episode_rewards = []
        
        # Statistics tracking
        self.max_density_reached = 0
        self.overcrowding_events = 0
        self.successful_dispersals = 0
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        # Reset timestep
        self.timestep = 0
        self.episode_rewards = []
        self.max_density_reached = 0
        self.overcrowding_events = 0
        
        # Initialize crowd density (simulate people entering from entrances)
        self.grid_density = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.float32)
        
        # Create initial crowd clusters (simulating entrance points)
        entrance_points = [
            (2, 2), (2, 17), (17, 2), (17, 17), (10, 2)  # 5 entrance points
        ]
        
        for ey, ex in entrance_points:
            # Add crowd around entrance
            crowd_size = self.np_random.uniform(3.0, 6.0)
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    y, x = ey + dy, ex + dx
                    if 0 <= y < self.GRID_HEIGHT and 0 <= x < self.GRID_WIDTH:
                        self.grid_density[y, x] = min(crowd_size, self.MAX_CROWD_PER_CELL)
        
        # Initialize velocity (crowds moving toward exits/gates)
        self.velocity_x = self.np_random.uniform(-0.3, 0.3, (self.GRID_HEIGHT, self.GRID_WIDTH))
        self.velocity_y = self.np_random.uniform(-0.3, 0.3, (self.GRID_HEIGHT, self.GRID_WIDTH))
        
        # Initialize gates (positioned at exits)
        # Gate: [x, y, is_open (1=open, 0=closed), capacity]
        self.gates = [
            [10, 0, 1.0, 5.0],   # Top center gate (open)
            [0, 10, 1.0, 5.0],   # Left center gate (open)
            [19, 10, 1.0, 5.0],  # Right center gate (open)
        ]
        
        # Initialize barriers (movable crowd control barriers)
        # Barrier: [x, y]
        self.barriers = [
            [5, 10],
            [15, 10],
            [10, 5],
            [10, 15],
        ]
        
        self.total_crowd = np.sum(self.grid_density)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.timestep += 1
        reward = 0.0
        
        # Execute action
        if action < 4:  # Move barrier
            barrier_id = action
            # Simple barrier movement logic (random direction for now)
            direction = self.np_random.integers(0, 4)
            self._move_barrier(barrier_id, direction)
            reward -= 0.1  # Small cost for moving barrier
            
        elif action < 7:  # Toggle gate
            gate_id = action - 4
            self._toggle_gate(gate_id)
            reward -= 0.05  # Small cost for gate action
            
        elif action < 11:  # Set flow direction
            flow_id = action - 7
            self._set_flow_direction(flow_id)
            reward -= 0.05
            
        else:  # Emergency call
            reward -= 1.0  # High cost for emergency
            self._emergency_response()
        
        # Simulate crowd dynamics
        self._update_crowd_dynamics()
        
        # Calculate rewards based on current state
        density_reward = self._calculate_density_reward()
        safety_reward = self._calculate_safety_reward()
        efficiency_reward = self._calculate_efficiency_reward()
        
        reward += density_reward + safety_reward + efficiency_reward
        
        # Check terminal conditions
        terminated = False
        truncated = False
        
        # Check for critical overcrowding
        max_density = np.max(self.grid_density)
        self.max_density_reached = max(self.max_density_reached, max_density)
        
        if max_density > self.CRITICAL_DENSITY:
            self.overcrowding_events += 1
            reward -= 50.0  # Large penalty for dangerous overcrowding
            terminated = True
        
        # Check for successful dispersal
        if self.total_crowd < 10.0:  # Most people have left
            reward += 100.0  # Large reward for success
            self.successful_dispersals += 1
            terminated = True
        
        # Check for max steps
        if self.timestep >= self.MAX_STEPS:
            truncated = True
        
        self.episode_rewards.append(reward)
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()
    
    def _move_barrier(self, barrier_id: int, direction: int):
        """Move barrier in specified direction (0:up, 1:down, 2:left, 3:right)"""
        if barrier_id >= len(self.barriers):
            return
        
        x, y = self.barriers[barrier_id]
        
        if direction == 0:  # Up
            y = max(0, y - 1)
        elif direction == 1:  # Down
            y = min(self.GRID_HEIGHT - 1, y + 1)
        elif direction == 2:  # Left
            x = max(0, x - 1)
        elif direction == 3:  # Right
            x = min(self.GRID_WIDTH - 1, x + 1)
        
        self.barriers[barrier_id] = [x, y]
    
    def _toggle_gate(self, gate_id: int):
        """Toggle gate between open and closed"""
        if gate_id < len(self.gates):
            self.gates[gate_id][2] = 1.0 - self.gates[gate_id][2]  # Toggle
    
    def _set_flow_direction(self, flow_id: int):
        """Set crowd flow direction (guide crowds toward specific areas)"""
        # Influence velocity in a region
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right
        dy, dx = directions[flow_id]
        
        # Apply flow influence to velocity
        self.velocity_x += dx * 0.1
        self.velocity_y += dy * 0.1
        
        # Clip velocities
        self.velocity_x = np.clip(self.velocity_x, -1.0, 1.0)
        self.velocity_y = np.clip(self.velocity_y, -1.0, 1.0)
    
    def _emergency_response(self):
        """Emergency response - open all gates and create evacuation paths"""
        for gate in self.gates:
            gate[2] = 1.0  # Open all gates
            gate[3] = 8.0  # Increase capacity
    
    def _update_crowd_dynamics(self):
        """Update crowd positions and densities based on movement"""
        new_density = np.copy(self.grid_density)
        
        # Simulate crowd movement
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid_density[y, x] > 0.1:
                    # Calculate movement toward nearest open gate
                    target_x, target_y = self._find_nearest_open_gate(x, y)
                    
                    # Update velocity toward target
                    dx = target_x - x
                    dy = target_y - y
                    dist = math.sqrt(dx*dx + dy*dy) + 0.01
                    
                    self.velocity_x[y, x] = 0.7 * self.velocity_x[y, x] + 0.3 * (dx / dist)
                    self.velocity_y[y, x] = 0.7 * self.velocity_y[y, x] + 0.3 * (dy / dist)
                    
                    # Move crowd
                    move_amount = min(0.5, self.grid_density[y, x] * 0.1)
                    
                    new_x = int(np.clip(x + self.velocity_x[y, x], 0, self.GRID_WIDTH - 1))
                    new_y = int(np.clip(y + self.velocity_y[y, x], 0, self.GRID_HEIGHT - 1))
                    
                    # Check if barrier blocks movement
                    if not self._is_blocked_by_barrier(x, y, new_x, new_y):
                        new_density[y, x] -= move_amount
                        new_density[new_y, new_x] += move_amount
        
        # Process people leaving through gates
        for gate in self.gates:
            gx, gy, is_open, capacity = gate
            gx, gy = int(gx), int(gy)
            
            if is_open > 0.5:  # Gate is open
                # Remove people near gate
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        cy, cx = gy + dy, gx + dx
                        if 0 <= cy < self.GRID_HEIGHT and 0 <= cx < self.GRID_WIDTH:
                            exit_amount = min(new_density[cy, cx], capacity * 0.1)
                            new_density[cy, cx] -= exit_amount
        
        # Add small random influx (new people entering)
        if self.timestep < self.MAX_STEPS * 0.7:  # Stop influx near end
            entrance_points = [(2, 2), (2, 17)]
            for ey, ex in entrance_points:
                if self.np_random.random() < 0.3:  # 30% chance
                    new_density[ey, ex] += self.np_random.uniform(0.5, 1.5)
        
        self.grid_density = np.clip(new_density, 0, self.MAX_CROWD_PER_CELL)
        self.total_crowd = np.sum(self.grid_density)
    
    def _find_nearest_open_gate(self, x: int, y: int) -> Tuple[int, int]:
        """Find the nearest open gate"""
        min_dist = float('inf')
        target_x, target_y = x, y
        
        for gate in self.gates:
            gx, gy, is_open, _ = gate
            if is_open > 0.5:  # Gate is open
                dist = math.sqrt((x - gx)**2 + (y - gy)**2)
                if dist < min_dist:
                    min_dist = dist
                    target_x, target_y = int(gx), int(gy)
        
        return target_x, target_y
    
    def _is_blocked_by_barrier(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """Check if movement is blocked by a barrier"""
        for barrier in self.barriers:
            bx, by = barrier
            # Simple check: if barrier is on the path
            if (x1 <= bx <= x2 or x2 <= bx <= x1) and (y1 <= by <= y2 or y2 <= by <= y1):
                if abs(bx - x2) <= 1 and abs(by - y2) <= 1:
                    return True
        return False
    
    def _calculate_density_reward(self) -> float:
        """Reward for maintaining optimal crowd density"""
        avg_density = np.mean(self.grid_density)
        max_density = np.max(self.grid_density)
        
        # Penalize high density
        density_penalty = 0.0
        if max_density > self.TARGET_DENSITY:
            density_penalty = -0.5 * (max_density - self.TARGET_DENSITY)
        
        # Reward for keeping average density low
        density_reward = 0.1 * (self.TARGET_DENSITY - avg_density) if avg_density < self.TARGET_DENSITY else 0
        
        return density_penalty + density_reward
    
    def _calculate_safety_reward(self) -> float:
        """Reward for maintaining safe conditions"""
        # Count cells with dangerous density
        dangerous_cells = np.sum(self.grid_density > (self.CRITICAL_DENSITY * 0.8))
        
        if dangerous_cells > 0:
            return -2.0 * dangerous_cells
        
        return 0.5  # Small reward for no dangerous conditions
    
    def _calculate_efficiency_reward(self) -> float:
        """Reward for efficient crowd flow"""
        # Reward for reducing total crowd over time
        crowd_reduction = -0.01 * self.total_crowd
        
        # Reward for open gates
        open_gates = sum(1 for gate in self.gates if gate[2] > 0.5)
        gate_reward = 0.1 * open_gates
        
        return crowd_reduction + gate_reward
    
    def _get_observation(self) -> np.ndarray:
        """Get the current observation"""
        # Normalize density
        norm_density = self.grid_density / self.MAX_CROWD_PER_CELL
        
        # Flatten grids
        obs = np.concatenate([
            norm_density.flatten(),
            self.velocity_x.flatten(),
            self.velocity_y.flatten(),
        ])
        
        # Add gate states
        gate_states = np.array([g[2] for g in self.gates], dtype=np.float32)
        obs = np.concatenate([obs, gate_states])
        
        # Add barrier positions (normalized)
        barrier_pos = np.array(self.barriers, dtype=np.float32).flatten()
        barrier_pos[::2] /= self.GRID_WIDTH  # Normalize x
        barrier_pos[1::2] /= self.GRID_HEIGHT  # Normalize y
        obs = np.concatenate([obs, barrier_pos])
        
        # Add timestep (normalized)
        timestep_norm = np.array([self.timestep / self.MAX_STEPS], dtype=np.float32)
        obs = np.concatenate([obs, timestep_norm])
        
        return obs.astype(np.float32)
    
    def _get_info(self) -> Dict:
        """Get additional information"""
        return {
            'timestep': self.timestep,
            'total_crowd': self.total_crowd,
            'max_density': np.max(self.grid_density),
            'avg_density': np.mean(self.grid_density),
            'open_gates': sum(1 for g in self.gates if g[2] > 0.5),
            'overcrowding_events': self.overcrowding_events,
            'max_density_reached': self.max_density_reached,
        }
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            if self.renderer is None:
                from .rendering import CrowdControlRenderer
                self.renderer = CrowdControlRenderer(self)
            
            return self.renderer.update_scene(
                self.grid_density,
                self.gates,
                self.barriers,
                self._get_info()
            )
        
        elif self.render_mode == "rgb_array":
            # Return RGB array for recording
            return self._render_rgb_array()
    
    def _render_rgb_array(self) -> np.ndarray:
        """Render as RGB array"""
        # Simple visualization as RGB array
        img = np.zeros((self.GRID_HEIGHT * 20, self.GRID_WIDTH * 20, 3), dtype=np.uint8)
        
        # Draw density
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                density = self.grid_density[y, x] / self.MAX_CROWD_PER_CELL
                color = int(density * 255)
                img[y*20:(y+1)*20, x*20:(x+1)*20] = [color, 0, 0]
        
        return img
    
    def close(self):
        """Clean up resources"""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

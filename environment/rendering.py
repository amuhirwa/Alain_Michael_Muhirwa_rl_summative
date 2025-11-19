"""
Advanced 3D Visualization for Crowd Control Environment using Panda3D
====================================================================

Features:
- Real-time 3D rendering of crowd agents
- Dynamic camera controls
- Visual representation of barriers, gates, and crowd density
- On-screen metrics display
- Heat map overlays for density visualization
"""

from direct.showbase.ShowBase import ShowBase
from panda3d.core import *
from direct.gui.OnscreenText import OnscreenText
from direct.task import Task
import numpy as np
import sys
from typing import Optional, List


class CrowdControlRenderer(ShowBase):
    """3D Renderer for Crowd Control Environment using Panda3D"""
    
    def __init__(self, env):
        super().__init__()
        
        self.env = env
        self.grid_width = env.GRID_WIDTH
        self.grid_height = env.GRID_HEIGHT
        
        # Store visualization objects
        self.crowd_nodes = []
        self.barrier_nodes = []
        self.gate_nodes = []
        self.density_markers = []
        
        # Setup the scene
        self._setup_camera()
        self._setup_lighting()
        self._setup_ground()
        self._initialize_scene_objects()
        self._setup_hud()
        
        # Control variables
        self.auto_rotate = False
        self.show_heat_map = True
        
        # Setup keyboard controls
        self._setup_controls()
    
    def _setup_camera(self):
        """Configure camera position and orientation"""
        self.cam.setPos(self.grid_width / 2, -30, 25)
        self.cam.lookAt(self.grid_width / 2, self.grid_height / 2, 0)
        
        # Enable camera movement
        self.disableMouse()
    
    def _setup_lighting(self):
        """Setup scene lighting"""
        # Ambient light
        ambient = AmbientLight("ambient")
        ambient.setColor((0.3, 0.3, 0.3, 1))
        ambient_np = self.render.attachNewNode(ambient)
        self.render.setLight(ambient_np)
        
        # Directional light (sun)
        sun = DirectionalLight("sun")
        sun.setColor((0.8, 0.8, 0.7, 1))
        sun_np = self.render.attachNewNode(sun)
        sun_np.setHpr(-45, -45, 0)
        self.render.setLight(sun_np)
        
        # Additional point light
        point = PointLight("point")
        point.setColor((0.5, 0.5, 0.5, 1))
        point_np = self.render.attachNewNode(point)
        point_np.setPos(self.grid_width / 2, self.grid_height / 2, 15)
        self.render.setLight(point_np)
    
    def _setup_ground(self):
        """Create the ground plane"""
        # Create ground grid
        self.ground = self.render.attachNewNode("ground")
        
        # Create checkered floor
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                card = self._create_card(x, y, 0, 1.0, 1.0)
                card.reparentTo(self.ground)
                
                # Color based on position (checkered pattern)
                if (x + y) % 2 == 0:
                    card.setColor(0.7, 0.7, 0.7, 1.0)
                else:
                    card.setColor(0.6, 0.6, 0.6, 1.0)
        
        # Create boundary walls
        self._create_walls()
    
    def _create_walls(self):
        """Create boundary walls"""
        wall_height = 2.0
        wall_thickness = 0.2
        
        # Create 4 walls
        walls_config = [
            (self.grid_width / 2, -wall_thickness / 2, self.grid_width, wall_thickness),  # Front
            (self.grid_width / 2, self.grid_height + wall_thickness / 2, self.grid_width, wall_thickness),  # Back
            (-wall_thickness / 2, self.grid_height / 2, wall_thickness, self.grid_height),  # Left
            (self.grid_width + wall_thickness / 2, self.grid_height / 2, wall_thickness, self.grid_height),  # Right
        ]
        
        for wx, wy, width, depth in walls_config:
            wall = self._create_box(wx, wy, wall_height / 2, width, depth, wall_height)
            wall.setColor(0.3, 0.3, 0.4, 1.0)
            wall.reparentTo(self.render)
    
    def _initialize_scene_objects(self):
        """Initialize all scene objects"""
        # Create density visualization markers
        for y in range(self.grid_height):
            row = []
            for x in range(self.grid_width):
                marker = self._create_cylinder(x + 0.5, y + 0.5, 0.5, 0.4, 1.0)
                marker.setColor(0, 0, 1, 0.5)
                marker.reparentTo(self.render)
                marker.hide()  # Initially hidden
                row.append(marker)
            self.density_markers.append(row)
        
        # Create barriers (movable walls)
        for i in range(self.env.NUM_BARRIERS):
            barrier = self._create_box(0, 0, 0.5, 0.8, 0.2, 1.0)
            barrier.setColor(1, 0.5, 0, 1)
            barrier.reparentTo(self.render)
            self.barrier_nodes.append(barrier)
        
        # Create gates
        for i in range(self.env.NUM_GATES):
            gate_frame = self._create_gate()
            gate_frame.reparentTo(self.render)
            self.gate_nodes.append(gate_frame)
    
    def _setup_hud(self):
        """Setup heads-up display for metrics"""
        self.hud_texts = []
        
        # Title
        title = OnscreenText(
            text="Crowd Control RL Agent",
            pos=(-1.3, 0.9),
            scale=0.07,
            fg=(1, 1, 1, 1),
            align=TextNode.ALeft
        )
        self.hud_texts.append(title)
        
        # Metrics
        self.timestep_text = OnscreenText(
            text="Timestep: 0",
            pos=(-1.3, 0.8),
            scale=0.05,
            fg=(1, 1, 1, 1),
            align=TextNode.ALeft
        )
        self.hud_texts.append(self.timestep_text)
        
        self.crowd_text = OnscreenText(
            text="Total Crowd: 0",
            pos=(-1.3, 0.7),
            scale=0.05,
            fg=(1, 1, 1, 1),
            align=TextNode.ALeft
        )
        self.hud_texts.append(self.crowd_text)
        
        self.density_text = OnscreenText(
            text="Max Density: 0.0",
            pos=(-1.3, 0.6),
            scale=0.05,
            fg=(1, 1, 1, 1),
            align=TextNode.ALeft
        )
        self.hud_texts.append(self.density_text)
        
        self.gates_text = OnscreenText(
            text="Open Gates: 0/3",
            pos=(-1.3, 0.5),
            scale=0.05,
            fg=(1, 1, 1, 1),
            align=TextNode.ALeft
        )
        self.hud_texts.append(self.gates_text)
        
        self.status_text = OnscreenText(
            text="Status: Running",
            pos=(-1.3, 0.4),
            scale=0.05,
            fg=(0, 1, 0, 1),
            align=TextNode.ALeft
        )
        self.hud_texts.append(self.status_text)
        
        # Controls help
        help_text = OnscreenText(
            text="Controls: [H]eat Map | [R]otate | [ESC]Quit",
            pos=(0, -0.95),
            scale=0.04,
            fg=(0.8, 0.8, 0.8, 1),
            align=TextNode.ACenter
        )
        self.hud_texts.append(help_text)
    
    def _setup_controls(self):
        """Setup keyboard controls"""
        self.accept("escape", sys.exit)
        self.accept("h", self._toggle_heat_map)
        self.accept("r", self._toggle_rotation)
        self.accept("arrow_up", self._move_camera, [0, 2, 0])
        self.accept("arrow_down", self._move_camera, [0, -2, 0])
        self.accept("arrow_left", self._move_camera, [-2, 0, 0])
        self.accept("arrow_right", self._move_camera, [2, 0, 0])
    
    def _toggle_heat_map(self):
        """Toggle heat map visualization"""
        self.show_heat_map = not self.show_heat_map
    
    def _toggle_rotation(self):
        """Toggle automatic camera rotation"""
        self.auto_rotate = not self.auto_rotate
    
    def _move_camera(self, dx, dy, dz):
        """Move camera by offset"""
        pos = self.cam.getPos()
        self.cam.setPos(pos[0] + dx, pos[1] + dy, pos[2] + dz)
    
    def update_scene(self, grid_density: np.ndarray, gates: List, barriers: List, info: dict):
        """Update and render the scene"""
        # Update density visualization
        self._update_density_visualization(grid_density)
        
        # Update barriers
        self._update_barriers(barriers)
        
        # Update gates
        self._update_gates(gates)
        
        # Update HUD
        self._update_hud(info)
        
        # Handle camera rotation
        if self.auto_rotate:
            angle = self.taskMgr.globalClock.getFrameTime() * 10
            radius = 30
            self.cam.setPos(
                self.grid_width / 2 + radius * np.cos(np.radians(angle)),
                self.grid_height / 2 + radius * np.sin(np.radians(angle)),
                25
            )
            self.cam.lookAt(self.grid_width / 2, self.grid_height / 2, 0)
        
        # Process one frame
        self.taskMgr.step()
        
        return None
    
    def _update_density_visualization(self, grid_density: np.ndarray):
        """Update crowd density visualization"""
        max_density = self.env.MAX_CROWD_PER_CELL
        
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                density = grid_density[y, x]
                marker = self.density_markers[y][x]
                
                if self.show_heat_map and density > 0.1:
                    marker.show()
                    
                    # Scale based on density
                    scale = min(1.0, density / max_density)
                    marker.setScale(scale, scale, scale * 2)
                    
                    # Color based on danger level
                    if density > self.env.CRITICAL_DENSITY * 0.8:
                        # Red for dangerous
                        marker.setColor(1, 0, 0, 0.7)
                    elif density > self.env.TARGET_DENSITY:
                        # Orange for moderate
                        marker.setColor(1, 0.5, 0, 0.6)
                    else:
                        # Blue for safe
                        marker.setColor(0, 0.5, 1, 0.5)
                else:
                    marker.hide()
    
    def _update_barriers(self, barriers: List):
        """Update barrier positions"""
        for i, barrier_pos in enumerate(barriers):
            if i < len(self.barrier_nodes):
                x, y = barrier_pos
                self.barrier_nodes[i].setPos(x + 0.5, y + 0.5, 0.5)
    
    def _update_gates(self, gates: List):
        """Update gate states"""
        for i, gate_data in enumerate(gates):
            if i < len(self.gate_nodes):
                x, y, is_open, _ = gate_data
                gate_node = self.gate_nodes[i]
                gate_node.setPos(x, y, 1.0)
                
                # Change color based on state
                if is_open > 0.5:
                    gate_node.setColor(0, 1, 0, 0.8)  # Green for open
                else:
                    gate_node.setColor(1, 0, 0, 0.8)  # Red for closed
    
    def _update_hud(self, info: dict):
        """Update HUD text"""
        self.timestep_text.setText(f"Timestep: {info['timestep']}")
        self.crowd_text.setText(f"Total Crowd: {info['total_crowd']:.1f}")
        self.density_text.setText(f"Max Density: {info['max_density']:.2f}")
        self.gates_text.setText(f"Open Gates: {info['open_gates']}/{self.env.NUM_GATES}")
        
        # Update status based on conditions
        if info['max_density'] > self.env.CRITICAL_DENSITY * 0.8:
            self.status_text.setText("Status: DANGER!")
            self.status_text.setFg((1, 0, 0, 1))
        elif info['max_density'] > self.env.TARGET_DENSITY * 1.5:
            self.status_text.setText("Status: Warning")
            self.status_text.setFg((1, 0.5, 0, 1))
        else:
            self.status_text.setText("Status: Safe")
            self.status_text.setFg((0, 1, 0, 1))
    
    # Helper methods to create 3D objects
    def _create_card(self, x: float, y: float, z: float, width: float, height: float):
        """Create a flat card (quad)"""
        card = CardMaker("card")
        card.setFrame(0, width, 0, height)
        node = self.render.attachNewNode(card.generate())
        node.setPos(x, y, z)
        node.setP(-90)  # Rotate to be horizontal
        return node
    
    def _create_box(self, x: float, y: float, z: float, width: float, depth: float, height: float):
        """Create a 3D box"""
        # Use CardMaker to create a simple box representation
        format = GeomVertexFormat.getV3()
        vdata = GeomVertexData('box', format, Geom.UHStatic)
        vertex = GeomVertexWriter(vdata, 'vertex')
        
        # Define 8 vertices of the box
        hw, hd, hh = width / 2, depth / 2, height / 2
        vertices = [
            (-hw, -hd, -hh), (hw, -hd, -hh), (hw, hd, -hh), (-hw, hd, -hh),  # Bottom
            (-hw, -hd, hh), (hw, -hd, hh), (hw, hd, hh), (-hw, hd, hh),  # Top
        ]
        
        for v in vertices:
            vertex.addData3(*v)
        
        # Create faces
        geom = Geom(vdata)
        
        # Define the 6 faces
        faces = [
            (0, 1, 2, 3),  # Bottom
            (4, 5, 6, 7),  # Top
            (0, 1, 5, 4),  # Front
            (2, 3, 7, 6),  # Back
            (0, 3, 7, 4),  # Left
            (1, 2, 6, 5),  # Right
        ]
        
        for face in faces:
            tri = GeomTriangles(Geom.UHStatic)
            tri.addVertices(face[0], face[1], face[2])
            tri.addVertices(face[0], face[2], face[3])
            geom.addPrimitive(tri)
        
        node = GeomNode('box')
        node.addGeom(geom)
        np = self.render.attachNewNode(node)
        np.setPos(x, y, z)
        
        return np
    
    def _create_cylinder(self, x: float, y: float, z: float, radius: float, height: float):
        """Create a cylinder (simplified as a box for now)"""
        # Simplified: use box representation
        return self._create_box(x, y, z, radius * 2, radius * 2, height)
    
    def _create_gate(self):
        """Create a gate structure"""
        gate = self.render.attachNewNode("gate")
        
        # Create two pillars
        pillar1 = self._create_box(-0.5, 0, 0.5, 0.2, 0.2, 2.0)
        pillar1.setColor(0.2, 0.6, 0.2, 1)
        pillar1.reparentTo(gate)
        
        pillar2 = self._create_box(0.5, 0, 0.5, 0.2, 0.2, 2.0)
        pillar2.setColor(0.2, 0.6, 0.2, 1)
        pillar2.reparentTo(gate)
        
        # Create top bar
        bar = self._create_box(0, 0, 2.0, 1.2, 0.1, 0.2)
        bar.setColor(0.2, 0.6, 0.2, 1)
        bar.reparentTo(gate)
        
        return gate
    
    def close(self):
        """Clean up and close the renderer"""
        try:
            self.destroy()
        except:
            pass

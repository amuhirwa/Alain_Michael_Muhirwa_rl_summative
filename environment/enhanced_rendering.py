"""
Enhanced 3D Rendering for Crowd Control Environment
===================================================

Visualizes individual agents with panic-based coloring and advanced features.

Novel Features:
- Individual agent visualization (not just density grid)
- Panic-level coloring (blue=calm, orange=stressed, red=panic)
- Real-time panic indicator in HUD
- Infrastructure state visualization (gate transitions, barrier cooldowns)
"""

from direct.showbase.ShowBase import ShowBase
from panda3d.core import (CardMaker, AmbientLight, DirectionalLight, PointLight, 
                          GeomVertexFormat, GeomVertexData, Geom, GeomTriangles, 
                          GeomVertexWriter, GeomNode, NodePath)
from direct.gui.OnscreenText import OnscreenText, TextNode
from direct.task import Task
import numpy as np
from typing import List, Dict


class EnhancedCrowdRenderer(ShowBase):
    """Enhanced 3D renderer for crowd control with individual agent visualization"""
    
    def __init__(self, env):
        super().__init__()
        
        self.env = env
        self.grid_width = env.GRID_WIDTH
        self.grid_height = env.GRID_HEIGHT
        
        # Agent visualization
        self.agent_nodes = []
        self.max_agent_nodes = 200
        
        # Scene objects
        self.gate_nodes = []
        self.barrier_nodes = []
        self.ground_node = None
        self.hud_texts = []
        
        # Settings
        self.show_heat_map = True
        self.auto_rotate = False
        self.rotation_speed = 5.0
        
        # Initialize scene
        self._setup_camera()
        self._setup_lights()
        self._initialize_scene_objects()
        self._setup_controls()
        self._setup_hud()
        
        # Add update task
        self.taskMgr.add(self._update_task, "updateTask")
        
        print("Enhanced 3D Renderer initialized")
        print("Controls:")
        print("  [H] - Toggle heat map")
        print("  [R] - Toggle auto-rotation")
        print("  [Arrows / WASD] - Move camera horizontally")
        print("  [Q/E or Page Up/Down] - Move camera up/down")
        print("  [ESC] - Exit")
    
    def _setup_camera(self):
        """Setup camera position and orientation"""
        # Position camera at 45-degree angle view
        # This gives a nice isometric-style view of the entire grid
        center_x = self.grid_width / 2
        center_y = self.grid_height / 2
        
        # Calculate position for 45-degree angle
        distance = 35  # Distance from center
        height = 30    # Height above ground (45-degree angle when distance ≈ height)
        
        self.camera.setPos(center_x, center_y - distance, height)
        self.camera.lookAt(center_x, center_y, 0)  # Look at center of grid
        self.camLens.setFov(60)
    
    def _setup_lights(self):
        """Setup scene lighting"""
        # Ambient light
        alight = AmbientLight('ambient')
        alight.setColor((0.4, 0.4, 0.45, 1))
        alnp = self.render.attachNewNode(alight)
        self.render.setLight(alnp)
        
        # Directional light (sun)
        dlight = DirectionalLight('sun')
        dlight.setColor((0.8, 0.8, 0.7, 1))
        dlnp = self.render.attachNewNode(dlight)
        dlnp.setHpr(45, -60, 0)
        self.render.setLight(dlnp)
        
        # Point light for better visibility
        plight = PointLight('point')
        plight.setColor((0.5, 0.5, 0.5, 1))
        plnp = self.render.attachNewNode(plight)
        plnp.setPos(10, 10, 15)
        self.render.setLight(plnp)
    
    def _load_model_or_fallback(self, model_path, width, depth, height):
        """
        Try to load a custom 3D model, fall back to procedural geometry
        
        Usage:
            # Place .egg or .bam files in assets/ folder
            agent_model = self._load_model_or_fallback("assets/person.egg", 0.3, 0.3, 0.6)
            gate_model = self._load_model_or_fallback("assets/gate.egg", 1.0, 1.0, 0.3)
        """
        try:
            model = self.loader.loadModel(model_path)
            model.setScale(width, depth, height)
            return model
        except:
            # Fall back to procedural geometry
            return self._create_procedural_box(width, depth, height)
    
    def _create_box(self, x, y, width, depth, height, z=0):
        """Helper to create a box using procedural geometry"""
        box = self._create_procedural_box(width, depth, height)
        box.setPos(x, y, z + height/2)
        return box
    
    def _create_procedural_box(self, width, depth, height):
        """Create a box using procedural geometry (internal method)"""
        # Create vertex data
        vformat = GeomVertexFormat.getV3n3c4()
        vdata = GeomVertexData('box', vformat, Geom.UHStatic)
        
        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        color = GeomVertexWriter(vdata, 'color')
        
        # Define 8 vertices of a unit cube
        # Front face
        vertex.addData3(-width/2, -depth/2, 0); normal.addData3(0, 0, -1); color.addData4(1, 1, 1, 1)
        vertex.addData3(width/2, -depth/2, 0); normal.addData3(0, 0, -1); color.addData4(1, 1, 1, 1)
        vertex.addData3(width/2, depth/2, 0); normal.addData3(0, 0, -1); color.addData4(1, 1, 1, 1)
        vertex.addData3(-width/2, depth/2, 0); normal.addData3(0, 0, -1); color.addData4(1, 1, 1, 1)
        
        # Back face
        vertex.addData3(-width/2, -depth/2, height); normal.addData3(0, 0, 1); color.addData4(1, 1, 1, 1)
        vertex.addData3(width/2, -depth/2, height); normal.addData3(0, 0, 1); color.addData4(1, 1, 1, 1)
        vertex.addData3(width/2, depth/2, height); normal.addData3(0, 0, 1); color.addData4(1, 1, 1, 1)
        vertex.addData3(-width/2, depth/2, height); normal.addData3(0, 0, 1); color.addData4(1, 1, 1, 1)
        
        # Create triangles
        tris = GeomTriangles(Geom.UHStatic)
        
        # Front face
        tris.addVertices(0, 1, 2); tris.addVertices(0, 2, 3)
        # Back face
        tris.addVertices(4, 6, 5); tris.addVertices(4, 7, 6)
        # Left face
        tris.addVertices(0, 3, 7); tris.addVertices(0, 7, 4)
        # Right face
        tris.addVertices(1, 5, 6); tris.addVertices(1, 6, 2)
        # Top face
        tris.addVertices(3, 2, 6); tris.addVertices(3, 6, 7)
        # Bottom face
        tris.addVertices(0, 4, 5); tris.addVertices(0, 5, 1)
        
        geom = Geom(vdata)
        geom.addPrimitive(tris)
        
        node = GeomNode('box')
        node.addGeom(geom)
        
        box_np = self.render.attachNewNode(node)
        
        return box_np
    
    def _initialize_scene_objects(self):
        """Initialize all scene objects"""
        # Ground plane
        cm = CardMaker('ground')
        cm.setFrame(0, self.grid_width, 0, self.grid_height)
        self.ground_node = self.render.attachNewNode(cm.generate())
        self.ground_node.setP(-90)
        self.ground_node.setColor(0.3, 0.35, 0.4, 1)
        
        # Create walls
        wall_height = 3.0
        wall_thickness = 0.3
        
        # Top wall
        top_wall = self._create_box(self.grid_width/2, 0, self.grid_width, wall_thickness, wall_height)
        top_wall.setColor(0.5, 0.5, 0.55, 1)
        top_wall.reparentTo(self.render)
        
        # Bottom wall
        bottom_wall = self._create_box(self.grid_width/2, self.grid_height, self.grid_width, wall_thickness, wall_height)
        bottom_wall.setColor(0.5, 0.5, 0.55, 1)
        bottom_wall.reparentTo(self.render)
        
        # Left wall
        left_wall = self._create_box(0, self.grid_height/2, wall_thickness, self.grid_height, wall_height)
        left_wall.setColor(0.5, 0.5, 0.55, 1)
        left_wall.reparentTo(self.render)
        
        # Right wall
        right_wall = self._create_box(self.grid_width, self.grid_height/2, wall_thickness, self.grid_height, wall_height)
        right_wall.setColor(0.5, 0.5, 0.55, 1)
        right_wall.reparentTo(self.render)
        
        # Create agent node pool
        for i in range(self.max_agent_nodes):
            agent_node = self._create_box(0, 0, 0.3, 0.3, 0.6, 0)
            agent_node.setColor(0.2, 0.5, 0.8, 0.9)
            agent_node.reparentTo(self.render)
            agent_node.hide()
            self.agent_nodes.append(agent_node)
        
        # Initialize gate nodes (will be updated in update_scene)
        for _ in range(self.env.NUM_GATES):
            gate = self._create_box(0, 0, 1.0, 1.0, 0.3, 0)
            gate.setColor(0.2, 0.8, 0.2, 0.8)
            gate.reparentTo(self.render)
            self.gate_nodes.append(gate)
        
        # Initialize barrier nodes
        for _ in range(self.env.NUM_BARRIERS):
            barrier = self._create_box(0, 0, 0.8, 0.8, 1.5, 0)
            barrier.setColor(0.9, 0.6, 0.1, 0.9)
            barrier.reparentTo(self.render)
            self.barrier_nodes.append(barrier)
    
    def _setup_controls(self):
        """Setup keyboard controls"""
        self.accept('h', self._toggle_heat_map)
        self.accept('r', self._toggle_rotation)
        self.accept('escape', self.userExit)
        
        # Arrow keys for horizontal movement
        self.accept('arrow_up', self._move_camera, [0, 2, 0])
        self.accept('arrow_down', self._move_camera, [0, -2, 0])
        self.accept('arrow_left', self._move_camera, [-2, 0, 0])
        self.accept('arrow_right', self._move_camera, [2, 0, 0])
        
        # Page Up/Down for vertical movement
        self.accept('page_up', self._move_camera, [0, 0, 2])
        self.accept('page_down', self._move_camera, [0, 0, -2])
        
        # W/A/S/D for alternative movement
        self.accept('w', self._move_camera, [0, 2, 0])
        self.accept('s', self._move_camera, [0, -2, 0])
        self.accept('a', self._move_camera, [-2, 0, 0])
        self.accept('d', self._move_camera, [2, 0, 0])
        self.accept('q', self._move_camera, [0, 0, 2])  # Up
        self.accept('e', self._move_camera, [0, 0, -2])  # Down
    
    def _setup_hud(self):
        """Setup HUD text displays"""
        # Title
        title = OnscreenText(
            text="ENHANCED CROWD CONTROL - Individual Agent Simulation",
            pos=(-1.3, 0.9),
            scale=0.06,
            fg=(1, 1, 1, 1),
            align=TextNode.ALeft,
            mayChange=False
        )
        self.hud_texts.append(title)
        
        # Agent count
        self.agent_count_text = OnscreenText(
            text="Agents: 0",
            pos=(-1.3, 0.8),
            scale=0.05,
            fg=(1, 1, 1, 1),
            align=TextNode.ALeft
        )
        self.hud_texts.append(self.agent_count_text)
        
        # Density indicator
        self.density_text = OnscreenText(
            text="Max Density: 0.0",
            pos=(-1.3, 0.7),
            scale=0.05,
            fg=(1, 1, 1, 1),
            align=TextNode.ALeft
        )
        self.hud_texts.append(self.density_text)
        
        # Panic indicator (NEW)
        self.panic_text = OnscreenText(
            text="Panic Level: 0.0",
            pos=(-1.3, 0.6),
            scale=0.05,
            fg=(0, 1, 0, 1),
            align=TextNode.ALeft
        )
        self.hud_texts.append(self.panic_text)
        
        # Gate status
        self.gate_status_text = OnscreenText(
            text="Gates: 0/0 Open",
            pos=(-1.3, 0.5),
            scale=0.05,
            fg=(1, 1, 1, 1),
            align=TextNode.ALeft
        )
        self.hud_texts.append(self.gate_status_text)
        
        # Throughput
        self.throughput_text = OnscreenText(
            text="Exited: 0",
            pos=(-1.3, 0.4),
            scale=0.05,
            fg=(1, 1, 1, 1),
            align=TextNode.ALeft
        )
        self.hud_texts.append(self.throughput_text)
        
        # Scenario info
        self.scenario_text = OnscreenText(
            text="Pattern: N/A | Difficulty: N/A",
            pos=(-1.3, 0.3),
            scale=0.05,
            fg=(0.7, 0.7, 1, 1),
            align=TextNode.ALeft
        )
        self.hud_texts.append(self.scenario_text)
        
        # Controls
        controls = OnscreenText(
            text="[H] Heat Map | [R] Rotate | [Arrows/WASD] Move | [Q/E or PgUp/PgDn] Height",
            pos=(0, -0.95),
            scale=0.045,
            fg=(0.8, 0.8, 0.8, 1),
            align=TextNode.ACenter,
            mayChange=False
        )
        self.hud_texts.append(controls)
    
    def update_scene(self, agents: List, gates: List, barriers: List, info: Dict):
        """Update scene with individual agents - NOVEL CONTRIBUTION"""
        
        # Update individual agent positions and colors based on panic
        for i, agent in enumerate(agents):
            if i < len(self.agent_nodes):
                node = self.agent_nodes[i]
                node.show()
                node.setPos(agent.x, agent.y, 0.3)
                
                # Color based on panic level (NOVEL: Panic visualization)
                panic = agent.panic_level
                if panic > 0.7:
                    # Red = high panic
                    node.setColor(1, 0, 0, 0.9)
                elif panic > 0.3:
                    # Orange = moderate stress
                    node.setColor(1, 0.5, 0, 0.9)
                else:
                    # Blue = calm
                    node.setColor(0.2, 0.5, 0.8, 0.9)
        
        # Hide unused agent nodes
        for i in range(len(agents), len(self.agent_nodes)):
            self.agent_nodes[i].hide()
        
        # Update gates
        for i, gate in enumerate(gates):
            if i < len(self.gate_nodes):
                gx, gy, is_open, capacity = gate
                self.gate_nodes[i].setPos(gx, gy, 0.15)
                
                # Color based on state
                if is_open > 0.5:
                    self.gate_nodes[i].setColor(0.2, 0.8, 0.2, 0.8)  # Green = open
                else:
                    self.gate_nodes[i].setColor(0.8, 0.2, 0.2, 0.8)  # Red = closed
        
        # Update barriers
        for i, (bx, by) in enumerate(barriers):
            if i < len(self.barrier_nodes):
                self.barrier_nodes[i].setPos(bx, by, 0.75)
                
                # Check if barrier is on cooldown
                if hasattr(self.env, 'barrier_move_cooldown'):
                    if self.env.barrier_move_cooldown[i] > 0:
                        # Yellow tint when on cooldown
                        self.barrier_nodes[i].setColor(1, 0.9, 0.3, 0.9)
                    else:
                        # Normal orange
                        self.barrier_nodes[i].setColor(0.9, 0.6, 0.1, 0.9)
        
        # Update HUD
        self._update_hud(info)
    
    def _update_hud(self, info: Dict):
        """Update HUD text with current info"""
        self.agent_count_text.setText(f"Agents: {info.get('total_agents', 0)}")
        
        max_density = info.get('max_density', 0.0)
        self.density_text.setText(f"Max Density: {max_density:.2f}")
        
        # Color-code density
        if max_density > self.env.CRITICAL_DENSITY * 0.8:
            self.density_text.setFg((1, 0, 0, 1))  # Red
        elif max_density > self.env.TARGET_DENSITY:
            self.density_text.setFg((1, 0.5, 0, 1))  # Orange
        else:
            self.density_text.setFg((0, 1, 0, 1))  # Green
        
        # Panic indicator (NOVEL)
        avg_panic = info.get('avg_panic', 0.0)
        max_panic = info.get('max_panic', 0.0)
        self.panic_text.setText(f"Panic: Avg {avg_panic:.2f} | Max {max_panic:.2f}")
        
        # Color-code panic
        if max_panic > 0.7:
            self.panic_text.setFg((1, 0, 0, 1))  # Red
        elif max_panic > 0.3:
            self.panic_text.setFg((1, 0.5, 0, 1))  # Orange
        else:
            self.panic_text.setFg((0, 1, 0, 1))  # Green
        
        open_gates = info.get('open_gates', 0)
        self.gate_status_text.setText(f"Gates: {open_gates}/{self.env.NUM_GATES} Open")
        
        exited = info.get('total_exited', 0)
        self.throughput_text.setText(f"Exited: {exited}")
        
        # Scenario info (NOVEL)
        pattern = info.get('pattern', 'N/A')
        difficulty = info.get('difficulty', 'N/A')
        self.scenario_text.setText(f"Pattern: {pattern.upper()} | Difficulty: {difficulty.upper()}")
    
    def _toggle_heat_map(self):
        """Toggle heat map visualization"""
        self.show_heat_map = not self.show_heat_map
        print(f"Heat map: {'ON' if self.show_heat_map else 'OFF'}")
    
    def _toggle_rotation(self):
        """Toggle automatic camera rotation"""
        self.auto_rotate = not self.auto_rotate
        print(f"Auto-rotation: {'ON' if self.auto_rotate else 'OFF'}")
    
    def _move_camera(self, dx, dy, dz):
        """Move camera by offset"""
        pos = self.camera.getPos()
        self.camera.setPos(pos.x + dx, pos.y + dy, pos.z + dz)
    
    def _update_task(self, task):
        """Update loop for animations"""
        if self.auto_rotate:
            # Rotate camera around center at elevated angle
            angle = task.time * self.rotation_speed
            radius = 30
            height = 25  # Elevated view
            self.camera.setPos(
                10 + radius * np.cos(np.radians(angle)),
                10 + radius * np.sin(np.radians(angle)),
                height
            )
            self.camera.lookAt(10, 10, 2)  # Look at center, slightly elevated
        
        return Task.cont
    
    def close(self):
        """Clean up and close renderer"""
        self.destroy()

# Adding Custom 3D Models to Enhanced Crowd Control Visualization

## Current Setup

The visualization currently uses **procedural geometry** (boxes created from code) to avoid dependencies on external model files. This ensures the demo works out of the box without requiring any assets.

## Why Procedural Geometry?

The original code tried to load `"models/box"` which doesn't exist in Panda3D by default, causing a black screen. Procedural geometry creates 3D shapes programmatically, ensuring the demo always works.

## Adding Custom Models (Future Enhancement)

When you want to use realistic 3D models, follow these steps:

### 1. Prepare Your Models

**Supported Formats:**

- `.egg` (Panda3D's native format)
- `.bam` (Panda3D's binary format)
- `.obj`, `.gltf`, `.fbx` (can be converted)

**Tools:**

- **Blender** (free) - Export as `.egg` using YABEE plugin
- **Maya** - Export as `.egg` using maya2egg
- **egg2bam** - Convert `.egg` to `.bam` for faster loading

### 2. Create Assets Folder

```powershell
mkdir assets
mkdir assets\models
mkdir assets\textures
```

### 3. Export Models from Blender

1. Create your model (person, gate, barrier, etc.)
2. Install YABEE plugin: https://github.com/09th/YABEE
3. File → Export → Panda3D (.egg)
4. Save to `assets/models/`

**Example exports:**

- `assets/models/person.egg` - Individual agent
- `assets/models/gate.egg` - Exit gate
- `assets/models/barrier.egg` - Crowd control barrier
- `assets/models/ground.egg` - Floor with texture

### 4. Modify enhanced_rendering.py

#### Option A: Use the Helper Method (Recommended)

The code already includes `_load_model_or_fallback()` method:

```python
# In _initialize_scene_objects(), replace procedural boxes with models:

# For agents
for i in range(self.max_agent_nodes):
    agent_node = self._load_model_or_fallback(
        "assets/models/person.egg",  # Try to load model
        0.3, 0.3, 0.6  # Fallback size if model not found
    )
    agent_node.setColor(0.2, 0.5, 0.8, 0.9)
    agent_node.reparentTo(self.render)
    agent_node.hide()
    self.agent_nodes.append(agent_node)

# For gates
for _ in range(self.env.NUM_GATES):
    gate = self._load_model_or_fallback(
        "assets/models/gate.egg",
        1.0, 1.0, 0.3
    )
    gate.setColor(0.2, 0.8, 0.2, 0.8)
    gate.reparentTo(self.render)
    self.gate_nodes.append(gate)

# For barriers
for _ in range(self.env.NUM_BARRIERS):
    barrier = self._load_model_or_fallback(
        "assets/models/barrier.egg",
        0.8, 0.8, 1.5
    )
    barrier.setColor(0.9, 0.6, 0.1, 0.9)
    barrier.reparentTo(self.render)
    self.barrier_nodes.append(barrier)
```

**Benefit:** If models are missing, it automatically falls back to procedural geometry.

#### Option B: Direct Model Loading

```python
# Load model directly (will crash if file missing)
agent_node = self.loader.loadModel("assets/models/person.egg")
agent_node.setScale(0.3, 0.3, 0.6)
```

### 5. Add Textures

```python
# In your Blender model, UV unwrap and assign textures
# Or apply textures in code:

from panda3d.core import Texture

tex = self.loader.loadTexture("assets/textures/person_skin.png")
agent_node.setTexture(tex)
```

### 6. Animation Support

If you export animated models:

```python
# Load animated model
person = self.loader.loadModel("assets/models/person_animated.egg")
person.setPlayRate(1.0, 'walk')  # Play walk animation

# Control animation based on agent state
if agent.panic_level > 0.7:
    person.setPlayRate(2.0, 'run')  # Run animation when panicking
else:
    person.setPlayRate(1.0, 'walk')  # Walk normally
```

### 7. Different Models for Different Panic Levels

```python
# In update_scene(), swap models based on panic:
panic = agent.panic_level

if panic > 0.7:
    model = self._load_model_or_fallback("assets/models/person_running.egg", 0.3, 0.3, 0.6)
elif panic > 0.3:
    model = self._load_model_or_fallback("assets/models/person_stressed.egg", 0.3, 0.3, 0.6)
else:
    model = self._load_model_or_fallback("assets/models/person_walking.egg", 0.3, 0.3, 0.6)
```

## Camera Position

The camera has been fixed to start in a better position:

```python
def _setup_camera(self):
    # Position: X=10 (center), Y=-30 (back), Z=25 (elevated)
    self.camera.setPos(10, -30, 25)
    self.camera.lookAt(10, 10, 2)  # Look at center
```

**Controls:**

- Arrow keys: Move camera
- `R`: Toggle auto-rotation (circular bird's-eye view)
- `H`: Toggle heat map (future feature)

## Example: Complete Model Integration

Here's a full example of replacing all geometry with custom models:

```python
# In enhanced_rendering.py

def _initialize_scene_objects(self):
    """Initialize all scene objects with custom models"""

    # Ground with texture
    ground = self._load_model_or_fallback(
        "assets/models/venue_floor.egg",
        self.grid_width, self.grid_height, 0.1
    )
    ground.setPos(self.grid_width/2, self.grid_height/2, 0)
    ground.reparentTo(self.render)

    # Walls
    for pos, rotation in [
        ((self.grid_width/2, 0, 1.5), (0, 0, 0)),  # Top
        ((self.grid_width/2, self.grid_height, 1.5), (0, 0, 180)),  # Bottom
        ((0, self.grid_height/2, 1.5), (0, 0, 90)),  # Left
        ((self.grid_width, self.grid_height/2, 1.5), (0, 0, -90))  # Right
    ]:
        wall = self._load_model_or_fallback(
            "assets/models/wall_section.egg",
            self.grid_width, 0.3, 3.0
        )
        wall.setPos(*pos)
        wall.setHpr(*rotation)
        wall.reparentTo(self.render)

    # Agent pool with variations
    self.agent_models = {
        'calm': self.loader.loadModel("assets/models/person_calm.egg"),
        'stressed': self.loader.loadModel("assets/models/person_stressed.egg"),
        'panic': self.loader.loadModel("assets/models/person_panic.egg")
    }

    for i in range(self.max_agent_nodes):
        agent = self.agent_models['calm'].copyTo(self.render)
        agent.setScale(0.3, 0.3, 0.6)
        agent.hide()
        self.agent_nodes.append(agent)

    # Gates (different models for open/closed)
    self.gate_models = {
        'open': self.loader.loadModel("assets/models/gate_open.egg"),
        'closed': self.loader.loadModel("assets/models/gate_closed.egg")
    }

    for _ in range(self.env.NUM_GATES):
        gate = self.gate_models['open'].copyTo(self.render)
        self.gate_nodes.append(gate)

    # Barriers
    for _ in range(self.env.NUM_BARRIERS):
        barrier = self._load_model_or_fallback(
            "assets/models/crowd_barrier.egg",
            0.8, 0.8, 1.5
        )
        barrier.reparentTo(self.render)
        self.barrier_nodes.append(barrier)
```

## Blender Model Guidelines

**For Agents (people):**

- Size: ~1.7 units tall (realistic human height)
- Rig: Simple skeleton for walking/running animations
- Variations: Create 3 versions (calm, stressed, panicking)
- Colors: Blue tints for calm, orange for stressed, red for panic

**For Gates:**

- Size: 2-3 units wide
- Two versions: open and closed
- Add lights or signs for extra detail

**For Barriers:**

- Size: 1-1.5 units tall
- Metal texture with caution stripes
- Modular design for stacking

**For Floor:**

- Size: Matches grid (20x20 units)
- Texture: Concrete, tiles, or venue floor
- Consider zones (entrances, exits, danger areas)

## Performance Considerations

**Model Complexity:**

- Keep polygon count low (<1000 per agent)
- Use level of detail (LOD) for distant objects
- Instance models rather than duplicating

**Optimization:**

```python
# Use instancing for repeated models
person_model = self.loader.loadModel("assets/models/person.egg")

for i in range(200):
    instance = person_model.instanceTo(self.render)  # Faster than copyTo
    self.agent_nodes.append(instance)
```

## Debugging Models

If models don't appear:

```python
# Check if model loaded
print(f"Model loaded: {person_model}")
print(f"Model bounds: {person_model.getBounds()}")
print(f"Model position: {person_model.getPos()}")

# Make model visible
person_model.show()
person_model.setColorScale(1, 1, 1, 1)  # Ensure not transparent
person_model.ls()  # List model hierarchy
```

## Resources

- **Panda3D Manual**: https://docs.panda3d.org/
- **YABEE (Blender Exporter)**: https://github.com/09th/YABEE
- **Free 3D Models**:
  - Sketchfab (many free CC models)
  - Mixamo (animated characters)
  - OpenGameArt.org
- **Panda3D Community**: https://discourse.panda3d.org/

## Summary

- ✅ **Current**: Procedural geometry (always works)
- 🎨 **Future**: Custom 3D models (optional, for realism)
- 🔄 **Hybrid**: Use `_load_model_or_fallback()` for best of both
- 📹 **Camera**: Fixed to start in good position (elevated bird's-eye view)
- ⚡ **Performance**: Use instancing for many agents

The system is designed to work immediately while allowing easy upgrades to realistic visuals when needed!

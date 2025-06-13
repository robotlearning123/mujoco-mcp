# Motion Control Demos for MuJoCo MCP

This directory contains motion control demonstrations showing how to control various robots through the MuJoCo MCP interface.

## Available Demos

### 1. `motion_control_demo.py` - Direct Control Demo
A standalone demo that directly controls robots via the viewer server connection.

**Features:**
- Load various robot models from MuJoCo Menagerie
- Predefined motion patterns (wave, walk, grasp, etc.)
- Interactive control mode
- Support for multiple robot types:
  - **Robotic Arms**: Franka Panda, UR5e, Kuka iiwa
  - **Quadrupeds**: Anymal C, Unitree Go2, Spot
  - **Humanoids**: Unitree G1, H1
  - **Grippers**: Robotiq, Shadow Hand

**Usage:**
```bash
# Make sure viewer server is running
python mujoco_viewer_server.py

# In another terminal, run the demo
python examples/motion_control_demo.py
```

### 2. `mcp_motion_control.py` - MCP Interface Demo
Demonstrates control through the MCP protocol interface.

**Features:**
- Natural language control
- MCP tool integration
- Demo sequences for basic scenes
- Compatible with Claude Desktop

**Usage:**
```bash
# Run directly
python examples/mcp_motion_control.py

# Or use through Claude Desktop with MCP configured
```

## Prerequisites

### 1. MuJoCo Menagerie
To use robot models from Menagerie:

```bash
# Clone MuJoCo Menagerie
git clone https://github.com/google-deepmind/mujoco_menagerie.git ~/mujoco_menagerie

# Or set environment variable to your installation
export MUJOCO_MENAGERIE_PATH=/path/to/mujoco_menagerie
```

### 2. Viewer Server
Always start the viewer server first:

```bash
python mujoco_viewer_server.py
```

## Motion Control Examples

### Basic Pendulum Control
```python
# Create pendulum
await handle_call_tool("create_scene", {"scene_type": "pendulum"})

# Set position
await handle_call_tool("set_joint_positions", {"positions": [1.57]})  # 90 degrees

# Step simulation
await handle_call_tool("step_simulation", {"steps": 100})
```

### Robot Arm Control
```python
demo = MotionControlDemo()
demo.load_model("franka_panda")
demo.go_home()  # Move to home position
demo.wave_motion(duration=5.0)  # Wave for 5 seconds
demo.circle_motion(radius=0.2, duration=10.0)  # Circular motion
```

### Natural Language Control
Through MCP interface:
- "Create a pendulum simulation"
- "Load Franka Panda robot"
- "Move the robot arm in a circle"
- "Make the robot wave"

## Supported Robot Models

| Model | Type | Joints | Demo Motions |
|-------|------|---------|--------------|
| franka_panda | Arm | 7 | wave, pick_place, circle |
| ur5e | Arm | 6 | wave, pick_place, spiral |
| kuka_iiwa | Arm | 7 | wave, figure8, reach |
| anymal_c | Quadruped | 12 | stand, walk, trot |
| go2 | Quadruped | 12 | stand, walk, jump |
| spot | Quadruped | 12 | stand, walk, dance |
| g1 | Humanoid | 37 | stand, wave_hand, walk |
| h1 | Humanoid | 25 | stand, balance, squat |
| shadow_hand | Hand | 24 | open, close, wave, grasp |

## Troubleshooting

1. **"Failed to connect to viewer server"**
   - Make sure `mujoco_viewer_server.py` is running
   - Check if port 8888 is available

2. **"MuJoCo Menagerie not found"**
   - Install MuJoCo Menagerie as shown above
   - Set `MUJOCO_MENAGERIE_PATH` environment variable

3. **"Model not found"**
   - Check if the model exists in your Menagerie installation
   - Verify the path in `robot_configs` dictionary

## Extending the Demos

To add new robot models:

1. Add configuration to `robot_configs` in `motion_control_demo.py`:
```python
"new_robot": {
    "path": "manufacturer_model/scene.xml",
    "type": "arm",  # or quadruped, humanoid, etc.
    "joints": 6,
    "home_position": [0, 0, 0, 0, 0, 0],
    "demo_motions": ["custom1", "custom2"]
}
```

2. Implement custom motion patterns:
```python
def custom_motion(self, duration: float = 5.0):
    """Custom motion pattern"""
    # Your motion control code here
```

3. Add to demo_map in `run_demo()` method.

## Advanced Features

- **Trajectory Planning**: Implement smooth trajectories between positions
- **Force Control**: Use torque control for compliant motions
- **Multi-Robot**: Control multiple robots simultaneously
- **Sensor Feedback**: Use state feedback for closed-loop control
# MuJoCo MCP API Reference

## Complete API Documentation for MuJoCo MCP v0.8.2

This document provides comprehensive API documentation for all MCP tools, Python modules, and interfaces.

---

## 📋 Table of Contents

1. [MCP Tools API](#mcp-tools-api)
2. [Python Modules API](#python-modules-api)
3. [Viewer Server API](#viewer-server-api)
4. [Advanced Controllers API](#advanced-controllers-api)
5. [Multi-Robot Coordination API](#multi-robot-coordination-api)
6. [Sensor Feedback API](#sensor-feedback-api)
7. [RL Integration API](#rl-integration-api)
8. [Visualization Tools API](#visualization-tools-api)

---

## 🔧 MCP Tools API

### Tool: `get_server_info`
Get information about the MuJoCo MCP server.

**Parameters:** None

**Returns:**
```json
{
  "name": "MuJoCo MCP Server",
  "version": "0.8.2",
  "description": "Control MuJoCo physics simulations through MCP",
  "status": "ready",
  "capabilities": ["create_scene", "step_simulation", "get_state", "reset", "close_viewer"]
}
```

---

### Tool: `create_scene`
Create a physics simulation scene.

**Parameters:**
- `scene_type` (string, required): Type of scene to create
  - Options: `"pendulum"`, `"double_pendulum"`, `"cart_pole"`, `"arm"`
- `parameters` (object, optional): Scene-specific parameters
  - For pendulum: `{"length": float, "mass": float, "damping": float}`
  - For double_pendulum: `{"length1": float, "length2": float, "mass1": float, "mass2": float}`
  - For cart_pole: `{"cart_mass": float, "pole_mass": float, "pole_length": float}`

**Returns:**
```json
{
  "success": true,
  "model_id": "pendulum_12345",
  "message": "Created pendulum scene successfully! Viewer window opened."
}
```

**Example:**
```json
{
  "tool": "create_scene",
  "arguments": {
    "scene_type": "pendulum",
    "parameters": {
      "length": 0.6,
      "mass": 0.5
    }
  }
}
```

---

### Tool: `step_simulation`
Advance the physics simulation forward.

**Parameters:**
- `model_id` (string, required): ID of the model to step
- `steps` (integer, optional): Number of simulation steps (default: 1)

**Returns:**
```json
{
  "success": true,
  "message": "Stepped simulation 100 steps"
}
```

---

### Tool: `get_state`
Get current state of the simulation.

**Parameters:**
- `model_id` (string, required): ID of the model to get state from

**Returns:**
```json
{
  "success": true,
  "state": {
    "time": 2.5,
    "qpos": [0.785, -0.123],  // Joint positions
    "qvel": [0.1, -0.05],      // Joint velocities
    "qacc": [0.01, -0.02],     // Joint accelerations
    "ctrl": [0.0, 0.0],        // Control signals
    "xpos": [[0, 0, 1], [0.5, 0, 0.5]]  // Body positions
  }
}
```

---

### Tool: `set_joint_positions`
Set joint positions for the robot.

**Parameters:**
- `model_id` (string, required): ID of the model
- `positions` (array, required): Array of joint positions in radians

**Returns:**
```json
{
  "success": true,
  "message": "Joint positions set successfully"
}
```

**Example:**
```json
{
  "tool": "set_joint_positions",
  "arguments": {
    "model_id": "franka_panda_123",
    "positions": [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
  }
}
```

---

### Tool: `reset_simulation`
Reset simulation to initial state.

**Parameters:**
- `model_id` (string, required): ID of the model to reset

**Returns:**
```json
{
  "success": true,
  "message": "Simulation reset to initial state"
}
```

---

### Tool: `execute_command`
Execute natural language commands.

**Parameters:**
- `command` (string, required): Natural language command
- `model_id` (string, optional): Target model ID

**Returns:**
```json
{
  "success": true,
  "result": "Command executed successfully",
  "actions_taken": ["created_scene", "set_position", "stepped_simulation"]
}
```

**Examples:**
```
"Create a pendulum simulation"
"Set the pendulum angle to 45 degrees"
"Load Franka Panda robot"
"Move the robot arm in a circle"
```

---

### Tool: `get_loaded_models`
List all active simulations.

**Parameters:** None

**Returns:**
```json
{
  "success": true,
  "models": {
    "pendulum_123": {
      "type": "pendulum",
      "created_time": 1623456789.0,
      "status": "running"
    },
    "franka_456": {
      "type": "robotic_arm",
      "created_time": 1623456890.0,
      "status": "paused"
    }
  }
}
```

---

### Tool: `close_viewer`
Close the MuJoCo GUI window.

**Parameters:**
- `model_id` (string, required): ID of the model viewer to close

**Returns:**
```json
{
  "success": true,
  "message": "Viewer closed"
}
```

---

## 🐍 Python Modules API

### Module: `mujoco_mcp.advanced_controllers`

#### Class: `PIDController`
Proportional-Integral-Derivative controller.

```python
from mujoco_mcp.advanced_controllers import PIDController, PIDConfig

# Create controller
config = PIDConfig(
    kp=1.0,          # Proportional gain
    ki=0.1,          # Integral gain  
    kd=0.05,         # Derivative gain
    max_output=100.0,  # Maximum output
    min_output=-100.0, # Minimum output
    windup_limit=100.0 # Anti-windup limit
)
controller = PIDController(config)

# Update control
error = target - current
output = controller.update(target, current, dt=0.02)

# Reset controller
controller.reset()
```

#### Class: `TrajectoryPlanner`
Advanced trajectory planning utilities.

```python
from mujoco_mcp.advanced_controllers import TrajectoryPlanner

planner = TrajectoryPlanner()

# Minimum jerk trajectory
positions, velocities, accelerations = planner.minimum_jerk_trajectory(
    start_pos=np.array([0, 0, 0]),
    end_pos=np.array([1, 0.5, 0.3]),
    duration=5.0,
    start_vel=None,  # Optional
    end_vel=None,    # Optional
    frequency=100.0  # Hz
)

# Spline trajectory through waypoints
waypoints = np.array([[0, 0], [1, 1], [2, 0], [3, 1]])
times = np.array([0, 1, 2, 3])
positions, velocities, accelerations = planner.spline_trajectory(
    waypoints, times, frequency=100.0
)
```

#### Class: `RobotController`
High-level robot controller combining multiple strategies.

```python
from mujoco_mcp.advanced_controllers import create_arm_controller

# Create pre-configured controller
controller = create_arm_controller("franka_panda")

# Set trajectory
controller.set_trajectory(waypoints, times)

# Get trajectory command
command = controller.get_trajectory_command(current_time)

# PID control
torques = controller.pid_control(target_positions, current_positions)

# Impedance control
forces = controller.impedance_control(
    current_pos, target_pos, current_vel,
    stiffness=np.array([100, 100, 100]),
    damping=np.array([10, 10, 10])
)
```

---

### Module: `mujoco_mcp.multi_robot_coordinator`

#### Class: `MultiRobotCoordinator`
Coordinate multiple robots for complex tasks.

```python
from mujoco_mcp.multi_robot_coordinator import MultiRobotCoordinator

coordinator = MultiRobotCoordinator()

# Add robots
coordinator.add_robot(
    robot_id="arm1",
    robot_type="franka_panda",
    capabilities={"manipulation": True, "mobility": False}
)

# Start coordination
coordinator.start_coordination()

# Cooperative manipulation
task_id = coordinator.cooperative_manipulation(
    robots=["arm1", "arm2"],
    target_object="assembly_part",
    approach_positions={
        "arm1": np.array([0.4, 0.2, 0.3]),
        "arm2": np.array([0.4, -0.2, 0.3])
    }
)

# Formation control
task_id = coordinator.formation_control(
    robots=["robot1", "robot2", "robot3"],
    formation_type="line",  # or "circle", "triangle"
    spacing=1.5
)

# Check task status
status = coordinator.get_task_status(task_id)

# Get robot status
robot_info = coordinator.get_robot_status("arm1")

# Stop coordination
coordinator.stop_coordination()
```

---

### Module: `mujoco_mcp.sensor_feedback`

#### Class: `SensorManager`
Manage multiple sensors and data collection.

```python
from mujoco_mcp.sensor_feedback import create_robot_sensor_suite

# Create sensor suite
sensor_manager = create_robot_sensor_suite("franka_panda", n_joints=7)

# Start sensing
sensor_manager.start_sensing()

# Get latest readings
readings = sensor_manager.get_latest_readings()

# Calibrate sensor
sensor_manager.calibrate_sensor("joint_positions", {
    "offset": [0.01, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0],
    "scale": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
})

# Stop sensing
sensor_manager.stop_sensing()
```

#### Class: `ClosedLoopController`
Closed-loop control with sensor feedback.

```python
from mujoco_mcp.sensor_feedback import create_feedback_controller

controller = create_feedback_controller("franka_panda")

# Set target state
controller.set_target({
    "joint_position": np.array([0, -0.785, 0, -1.57, 0, 1.57, 0]),
    "joint_velocity": np.zeros(7)
})

# Update from sensors
controller.update_state(sensor_readings)

# Compute control
commands = controller.compute_control()
```

---

### Module: `mujoco_mcp.rl_integration`

#### Function: `create_reaching_env`
Create reinforcement learning environment for reaching tasks.

```python
from mujoco_mcp.rl_integration import create_reaching_env, RLTrainer

# Create environment
env = create_reaching_env("franka_panda")

# Create trainer
trainer = RLTrainer(env)

# Run baseline
baseline_results = trainer.random_policy_baseline(num_episodes=10)

# Evaluate custom policy
def my_policy(observation):
    # Your policy logic
    return action

results = trainer.evaluate_policy(my_policy, num_episodes=10)

# Save training data
trainer.save_training_data("training_results.json")
```

#### Class: `MuJoCoRLEnvironment`
Gymnasium-compatible RL environment.

```python
from mujoco_mcp.rl_integration import MuJoCoRLEnvironment, RLConfig

config = RLConfig(
    robot_type="franka_panda",
    task_type="reaching",
    max_episode_steps=1000,
    reward_scale=1.0,
    action_space_type="continuous",
    observation_space_size=20,
    action_space_size=7
)

env = MuJoCoRLEnvironment(config)

# Standard Gymnasium interface
obs, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

---

### Module: `mujoco_mcp.visualization_tools`

#### Class: `RobotStateMonitor`
Monitor and visualize robot state in real-time.

```python
from mujoco_mcp.visualization_tools import RobotStateMonitor

monitor = RobotStateMonitor(viewer_client)

# Start monitoring
monitor.start_monitoring("robot_model", update_rate=50.0)

# Show visualizations
monitor.show_joint_positions()
monitor.show_dashboard()

# Export data
monitor.export_data("robot_data.json")  # or .npz

# Analyze performance
analysis = monitor.analyze_performance()

# Stop monitoring
monitor.stop_monitoring()
```

#### Class: `TrajectoryVisualizer`
Visualize robot trajectories in 3D.

```python
from mujoco_mcp.visualization_tools import TrajectoryVisualizer

viz = TrajectoryVisualizer()

# Add trajectories
viz.add_trajectory("planned", planned_positions, timestamps)
viz.add_trajectory("actual", actual_positions)

# Create 3D plot
fig = viz.create_3d_plot()
viz.show_plot()

# Animate trajectory
viz.animate_trajectory("planned", speed=1.0)
```

---

## 🖥️ Viewer Server API

### Enhanced Viewer Server Commands

The viewer server accepts JSON commands over socket (port 8888).

#### Command: `ping`
```json
{"type": "ping"}
```

Response:
```json
{
  "success": true,
  "pong": true,
  "server_info": {
    "version": "0.8.2-enhanced",
    "models_loaded": 2,
    "active_connections": 5,
    "cpu_usage": 15.2,
    "memory_usage": 256.8
  }
}
```

#### Command: `load_model`
```json
{
  "type": "load_model",
  "model_id": "my_robot",
  "model_xml": "<mujoco>...</mujoco>"  // or file path
}
```

#### Command: `get_state`
```json
{
  "type": "get_state",
  "model_id": "my_robot"
}
```

#### Command: `set_joint_positions`
```json
{
  "type": "set_joint_positions",
  "model_id": "my_robot",
  "positions": [0.0, 0.785, 0.0, -1.57, 0.0]
}
```

#### Command: `get_diagnostics`
```json
{"type": "get_diagnostics"}
```

Response includes comprehensive server health metrics.

---

## 📊 Error Codes

### Standard MCP Error Codes
- `-32700`: Parse error
- `-32600`: Invalid request
- `-32601`: Method not found
- `-32602`: Invalid params
- `-32603`: Internal error

### Custom Error Codes
- `-32001`: Viewer server not connected
- `-32002`: Model not found
- `-32003`: Invalid model configuration
- `-32004`: Simulation error
- `-32005`: Resource limit exceeded

---

## 🔌 Integration Examples

### Complete System Integration
```python
import asyncio
from mujoco_mcp.multi_robot_coordinator import MultiRobotCoordinator
from mujoco_mcp.visualization_tools import RobotStateMonitor

async def main():
    # Initialize system
    coordinator = MultiRobotCoordinator()
    monitor = RobotStateMonitor(coordinator.viewer_client)
    
    # Add robots
    coordinator.add_robot("arm1", "franka_panda", {"manipulation": True})
    coordinator.add_robot("arm2", "ur5e", {"manipulation": True})
    
    # Start systems
    coordinator.start_coordination()
    monitor.start_monitoring("arm1")
    
    # Execute cooperative task
    task_id = coordinator.cooperative_manipulation(
        robots=["arm1", "arm2"],
        target_object="box",
        approach_positions={
            "arm1": np.array([0.3, 0.1, 0.5]),
            "arm2": np.array([0.3, -0.1, 0.5])
        }
    )
    
    # Monitor execution
    while True:
        status = coordinator.get_task_status(task_id)
        if status in ["completed", "failed"]:
            break
        await asyncio.sleep(0.1)
    
    # Export results
    monitor.export_data("task_results.json")
    analysis = monitor.analyze_performance()
    
    # Cleanup
    coordinator.stop_coordination()
    monitor.stop_monitoring()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 📚 Additional Resources

- [MCP Specification](https://modelcontextprotocol.io/)
- [MuJoCo Python API](https://mujoco.readthedocs.io/en/stable/python.html)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

For more examples and tutorials, see the [examples/](examples/) directory.
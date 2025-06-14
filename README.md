# MuJoCo MCP - Enterprise Robotics Simulation Platform

[![Version](https://img.shields.io/badge/version-0.8.2-blue.svg)](CHANGELOG.md)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-2.3.0%2B-green.svg)](https://github.com/google-deepmind/mujoco)
[![MCP](https://img.shields.io/badge/MCP-2024--11--05-purple.svg)](https://modelcontextprotocol.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

🤖 Advanced robotics simulation platform that enables AI assistants to control complex physics simulations through natural language. Built on MuJoCo physics engine and Model Context Protocol for seamless integration with Claude Desktop and other MCP clients.

🚀 **[Quick Start](#quick-start)** | 📚 **[Documentation](DOCUMENTATION_INDEX.md)** | 🏗️ **[Architecture](ARCHITECTURE.md)** | 🔧 **[API Reference](API_REFERENCE.md)** | 🎯 **[Advanced Features](ADVANCED_FEATURES_GUIDE.md)**

## 🌟 Features

### Core Capabilities
- **Natural Language Control**: Control robots using plain English commands
- **Real-time Visualization**: Native MuJoCo viewer with interactive GUI
- **MCP Standard Compliance**: Full Model Context Protocol implementation
- **Cross-Platform Support**: Works on macOS, Linux, and Windows

### Advanced Features (v0.8.2)
- **🎛️ Advanced Control Algorithms**: PID, trajectory planning, optimization control
- **🤖 Multi-Robot Coordination**: Formation control, cooperative manipulation
- **🔬 Sensor Feedback Systems**: Closed-loop control with multi-modal sensors
- **🧠 RL Integration**: Gymnasium-compatible reinforcement learning environments
- **📊 Physics Benchmarking**: Performance, accuracy, and scalability testing
- **📈 Real-time Monitoring**: Advanced visualization and analytics tools
- **🚀 Production Ready**: Enhanced server with connection pooling and diagnostics

## Quick Start

### 1. Install Dependencies

```bash
pip install mujoco mcp numpy
```

### 2. Install MuJoCo MCP

```bash
pip install -e .
```

### 3. Start the Viewer Server

```bash
python mujoco_viewer_server.py
```

### 4. Configure Claude Desktop

Add to your Claude Desktop config:

```json
{
  "mcpServers": {
    "mujoco-mcp": {
      "command": "python",
      "args": ["-m", "mujoco_mcp"],
      "env": {
        "PYTHONPATH": "./src"
      }
    }
  }
}
```

### 5. Start Using Natural Language Commands

In Claude Desktop:
```
"Create a pendulum simulation"
"Set the pendulum angle to 45 degrees"
"Step the simulation 100 times"
"Show me the current state"
```

## 📝 Example Usage

### Basic Physics Simulations
```
# Simple pendulum
"Create a pendulum simulation"
"Set the pendulum to 90 degrees and let it swing"

# Double pendulum (chaotic motion)
"Create a double pendulum"
"Give it a small push and watch the chaos"

# Cart-pole balancing
"Create a cart pole simulation"
"Try to balance the pole"
```

### Advanced Robot Control
```
# Load robot from MuJoCo Menagerie
"Load a Franka Panda robot"
"Move the robot arm in a circle"
"Set all joints to home position"

# Multi-robot coordination
"Create two robot arms side by side"
"Make them work together to lift a box"

# Walking robots
"Load the Unitree Go2 quadruped"
"Make it walk forward"
```

### Reinforcement Learning
```python
from mujoco_mcp.rl_integration import create_reaching_env

# Create RL environment
env = create_reaching_env("franka_panda")

# Train your agent
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()  # Your policy here
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

## 🛠️ MCP Tools Available

| Tool | Description | Example |
|------|-------------|---------|
| `get_server_info` | Get server status | Returns version, capabilities |
| `create_scene` | Create physics simulation | `{"scene_type": "pendulum"}` |
| `step_simulation` | Advance simulation | `{"steps": 100}` |
| `get_state` | Get current state | Returns positions, velocities |
| `set_joint_positions` | Control joints | `{"positions": [0, 0.785, 0]}` |
| `reset_simulation` | Reset to initial | Resets physics state |
| `execute_command` | Natural language | `{"command": "move arm up"}` |
| `get_loaded_models` | List active models | Returns all loaded models |
| `close_viewer` | Close GUI window | Closes visualization |

## 🚀 Advanced Setup

### Install MuJoCo Menagerie (for robot models)
```bash
git clone https://github.com/google-deepmind/mujoco_menagerie.git ~/mujoco_menagerie
export MUJOCO_MENAGERIE_PATH=~/mujoco_menagerie
```

### Use Enhanced Production Server
```bash
# For better performance and reliability
/opt/miniconda3/bin/mjpython mujoco_viewer_server_enhanced.py --port 8888
```

### Run Comprehensive Tests
```bash
# Test basic functionality
python scripts/quick_internal_test.py

# Test advanced features
python test_advanced_features.py

# Run benchmarks
python benchmarks/physics_benchmarks.py
```

## 📚 Documentation

- **[Documentation Index](DOCUMENTATION_INDEX.md)** - Complete guide to all docs
- **[Architecture Guide](ARCHITECTURE.md)** - System design and components
- **[API Reference](API_REFERENCE.md)** - Complete API documentation
- **[Advanced Features](ADVANCED_FEATURES_GUIDE.md)** - Controllers, RL, multi-robot
- **[Motion Control Examples](examples/README_MOTION_CONTROL.md)** - Robot demos
- **[Testing Summary](TESTING_SUMMARY.md)** - Test coverage and results
- **[Changelog](CHANGELOG.md)** - Version history

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 🐛 Troubleshooting

### Common Issues

1. **"Failed to connect to viewer server"**
   - Make sure `mujoco_viewer_server.py` is running
   - Check port 8888 is available
   - On macOS, use `/opt/miniconda3/bin/mjpython`

2. **"Model not found"**
   - Install MuJoCo Menagerie for robot models
   - Check file paths in configurations

3. **Performance issues**
   - Use the enhanced viewer server
   - Enable connection pooling
   - Check system resources

For more help, see the [Documentation Index](DOCUMENTATION_INDEX.md).

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [MuJoCo](https://mujoco.org/) physics engine by Google DeepMind
- [Model Context Protocol](https://modelcontextprotocol.io/) by Anthropic
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) for robot models

---

Built with ❤️ for the robotics and AI community

# MuJoCo MCP - Model Context Protocol for MuJoCo

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-2.3.0%2B-green.svg)](https://github.com/google-deepmind/mujoco)
[![MCP](https://img.shields.io/badge/MCP-2024--11--05-blue.svg)](https://modelcontextprotocol.io)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.7.0-orange.svg)](CHANGELOG.md)

MuJoCo MCP enables AI agents to control MuJoCo physics simulations through the Model Context Protocol. It provides a bridge between language models (like Claude) and the MuJoCo physics engine, allowing natural language control of robotic simulations.

## 🌟 Key Features

### v0.7.0 Highlights
- **🚀 Enhanced Socket Architecture**: Multi-threaded server supporting concurrent connections
- **📡 Remote Viewer Mode**: External MuJoCo GUI process with real-time visualization
- **🔌 Robust Communication**: Dynamic buffer management for large models
- **🛡️ Improved Stability**: Better error handling and connection recovery
- **🤖 Multi-Model Support**: Architecture ready for multiple simultaneous simulations

### Core Capabilities
- **Natural Language Control**: Control simulations using plain English commands
- **Real-time Visualization**: Native MuJoCo viewer with full GUI capabilities
- **MCP Standard Compliance**: Full implementation of Model Context Protocol
- **Cross-Platform**: Works on macOS, Linux, and Windows

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** with pip
2. **MuJoCo** (automatically installed with pip)
3. **Claude Desktop** or any MCP-compatible client

### Installation

```bash
# Clone the repository
git clone https://github.com/robotlearning123/mujoco-mcp.git
cd mujoco-mcp

# Install in development mode
pip install -e .
```

### Two-Step Usage

#### Step 1: Start the MuJoCo Viewer Server

```bash
# On macOS (required):
mjpython mujoco_viewer_server.py

# On Linux/Windows:
python mujoco_viewer_server.py
```

This starts the viewer server on `localhost:8888`, ready to display simulations.

#### Step 2: Configure Claude Desktop

Add to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "mujoco-mcp": {
      "command": "python",
      "args": ["path/to/mcp_server_remote.py"],
      "cwd": "path/to/mujoco-mcp",
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

## 🎮 Usage Examples

### Basic Commands in Claude

```
Human: Create a pendulum simulation

Claude: I'll create a pendulum simulation for you.
[Creates pendulum and opens MuJoCo viewer window]

Human: Set the pendulum angle to 45 degrees

Claude: I'll set the pendulum to 45 degrees.
[Pendulum moves to the specified angle]

Human: Reset the simulation

Claude: I've reset the simulation to its initial state.
[Simulation resets]
```

## 🛠️ Available MCP Tools

### Core Tools
- **get_server_info** - Get server information and capabilities
- **create_scene** - Create physics scenes (pendulum, double_pendulum, cart_pole, robotic_arm)
- **step_simulation** - Advance simulation time
- **get_state** - Get current simulation state
- **set_joint_positions** - Control joint angles
- **reset_simulation** - Reset to initial state
- **execute_command** - Natural language command interface
- **get_loaded_models** - List active simulations

### Supported Scenes
1. **Single Pendulum** - Classic pendulum with damping
2. **Double Pendulum** - Chaotic double pendulum system
3. **Cart-Pole** - Classic control problem
4. **Robotic Arm** - Multi-joint robotic arm (coming soon)

## 🏗️ Architecture

```
Claude Desktop ←→ MCP Server ←→ Socket (8888) ←→ MuJoCo Viewer
     (MCP)       (Python)        (TCP/IP)         (mjpython)
```

The architecture follows the successful pattern used by Blender MCP and Figma MCP:
- External application control via sockets
- Process separation for stability
- Real-time bidirectional communication

## 🐛 Troubleshooting

### Common Issues

1. **"Failed to connect to viewer"**
   - Ensure `mujoco_viewer_server.py` is running
   - Check if port 8888 is available
   - On macOS, use `mjpython` instead of `python`

2. **"Another viewer is already open"**
   - MuJoCo limitation: one viewer per process
   - Restart the viewer server

3. **Connection timeouts**
   - Check firewall settings
   - Ensure localhost connections are allowed

### Logs
```bash
# MCP Server logs
tail -f ~/Library/Logs/Claude/mcp-server-mujoco-mcp.log

# Viewer server logs (in terminal where started)
```

## 📊 Performance

- **Connection Latency**: <100ms (local socket)
- **Command Response**: <50ms typical
- **Simulation Rate**: 60Hz+ (hardware dependent)
- **Buffer Size**: Dynamic, up to 1MB for complex models

## 🔮 Future Roadmap

### v0.8.0 (Planned)
- [ ] Multi-window support (process separation)
- [ ] Advanced physics constraints
- [ ] Contact force visualization
- [ ] Performance monitoring tools

### v0.9.0 (Planned)
- [ ] Cloud deployment support
- [ ] Collaborative simulations
- [ ] Custom MJCF model loading
- [ ] Integration with MuJoCo MPC

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup
```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest tests/

# Check code style
python -m flake8 src/
```

## 📚 Documentation

- [CHANGELOG.md](CHANGELOG.md) - Version history
- [CLAUDE.md](CLAUDE.md) - AI assistant instructions
- [CONFIG.md](CONFIG.md) - Configuration guide
- [API Documentation](docs/api.md) - Detailed API reference

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [MuJoCo](https://github.com/google-deepmind/mujoco) by Google DeepMind
- [Model Context Protocol](https://modelcontextprotocol.io) by Anthropic
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- Inspired by [Blender MCP](https://github.com/ahujasid/blender-mcp) architecture

---

Made with ❤️ for the robotics and AI community
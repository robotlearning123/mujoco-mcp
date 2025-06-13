# MuJoCo MCP - Model Context Protocol for MuJoCo

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-2.3.0%2B-green.svg)](https://github.com/google-deepmind/mujoco)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Control MuJoCo physics simulations through natural language using the Model Context Protocol.

## Features

- **Natural Language Control**: Control robots using plain English
- **Real-time Visualization**: Native MuJoCo viewer with GUI
- **MCP Standard**: Full Model Context Protocol implementation
- **Cross-Platform**: Works on macOS, Linux, and Windows

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

### 5. Use Natural Language Commands

In Claude Desktop:
- "Create a pendulum simulation"
- "Set the pendulum angle to 45 degrees"
- "Create a robotic arm scene"

## MCP Tools

- `get_server_info` - Get server information
- `create_scene` - Create a physics simulation
- `step_simulation` - Advance simulation time
- `get_state` - Get current simulation state
- `set_joint_positions` - Control joint angles
- `reset_simulation` - Reset to initial state
- `close_viewer` - Close viewer window

## Development

Run tests:

```bash
python scripts/quick_internal_test.py
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

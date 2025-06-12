# MuJoCo MCP Configuration Guide

This guide explains how to configure MuJoCo MCP for various AI development environments.

## Claude Desktop Configuration

### Method 1: Using claude_desktop_config.json

1. Copy the `claude_desktop_config.json` file to your Claude Desktop configuration directory:
   - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
   - **Linux**: `~/.config/Claude/claude_desktop_config.json`

2. Restart Claude Desktop

### Method 2: Manual Configuration

Add the following to your Claude Desktop settings:

```json
{
  "mcpServers": {
    "mujoco-mcp": {
      "command": "python",
      "args": ["-m", "mujoco_mcp.server"],
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

## Cursor Configuration

### Using .cursorrules

The `.cursorrules` file is already included in the project. Cursor will automatically detect and use it.

### Manual MCP Configuration

Add to your Cursor settings.json:

```json
{
  "mcp.servers": {
    "mujoco-mcp": {
      "command": "python",
      "args": ["-m", "mujoco_mcp.server"],
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

## VS Code Configuration

Add to your `.vscode/settings.json`:

```json
{
  "mcp.servers": {
    "mujoco-mcp": {
      "command": "python",
      "args": ["-m", "mujoco_mcp.server"],
      "cwd": "${workspaceFolder}",
      "env": {
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": "${workspaceFolder}/src"
      }
    }
  }
}
```

## Environment Variables

You can configure the MuJoCo MCP server using environment variables:

- `MUJOCO_MCP_HOST`: Server host (default: "localhost")
- `MUJOCO_MCP_PORT`: Server port (default: 8000)
- `MUJOCO_MCP_LOG_LEVEL`: Logging level (default: "INFO")
- `MUJOCO_MCP_MAX_SIMULATIONS`: Maximum concurrent simulations (default: 10)
- `MUJOCO_MCP_TIMEOUT`: Request timeout in seconds (default: 30)

Example:
```bash
export MUJOCO_MCP_LOG_LEVEL=DEBUG
export MUJOCO_MCP_PORT=8080
python -m mujoco_mcp.server
```

## Docker Configuration

For containerized deployments:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install MuJoCo dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy and install MuJoCo MCP
COPY . .
RUN pip install -e .

# Expose MCP port
EXPOSE 8000

# Start the server
CMD ["python", "-m", "mujoco_mcp.server"]
```

## Advanced Configuration

### Custom Server Script

Create a custom server script for specific configurations:

```python
#!/usr/bin/env python3
"""Custom MuJoCo MCP server with specific settings."""

from mujoco_mcp import MuJoCoMCPServer

# Custom configuration
config = {
    "max_simulations": 5,
    "default_timestep": 0.001,
    "visualization_enabled": True,
    "rl_algorithms": ["PPO", "SAC", "TD3"],
    "security": {
        "max_force": 100.0,
        "max_velocity": 10.0,
        "rate_limit": 1000  # requests per minute
    }
}

# Start server with custom config
server = MuJoCoMCPServer(config=config)
server.start(host="0.0.0.0", port=8000)
```

### Multi-Instance Configuration

For running multiple MuJoCo MCP instances:

```json
{
  "mcpServers": {
    "mujoco-mcp-1": {
      "command": "python",
      "args": ["-m", "mujoco_mcp.server", "--port", "8001"],
      "metadata": {"instance": "robot_arm"}
    },
    "mujoco-mcp-2": {
      "command": "python",
      "args": ["-m", "mujoco_mcp.server", "--port", "8002"],
      "metadata": {"instance": "humanoid"}
    }
  }
}
```

## Troubleshooting

### Common Issues

1. **Server not starting**: Ensure MuJoCo is properly installed:
   ```bash
   python -c "import mujoco; print(mujoco.__version__)"
   ```

2. **Connection refused**: Check if the port is already in use:
   ```bash
   lsof -i :8000  # macOS/Linux
   netstat -an | findstr :8000  # Windows
   ```

3. **Module not found**: Ensure the package is installed:
   ```bash
   pip install -e /path/to/mujoco-mcp
   ```

### Debug Mode

Enable debug logging for troubleshooting:

```bash
MUJOCO_MCP_LOG_LEVEL=DEBUG python -m mujoco_mcp.server
```

## Security Considerations

When deploying MuJoCo MCP:

1. **Local Development**: Use default localhost configuration
2. **Network Access**: If exposing to network, use authentication:
   ```python
   server = MuJoCoMCPServer(auth_required=True, auth_token="your-secret-token")
   ```
3. **Rate Limiting**: Configure appropriate rate limits for your use case
4. **Force Limits**: Adjust safety limits based on your simulation requirements

## Getting Help

- GitHub Issues: https://github.com/mujoco-mcp/mujoco-mcp/issues
- Documentation: https://github.com/mujoco-mcp/mujoco-mcp/wiki
- Examples: See the `examples/` directory for usage examples
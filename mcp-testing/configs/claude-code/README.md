# Claude Code MCP Configuration

## Setup

1. **Install Claude Code CLI**:
   ```bash
   # macOS
   brew install claude-code
   
   # Linux/Windows
   pip install claude-code
   ```

2. **Configure MCP Server**:
   ```bash
   # Copy configuration to Claude Code config directory
   cp mcp.json ~/.claude/mcp.json
   
   # Or set CLAUDE_MCP_CONFIG environment variable
   export CLAUDE_MCP_CONFIG="$(pwd)/mcp.json"
   ```

3. **Test Connection**:
   ```bash
   claude-code --test-mcp mujoco-mcp
   ```

## Environment Variables

- `REPO_ROOT`: Path to mujoco-mcp repository root
- `MUJOCO_GL`: Graphics backend (osmesa, egl, glfw)
- `MUJOCO_MCP_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

## Usage Examples

```bash
# Basic functionality test
claude-code "List available MuJoCo MCP tools"

# Create a physics scene
claude-code "Create a simple pendulum scene using MuJoCo MCP"

# Run simulation
claude-code "Run a 1000-step simulation and get the results"
```

## Troubleshooting

### Common Issues

1. **Server not found**: Ensure Python path includes mujoco_mcp module
2. **Timeout errors**: Increase timeout in configuration
3. **Graphics issues**: Set appropriate MUJOCO_GL backend

### Debug Mode

```bash
# Enable debug logging
export MUJOCO_MCP_LOG_LEVEL=DEBUG
claude-code --verbose "Test MuJoCo MCP connection"
```
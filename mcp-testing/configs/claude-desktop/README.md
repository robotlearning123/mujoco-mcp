# Claude Desktop MCP Configuration

## Setup

1. **Install Claude Desktop**:
   - Download from [Claude Desktop](https://claude.ai/download)
   - Available for macOS and Windows

2. **Configure MCP Server**:

   **macOS**:
   ```bash
   # Copy configuration to Claude Desktop config directory
   cp claude_desktop_config.json ~/Library/Application\ Support/Claude/claude_desktop_config.json
   ```

   **Windows**:
   ```bash
   # Copy to Windows AppData
   cp claude_desktop_config.json %APPDATA%/Claude/claude_desktop_config.json
   ```

3. **Install Dependencies**:
   ```bash
   # Ensure mujoco-mcp is installed in the Python environment
   pip install -e .
   
   # For graphics support
   pip install PyOpenGL PyOpenGL_accelerate
   ```

4. **Restart Claude Desktop**

## Configuration Details

The configuration uses:
- **osmesa**: Software-based OpenGL rendering (most compatible)
- **PYTHONUNBUFFERED**: Ensures real-time logging
- **INFO level**: Balanced logging for debugging

## Usage

1. **Open Claude Desktop**
2. **Start a conversation**
3. **Use MCP commands**:
   ```
   Create a pendulum simulation using MuJoCo MCP
   
   Step the simulation forward 100 steps
   
   Get the current state of the simulation
   
   Reset the simulation
   ```

## Verification

After setup, verify the MCP server is working:

1. Start Claude Desktop
2. Look for MCP status in the interface
3. Try: "List the available MuJoCo MCP tools"

## Platform-Specific Notes

### macOS
- Works with both Intel and Apple Silicon
- Graphics should work out of the box with osmesa
- May prompt for Python permissions

### Windows  
- Requires WSL2 for best compatibility
- Native Windows support with osmesa backend
- May need Visual C++ Redistributable

### Linux
- Claude Desktop not officially supported on Linux
- Use Claude Code or Cursor instead

## Troubleshooting

### Server Not Starting
1. Check Python path: `which python`
2. Verify installation: `python -m mujoco_mcp --help`
3. Check permissions on config file

### Graphics Issues
- Switch to `MUJOCO_GL=osmesa` for software rendering
- Install graphics drivers if using EGL/GLFW

### Connection Lost
- Check if viewer server is running: `mujoco-mcp-viewer`
- Restart Claude Desktop
- Clear app data and reconfigure

## Logging

Enable debug logging by modifying config:
```json
{
  "env": {
    "MUJOCO_MCP_LOG_LEVEL": "DEBUG"
  }
}
```

Logs can be found:
- **macOS**: `~/Library/Logs/Claude/`
- **Windows**: `%LOCALAPPDATA%/Claude/logs/`
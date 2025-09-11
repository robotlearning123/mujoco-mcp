# Cursor MCP Configuration

## Setup

1. **Install Cursor**:
   - Download from [cursor.sh](https://cursor.sh)
   - Available for macOS, Linux, and Windows

2. **Configure MCP Server**:
   
   **Option A: Workspace Settings**
   ```bash
   # Copy to .cursor/ in your project
   mkdir -p .cursor
   cp settings.json .cursor/settings.json
   ```
   
   **Option B: Global Settings**
   ```bash
   # macOS
   cp settings.json ~/Library/Application\ Support/Cursor/User/settings.json
   
   # Linux
   cp settings.json ~/.config/Cursor/User/settings.json
   
   # Windows
   cp settings.json %APPDATA%/Cursor/User/settings.json
   ```

3. **Restart Cursor** and open your mujoco-mcp project

4. **Verify Setup**:
   - Check status bar for MCP indicator
   - Use Cmd/Ctrl+Shift+P → "MCP: Show Status"

## Features

- **Auto-start**: MCP server starts automatically with workspace
- **Status indicator**: Shows connection status in status bar
- **Integrated debugging**: Built-in MCP debugging tools
- **Hot reload**: Automatically restarts server on code changes

## Usage

### Via Chat Interface
```
@mujoco-mcp create_scene pendulum
@mujoco-mcp step_simulation model_id=pendulum steps=100
@mujoco-mcp get_state model_id=pendulum
```

### Via Command Palette
- Cmd/Ctrl+Shift+P → "MCP: Create Scene"
- Cmd/Ctrl+Shift+P → "MCP: Step Simulation"
- Cmd/Ctrl+Shift+P → "MCP: Get State"

## Troubleshooting

### MCP Server Not Starting
1. Check Python path is correct
2. Verify mujoco_mcp is installed: `pip show mujoco-mcp`
3. Check logs in Cursor Developer Tools (Help → Toggle Developer Tools)

### Graphics Issues
- **Linux**: Install EGL drivers: `sudo apt-get install libegl1-mesa-dev`
- **Windows**: Use WSL2 with X11 forwarding
- **macOS**: Should work out of the box

### Connection Timeouts
- Increase timeout in settings.json
- Check if MuJoCo viewer server is responding: `mujoco-mcp-viewer --test`

## Debug Mode

Enable detailed logging:
```json
{
  "mcp.debug": true,
  "mcp.logLevel": "debug"
}
```

Then check Cursor console for detailed MCP communication logs.
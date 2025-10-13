# 🧪 Testing Your MuJoCo MCP Tools Step-by-Step

## When your server is properly connected to Claude Code, you can test each tool:

### 1. **get_server_info**
```
Ask Claude: "Use the get_server_info tool to show me server details"
Expected: Server name, version, capabilities
```

### 2. **create_scene** 
```
Ask Claude: "Create a pendulum scene using create_scene tool"
Expected: Scene created successfully with model details
```

### 3. **step_simulation**
```
Ask Claude: "Step the pendulum simulation 5 times"
Expected: Simulation advances, physics updates
```

### 4. **get_state**
```
Ask Claude: "Get the current state of the pendulum simulation"
Expected: JSON with positions, velocities, etc.
```

### 5. **reset_simulation**
```
Ask Claude: "Reset the pendulum to initial state" 
Expected: Simulation resets to starting conditions
```

### 6. **close_viewer**
```
Ask Claude: "Close the MuJoCo viewer window"
Expected: Viewer window closes cleanly
```

## 🎯 Testing Workflow

1. **Server Status**: "Check if mujoco-mcp server is connected"
2. **Tool Discovery**: "List all available MuJoCo tools" 
3. **Basic Function**: "Get server info"
4. **Model Loading**: "Create a scene"
5. **Simulation**: "Step and check state"
6. **Cleanup**: "Reset and close viewer"

## ✅ Success Indicators

- ✅ Tools are discoverable
- ✅ Scene creation works
- ✅ Physics simulation responds
- ✅ State data is returned
- ✅ Reset functionality works
- ✅ Viewer management works

## ❌ Common Issues

- **Server not found**: Check configuration and restart
- **Tool timeout**: MuJoCo might be slow to start
- **Import errors**: Check Python path and dependencies
- **Permission issues**: Check file/directory access
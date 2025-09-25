# 🔧 MCP Server Troubleshooting Guide

## 1. **Check Server Can Start**
```bash
cd /path/to/mujoco-mcp
python -m mujoco_mcp --check
```
**Expected**: Configuration check passes

## 2. **Test Server Manually**
```bash
python -m mujoco_mcp
# Should start and wait for input
# Press Ctrl+C to stop
```

## 3. **Check Dependencies**
```bash
pip list | grep -E "(mujoco|mcp)"
```
**Expected**: mujoco and mcp packages installed

## 4. **Test with Demo Script**
```bash
python demo_simple_mcp.py
```
**Expected**: Successfully connects and lists tools

## 5. **Check Claude Desktop Logs**
- **macOS**: `~/Library/Logs/Claude Desktop/`
- **Windows**: `%LOCALAPPDATA%/Claude Desktop/logs/`
- **Linux**: `~/.cache/claude-desktop/logs/`

Look for MCP server connection errors.

## 6. **Verify Configuration Path**
```bash
# Check if config file exists
ls -la ~/.config/claude-desktop/claude_desktop_config.json

# View contents
cat ~/.config/claude-desktop/claude_desktop_config.json
```

## 7. **Test JSON-RPC Communication**
```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}' | python -m mujoco_mcp
```
**Expected**: JSON response with server info

## 🚨 Common Issues & Fixes

### **"Server not found"**
- ✅ Check configuration file exists
- ✅ Restart Claude Desktop after config changes
- ✅ Verify file paths in configuration

### **"Module not found"** 
- ✅ Check PYTHONPATH in config
- ✅ Install mujoco-mcp in development mode: `pip install -e .`
- ✅ Verify src/mujoco_mcp/__init__.py exists

### **"Connection timeout"**
- ✅ MuJoCo might be slow to start
- ✅ Check system resources
- ✅ Try increasing timeout in configuration

### **"Tool execution fails"**
- ✅ Check MuJoCo installation: `python -c "import mujoco; print(mujoco.__version__)"`
- ✅ Check display/graphics setup
- ✅ Review server logs for errors

## ✅ Success Test Checklist

- [ ] Server starts without errors
- [ ] Configuration file is correctly placed  
- [ ] Claude Desktop connects to server
- [ ] Tools are discoverable via ListMcpResourcesTool
- [ ] get_server_info returns server details
- [ ] create_scene successfully creates models
- [ ] Simulation tools (step, get_state) work
- [ ] No errors in Claude Desktop logs
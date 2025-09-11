# MuJoCo MCP Multi-Client Testing Framework

Tests MuJoCo MCP server compatibility across multiple AI clients and platforms.

## Supported Clients
- **Claude Code** - CLI tool
- **Cursor** - AI code editor  
- **Claude Desktop** - Desktop app
- **ChatGPT** - With MCP support

## Quick Start
```bash
# Test all clients
./mcp-testing/scripts/test-client.sh all

# Test specific client
./mcp-testing/scripts/test-client.sh claude-code
```

## Features
- Cross-platform testing (macOS, Linux, Windows)
- Automated configuration setup
- MCP protocol compliance validation
- Graphics backend testing (osmesa, EGL, GLFW)
#!/bin/bash
# MuJoCo MCP Remote Quick Start Script

echo "🚀 MuJoCo MCP Remote Quick Start"
echo "================================"

# Enter correct directory
cd "$(dirname "$0")"

# Check Python
echo "📍 Current directory: $(pwd)"
echo "🐍 Python path: $(which python)"
echo "🐍 Python version: $(python --version)"

# Run startup script
echo ""
echo "🔧 Starting system..."
python start_mujoco_system.py

# If failed, provide alternative
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Automatic startup failed, please manually execute the following steps:"
    echo ""
    echo "1. Start Viewer Server:"
    echo "   python mujoco_viewer_server.py"
    echo ""
    echo "2. Test in Claude Desktop:"
    echo "   - Restart Claude Desktop"
    echo "   - Input: 'What MCP servers are available?'"
    echo ""
fi
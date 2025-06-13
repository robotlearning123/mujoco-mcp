#!/bin/bash
# MuJoCo MCP Remote Quick Start Script

echo "🚀 MuJoCo MCP Remote 快速启动"
echo "================================"

# 进入正确的目录
cd "$(dirname "$0")"

# 检查Python
echo "📍 当前目录: $(pwd)"
echo "🐍 Python路径: $(which python)"
echo "🐍 Python版本: $(python --version)"

# 运行启动脚本
echo ""
echo "🔧 启动系统..."
python start_mujoco_system.py

# 如果失败，提供备用方案
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 自动启动失败，请手动执行以下步骤："
    echo ""
    echo "1. 启动Viewer Server:"
    echo "   python mujoco_viewer_server.py"
    echo ""
    echo "2. 在Claude Desktop中测试:"
    echo "   - 重启Claude Desktop"
    echo "   - 输入: 'What MCP servers are available?'"
    echo ""
fi
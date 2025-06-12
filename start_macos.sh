#!/bin/bash
# macOS专用启动脚本 - 使用mjpython

echo "🍎 MuJoCo MCP macOS 启动脚本"
echo "================================"

# 进入正确的目录
cd "$(dirname "$0")"

# 检查mjpython
echo "🔍 检查mjpython..."
MJPYTHON=$(which mjpython)

if [ -z "$MJPYTHON" ]; then
    echo "❌ 找不到mjpython"
    echo "   请确保已安装MuJoCo并添加到PATH"
    echo "   通常在: /opt/miniconda3/bin/mjpython"
    exit 1
fi

echo "✅ 找到mjpython: $MJPYTHON"

# 检查Python（用于MCP服务器）
echo "🐍 Python路径: $(which python)"

# 运行macOS修复脚本
echo ""
echo "🔧 运行macOS修复程序..."
python macos_fix.py
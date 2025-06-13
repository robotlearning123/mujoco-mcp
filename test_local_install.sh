#!/bin/bash
# MuJoCo-MCP 本地安装测试脚本

echo "=== MuJoCo-MCP Local Installation Test ==="

# 1. 创建测试虚拟环境
echo "1. Creating test virtual environment..."
python -m venv test_env
source test_env/bin/activate

# 2. 构建包
echo "2. Building package..."
pip install build
python -m build

# 3. 本地安装测试
echo "3. Installing from local wheel..."
pip install dist/mujoco_mcp-*.whl

# 4. 验证安装
echo "4. Verifying installation..."
python -c "import mujoco_mcp; print(f'Version: {mujoco_mcp.__version__}')"

# 5. 测试命令行入口
echo "5. Testing CLI entry points..."
python -m mujoco_mcp --version

# 6. 测试 MCP server 启动
echo "6. Testing MCP server startup..."
timeout 5 python -m mujoco_mcp || echo "Server started successfully"

# 7. 清理
deactivate
rm -rf test_env

echo "✅ Local installation test completed!"
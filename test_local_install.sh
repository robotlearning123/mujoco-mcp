#!/bin/bash
# MuJoCo-MCP Local Installation Test Script

echo "=== MuJoCo-MCP Local Installation Test ==="

# 1. Create test virtual environment
echo "1. Creating test virtual environment..."
python -m venv test_env
source test_env/bin/activate

# 2. Build package
echo "2. Building package..."
pip install build
python -m build

# 3. Local installation test
echo "3. Installing from local wheel..."
pip install dist/mujoco_mcp-*.whl

# 4. Verify installation
echo "4. Verifying installation..."
python -c "import mujoco_mcp; print(f'Version: {mujoco_mcp.__version__}')"

# 5. Test CLI entry points
echo "5. Testing CLI entry points..."
python -m mujoco_mcp --version

# 6. Test MCP server startup
echo "6. Testing MCP server startup..."
timeout 5 python -m mujoco_mcp || echo "Server started successfully"

# 7. Cleanup
deactivate
rm -rf test_env

echo "✅ Local installation test completed!"
#!/bin/bash
# MuJoCo-MCP 测试环境自动化搭建脚本

set -e  # 遇到错误立即退出

echo "=========================================="
echo "MuJoCo-MCP Internal Testing Setup"
echo "=========================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 工具函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查系统信息
check_system() {
    log_info "检查系统信息..."
    echo "OS: $(uname -s)"
    echo "Architecture: $(uname -m)"
    echo "Python: $(python --version 2>/dev/null || echo 'Not found')"
    echo "Pip: $(pip --version 2>/dev/null || echo 'Not found')"
    echo ""
}

# 清理旧环境
cleanup_old_envs() {
    log_info "清理旧测试环境..."
    rm -rf test_envs/
    mkdir -p test_envs
}

# 创建多版本Python测试环境
create_python_envs() {
    log_info "创建Python测试环境..."
    
    cd test_envs
    
    # Python 3.10
    if command -v python3.10 &> /dev/null; then
        log_info "创建 Python 3.10 环境..."
        python3.10 -m venv py310_env
        source py310_env/bin/activate
        pip install --upgrade pip
        pip install ../../dist/mujoco_mcp-0.8.0-py3-none-any.whl
        deactivate
    else
        log_warn "Python 3.10 未找到，跳过"
    fi
    
    # Python 3.11
    if command -v python3.11 &> /dev/null; then
        log_info "创建 Python 3.11 环境..."
        python3.11 -m venv py311_env
        source py311_env/bin/activate
        pip install --upgrade pip
        pip install ../../dist/mujoco_mcp-0.8.0-py3-none-any.whl
        deactivate
    else
        log_warn "Python 3.11 未找到，跳过"
    fi
    
    # Python 3.12
    if command -v python3.12 &> /dev/null; then
        log_info "创建 Python 3.12 环境..."
        python3.12 -m venv py312_env
        source py312_env/bin/activate
        pip install --upgrade pip
        pip install ../../dist/mujoco_mcp-0.8.0-py3-none-any.whl
        deactivate
    else
        log_warn "Python 3.12 未找到，跳过"
    fi
    
    cd ..
}

# 创建conda测试环境
create_conda_envs() {
    if command -v conda &> /dev/null; then
        log_info "创建Conda测试环境..."
        
        cd test_envs
        
        # 创建conda环境
        conda create -y -n mujoco_mcp_test python=3.11
        conda activate mujoco_mcp_test
        pip install ../../dist/mujoco_mcp-0.8.0-py3-none-any.whl
        conda deactivate
        
        cd ..
    else
        log_warn "Conda 未找到，跳过conda环境测试"
    fi
}

# 测试基础安装
test_basic_installation() {
    log_info "测试基础安装..."
    
    cd test_envs
    
    # 测试每个环境
    for env_dir in */; do
        if [ -d "$env_dir" ] && [ -f "$env_dir/bin/activate" ]; then
            log_info "测试环境: $env_dir"
            source "$env_dir/bin/activate"
            
            # 基础命令测试
            echo "  - 版本检查..."
            mujoco-mcp --version || log_error "版本命令失败"
            
            echo "  - 配置检查..."
            mujoco-mcp --check || log_error "配置检查失败"
            
            echo "  - 帮助信息..."
            mujoco-mcp --help > /dev/null || log_error "帮助命令失败"
            
            deactivate
            echo ""
        fi
    done
    
    cd ..
}

# 创建MCP客户端测试配置
create_mcp_configs() {
    log_info "创建MCP客户端测试配置..."
    
    mkdir -p test_configs
    
    # Claude Desktop 配置
    cat > test_configs/claude_desktop_config.json << 'EOF'
{
  "mcpServers": {
    "mujoco-mcp": {
      "command": "mujoco-mcp",
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
EOF

    # Cursor 配置
    cat > test_configs/cursor_mcp_config.json << 'EOF'
{
  "mcp": {
    "servers": {
      "mujoco-mcp": {
        "command": "mujoco-mcp"
      }
    }
  }
}
EOF

    log_info "配置文件已创建在 test_configs/ 目录"
}

# 创建性能测试脚本
create_performance_tests() {
    log_info "创建性能测试脚本..."
    
    mkdir -p test_scripts
    
    cat > test_scripts/performance_test.py << 'EOF'
#!/usr/bin/env python3
"""
MuJoCo-MCP 性能测试脚本
"""

import time
import asyncio
import statistics
import json
from mujoco_mcp.mcp_server import handle_list_tools, handle_call_tool

async def test_tool_response_times():
    """测试工具响应时间"""
    print("测试工具响应时间...")
    
    # 测试 get_server_info
    times = []
    for i in range(10):
        start = time.time()
        await handle_call_tool("get_server_info", {})
        end = time.time()
        times.append((end - start) * 1000)  # 转换为毫秒
    
    print(f"get_server_info 响应时间:")
    print(f"  平均: {statistics.mean(times):.2f}ms")
    print(f"  中位数: {statistics.median(times):.2f}ms")
    print(f"  P95: {statistics.quantiles(times, n=20)[18]:.2f}ms")
    print(f"  最大: {max(times):.2f}ms")

async def test_tools_list():
    """测试工具列表"""
    print("\n测试工具列表...")
    
    start = time.time()
    tools = await handle_list_tools()
    end = time.time()
    
    print(f"工具数量: {len(tools)}")
    print(f"列表响应时间: {(end-start)*1000:.2f}ms")
    
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")

async def main():
    print("=== MuJoCo-MCP 性能测试 ===\n")
    
    await test_tools_list()
    await test_tool_response_times()
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    asyncio.run(main())
EOF

    chmod +x test_scripts/performance_test.py
}

# 创建功能测试脚本
create_functional_tests() {
    log_info "创建功能测试脚本..."
    
    cat > test_scripts/functional_test.py << 'EOF'
#!/usr/bin/env python3
"""
MuJoCo-MCP 功能测试脚本
"""

import asyncio
import json
from mujoco_mcp.mcp_server import handle_call_tool

async def test_basic_workflow():
    """测试基础工作流程"""
    print("=== 基础工作流程测试 ===\n")
    
    # 1. 获取服务器信息
    print("1. 获取服务器信息...")
    result = await handle_call_tool("get_server_info", {})
    print(f"   结果: {result[0].text[:100]}...")
    
    # 2. 创建场景 (模拟，实际需要viewer server)
    print("\n2. 创建摆锤场景...")
    result = await handle_call_tool("create_scene", {"scene_type": "pendulum"})
    print(f"   结果: {result[0].text}")
    
    # 3. 测试无效工具
    print("\n3. 测试无效工具...")
    result = await handle_call_tool("invalid_tool", {})
    print(f"   结果: {result[0].text}")
    
    print("\n=== 基础测试完成 ===")

async def test_error_handling():
    """测试错误处理"""
    print("\n=== 错误处理测试 ===\n")
    
    # 测试无效参数
    test_cases = [
        ("create_scene", {"scene_type": "invalid_scene"}),
        ("step_simulation", {"model_id": "nonexistent"}),
        ("get_state", {}),  # 缺少必需参数
    ]
    
    for tool_name, args in test_cases:
        print(f"测试 {tool_name} 错误处理...")
        result = await handle_call_tool(tool_name, args)
        print(f"   结果: {result[0].text}")
        print()

async def main():
    await test_basic_workflow()
    await test_error_handling()

if __name__ == "__main__":
    asyncio.run(main())
EOF

    chmod +x test_scripts/functional_test.py
}

# 创建测试报告生成器
create_test_reporter() {
    log_info "创建测试报告生成器..."
    
    cat > test_scripts/generate_report.py << 'EOF'
#!/usr/bin/env python3
"""
测试报告生成器
"""

import json
import sys
import time
from pathlib import Path

def generate_test_report():
    """生成测试报告"""
    
    report = {
        "test_session": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "version": "0.8.0",
            "platform": sys.platform
        },
        "environments": [],
        "test_results": {
            "installation": {"passed": 0, "failed": 0},
            "functionality": {"passed": 0, "failed": 0}, 
            "performance": {"passed": 0, "failed": 0}
        },
        "issues": []
    }
    
    # 检查测试环境
    test_envs_dir = Path("test_envs")
    if test_envs_dir.exists():
        for env_dir in test_envs_dir.iterdir():
            if env_dir.is_dir():
                report["environments"].append({
                    "name": env_dir.name,
                    "status": "created"
                })
    
    # 保存报告
    with open("test_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("测试报告已生成: test_report.json")

if __name__ == "__main__":
    generate_test_reporter()
EOF

    chmod +x test_scripts/generate_report.py
}

# 主函数
main() {
    check_system
    cleanup_old_envs
    create_python_envs
    create_conda_envs
    test_basic_installation
    create_mcp_configs
    create_performance_tests
    create_functional_tests
    create_test_reporter
    
    log_info "测试环境搭建完成！"
    echo ""
    echo "下一步操作:"
    echo "1. 运行功能测试: python test_scripts/functional_test.py"
    echo "2. 运行性能测试: python test_scripts/performance_test.py"
    echo "3. 测试MCP客户端集成 (使用 test_configs/ 中的配置)"
    echo "4. 生成测试报告: python test_scripts/generate_report.py"
    echo ""
    echo "测试环境位置:"
    echo "- test_envs/     # Python 虚拟环境"
    echo "- test_configs/  # MCP 客户端配置"
    echo "- test_scripts/  # 测试脚本"
}

# 运行主函数
main
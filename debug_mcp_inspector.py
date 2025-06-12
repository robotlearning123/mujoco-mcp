#!/usr/bin/env python3
"""
MCP Inspector 调试辅助脚本
提供便捷的MCP Inspector测试和调试功能
"""

import subprocess
import sys
import os
import json
import time
from typing import List, Dict, Any

def check_prerequisites():
    """检查前置条件"""
    print("🔍 检查前置条件...")
    
    # 检查npx
    try:
        result = subprocess.run(['npx', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ npx 版本: {result.stdout.strip()}")
        else:
            print("   ❌ npx 不可用")
            return False
    except FileNotFoundError:
        print("   ❌ npx 未安装")
        return False
    
    # 检查Python MCP模块
    try:
        import mcp
        print(f"   ✅ MCP Python SDK 可用")
    except ImportError:
        print("   ❌ MCP Python SDK 未安装")
        print("      请运行: pip install mcp")
        return False
    
    # 检查MuJoCo
    try:
        import mujoco
        print(f"   ✅ MuJoCo 版本: {mujoco.__version__}")
    except ImportError:
        print("   ❌ MuJoCo 未安装")
        return False
    
    # 检查服务器文件
    server_file = "mcp_server_remote.py"
    if os.path.exists(server_file):
        print(f"   ✅ MCP服务器文件存在: {server_file}")
    else:
        print(f"   ❌ MCP服务器文件不存在: {server_file}")
        return False
    
    return True

def start_viewer_server(port=8888):
    """启动MuJoCo Viewer Server"""
    print(f"🖥️ 启动 MuJoCo Viewer Server (端口 {port})...")
    
    try:
        # 检查端口是否已被占用
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        
        if result == 0:
            print(f"   ⚠️  端口 {port} 已被占用，Viewer Server可能已在运行")
            return True
        
        # 启动服务器
        print(f"   启动命令: python mujoco_viewer_server.py --port {port}")
        print("   注意: 请在另一个终端手动运行上述命令")
        return True
        
    except Exception as e:
        print(f"   ❌ 检查端口失败: {e}")
        return False

def run_mcp_inspector(interactive=True):
    """运行MCP Inspector"""
    print("🔧 启动 MCP Inspector...")
    
    if interactive:
        print("   模式: 交互式")
        cmd = [
            'npx', 
            '@modelcontextprotocol/inspector',
            'python', 
            'mcp_server_remote.py'
        ]
    else:
        print("   模式: 命令行")
        cmd = [
            'npx', 
            '@modelcontextprotocol/inspector',
            '--help'
        ]
    
    print(f"   命令: {' '.join(cmd)}")
    
    if interactive:
        print("\n   🎮 Inspector将在浏览器中打开")
        print("   📍 通常地址: http://localhost:3000")
        print("   ⏹️  按 Ctrl+C 停止")
        
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            print("\n   停止 MCP Inspector")
        except Exception as e:
            print(f"   ❌ 启动失败: {e}")
    else:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            print(f"   输出: {result.stdout}")
            if result.stderr:
                print(f"   错误: {result.stderr}")
        except Exception as e:
            print(f"   ❌ 运行失败: {e}")

def test_server_direct():
    """直接测试MCP服务器"""
    print("🧪 直接测试 MCP 服务器...")
    
    sys.path.insert(0, 'src')
    
    try:
        from mujoco_mcp.remote_server import MuJoCoRemoteServer
        
        server = MuJoCoRemoteServer()
        
        # 测试服务器信息
        print("   测试 get_server_info...")
        result = server.call_tool('get_server_info', {})
        print(f"   ✅ 服务器信息: {result.get('name', 'unknown')} v{result.get('version', 'unknown')}")
        
        # 测试工具列表
        print("   测试工具列表...")
        tools = server.get_tools()
        print(f"   ✅ 可用工具: {len(tools)} 个")
        for tool in tools:
            print(f"      - {tool['name']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 直接测试失败: {e}")
        return False

def create_test_scenario():
    """创建测试场景脚本"""
    print("📝 创建测试场景...")
    
    scenario_script = """
# MCP Inspector 测试场景

## 基础测试
1. 连接到服务器后，首先运行：
   get_server_info

2. 创建简单场景：
   create_scene
   参数: {"scene_type": "pendulum"}

3. 获取状态：
   get_state
   参数: {"model_id": "<从create_scene获取的ID>"}

## 高级测试
4. 设置关节位置：
   set_joint_positions
   参数: {"model_id": "<model_id>", "positions": [0.5]}

5. 重置仿真：
   reset_simulation
   参数: {"model_id": "<model_id>"}

6. 自然语言测试：
   execute_command
   参数: {"command": "create pendulum"}

## 错误测试
7. 测试无效工具：
   invalid_tool_name

8. 测试无效参数：
   create_scene
   参数: {"invalid_param": "test"}
"""
    
    with open("mcp_inspector_test_scenario.md", "w", encoding="utf-8") as f:
        f.write(scenario_script)
    
    print("   ✅ 测试场景已保存到: mcp_inspector_test_scenario.md")

def main():
    """主函数"""
    print("🚀 MCP Inspector 调试辅助工具")
    print("版本: v0.6.2")
    print("时间:", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("="*50)
    
    # 检查前置条件
    if not check_prerequisites():
        print("\n❌ 前置条件检查失败，请解决上述问题后重试")
        return 1
    
    print("\n" + "="*50)
    
    # 提供选项菜单
    while True:
        print("\n🎯 选择操作:")
        print("1. 启动 MCP Inspector (交互式)")
        print("2. 直接测试 MCP 服务器")
        print("3. 检查 Viewer Server 状态")
        print("4. 创建测试场景文档")
        print("5. 显示使用说明")
        print("0. 退出")
        
        choice = input("\n请选择 (0-5): ").strip()
        
        if choice == "1":
            print("\n⚠️  确保已启动 MuJoCo Viewer Server:")
            print("   python mujoco_viewer_server.py")
            input("按回车继续...")
            run_mcp_inspector(interactive=True)
        
        elif choice == "2":
            test_server_direct()
        
        elif choice == "3":
            start_viewer_server()
        
        elif choice == "4":
            create_test_scenario()
        
        elif choice == "5":
            print_usage_instructions()
        
        elif choice == "0":
            print("👋 再见!")
            break
        
        else:
            print("❌ 无效选择，请重试")

def print_usage_instructions():
    """打印使用说明"""
    print("""
📖 使用说明

🔧 完整测试流程:

1. 启动 MuJoCo Viewer Server:
   终端1: python mujoco_viewer_server.py

2. 启动 MCP Inspector:
   终端2: python debug_mcp_inspector.py
   选择选项1

3. 在浏览器中测试:
   - 访问 http://localhost:3000
   - 连接到服务器
   - 按照测试场景进行测试

🐛 调试技巧:

- 查看服务器日志输出
- 检查浏览器开发者工具
- 使用选项2进行直接测试
- 确保端口8888未被占用

📚 参考文档:
- MCP规范: https://spec.modelcontextprotocol.io/
- MCP Inspector: https://github.com/modelcontextprotocol/inspector
""")

if __name__ == "__main__":
    try:
        result = main()
        sys.exit(result or 0)
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，程序退出")
        sys.exit(0)
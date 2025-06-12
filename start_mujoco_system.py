#!/usr/bin/env python3
"""
MuJoCo MCP Remote 系统自动启动脚本
解决连接问题，提供完整的系统启动和诊断功能
"""

import os
import sys
import subprocess
import socket
import time
import json
import signal
import threading
from typing import Optional, Dict, Any

def check_port_available(port: int, host: str = 'localhost') -> bool:
    """检查端口是否可用"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False

def kill_process_on_port(port: int):
    """杀死占用指定端口的进程"""
    try:
        result = subprocess.run(['lsof', '-ti', f':{port}'], 
                              capture_output=True, text=True)
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                try:
                    subprocess.run(['kill', '-9', pid])
                    print(f"   🔧 已终止占用端口{port}的进程: {pid}")
                except:
                    pass
        return True
    except:
        return False

def check_dependencies():
    """检查系统依赖"""
    print("🔍 检查系统依赖...")
    
    # 检查Python模块
    required_modules = ['mujoco', 'numpy', 'mcp']
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError:
            print(f"   ❌ {module} (缺失)")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n⚠️  请安装缺失的模块:")
        print(f"   pip install {' '.join(missing_modules)}")
        return False
    
    # 检查文件存在
    required_files = [
        'mujoco_viewer_server.py',
        'mcp_server_remote.py',
        'src/mujoco_mcp/__init__.py'
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} (不存在)")
            return False
    
    return True

def start_viewer_server(port: int = 8888) -> Optional[subprocess.Popen]:
    """启动MuJoCo Viewer Server"""
    print(f"🖥️ 启动MuJoCo Viewer Server (端口 {port})...")
    
    # 检查端口是否被占用
    if check_port_available(port):
        print(f"   ⚠️  端口 {port} 已被占用")
        response = input("   是否终止占用进程? (y/n): ")
        if response.lower() == 'y':
            kill_process_on_port(port)
            time.sleep(2)
        else:
            return None
    
    try:
        # 启动viewer server
        cmd = [sys.executable, 'mujoco_viewer_server.py', '--port', str(port)]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 等待启动
        print("   等待服务器启动...")
        for i in range(10):  # 最多等待10秒
            if check_port_available(port):
                print(f"   ✅ Viewer Server启动成功 (PID: {process.pid})")
                return process
            time.sleep(1)
            print(f"   ⏳ 等待中... ({i+1}/10)")
        
        print("   ❌ Viewer Server启动超时")
        process.terminate()
        return None
        
    except Exception as e:
        print(f"   ❌ 启动失败: {e}")
        return None

def test_viewer_connection(port: int = 8888) -> bool:
    """测试viewer连接"""
    print(f"🔗 测试Viewer Server连接 (端口 {port})...")
    
    try:
        import socket
        import json
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect(('localhost', port))
        
        # 发送ping命令
        ping_cmd = json.dumps({"type": "ping"})
        sock.send(ping_cmd.encode('utf-8'))
        
        # 接收响应
        response = sock.recv(1024)
        result = json.loads(response.decode('utf-8'))
        
        sock.close()
        
        if result.get("success") and result.get("pong"):
            print("   ✅ Viewer Server响应正常")
            return True
        else:
            print(f"   ❌ Viewer Server响应异常: {result}")
            return False
            
    except Exception as e:
        print(f"   ❌ 连接测试失败: {e}")
        return False

def check_claude_config():
    """检查Claude Desktop配置"""
    print("📋 检查Claude Desktop配置...")
    
    config_paths = [
        "/Users/robert/Library/Application Support/Claude/claude_desktop_config.json",
        "~/.config/claude-desktop/config.json"
    ]
    
    for path in config_paths:
        expanded_path = os.path.expanduser(path)
        if os.path.exists(expanded_path):
            print(f"   ✅ 找到配置文件: {expanded_path}")
            
            try:
                with open(expanded_path, 'r') as f:
                    config = json.load(f)
                
                mcp_servers = config.get('mcpServers', {})
                if 'mujoco-mcp-remote' in mcp_servers:
                    print("   ✅ 找到mujoco-mcp-remote配置")
                    server_config = mcp_servers['mujoco-mcp-remote']
                    
                    # 检查命令路径
                    command = server_config.get('command', '')
                    args = server_config.get('args', [])
                    if args and os.path.exists(args[0]):
                        print("   ✅ MCP服务器文件存在")
                    else:
                        print("   ❌ MCP服务器文件路径错误")
                        print(f"      配置路径: {args[0] if args else 'N/A'}")
                        print(f"      当前目录: {os.getcwd()}")
                        return False
                else:
                    print("   ❌ 未找到mujoco-mcp-remote配置")
                    return False
                    
                return True
                
            except Exception as e:
                print(f"   ❌ 配置文件解析失败: {e}")
                return False
    
    print("   ❌ 未找到Claude Desktop配置文件")
    return False

def update_claude_config():
    """更新Claude Desktop配置"""
    print("🔧 更新Claude Desktop配置...")
    
    config_path = "/Users/robert/Library/Application Support/Claude/claude_desktop_config.json"
    
    # 读取现有配置
    config = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except:
            config = {}
    
    # 更新配置
    current_dir = os.getcwd()
    config.setdefault('mcpServers', {})
    config['mcpServers']['mujoco-mcp-remote'] = {
        "command": sys.executable,
        "args": [os.path.join(current_dir, "mcp_server_remote.py")],
        "cwd": current_dir,
        "env": {
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": os.path.join(current_dir, "src"),
            "MUJOCO_MCP_LOG_LEVEL": "INFO",
            "PATH": os.environ.get("PATH", "")
        }
    }
    
    # 保存配置
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print("   ✅ 配置更新成功")
        return True
    except Exception as e:
        print(f"   ❌ 配置更新失败: {e}")
        return False

def create_simple_test():
    """创建简单测试"""
    print("🧪 创建系统测试...")
    
    try:
        # 导入测试模块
        sys.path.insert(0, 'src')
        from mujoco_mcp.remote_server import MuJoCoRemoteServer
        
        server = MuJoCoRemoteServer()
        
        # 测试基本功能
        result = server.call_tool('get_server_info', {})
        if result.get('name') == 'mujoco-mcp-remote':
            print("   ✅ MCP服务器模块正常")
            return True
        else:
            print(f"   ❌ MCP服务器测试失败: {result}")
            return False
            
    except Exception as e:
        print(f"   ❌ 模块测试失败: {e}")
        return False

def main():
    """主启动流程"""
    print("🚀 MuJoCo MCP Remote 系统启动器")
    print("=" * 50)
    
    viewer_process = None
    
    try:
        # 1. 检查依赖
        if not check_dependencies():
            print("\n❌ 依赖检查失败，请解决后重试")
            return 1
        
        # 2. 创建简单测试
        if not create_simple_test():
            print("\n❌ 模块测试失败")
            return 1
        
        # 3. 启动Viewer Server
        viewer_process = start_viewer_server()
        if not viewer_process:
            print("\n❌ Viewer Server启动失败")
            return 1
        
        # 4. 测试连接
        if not test_viewer_connection():
            print("\n❌ Viewer连接测试失败")
            return 1
        
        # 5. 检查Claude配置
        if not check_claude_config():
            print("\n🔧 配置有问题，正在修复...")
            if not update_claude_config():
                print("\n❌ 配置修复失败")
                return 1
        
        print("\n" + "=" * 50)
        print("🎉 系统启动成功！")
        print("\n📋 接下来的步骤:")
        print("1. 重启Claude Desktop (Cmd+Q 然后重新打开)")
        print("2. 在Claude Desktop中测试: 'What MCP servers are available?'")
        print("3. 创建场景: 'Create a pendulum simulation'")
        print("\n⏹️  按 Ctrl+C 停止Viewer Server")
        
        # 保持运行
        while True:
            time.sleep(1)
            if viewer_process and viewer_process.poll() is not None:
                print("\n⚠️  Viewer Server进程已退出")
                break
    
    except KeyboardInterrupt:
        print("\n\n🛑 用户中断")
    
    except Exception as e:
        print(f"\n❌ 启动过程出错: {e}")
        return 1
    
    finally:
        # 清理
        if viewer_process:
            print("🧹 清理Viewer Server进程...")
            try:
                viewer_process.terminate()
                viewer_process.wait(timeout=5)
            except:
                try:
                    viewer_process.kill()
                except:
                    pass
    
    print("👋 再见!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
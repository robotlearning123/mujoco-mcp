#!/usr/bin/env python3
"""
macOS MuJoCo修复脚本
解决mjpython和连接问题
"""

import os
import sys
import subprocess
import time
import signal

def kill_existing_processes():
    """杀死现有的viewer进程"""
    print("🧹 清理现有进程...")
    try:
        # 获取占用8888端口的进程
        result = subprocess.run(['lsof', '-ti', ':8888'], 
                              capture_output=True, text=True)
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGTERM)
                    print(f"   ✅ 终止进程: {pid}")
                except:
                    pass
            time.sleep(2)
    except:
        pass

def start_viewer_with_mjpython():
    """使用mjpython启动viewer"""
    print("\n🚀 使用mjpython启动MuJoCo Viewer Server...")
    
    # 检查mjpython
    mjpython_path = subprocess.run(['which', 'mjpython'], 
                                  capture_output=True, text=True).stdout.strip()
    
    if not mjpython_path:
        print("❌ 找不到mjpython，请安装MuJoCo")
        return None
    
    print(f"   ✅ 找到mjpython: {mjpython_path}")
    
    # 启动viewer server
    cmd = [mjpython_path, 'mujoco_viewer_server.py', '--port', '8888']
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print(f"   ✅ Viewer Server已启动 (PID: {process.pid})")
        return process
    except Exception as e:
        print(f"   ❌ 启动失败: {e}")
        return None

def update_claude_config_for_macos():
    """更新Claude配置使用mjpython"""
    print("\n🔧 更新Claude配置...")
    
    import json
    config_path = "/Users/robert/Library/Application Support/Claude/claude_desktop_config.json"
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 检查是否需要使用mjpython
        mjpython_path = subprocess.run(['which', 'mjpython'], 
                                      capture_output=True, text=True).stdout.strip()
        
        if mjpython_path and 'mujoco-mcp-remote' in config.get('mcpServers', {}):
            # 更新为mjpython
            config['mcpServers']['mujoco-mcp-remote']['command'] = mjpython_path
            config['mcpServers']['mujoco-mcp-remote']['env']['MUJOCO_GL'] = 'glfw'
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print("   ✅ 配置已更新为使用mjpython")
            return True
    except Exception as e:
        print(f"   ❌ 配置更新失败: {e}")
    
    return False

def test_connection():
    """测试连接"""
    print("\n🔗 测试连接...")
    
    import socket
    import json
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect(('localhost', 8888))
        
        # 发送ping
        ping_cmd = json.dumps({"type": "ping"})
        sock.send(ping_cmd.encode('utf-8'))
        
        # 接收响应
        response = sock.recv(1024)
        result = json.loads(response.decode('utf-8'))
        sock.close()
        
        if result.get("success"):
            print("   ✅ Viewer Server响应正常")
            return True
    except Exception as e:
        print(f"   ❌ 连接测试失败: {e}")
    
    return False

def main():
    """主函数"""
    print("🔧 macOS MuJoCo修复程序")
    print("="*50)
    
    # 1. 清理现有进程
    kill_existing_processes()
    
    # 2. 使用mjpython启动viewer
    viewer_process = start_viewer_with_mjpython()
    if not viewer_process:
        print("\n❌ 无法启动Viewer Server")
        return 1
    
    # 等待启动
    time.sleep(3)
    
    # 3. 测试连接
    if not test_connection():
        print("\n❌ 连接测试失败")
        viewer_process.terminate()
        return 1
    
    # 4. 更新配置
    update_claude_config_for_macos()
    
    print("\n" + "="*50)
    print("🎉 修复完成！")
    print("\n📋 接下来：")
    print("1. 重启Claude Desktop")
    print("2. 测试创建场景: 'Create a pendulum simulation'")
    print("\n⏹️  按 Ctrl+C 停止Viewer Server")
    
    try:
        # 保持运行
        while True:
            time.sleep(1)
            if viewer_process.poll() is not None:
                print("\n⚠️  Viewer Server进程已退出")
                break
    except KeyboardInterrupt:
        print("\n👋 停止Viewer Server...")
        viewer_process.terminate()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
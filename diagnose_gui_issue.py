#!/usr/bin/env python3
"""
诊断MuJoCo GUI显示问题
"""

import os
import sys
import subprocess
import socket
import json

def check_mjpython():
    """检查mjpython"""
    print("1️⃣ 检查mjpython...")
    result = subprocess.run(['which', 'mjpython'], capture_output=True, text=True)
    if result.returncode == 0:
        mjpython_path = result.stdout.strip()
        print(f"   ✅ mjpython路径: {mjpython_path}")
        
        # 检查版本
        version_result = subprocess.run([mjpython_path, '-c', 'import mujoco; print(mujoco.__version__)'], 
                                      capture_output=True, text=True)
        if version_result.returncode == 0:
            print(f"   ✅ MuJoCo版本: {version_result.stdout.strip()}")
        return True
    else:
        print("   ❌ 找不到mjpython")
        return False

def check_viewer_server():
    """检查viewer server"""
    print("\n2️⃣ 检查Viewer Server...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('localhost', 8888))
        sock.close()
        
        if result == 0:
            print("   ✅ Viewer Server运行在端口8888")
            return True
        else:
            print("   ❌ Viewer Server未运行")
            return False
    except:
        print("   ❌ 无法检查Viewer Server")
        return False

def test_viewer_connection():
    """测试viewer连接"""
    print("\n3️⃣ 测试Viewer连接...")
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
            print("   ✅ Viewer响应正常")
            print(f"   详情: {result}")
            return True
        else:
            print("   ❌ Viewer响应异常")
            return False
    except Exception as e:
        print(f"   ❌ 连接测试失败: {e}")
        return False

def check_display_env():
    """检查显示环境"""
    print("\n4️⃣ 检查显示环境...")
    
    # 检查DISPLAY变量（如果通过SSH）
    display = os.environ.get('DISPLAY', 'Not set')
    print(f"   DISPLAY: {display}")
    
    # 检查MUJOCO_GL
    mujoco_gl = os.environ.get('MUJOCO_GL', 'Not set')
    print(f"   MUJOCO_GL: {mujoco_gl}")
    
    # 检查是否在SSH会话中
    ssh_connection = os.environ.get('SSH_CONNECTION', None)
    if ssh_connection:
        print("   ⚠️  检测到SSH连接，可能无法显示GUI")
    else:
        print("   ✅ 本地会话，应该可以显示GUI")

def check_processes():
    """检查相关进程"""
    print("\n5️⃣ 检查运行的进程...")
    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
    processes = result.stdout.splitlines()
    
    mujoco_processes = [p for p in processes if 'mujoco' in p.lower() and 'grep' not in p]
    mjpython_processes = [p for p in processes if 'mjpython' in p and 'grep' not in p]
    
    if mujoco_processes:
        print(f"   找到{len(mujoco_processes)}个MuJoCo进程")
        for p in mujoco_processes[:3]:  # 只显示前3个
            print(f"   - {p[:100]}...")
    
    if mjpython_processes:
        print(f"   找到{len(mjpython_processes)}个mjpython进程")

def main():
    print("🔍 MuJoCo GUI诊断工具")
    print("="*50)
    
    # 运行所有检查
    mjpython_ok = check_mjpython()
    viewer_ok = check_viewer_server()
    
    if viewer_ok:
        test_viewer_connection()
    
    check_display_env()
    check_processes()
    
    # 诊断结果
    print("\n" + "="*50)
    print("📊 诊断结果：")
    
    if not mjpython_ok:
        print("\n❌ 问题: mjpython未安装")
        print("   解决: pip install mujoco")
    
    if not viewer_ok:
        print("\n❌ 问题: Viewer Server未运行")
        print("   解决: mjpython mujoco_viewer_server.py --port 8888")
    
    if mjpython_ok and viewer_ok:
        print("\n✅ 基础环境正常")
        print("\n🔧 尝试以下步骤：")
        print("1. 重启Claude Desktop")
        print("2. 在Claude中运行: Create a pendulum simulation")
        print("3. 检查是否有窗口弹出")
        print("\n如果仍无GUI，运行：")
        print("   mjpython test_gui_direct.py")

if __name__ == "__main__":
    main()
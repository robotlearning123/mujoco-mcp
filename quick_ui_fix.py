#!/usr/bin/env python3
"""
快速修复MuJoCo UI显示问题
确保Claude Desktop可以实时控制MuJoCo机器人
"""

import os
import sys
import subprocess
import time
import signal

def kill_all_mujoco_processes():
    """清理所有MuJoCo相关进程"""
    print("🧹 清理现有进程...")
    subprocess.run(['pkill', '-f', 'mujoco_viewer_server'], capture_output=True)
    subprocess.run(['pkill', '-f', 'mjpython'], capture_output=True)
    time.sleep(1)

def start_mujoco_ui():
    """启动MuJoCo UI界面"""
    print("\n🚀 启动MuJoCo UI界面...")
    
    # 确认mjpython路径
    mjpython = subprocess.run(['which', 'mjpython'], 
                            capture_output=True, text=True).stdout.strip()
    
    if not mjpython:
        print("❌ 错误：找不到mjpython")
        print("   请运行: pip install mujoco")
        return False
    
    print(f"✅ 找到mjpython: {mjpython}")
    
    # 启动viewer server
    cmd = [mjpython, 'mujoco_viewer_server.py', '--port', '8888']
    print(f"📺 启动命令: {' '.join(cmd)}")
    
    process = subprocess.Popen(cmd)
    print(f"✅ MuJoCo Viewer已启动 (PID: {process.pid})")
    
    return process

def test_connection():
    """测试连接"""
    print("\n🔗 测试连接...")
    time.sleep(3)  # 等待启动
    
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        sock.connect(('localhost', 8888))
        sock.close()
        print("✅ 连接成功！")
        return True
    except:
        print("❌ 连接失败")
        return False

def main():
    print("🎯 MuJoCo UI 快速修复工具")
    print("目标：让Claude Desktop能实时控制MuJoCo机器人")
    print("="*50)
    
    # 1. 清理
    kill_all_mujoco_processes()
    
    # 2. 启动UI
    viewer_process = start_mujoco_ui()
    if not viewer_process:
        return 1
    
    # 3. 测试
    if not test_connection():
        print("\n⚠️  连接测试失败，但UI可能已启动")
        print("   请检查是否有MuJoCo窗口弹出")
    
    print("\n" + "="*50)
    print("📋 现在请执行以下步骤：")
    print("\n1. 重启Claude Desktop (Cmd+Q 然后重新打开)")
    print("\n2. 在Claude中测试命令：")
    print("   - 'Create a pendulum simulation'")
    print("   - 'Create a robotic arm scene'") 
    print("\n3. 你应该看到：")
    print("   - MuJoCo GUI窗口自动弹出")
    print("   - 机器人在窗口中实时运动")
    print("   - 可以用鼠标拖动视角")
    print("\n⏹️  按 Ctrl+C 停止")
    
    try:
        viewer_process.wait()
    except KeyboardInterrupt:
        print("\n\n👋 正在停止...")
        viewer_process.terminate()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
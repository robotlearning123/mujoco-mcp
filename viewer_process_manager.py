#!/usr/bin/env python3
"""
MuJoCo Viewer 进程管理器
提供自动启动、健康检查和故障恢复功能
"""

import os
import sys
import subprocess
import socket
import json
import time
import threading
import atexit
from typing import Optional, Dict, Any

class ViewerProcessManager:
    """Viewer进程管理器"""
    
    def __init__(self, port: int = 8888, auto_restart: bool = True):
        self.port = port
        self.auto_restart = auto_restart
        self.process: Optional[subprocess.Popen] = None
        self.running = False
        self.health_check_thread = None
        self.restart_count = 0
        self.max_restarts = 5
        
        # 注册退出清理
        atexit.register(self.cleanup)
    
    def is_port_open(self) -> bool:
        """检查端口是否开放"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', self.port))
            sock.close()
            return result == 0
        except:
            return False
    
    def ping_viewer(self) -> bool:
        """Ping viewer服务器"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            sock.connect(('localhost', self.port))
            
            # 发送ping
            ping_cmd = json.dumps({"type": "ping"})
            sock.send(ping_cmd.encode('utf-8'))
            
            # 接收响应
            response = sock.recv(1024)
            result = json.loads(response.decode('utf-8'))
            sock.close()
            
            return result.get("success", False) and result.get("pong", False)
        except:
            return False
    
    def start_viewer(self) -> bool:
        """启动viewer进程"""
        if self.process and self.process.poll() is None:
            print(f"Viewer already running (PID: {self.process.pid})")
            return True
        
        try:
            # 杀死占用端口的进程
            if self.is_port_open():
                self._kill_port_process()
                time.sleep(1)
            
            # 启动新进程
            cmd = [sys.executable, 'mujoco_viewer_server.py', '--port', str(self.port)]
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 等待启动
            for _ in range(10):
                if self.is_port_open():
                    time.sleep(0.5)  # 额外等待
                    if self.ping_viewer():
                        print(f"✅ Viewer started successfully (PID: {self.process.pid})")
                        self.running = True
                        self.restart_count = 0
                        return True
                time.sleep(1)
            
            # 启动失败
            self.stop_viewer()
            return False
            
        except Exception as e:
            print(f"❌ Failed to start viewer: {e}")
            return False
    
    def stop_viewer(self):
        """停止viewer进程"""
        self.running = False
        
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except:
                try:
                    self.process.kill()
                except:
                    pass
            self.process = None
    
    def _kill_port_process(self):
        """杀死占用端口的进程"""
        try:
            result = subprocess.run(
                ['lsof', '-ti', f':{self.port}'],
                capture_output=True,
                text=True
            )
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    subprocess.run(['kill', '-9', pid])
        except:
            pass
    
    def health_check_loop(self):
        """健康检查循环"""
        while self.running:
            time.sleep(5)  # 每5秒检查一次
            
            if not self.ping_viewer():
                print(f"⚠️  Viewer health check failed")
                
                if self.auto_restart and self.restart_count < self.max_restarts:
                    print(f"🔄 Attempting restart ({self.restart_count + 1}/{self.max_restarts})")
                    self.restart_count += 1
                    
                    self.stop_viewer()
                    time.sleep(2)
                    
                    if self.start_viewer():
                        print("✅ Viewer restarted successfully")
                    else:
                        print("❌ Failed to restart viewer")
                        self.running = False
                else:
                    print("❌ Max restarts reached or auto-restart disabled")
                    self.running = False
    
    def start_with_health_check(self) -> bool:
        """启动viewer并开始健康检查"""
        if not self.start_viewer():
            return False
        
        # 启动健康检查线程
        self.health_check_thread = threading.Thread(
            target=self.health_check_loop,
            daemon=True
        )
        self.health_check_thread.start()
        
        return True
    
    def cleanup(self):
        """清理资源"""
        self.stop_viewer()

# 全局实例
_viewer_manager: Optional[ViewerProcessManager] = None

def get_viewer_manager(port: int = 8888) -> ViewerProcessManager:
    """获取viewer管理器单例"""
    global _viewer_manager
    if _viewer_manager is None:
        _viewer_manager = ViewerProcessManager(port)
    return _viewer_manager

def ensure_viewer_running(port: int = 8888) -> bool:
    """确保viewer正在运行"""
    manager = get_viewer_manager(port)
    
    # 先检查是否已经在运行
    if manager.ping_viewer():
        return True
    
    # 尝试启动
    return manager.start_with_health_check()

if __name__ == "__main__":
    # 测试脚本
    print("🧪 Testing Viewer Process Manager")
    
    manager = ViewerProcessManager()
    
    if manager.start_with_health_check():
        print("✅ Viewer started with health monitoring")
        print("Press Ctrl+C to stop...")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 Stopping...")
    else:
        print("❌ Failed to start viewer")
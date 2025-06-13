"""
MuJoCo Viewer Client - Socket通信模块
连接到独立的MuJoCo Viewer Server进程
增强版：包含自动重连、健康检查和进程管理
"""

import json
import socket
import logging
import time
import subprocess
import sys
import os
from typing import Dict, Any, Optional

logger = logging.getLogger("mujoco_mcp.viewer_client")

class MuJoCoViewerClient:
    """连接到MuJoCo Viewer Server的客户端 - 增强版"""
    
    def __init__(self, host: str = 'localhost', port: int = 8888):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        self.auto_start = True  # 自动启动viewer server
        self.reconnect_attempts = 3
        self.reconnect_delay = 2.0
    
    def connect(self) -> bool:
        """连接到MuJoCo Viewer Server - 支持自动启动和重试"""
        for attempt in range(self.reconnect_attempts):
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.settimeout(15.0)  # Increased timeout for model replacement
                self.socket.connect((self.host, self.port))
                self.connected = True
                logger.info(f"Connected to MuJoCo Viewer Server at {self.host}:{self.port}")
                return True
            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                
                # 第一次失败后尝试启动viewer server
                if attempt == 0 and self.auto_start:
                    logger.info("Attempting to start MuJoCo Viewer Server...")
                    if self._start_viewer_server():
                        time.sleep(3)  # 等待服务器启动
                        continue
                
                if attempt < self.reconnect_attempts - 1:
                    time.sleep(self.reconnect_delay)
        
        logger.error(f"Failed to connect after {self.reconnect_attempts} attempts")
        self.connected = False
        return False
    
    def disconnect(self):
        """断开连接"""
        if self.socket:
            self.socket.close()
            self.socket = None
        self.connected = False
        logger.info("Disconnected from MuJoCo Viewer Server")
    
    def send_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """发送命令到viewer server并获取响应"""
        if not self.connected or not self.socket:
            return {"success": False, "error": "Not connected to viewer server"}
        
        try:
            # 发送命令
            command_json = json.dumps(command)
            self.socket.send(command_json.encode('utf-8'))
            
            # 接收响应 - 支持更大的消息
            response_data = b""
            while True:
                chunk = self.socket.recv(8192)
                if not chunk:
                    break
                response_data += chunk
                
                # 检查是否收到完整的JSON (以换行符结束)
                if response_data.endswith(b'\n'):
                    break
                
                # 防止无限等待
                if len(response_data) > 1024 * 1024:  # 1MB限制
                    raise ValueError("Response too large")
            
            response = json.loads(response_data.decode('utf-8').strip())
            
            return response
        except Exception as e:
            logger.error(f"Failed to send command: {e}")
            return {"success": False, "error": str(e)}
    
    def ping(self) -> bool:
        """测试连接是否正常 - 增强版"""
        if not self.connected:
            # 尝试重连
            if not self.connect():
                return False
        
        try:
            response = self.send_command({"type": "ping"})
            return response.get("success", False)
        except:
            # 连接可能已断开，尝试重连
            self.connected = False
            if self.connect():
                response = self.send_command({"type": "ping"})
                return response.get("success", False)
            return False
    
    def load_model(self, model_source: str, model_id: str = None) -> Dict[str, Any]:
        """加载MuJoCo模型到viewer
        
        Args:
            model_source: XML字符串或XML文件路径
            model_id: 模型ID
        """
        cmd = {
            "type": "load_model",
            "model_xml": model_source  # 保持向后兼容，但实际可以是文件路径
        }
        if model_id:
            cmd["model_id"] = model_id
        return self.send_command(cmd)
    
    def replace_model(self, model_source: str, model_id: str = None) -> Dict[str, Any]:
        """替换当前模型（关闭现有viewer并加载新模型）
        
        Args:
            model_source: XML字符串或XML文件路径
            model_id: 模型ID
        """
        cmd = {
            "type": "replace_model",
            "model_xml": model_source  # 保持向后兼容，但实际可以是文件路径
        }
        if model_id:
            cmd["model_id"] = model_id
        return self.send_command(cmd)
    
    def start_viewer(self) -> Dict[str, Any]:
        """启动viewer GUI"""
        return self.send_command({"type": "start_viewer"})
    
    def get_state(self, model_id: str = None) -> Dict[str, Any]:
        """获取仿真状态"""
        cmd = {"type": "get_state"}
        if model_id:
            cmd["model_id"] = model_id
        return self.send_command(cmd)
    
    def set_control(self, control: list) -> Dict[str, Any]:
        """设置控制输入"""
        return self.send_command({
            "type": "set_control",
            "control": control
        })
    
    def set_joint_positions(self, positions: list, model_id: str = None) -> Dict[str, Any]:
        """设置关节位置"""
        cmd = {
            "type": "set_joint_positions",
            "positions": positions
        }
        if model_id:
            cmd["model_id"] = model_id
        return self.send_command(cmd)
    
    def reset_simulation(self, model_id: str = None) -> Dict[str, Any]:
        """重置仿真"""
        cmd = {"type": "reset"}
        if model_id:
            cmd["model_id"] = model_id
        return self.send_command(cmd)
    
    def close_viewer(self) -> Dict[str, Any]:
        """关闭viewer GUI窗口"""
        return self.send_command({"type": "close_viewer"})
    
    def shutdown_server(self) -> Dict[str, Any]:
        """关闭整个viewer服务器"""
        return self.send_command({"type": "shutdown_server"})
    
    def capture_render(self, model_id: str = None, width: int = 640, height: int = 480) -> Dict[str, Any]:
        """捕获当前渲染的图像"""
        cmd = {
            "type": "capture_render",
            "width": width,
            "height": height
        }
        if model_id:
            cmd["model_id"] = model_id
        return self.send_command(cmd)
    
    def _start_viewer_server(self) -> bool:
        """尝试启动MuJoCo Viewer Server - 支持macOS mjpython"""
        try:
            # 查找viewer server脚本
            script_paths = [
                'mujoco_viewer_server.py',
                os.path.join(os.path.dirname(__file__), '..', '..', 'mujoco_viewer_server.py'),
                os.path.join(os.getcwd(), 'mujoco_viewer_server.py')
            ]
            
            viewer_script = None
            for path in script_paths:
                if os.path.exists(path):
                    viewer_script = os.path.abspath(path)
                    break
            
            if not viewer_script:
                logger.error("Could not find mujoco_viewer_server.py")
                return False
            
            # 检查是否需要使用mjpython (macOS)
            python_executable = sys.executable
            if sys.platform == 'darwin':  # macOS
                # 尝试找mjpython
                mjpython_result = subprocess.run(['which', 'mjpython'], 
                                               capture_output=True, text=True)
                if mjpython_result.returncode == 0:
                    mjpython_path = mjpython_result.stdout.strip()
                    if mjpython_path:
                        python_executable = mjpython_path
                        logger.info(f"Using mjpython for macOS: {mjpython_path}")
                else:
                    logger.warning("mjpython not found on macOS, viewer may not work properly")
            
            # 启动进程
            cmd = [python_executable, viewer_script, '--port', str(self.port)]
            logger.info(f"Starting viewer with command: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True  # 独立进程组
            )
            
            logger.info(f"Started MuJoCo Viewer Server (PID: {process.pid})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start viewer server: {e}")
            return False
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """获取连接诊断信息"""
        diagnostics = {
            "host": self.host,
            "port": self.port,
            "connected": self.connected,
            "socket_alive": self.socket is not None,
            "ping_result": False,
            "viewer_process": self._check_viewer_process()
        }
        
        if self.connected:
            diagnostics["ping_result"] = self.ping()
        
        return diagnostics
    
    def _check_viewer_process(self) -> bool:
        """检查viewer进程是否运行"""
        try:
            # 使用lsof检查端口
            result = subprocess.run(
                ['lsof', '-ti', f':{self.port}'],
                capture_output=True,
                text=True
            )
            return bool(result.stdout.strip())
        except:
            return False

class ViewerManager:
    """管理多个viewer客户端连接"""
    
    def __init__(self):
        self.clients = {}  # model_id -> ViewerClient
        self.default_port = 8888
    
    def create_client(self, model_id: str, port: Optional[int] = None) -> bool:
        """为特定模型创建viewer客户端"""
        if port is None:
            port = self.default_port
        
        client = MuJoCoViewerClient(port=port)
        if client.connect():
            self.clients[model_id] = client
            logger.info(f"Created viewer client for model {model_id}")
            return True
        else:
            logger.error(f"Failed to create viewer client for model {model_id}")
            return False
    
    def get_client(self, model_id: str) -> Optional[MuJoCoViewerClient]:
        """获取指定模型的viewer客户端"""
        return self.clients.get(model_id)
    
    def remove_client(self, model_id: str):
        """移除viewer客户端"""
        if model_id in self.clients:
            self.clients[model_id].disconnect()
            del self.clients[model_id]
            logger.info(f"Removed viewer client for model {model_id}")
    
    def disconnect_all(self):
        """断开所有连接"""
        for model_id in list(self.clients.keys()):
            self.remove_client(model_id)

# 全局viewer管理器实例
viewer_manager = ViewerManager()

# 诊断信息获取函数
def get_system_diagnostics() -> Dict[str, Any]:
    """获取系统诊断信息"""
    diagnostics = {
        "viewer_manager": {
            "active_clients": len(viewer_manager.clients),
            "client_ids": list(viewer_manager.clients.keys()),
            "default_port": viewer_manager.default_port
        },
        "clients": {}
    }
    
    for model_id, client in viewer_manager.clients.items():
        diagnostics["clients"][model_id] = client.get_diagnostics()
    
    return diagnostics

def get_viewer_client(model_id: str) -> Optional[MuJoCoViewerClient]:
    """获取指定模型的viewer客户端的便捷函数"""
    return viewer_manager.get_client(model_id)

def ensure_viewer_connection(model_id: str) -> bool:
    """确保viewer连接存在的便捷函数 - 增强版"""
    client = viewer_manager.get_client(model_id)
    if client and client.connected and client.ping():
        return True
    
    # 如果连接不存在或已断开，尝试重新连接
    logger.info(f"Creating new viewer connection for model {model_id}")
    
    # 多次尝试
    for attempt in range(3):
        if viewer_manager.create_client(model_id):
            return True
        logger.warning(f"Connection attempt {attempt + 1} failed for model {model_id}")
        if attempt < 2:
            time.sleep(2)
    
    # 最后提供详细诊断
    client = viewer_manager.get_client(model_id)
    if client:
        diagnostics = client.get_diagnostics()
        logger.error(f"Connection diagnostics: {diagnostics}")
    
    return False
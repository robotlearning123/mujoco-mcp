"""
MuJoCo Viewer Client - Socket通信模块
连接到独立的MuJoCo Viewer Server进程
"""

import json
import socket
import logging
import time
from typing import Dict, Any, Optional

logger = logging.getLogger("mujoco_mcp.viewer_client")

class MuJoCoViewerClient:
    """连接到MuJoCo Viewer Server的客户端"""
    
    def __init__(self, host: str = 'localhost', port: int = 8888):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
    
    def connect(self) -> bool:
        """连接到MuJoCo Viewer Server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
            logger.info(f"Connected to MuJoCo Viewer Server at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to viewer server: {e}")
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
            
            # 接收响应
            response_data = self.socket.recv(4096)
            response = json.loads(response_data.decode('utf-8').strip())
            
            return response
        except Exception as e:
            logger.error(f"Failed to send command: {e}")
            return {"success": False, "error": str(e)}
    
    def ping(self) -> bool:
        """测试连接是否正常"""
        response = self.send_command({"type": "ping"})
        return response.get("success", False)
    
    def load_model(self, model_xml: str) -> Dict[str, Any]:
        """加载MuJoCo模型到viewer"""
        return self.send_command({
            "type": "load_model",
            "model_xml": model_xml
        })
    
    def start_viewer(self) -> Dict[str, Any]:
        """启动viewer GUI"""
        return self.send_command({"type": "start_viewer"})
    
    def get_state(self) -> Dict[str, Any]:
        """获取仿真状态"""
        return self.send_command({"type": "get_state"})
    
    def set_control(self, control: list) -> Dict[str, Any]:
        """设置控制输入"""
        return self.send_command({
            "type": "set_control",
            "control": control
        })
    
    def set_joint_positions(self, positions: list) -> Dict[str, Any]:
        """设置关节位置"""
        return self.send_command({
            "type": "set_joint_positions",
            "positions": positions
        })
    
    def reset_simulation(self) -> Dict[str, Any]:
        """重置仿真"""
        return self.send_command({"type": "reset"})

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

def get_viewer_client(model_id: str) -> Optional[MuJoCoViewerClient]:
    """获取指定模型的viewer客户端的便捷函数"""
    return viewer_manager.get_client(model_id)

def ensure_viewer_connection(model_id: str) -> bool:
    """确保viewer连接存在的便捷函数"""
    client = viewer_manager.get_client(model_id)
    if client and client.connected and client.ping():
        return True
    
    # 如果连接不存在或已断开，尝试重新连接
    logger.info(f"Creating new viewer connection for model {model_id}")
    return viewer_manager.create_client(model_id)
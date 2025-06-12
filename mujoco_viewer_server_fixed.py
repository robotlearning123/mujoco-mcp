#!/usr/bin/env mjpython
"""
MuJoCo Viewer Server - 增强版
支持多模型并发管理
使用官方 mujoco.viewer.launch_passive() API
通过Socket与MCP服务器通信

修复的问题:
1. 支持多个并发连接
2. 增加接收缓冲区大小
3. 改进错误处理和超时管理
4. 支持多个模型的独立管理
"""

import time
import json
import socket
import threading
import logging
import sys
import os
from typing import Dict, Any, Optional
import uuid

import mujoco
import mujoco.viewer
import numpy as np

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mujoco_viewer_server")

class ModelViewer:
    """单个模型的viewer管理器"""
    def __init__(self, model_id: str, model_xml: str):
        self.model_id = model_id
        self.model = None
        self.data = None
        self.viewer = None
        self.simulation_running = False
        self.created_time = time.time()
        
        # 加载模型
        self.model = mujoco.MjModel.from_xml_string(model_xml)
        self.data = mujoco.MjData(self.model)
        
        # 启动viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        # 启动仿真循环
        self.simulation_running = True
        self.sim_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.sim_thread.start()
        
        logger.info(f"Created ModelViewer for {model_id}")
    
    def _simulation_loop(self):
        """仿真循环"""
        while self.simulation_running and self.viewer and self.viewer.is_running():
            with self.viewer.lock():
                mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            time.sleep(0.002)  # ~500Hz
    
    def get_state(self) -> Dict[str, Any]:
        """获取状态"""
        with self.viewer.lock():
            state = {
                "time": self.data.time,
                "qpos": self.data.qpos.tolist(),
                "qvel": self.data.qvel.tolist(),
                "ctrl": self.data.ctrl.tolist()
            }
        return state
    
    def set_joint_positions(self, positions: list) -> bool:
        """设置关节位置"""
        with self.viewer.lock():
            for i, pos in enumerate(positions[:self.model.nq]):
                self.data.qpos[i] = pos
            mujoco.mj_forward(self.model, self.data)
        return True
    
    def reset(self):
        """重置仿真"""
        with self.viewer.lock():
            mujoco.mj_resetData(self.model, self.data)
        
    def close(self):
        """关闭viewer"""
        self.simulation_running = False
        if self.viewer:
            # viewer会自动清理
            self.viewer = None

class MuJoCoViewerServer:
    """增强版MuJoCo Viewer服务器 - 支持多模型"""
    
    def __init__(self, port: int = 8888):
        self.port = port
        self.running = False
        self.socket_server = None
        
        # 模型管理器 - 支持多个模型
        self.models: Dict[str, ModelViewer] = {}
        self.models_lock = threading.Lock()
        
        # 客户端管理
        self.client_threads = []
        
    def handle_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """处理命令 - 支持多模型"""
        cmd_type = command.get("type")
        
        try:
            if cmd_type == "load_model":
                model_id = command.get("model_id", str(uuid.uuid4()))
                model_xml = command.get("model_xml")
                
                with self.models_lock:
                    # 如果模型已存在，先关闭
                    if model_id in self.models:
                        self.models[model_id].close()
                        del self.models[model_id]
                    
                    # 创建新模型
                    self.models[model_id] = ModelViewer(model_id, model_xml)
                
                return {
                    "success": True,
                    "model_id": model_id,
                    "model_info": {
                        "nq": self.models[model_id].model.nq,
                        "nv": self.models[model_id].model.nv,
                        "nbody": self.models[model_id].model.nbody
                    }
                }
                
            elif cmd_type == "start_viewer":
                # 兼容旧版本，但viewer已经在load_model时启动
                return {"success": True, "message": "Viewer already started"}
                
            elif cmd_type == "get_state":
                model_id = command.get("model_id")
                if model_id not in self.models:
                    return {"success": False, "error": f"Model {model_id} not found"}
                
                state = self.models[model_id].get_state()
                return {"success": True, **state}
                
            elif cmd_type == "set_joint_positions":
                model_id = command.get("model_id")
                positions = command.get("positions", [])
                
                if model_id not in self.models:
                    return {"success": False, "error": f"Model {model_id} not found"}
                
                self.models[model_id].set_joint_positions(positions)
                return {"success": True, "positions_set": positions}
                
            elif cmd_type == "reset":
                model_id = command.get("model_id")
                if model_id not in self.models:
                    return {"success": False, "error": f"Model {model_id} not found"}
                
                self.models[model_id].reset()
                return {"success": True}
                
            elif cmd_type == "close_model":
                model_id = command.get("model_id")
                with self.models_lock:
                    if model_id in self.models:
                        self.models[model_id].close()
                        del self.models[model_id]
                return {"success": True}
                
            elif cmd_type == "list_models":
                models_info = {}
                with self.models_lock:
                    for model_id, viewer in self.models.items():
                        models_info[model_id] = {
                            "created_time": viewer.created_time,
                            "viewer_running": viewer.viewer and viewer.viewer.is_running()
                        }
                return {"success": True, "models": models_info}
                
            elif cmd_type == "ping":
                return {
                    "success": True, 
                    "pong": True,
                    "models_count": len(self.models),
                    "server_running": self.running
                }
                
            else:
                return {"success": False, "error": f"Unknown command: {cmd_type}"}
                
        except Exception as e:
            logger.error(f"Error handling command {cmd_type}: {e}")
            return {"success": False, "error": str(e)}
    
    def handle_client(self, client_socket: socket.socket, address):
        """处理单个客户端连接 - 在独立线程中"""
        logger.info(f"Client connected from {address}")
        
        # 设置更大的接收缓冲区
        client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
        
        try:
            while self.running:
                # 接收数据 - 支持大消息
                data = b""
                while True:
                    chunk = client_socket.recv(8192)
                    if not chunk:
                        if data:
                            break
                        else:
                            # 连接关闭
                            return
                    data += chunk
                    
                    # 检查是否收到完整的JSON
                    try:
                        json.loads(data.decode('utf-8'))
                        break
                    except:
                        # 继续接收
                        if len(data) > 1024 * 1024:  # 1MB限制
                            raise ValueError("Message too large")
                        continue
                
                # 解析命令
                command = json.loads(data.decode('utf-8'))
                logger.debug(f"Received command: {command.get('type', 'unknown')}")
                
                # 处理命令
                response = self.handle_command(command)
                
                # 发送响应
                response_json = json.dumps(response) + '\n'
                client_socket.send(response_json.encode('utf-8'))
                
        except Exception as e:
            logger.error(f"Error handling client {address}: {e}")
            try:
                error_response = {"success": False, "error": str(e)}
                client_socket.send(json.dumps(error_response).encode('utf-8'))
            except:
                pass
        finally:
            client_socket.close()
            logger.info(f"Client {address} disconnected")
    
    def start_socket_server(self):
        """启动Socket服务器 - 支持多连接"""
        self.socket_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        self.socket_server.bind(('localhost', self.port))
        self.socket_server.listen(10)  # 支持多个连接
        logger.info(f"MuJoCo Viewer Server listening on port {self.port}")
        
        while self.running:
            try:
                client_socket, address = self.socket_server.accept()
                
                # 为每个客户端创建独立线程
                client_thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_socket, address),
                    daemon=True
                )
                client_thread.start()
                self.client_threads.append(client_thread)
                
            except Exception as e:
                if self.running:
                    logger.error(f"Error accepting connection: {e}")
    
    def start(self):
        """启动服务器"""
        self.running = True
        logger.info("Starting Enhanced MuJoCo Viewer Server...")
        
        try:
            self.start_socket_server()
        except KeyboardInterrupt:
            logger.info("Server interrupted by user")
        finally:
            self.stop()
    
    def stop(self):
        """停止服务器"""
        logger.info("Stopping MuJoCo Viewer Server...")
        self.running = False
        
        # 关闭所有模型
        with self.models_lock:
            for model_id in list(self.models.keys()):
                self.models[model_id].close()
            self.models.clear()
        
        # 关闭socket
        if self.socket_server:
            try:
                self.socket_server.close()
            except:
                pass
        
        logger.info("Server stopped")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced MuJoCo Viewer Server")
    parser.add_argument("--port", type=int, default=8888, help="Socket server port")
    args = parser.parse_args()
    
    server = MuJoCoViewerServer(port=args.port)
    server.start()

if __name__ == "__main__":
    main()
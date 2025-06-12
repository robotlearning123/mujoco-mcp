#!/usr/bin/env python3
"""
MuJoCo Viewer Server - 独立GUI进程
使用官方 mujoco.viewer.launch_passive() API
通过Socket与MCP服务器通信
"""

import time
import json
import socket
import threading
import logging
import sys
import os
from typing import Dict, Any, Optional

import mujoco
import mujoco.viewer
import numpy as np

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mujoco_viewer_server")

class MuJoCoViewerServer:
    """MuJoCo Viewer服务器，使用官方launch_passive API"""
    
    def __init__(self, port: int = 8888):
        self.port = port
        self.model = None
        self.data = None
        self.viewer = None
        self.running = False
        self.socket_server = None
        self.client_socket = None
        
        # 仿真参数
        self.simulation_running = False
        self.target_fps = 60
        self.timestep = 1.0 / self.target_fps
    
    def load_model(self, model_xml: str) -> Dict[str, Any]:
        """加载MuJoCo模型"""
        try:
            self.model = mujoco.MjModel.from_xml_string(model_xml)
            self.data = mujoco.MjData(self.model)
            logger.info(f"Model loaded successfully: {self.model.nq} DOF")
            
            return {
                "success": True,
                "model_info": {
                    "nq": self.model.nq,
                    "nv": self.model.nv,
                    "nbody": self.model.nbody,
                    "njnt": self.model.njnt,
                    "timestep": self.model.opt.timestep
                }
            }
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return {"success": False, "error": str(e)}
    
    def start_viewer(self) -> Dict[str, Any]:
        """启动被动模式viewer"""
        if not self.model or not self.data:
            return {"success": False, "error": "No model loaded"}
        
        try:
            # 使用官方launch_passive API
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            logger.info("MuJoCo viewer launched in passive mode")
            
            # 启动仿真循环
            self.simulation_running = True
            threading.Thread(target=self._simulation_loop, daemon=True).start()
            
            return {
                "success": True,
                "viewer_info": {
                    "mode": "passive",
                    "thread_safe": True,
                    "fps_target": self.target_fps
                }
            }
        except Exception as e:
            logger.error(f"Failed to start viewer: {e}")
            return {"success": False, "error": str(e)}
    
    def _simulation_loop(self):
        """仿真主循环"""
        last_time = time.time()
        
        while self.simulation_running and self.viewer and self.viewer.is_running():
            loop_start = time.time()
            
            # 线程安全的仿真步进
            with self.viewer.lock():
                mujoco.mj_step(self.model, self.data)
                
                # 可以在这里修改viewer选项
                # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
            
            # 同步状态到viewer
            self.viewer.sync()
            
            # 控制帧率
            elapsed = time.time() - loop_start
            sleep_time = self.timestep - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            current_time = time.time()
            if current_time - last_time > 1.0:  # 每秒更新一次状态
                last_time = current_time
        
        logger.info("Simulation loop ended")
        self.simulation_running = False
    
    def get_state(self) -> Dict[str, Any]:
        """获取当前仿真状态"""
        if not self.data:
            return {"success": False, "error": "No simulation data"}
        
        try:
            with self.viewer.lock() if self.viewer else threading.Lock():
                state = {
                    "success": True,
                    "time": self.data.time,
                    "qpos": self.data.qpos.tolist(),
                    "qvel": self.data.qvel.tolist(),
                    "qacc": self.data.qacc.tolist() if hasattr(self.data, 'qacc') else [],
                    "ctrl": self.data.ctrl.tolist() if self.model.nu > 0 else [],
                    "energy": {
                        "kinetic": float(np.sum(0.5 * self.data.qvel * (self.data.qM @ self.data.qvel))),
                        "potential": float(self.data.energy[1]) if hasattr(self.data, 'energy') else 0.0
                    }
                }
            
            return state
        except Exception as e:
            logger.error(f"Failed to get state: {e}")
            return {"success": False, "error": str(e)}
    
    def set_control(self, control: list) -> Dict[str, Any]:
        """设置控制输入"""
        if not self.data:
            return {"success": False, "error": "No simulation data"}
        
        try:
            with self.viewer.lock() if self.viewer else threading.Lock():
                if len(control) != self.model.nu:
                    return {"success": False, "error": f"Control size mismatch: expected {self.model.nu}, got {len(control)}"}
                
                self.data.ctrl[:] = control
            
            return {"success": True, "control_applied": control}
        except Exception as e:
            logger.error(f"Failed to set control: {e}")
            return {"success": False, "error": str(e)}
    
    def set_joint_positions(self, positions: list) -> Dict[str, Any]:
        """设置关节位置"""
        if not self.data:
            return {"success": False, "error": "No simulation data"}
        
        try:
            with self.viewer.lock() if self.viewer else threading.Lock():
                if len(positions) != self.model.nq:
                    return {"success": False, "error": f"Position size mismatch: expected {self.model.nq}, got {len(positions)}"}
                
                self.data.qpos[:] = positions
                self.data.qvel[:] = 0  # 重置速度
                mujoco.mj_forward(self.model, self.data)  # 更新状态
            
            return {"success": True, "positions_set": positions}
        except Exception as e:
            logger.error(f"Failed to set joint positions: {e}")
            return {"success": False, "error": str(e)}
    
    def reset_simulation(self) -> Dict[str, Any]:
        """重置仿真"""
        if not self.model or not self.data:
            return {"success": False, "error": "No simulation loaded"}
        
        try:
            with self.viewer.lock() if self.viewer else threading.Lock():
                mujoco.mj_resetData(self.model, self.data)
            
            logger.info("Simulation reset")
            return {"success": True, "time": self.data.time}
        except Exception as e:
            logger.error(f"Failed to reset simulation: {e}")
            return {"success": False, "error": str(e)}
    
    def handle_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """处理来自MCP服务器的命令"""
        cmd_type = command.get("type")
        
        if cmd_type == "load_model":
            return self.load_model(command.get("model_xml", ""))
        elif cmd_type == "start_viewer":
            return self.start_viewer()
        elif cmd_type == "get_state":
            return self.get_state()
        elif cmd_type == "set_control":
            return self.set_control(command.get("control", []))
        elif cmd_type == "set_joint_positions":
            return self.set_joint_positions(command.get("positions", []))
        elif cmd_type == "reset":
            return self.reset_simulation()
        elif cmd_type == "ping":
            return {"success": True, "pong": True, "viewer_running": self.viewer.is_running() if self.viewer else False}
        else:
            return {"success": False, "error": f"Unknown command: {cmd_type}"}
    
    def start_socket_server(self):
        """启动Socket服务器监听MCP连接"""
        self.socket_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            self.socket_server.bind(('localhost', self.port))
            self.socket_server.listen(1)
            logger.info(f"MuJoCo Viewer Server listening on port {self.port}")
            
            while self.running:
                try:
                    client_socket, address = self.socket_server.accept()
                    logger.info(f"MCP client connected from {address}")
                    self.client_socket = client_socket
                    
                    # 处理客户端请求
                    while self.running:
                        try:
                            data = client_socket.recv(4096)
                            if not data:
                                break
                            
                            # 解析JSON命令
                            command = json.loads(data.decode('utf-8'))
                            logger.info(f"Received command: {command.get('type', 'unknown')}")
                            
                            # 处理命令
                            response = self.handle_command(command)
                            
                            # 发送响应
                            response_json = json.dumps(response) + '\n'
                            client_socket.send(response_json.encode('utf-8'))
                            
                        except json.JSONDecodeError as e:
                            error_response = {"success": False, "error": f"Invalid JSON: {e}"}
                            client_socket.send(json.dumps(error_response).encode('utf-8'))
                        except Exception as e:
                            logger.error(f"Error handling request: {e}")
                            error_response = {"success": False, "error": str(e)}
                            client_socket.send(json.dumps(error_response).encode('utf-8'))
                
                except socket.error as e:
                    if self.running:
                        logger.error(f"Socket error: {e}")
                    break
                finally:
                    if self.client_socket:
                        self.client_socket.close()
                        self.client_socket = None
        
        except Exception as e:
            logger.error(f"Failed to start socket server: {e}")
        finally:
            if self.socket_server:
                self.socket_server.close()
    
    def start(self):
        """启动MuJoCo Viewer服务器"""
        self.running = True
        logger.info("Starting MuJoCo Viewer Server...")
        
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
        self.simulation_running = False
        
        if self.client_socket:
            self.client_socket.close()
        
        if self.socket_server:
            self.socket_server.close()
        
        if self.viewer:
            # viewer会自动清理
            pass

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MuJoCo Viewer Server")
    parser.add_argument("--port", type=int, default=8888, help="Socket server port")
    args = parser.parse_args()
    
    server = MuJoCoViewerServer(port=args.port)
    server.start()

if __name__ == "__main__":
    main()
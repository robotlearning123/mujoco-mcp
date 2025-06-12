#!/usr/bin/env mjpython
"""
MuJoCo Viewer Server - 独立GUI进程
使用官方 mujoco.viewer.launch_passive() API
通过Socket与MCP服务器通信

macOS注意：根据官方文档，必须使用mjpython运行此脚本
官方文档: https://mujoco.readthedocs.io/en/stable/python.html
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
        
        # 线程安全控制
        self._simulation_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._command_lock = threading.Lock()
    
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
        """启动被动模式viewer - 增强版本"""
        if not self.model or not self.data:
            return {"success": False, "error": "No model loaded"}
        
        # 检查是否已经有viewer运行
        if self.viewer and self.viewer.is_running():
            logger.info("Viewer already running, reusing existing instance")
            return {
                "success": True,
                "viewer_info": {
                    "mode": "passive",
                    "thread_safe": True,
                    "fps_target": self.target_fps,
                    "status": "already_running"
                }
            }
        
        try:
            # 使用官方launch_passive API
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            
            # 等待viewer完全初始化
            time.sleep(0.1)
            
            if not self.viewer.is_running():
                raise RuntimeError("Viewer failed to start properly")
            
            logger.info("MuJoCo viewer launched successfully in passive mode")
            
            # 启动仿真循环
            if not self.simulation_running:
                self.simulation_running = True
                threading.Thread(target=self._simulation_loop, daemon=True).start()
                logger.info("Simulation loop started")
            
            return {
                "success": True,
                "viewer_info": {
                    "mode": "passive",
                    "thread_safe": True,
                    "fps_target": self.target_fps,
                    "status": "newly_started",
                    "viewer_running": self.viewer.is_running()
                }
            }
        except Exception as e:
            logger.error(f"Failed to start viewer: {e}")
            self.viewer = None
            return {"success": False, "error": str(e)}
    
    def _simulation_loop(self):
        """仿真主循环 - 优化版本"""
        last_time = time.time()
        frame_count = 0
        
        logger.info(f"Starting simulation loop at {self.target_fps} FPS")
        
        try:
            while self.simulation_running and self.viewer and self.viewer.is_running():
                loop_start = time.time()
                
                # 线程安全的仿真步进
                with self._simulation_lock:
                    if self.viewer and self.viewer.is_running():
                        with self.viewer.lock():
                            # 执行仿真步进
                            mujoco.mj_step(self.model, self.data)
                            
                            # 可选：启用接触点视化
                            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
                        
                        # 同步状态到viewer
                        self.viewer.sync()
                
                frame_count += 1
                
                # 控制帧率（更精确的定时）
                elapsed = time.time() - loop_start
                sleep_time = self.timestep - elapsed
                if sleep_time > 0.001:  # 避免太短的睡眠
                    time.sleep(sleep_time)
                
                # 每秒打印一次性能统计
                current_time = time.time()
                if current_time - last_time >= 1.0:
                    actual_fps = frame_count / (current_time - last_time)
                    logger.debug(f"Simulation FPS: {actual_fps:.1f} (target: {self.target_fps})")
                    last_time = current_time
                    frame_count = 0
                    
        except Exception as e:
            logger.error(f"Simulation loop error: {e}")
        finally:
            logger.info("Simulation loop ended")
            self.simulation_running = False
    
    def get_state(self) -> Dict[str, Any]:
        """获取当前仿真状态 - 线程安全版本"""
        if not self.data:
            return {"success": False, "error": "No simulation data"}
        
        try:
            with self._state_lock:
                # 在获取状态时防止仿真步进的干扰
                if self.viewer and self.viewer.is_running():
                    with self.viewer.lock():
                        state_data = self._extract_state_data()
                else:
                    # Fallback：如果viewer不可用，直接读取数据
                    state_data = self._extract_state_data()
                
                return {"success": True, **state_data}
                
        except Exception as e:
            logger.error(f"Failed to get state: {e}")
            return {"success": False, "error": str(e)}
    
    def _extract_state_data(self) -> Dict[str, Any]:
        """提取状态数据的内部方法"""
        # 计算动能（更安全的方式）
        try:
            # 计算质量矩阵
            M = np.zeros((self.model.nv, self.model.nv))
            mujoco.mj_fullM(self.model, M, self.data.qM)
            kinetic_energy = 0.5 * np.dot(self.data.qvel, np.dot(M, self.data.qvel))
        except:
            kinetic_energy = 0.0
        
        # 计算势能（如果可用）
        try:
            potential_energy = -np.dot(self.data.qpos[:3] if self.model.nq >= 3 else self.data.qpos, 
                                     [0, 0, -9.81][:len(self.data.qpos)])
        except:
            potential_energy = 0.0
        
        return {
            "time": float(self.data.time),
            "qpos": self.data.qpos.tolist(),
            "qvel": self.data.qvel.tolist(),
            "qacc": self.data.qacc.tolist() if hasattr(self.data, 'qacc') else [],
            "ctrl": self.data.ctrl.tolist() if self.model.nu > 0 else [],
            "energy": {
                "kinetic": float(kinetic_energy),
                "potential": float(potential_energy)
            },
            "viewer_running": self.viewer.is_running() if self.viewer else False,
            "simulation_running": self.simulation_running
        }
    
    def set_control(self, control: list) -> Dict[str, Any]:
        """设置控制输入 - 线程安全版本"""
        if not self.data:
            return {"success": False, "error": "No simulation data"}
        
        # 参数验证
        if len(control) != self.model.nu:
            return {"success": False, "error": f"Control size mismatch: expected {self.model.nu}, got {len(control)}"}
        
        try:
            with self._command_lock:
                if self.viewer and self.viewer.is_running():
                    with self.viewer.lock():
                        self.data.ctrl[:] = control
                else:
                    # Fallback：viewer不可用时直接设置
                    self.data.ctrl[:] = control
            
            logger.debug(f"Control applied: {control}")
            return {"success": True, "control_applied": control}
        except Exception as e:
            logger.error(f"Failed to set control: {e}")
            return {"success": False, "error": str(e)}
    
    def set_joint_positions(self, positions: list) -> Dict[str, Any]:
        """设置关节位置 - 线程安全版本"""
        if not self.data:
            return {"success": False, "error": "No simulation data"}
        
        # 参数验证
        if len(positions) != self.model.nq:
            return {"success": False, "error": f"Position size mismatch: expected {self.model.nq}, got {len(positions)}"}
        
        try:
            with self._command_lock:
                if self.viewer and self.viewer.is_running():
                    with self.viewer.lock():
                        self.data.qpos[:] = positions
                        self.data.qvel[:] = 0  # 重置速度
                        mujoco.mj_forward(self.model, self.data)  # 更新状态
                else:
                    # Fallback：viewer不可用时直接设置
                    self.data.qpos[:] = positions
                    self.data.qvel[:] = 0
                    mujoco.mj_forward(self.model, self.data)
            
            logger.debug(f"Joint positions set: {positions}")
            return {"success": True, "positions_set": positions}
        except Exception as e:
            logger.error(f"Failed to set joint positions: {e}")
            return {"success": False, "error": str(e)}
    
    def reset_simulation(self) -> Dict[str, Any]:
        """重置仿真 - 线程安全版本"""
        if not self.model or not self.data:
            return {"success": False, "error": "No simulation loaded"}
        
        try:
            with self._command_lock:
                if self.viewer and self.viewer.is_running():
                    with self.viewer.lock():
                        mujoco.mj_resetData(self.model, self.data)
                else:
                    # Fallback：viewer不可用时直接重置
                    mujoco.mj_resetData(self.model, self.data)
            
            logger.info("Simulation reset to initial state")
            return {"success": True, "time": float(self.data.time), "message": "Simulation reset to initial state"}
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
            viewer_status = {
                "viewer_exists": self.viewer is not None,
                "viewer_running": self.viewer.is_running() if self.viewer else False,
                "simulation_running": self.simulation_running,
                "model_loaded": self.model is not None
            }
            return {"success": True, "pong": True, **viewer_status}
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
        """停止服务器 - 增强清理"""
        logger.info("Stopping MuJoCo Viewer Server...")
        self.running = False
        self.simulation_running = False
        
        # 关闭客户端连接
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass
            self.client_socket = None
        
        # 关闭Socket服务器
        if self.socket_server:
            try:
                self.socket_server.close()
            except:
                pass
            self.socket_server = None
        
        # 清理viewer（它会自动清理，但我们可以确保状态正确）
        if self.viewer:
            try:
                # viewer.close() 如果存在的话
                pass
            except:
                pass
            self.viewer = None
        
        logger.info("MuJoCo Viewer Server stopped successfully")

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
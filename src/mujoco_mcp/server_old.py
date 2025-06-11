import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Set, Union

from mcp.server.fastmcp import FastMCP
from .simulation import MuJoCoSimulation
from .enhanced_auth_manager import EnhancedAuthManager

class MuJoCoMCPServer(FastMCP):
    """MuJoCo模型上下文协议服务器实现"""
    
    def __init__(self, host: str = "localhost", port: int = 7777, auth_manager: Optional[EnhancedAuthManager] = None):
        """
        初始化MuJoCo MCP服务器
        
        Args:
            host: 服务器主机名
            port: 服务器端口
            auth_manager: 授权管理器，如果为None则使用默认管理器
        """
        super().__init__()
        self.simulations: Dict[str, MuJoCoSimulation] = {}
        self.logger = logging.getLogger("mujoco_mcp.server")
        self.auth_manager = auth_manager or EnhancedAuthManager()
        
        # 性能监控
        self.performance_monitor = PerformanceMonitor()
        
        # 注册资源和工具
        self._register_resources()
        self._register_tools()

    def _register_resources(self):
        """注册MCP资源"""
        # 基础资源
        self.register_resource("joint_positions", self._get_joint_positions)
        self.register_resource("joint_velocities", self._get_joint_velocities)
        self.register_resource("simulation_info", self._get_simulation_info)
        self.register_resource("sensor_data", self._get_sensor_data)
        self.register_resource("rigid_body_states", self._get_rigid_body_states)
        
        # 新增高级资源
        self.register_resource("scene_objects", self._get_scene_objects)
        self.register_resource("object_properties", self._get_object_properties)
        self.register_resource("robot_state", self._get_robot_state)
        self.register_resource("trajectory_info", self._get_trajectory_info)
    
    def _register_tools(self):
        """注册MCP工具"""
        # 基础工具
        self.register_tool("start_simulation", self._start_simulation)
        self.register_tool("step_simulation", self._step_simulation)
        self.register_tool("reset_simulation", self._reset_simulation)
        self.register_tool("set_joint_positions", self._set_joint_positions)
        self.register_tool("set_joint_velocities", self._set_joint_velocities)
        self.register_tool("apply_control", self._apply_control)
        self.register_tool("delete_simulation", self._delete_simulation)
        
        # 新增高级工具
        self.register_tool("move_to_position", self._move_to_position)
        self.register_tool("move_to_object", self._move_to_object)
        self.register_tool("grasp_object", self._grasp_object)
        self.register_tool("release_object", self._release_object)
        self.register_tool("apply_force", self._apply_force)
        self.register_tool("interact_with_object", self._interact_with_object)
        self.register_tool("plan_trajectory", self._plan_trajectory)
    
    def _get_simulation(self, sim_id: str) -> MuJoCoSimulation:
        """
        获取模拟实例
        
        Args:
            sim_id: 模拟ID
            
        Returns:
            MuJoCoSimulation: 模拟实例
            
        Raises:
            ValueError: 如果找不到模拟ID
        """
        if sim_id not in self.simulations:
            raise ValueError(f"找不到模拟ID: {sim_id}")
        return self.simulations[sim_id]
    
    def _authorize_client_operation(self, 
                                   client_id: str, 
                                   operation: str, 
                                   resource: Optional[str] = None,
                                   parameters: Optional[Dict[str, Any]] = None) -> bool:
        """
        验证客户端操作授权
        
        Args:
            client_id: 客户端ID
            operation: 操作名称
            resource: 资源名称 (可选)
            parameters: 操作参数 (可选)
            
        Returns:
            bool: 是否授权
        """
        # 默认授权如果没有授权管理器
        if not self.auth_manager:
            return True
            
        # 请求授权
        authorized, message, request_id = self.auth_manager.request_authorization(
            client_id, operation, resource, parameters
        )
        
        if not authorized:
            self.logger.warning(f"拒绝未授权的操作: 客户端={client_id}, 操作={operation}, 消息={message}")
            
        return authorized
        
    # 资源处理函数
    def _get_joint_positions(self, sim_id: str, client_id: str) -> Dict[str, Any]:
        """
        获取关节位置
        
        Args:
            sim_id: 模拟ID
            client_id: 客户端ID
            
        Returns:
            Dict[str, Any]: 关节位置
        """
        if not self._authorize_client_operation(client_id, "get_resource", "joint_positions", {"sim_id": sim_id}):
            return {"error": "未授权访问"}
            
        sim = self._get_simulation(sim_id)
        return {"positions": sim.get_joint_positions().tolist(), "names": sim.get_joint_names()}
    
    def _get_joint_velocities(self, sim_id: str, client_id: str) -> Dict[str, Any]:
        """
        获取关节速度
        
        Args:
            sim_id: 模拟ID
            client_id: 客户端ID
            
        Returns:
            Dict[str, Any]: 关节速度
        """
        if not self._authorize_client_operation(client_id, "get_resource", "joint_velocities", {"sim_id": sim_id}):
            return {"error": "未授权访问"}
            
        sim = self._get_simulation(sim_id)
        return {"velocities": sim.get_joint_velocities().tolist(), "names": sim.get_joint_names()}
    
    def _get_simulation_info(self, sim_id: str, client_id: str) -> Dict[str, Any]:
        """
        获取模拟信息
        
        Args:
            sim_id: 模拟ID
            client_id: 客户端ID
            
        Returns:
            Dict[str, Any]: 模拟信息
        """
        if not self._authorize_client_operation(client_id, "get_resource", "simulation_info", {"sim_id": sim_id}):
            return {"error": "未授权访问"}
            
        sim = self._get_simulation(sim_id)
        return {
            "id": sim_id,
            "time": sim.get_time(),
            "timestep": sim.get_timestep(),
            "num_joints": sim.get_num_joints(),
            "num_actuators": sim.get_num_actuators(),
            "joint_names": sim.get_joint_names(),
            "model_name": sim.get_model_name(),
            "is_initialized": sim.is_initialized()
        }
    
    def _get_sensor_data(self, sim_id: str, client_id: str) -> Dict[str, Any]:
        """
        获取传感器数据
        
        Args:
            sim_id: 模拟ID
            client_id: 客户端ID
            
        Returns:
            Dict[str, Any]: 传感器数据
        """
        if not self._authorize_client_operation(client_id, "get_resource", "sensor_data", {"sim_id": sim_id}):
            return {"error": "未授权访问"}
            
        sim = self._get_simulation(sim_id)
        return sim.get_sensor_data()
    
    def _get_rigid_body_states(self, sim_id: str, client_id: str) -> Dict[str, Any]:
        """
        获取刚体状态
        
        Args:
            sim_id: 模拟ID
            client_id: 客户端ID
            
        Returns:
            Dict[str, Any]: 刚体状态
        """
        if not self._authorize_client_operation(client_id, "get_resource", "rigid_body_states", {"sim_id": sim_id}):
            return {"error": "未授权访问"}
            
        sim = self._get_simulation(sim_id)
        return sim.get_rigid_body_states()
    
    # 工具处理函数
    def _start_simulation(self, client_id: str, model_xml: str = None, model_path: str = None) -> Dict[str, Any]:
        """
        启动新的模拟
        
        Args:
            client_id: 客户端ID
            model_xml: 模型XML (可选)
            model_path: 模型文件路径 (可选)
            
        Returns:
            Dict[str, Any]: 包含模拟ID的响应
            
        Raises:
            ValueError: 如果既没有提供model_xml也没有提供model_path
        """
        if not model_xml and not model_path:
            return {"error": "必须提供model_xml或model_path"}
        
        params = {"model_xml": model_xml, "model_path": model_path}
        if not self._authorize_client_operation(client_id, "start_simulation", None, params):
            return {"error": "未授权访问"}
        
        try:
            # 创建新的模拟实例
            sim = MuJoCoSimulation()
            
            # 根据提供的参数初始化模拟
            if model_xml:
                sim.load_from_xml_string(model_xml)
            elif model_path:
                sim.load_from_file(model_path)
            
            # 生成唯一的模拟ID
            sim_id = f"sim_{int(time.time())}_{len(self.simulations)}"
            
            # 存储模拟实例
            self.simulations[sim_id] = sim
            
            self.logger.info(f"已启动新的模拟: {sim_id}")
            return {"sim_id": sim_id}
        except Exception as e:
            self.logger.error(f"启动模拟时出错: {str(e)}")
            return {"error": f"启动模拟时出错: {str(e)}"}
    
    def _step_simulation(self, client_id: str, sim_id: str, num_steps: int = 1) -> Dict[str, Any]:
        """
        推进模拟
        
        Args:
            client_id: 客户端ID
            sim_id: 模拟ID
            num_steps: 要执行的步数
            
        Returns:
            Dict[str, Any]: 步骤执行结果
        """
        params = {"sim_id": sim_id, "num_steps": num_steps}
        if not self._authorize_client_operation(client_id, "step_simulation", None, params):
            return {"error": "未授权访问"}
            
        try:
            sim = self._get_simulation(sim_id)
            
            # 执行指定步数
            for _ in range(num_steps):
                sim.step()
            
            return {
                "time": sim.get_time(),
                "steps_executed": num_steps
            }
        except Exception as e:
            self.logger.error(f"步进模拟时出错: {str(e)}")
            return {"error": f"步进模拟时出错: {str(e)}"}
    
    def _reset_simulation(self, client_id: str, sim_id: str) -> Dict[str, Any]:
        """
        重置模拟
        
        Args:
            client_id: 客户端ID
            sim_id: 模拟ID
            
        Returns:
            Dict[str, Any]: 重置结果
        """
        params = {"sim_id": sim_id}
        if not self._authorize_client_operation(client_id, "reset_simulation", None, params):
            return {"error": "未授权访问"}
            
        try:
            sim = self._get_simulation(sim_id)
            sim.reset()
            
            return {"success": True, "time": 0.0}
        except Exception as e:
            self.logger.error(f"重置模拟时出错: {str(e)}")
            return {"error": f"重置模拟时出错: {str(e)}"}
    
    def _set_joint_positions(self, client_id: str, sim_id: str, positions: List[float], joint_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        设置关节位置
        
        Args:
            client_id: 客户端ID
            sim_id: 模拟ID
            positions: 关节位置列表
            joint_names: 关节名称列表 (可选)
            
        Returns:
            Dict[str, Any]: 设置结果
        """
        params = {"sim_id": sim_id, "positions": positions, "joint_names": joint_names}
        if not self._authorize_client_operation(client_id, "set_joint_positions", None, params):
            return {"error": "未授权访问"}
            
        try:
            sim = self._get_simulation(sim_id)
            
            if joint_names:
                # 设置指定关节的位置
                sim.set_joint_positions_by_name(positions, joint_names)
            else:
                # 设置所有关节的位置
                sim.set_joint_positions(positions)
            
            return {"success": True}
        except Exception as e:
            self.logger.error(f"设置关节位置时出错: {str(e)}")
            return {"error": f"设置关节位置时出错: {str(e)}"}
    
    def _set_joint_velocities(self, client_id: str, sim_id: str, velocities: List[float], joint_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        设置关节速度
        
        Args:
            client_id: 客户端ID
            sim_id: 模拟ID
            velocities: 关节速度列表
            joint_names: 关节名称列表 (可选)
            
        Returns:
            Dict[str, Any]: 设置结果
        """
        params = {"sim_id": sim_id, "velocities": velocities, "joint_names": joint_names}
        if not self._authorize_client_operation(client_id, "set_joint_velocities", None, params):
            return {"error": "未授权访问"}
            
        try:
            sim = self._get_simulation(sim_id)
            
            if joint_names:
                # 设置指定关节的速度
                sim.set_joint_velocities_by_name(velocities, joint_names)
            else:
                # 设置所有关节的速度
                sim.set_joint_velocities(velocities)
            
            return {"success": True}
        except Exception as e:
            self.logger.error(f"设置关节速度时出错: {str(e)}")
            return {"error": f"设置关节速度时出错: {str(e)}"}
    
    def _apply_control(self, client_id: str, sim_id: str, controls: List[float]) -> Dict[str, Any]:
        """
        应用控制输入
        
        Args:
            client_id: 客户端ID
            sim_id: 模拟ID
            controls: 控制输入列表
            
        Returns:
            Dict[str, Any]: 操作结果
        """
        params = {"sim_id": sim_id, "controls": controls}
        if not self._authorize_client_operation(client_id, "apply_control", None, params):
            return {"error": "未授权访问"}
            
        try:
            sim = self._get_simulation(sim_id)
            sim.apply_control(controls)
            
            return {"success": True}
        except Exception as e:
            self.logger.error(f"应用控制输入时出错: {str(e)}")
            return {"error": f"应用控制输入时出错: {str(e)}"}
    
    def _delete_simulation(self, client_id: str, sim_id: str) -> Dict[str, Any]:
        """
        删除模拟
        
        Args:
            client_id: 客户端ID
            sim_id: 模拟ID
            
        Returns:
            Dict[str, Any]: 删除结果
        """
        params = {"sim_id": sim_id}
        if not self._authorize_client_operation(client_id, "delete_simulation", None, params):
            return {"error": "未授权访问"}
            
        try:
            if sim_id not in self.simulations:
                return {"error": f"找不到模拟ID: {sim_id}"}
            
            # 删除模拟实例
            del self.simulations[sim_id]
            
            self.logger.info(f"已删除模拟: {sim_id}")
            return {"success": True}
        except Exception as e:
            self.logger.error(f"删除模拟时出错: {str(e)}")
            return {"error": f"删除模拟时出错: {str(e)}"}
    
    def get_simulation_ids(self) -> List[str]:
        """
        获取所有活动模拟的ID
        
        Returns:
            List[str]: 模拟ID列表
        """
        return list(self.simulations.keys())
    
    def shutdown(self):
        """关闭服务器并清理资源"""
        # 关闭所有模拟
        for sim_id in list(self.simulations.keys()):
            try:
                del self.simulations[sim_id]
            except Exception as e:
                self.logger.error(f"关闭模拟时出错: {sim_id}, {str(e)}")
        
        # 关闭服务器
        super().shutdown()
        self.logger.info("MuJoCo MCP服务器已关闭")

    def _get_scene_objects(self, request):
        """
        获取场景中的所有对象
        
        Args:
            request: MCP请求，包含sim_id参数
            
        Returns:
            包含场景对象信息的响应
        """
        try:
            self.performance_monitor.start_timer("get_scene_objects")
            sim_id = request.parameters.get("sim_id")
            
            simulation = self._get_simulation(sim_id)
            objects = simulation.get_scene_objects()
            
            self.performance_monitor.end_timer("get_scene_objects")
            return {
                "success": True,
                "data": {
                    "objects": objects
                }
            }
        except Exception as e:
            self.logger.error(f"获取场景对象失败: {str(e)}")
            return {
                "success": False,
                "error": f"获取场景对象失败: {str(e)}"
            }
    
    def _get_object_properties(self, request):
        """
        获取指定对象的属性
        
        Args:
            request: MCP请求，包含sim_id和object_id参数
            
        Returns:
            包含对象属性的响应
        """
        try:
            self.performance_monitor.start_timer("get_object_properties")
            sim_id = request.parameters.get("sim_id")
            object_id = request.parameters.get("object_id")
            
            simulation = self._get_simulation(sim_id)
            properties = simulation.get_object_properties(object_id)
            
            self.performance_monitor.end_timer("get_object_properties")
            return {
                "success": True,
                "data": properties
            }
        except Exception as e:
            self.logger.error(f"获取对象属性失败: {str(e)}")
            return {
                "success": False,
                "error": f"获取对象属性失败: {str(e)}"
            }
    
    def _get_robot_state(self, request):
        """
        获取机器人状态
        
        Args:
            request: MCP请求，包含sim_id和robot_id参数
            
        Returns:
            包含机器人状态的响应
        """
        try:
            self.performance_monitor.start_timer("get_robot_state")
            sim_id = request.parameters.get("sim_id")
            robot_id = request.parameters.get("robot_id")
            
            simulation = self._get_simulation(sim_id)
            robot_state = simulation.get_robot_state(robot_id)
            
            self.performance_monitor.end_timer("get_robot_state")
            return {
                "success": True,
                "data": robot_state
            }
        except Exception as e:
            self.logger.error(f"获取机器人状态失败: {str(e)}")
            return {
                "success": False,
                "error": f"获取机器人状态失败: {str(e)}"
            }
    
    def _get_trajectory_info(self, request):
        """
        获取轨迹信息
        
        Args:
            request: MCP请求，包含sim_id和trajectory_id参数
            
        Returns:
            包含轨迹信息的响应
        """
        try:
            self.performance_monitor.start_timer("get_trajectory_info")
            sim_id = request.parameters.get("sim_id")
            trajectory_id = request.parameters.get("trajectory_id")
            
            simulation = self._get_simulation(sim_id)
            trajectory_info = simulation.get_trajectory_info(trajectory_id)
            
            self.performance_monitor.end_timer("get_trajectory_info")
            return {
                "success": True,
                "data": trajectory_info
            }
        except Exception as e:
            self.logger.error(f"获取轨迹信息失败: {str(e)}")
            return {
                "success": False,
                "error": f"获取轨迹信息失败: {str(e)}"
            }
    
    def _move_to_position(self, request):
        """
        移动机器人到指定位置
        
        Args:
            request: MCP请求，包含sim_id, robot_id和position参数
            
        Returns:
            操作结果响应
        """
        try:
            self.performance_monitor.start_timer("move_to_position")
            
            # 授权验证
            auth_result, auth_msg = self.auth_manager.handle_auth_request(request)
            if not auth_result:
                return {
                    "success": False,
                    "error": f"操作未授权: {auth_msg}"
                }
            
            sim_id = request.parameters.get("sim_id")
            robot_id = request.parameters.get("robot_id")
            position = request.parameters.get("position")
            speed = request.parameters.get("speed", 1.0)
            
            simulation = self._get_simulation(sim_id)
            result = simulation.move_robot_to_position(robot_id, position, speed)
            
            self.performance_monitor.end_timer("move_to_position")
            return {
                "success": True,
                "data": {
                    "result": result
                }
            }
        except Exception as e:
            self.logger.error(f"移动到位置失败: {str(e)}")
            return {
                "success": False,
                "error": f"移动到位置失败: {str(e)}"
            }
    
    def _move_to_object(self, request):
        """
        移动机器人到指定对象
        
        Args:
            request: MCP请求，包含sim_id, robot_id和object_id参数
            
        Returns:
            操作结果响应
        """
        try:
            self.performance_monitor.start_timer("move_to_object")
            
            # 授权验证
            auth_result, auth_msg = self.auth_manager.handle_auth_request(request)
            if not auth_result:
                return {
                    "success": False,
                    "error": f"操作未授权: {auth_msg}"
                }
            
            sim_id = request.parameters.get("sim_id")
            robot_id = request.parameters.get("robot_id")
            object_id = request.parameters.get("object_id")
            offset = request.parameters.get("offset", [0.0, 0.0, 0.1])  # 默认偏移量
            speed = request.parameters.get("speed", 1.0)
            
            simulation = self._get_simulation(sim_id)
            result = simulation.move_robot_to_object(robot_id, object_id, offset, speed)
            
            self.performance_monitor.end_timer("move_to_object")
            return {
                "success": True,
                "data": {
                    "result": result
                }
            }
        except Exception as e:
            self.logger.error(f"移动到对象失败: {str(e)}")
            return {
                "success": False,
                "error": f"移动到对象失败: {str(e)}"
            }
    
    def _grasp_object(self, request):
        """
        抓取对象
        
        Args:
            request: MCP请求，包含sim_id, robot_id和object_id参数
            
        Returns:
            操作结果响应
        """
        try:
            self.performance_monitor.start_timer("grasp_object")
            
            # 授权验证
            auth_result, auth_msg = self.auth_manager.handle_auth_request(request)
            if not auth_result:
                return {
                    "success": False,
                    "error": f"操作未授权: {auth_msg}"
                }
            
            sim_id = request.parameters.get("sim_id")
            robot_id = request.parameters.get("robot_id")
            object_id = request.parameters.get("object_id")
            force = request.parameters.get("force", 50.0)  # 默认抓取力
            
            simulation = self._get_simulation(sim_id)
            result = simulation.grasp_object(robot_id, object_id, force)
            
            self.performance_monitor.end_timer("grasp_object")
            return {
                "success": True,
                "data": {
                    "result": result
                }
            }
        except Exception as e:
            self.logger.error(f"抓取对象失败: {str(e)}")
            return {
                "success": False,
                "error": f"抓取对象失败: {str(e)}"
            }
    
    def _release_object(self, request):
        """
        释放对象
        
        Args:
            request: MCP请求，包含sim_id和robot_id参数
            
        Returns:
            操作结果响应
        """
        try:
            self.performance_monitor.start_timer("release_object")
            
            # 授权验证
            auth_result, auth_msg = self.auth_manager.handle_auth_request(request)
            if not auth_result:
                return {
                    "success": False,
                    "error": f"操作未授权: {auth_msg}"
                }
            
            sim_id = request.parameters.get("sim_id")
            robot_id = request.parameters.get("robot_id")
            
            simulation = self._get_simulation(sim_id)
            result = simulation.release_object(robot_id)
            
            self.performance_monitor.end_timer("release_object")
            return {
                "success": True,
                "data": {
                    "result": result
                }
            }
        except Exception as e:
            self.logger.error(f"释放对象失败: {str(e)}")
            return {
                "success": False,
                "error": f"释放对象失败: {str(e)}"
            }
    
    def _apply_force(self, request):
        """
        应用力到物体
        
        Args:
            request: MCP请求，包含sim_id, object_id, force和point参数
            
        Returns:
            操作结果响应
        """
        try:
            self.performance_monitor.start_timer("apply_force")
            
            # 授权验证
            auth_result, auth_msg = self.auth_manager.handle_auth_request(request)
            if not auth_result:
                return {
                    "success": False,
                    "error": f"操作未授权: {auth_msg}"
                }
            
            sim_id = request.parameters.get("sim_id")
            object_id = request.parameters.get("object_id")
            force = request.parameters.get("force")  # [fx, fy, fz]
            point = request.parameters.get("point", None)  # 可选，施力点
            
            simulation = self._get_simulation(sim_id)
            result = simulation.apply_force(object_id, force, point)
            
            self.performance_monitor.end_timer("apply_force")
            return {
                "success": True,
                "data": {
                    "result": result
                }
            }
        except Exception as e:
            self.logger.error(f"应用力失败: {str(e)}")
            return {
                "success": False,
                "error": f"应用力失败: {str(e)}"
            }
    
    def _interact_with_object(self, request):
        """
        与对象交互
        
        Args:
            request: MCP请求，包含sim_id, object_id, interaction_type和parameters参数
            
        Returns:
            操作结果响应
        """
        try:
            self.performance_monitor.start_timer("interact_with_object")
            
            # 授权验证
            auth_result, auth_msg = self.auth_manager.handle_auth_request(request)
            if not auth_result:
                return {
                    "success": False,
                    "error": f"操作未授权: {auth_msg}"
                }
            
            sim_id = request.parameters.get("sim_id")
            object_id = request.parameters.get("object_id")
            interaction_type = request.parameters.get("interaction_type")
            parameters = request.parameters.get("parameters", {})
            
            simulation = self._get_simulation(sim_id)
            result = simulation.interact_with_object(object_id, interaction_type, parameters)
            
            self.performance_monitor.end_timer("interact_with_object")
            return {
                "success": True,
                "data": {
                    "result": result
                }
            }
        except Exception as e:
            self.logger.error(f"对象交互失败: {str(e)}")
            return {
                "success": False,
                "error": f"对象交互失败: {str(e)}"
            }
    
    def _plan_trajectory(self, request):
        """
        规划轨迹
        
        Args:
            request: MCP请求，包含sim_id, robot_id, start_position, target_position参数
            
        Returns:
            包含轨迹信息的响应
        """
        try:
            self.performance_monitor.start_timer("plan_trajectory")
            
            # 授权验证
            auth_result, auth_msg = self.auth_manager.handle_auth_request(request)
            if not auth_result:
                return {
                    "success": False,
                    "error": f"操作未授权: {auth_msg}"
                }
            
            sim_id = request.parameters.get("sim_id")
            robot_id = request.parameters.get("robot_id")
            start_position = request.parameters.get("start_position")
            target_position = request.parameters.get("target_position")
            avoid_objects = request.parameters.get("avoid_objects", [])
            max_velocity = request.parameters.get("max_velocity", 1.0)
            
            simulation = self._get_simulation(sim_id)
            trajectory = simulation.plan_trajectory(
                robot_id, start_position, target_position, avoid_objects, max_velocity
            )
            
            self.performance_monitor.end_timer("plan_trajectory")
            return {
                "success": True,
                "data": {
                    "trajectory_id": trajectory["id"],
                    "waypoints": trajectory["waypoints"],
                    "duration": trajectory["duration"]
                }
            }
        except Exception as e:
            self.logger.error(f"轨迹规划失败: {str(e)}")
            return {
                "success": False,
                "error": f"轨迹规划失败: {str(e)}"
            }
    
    def get_performance_metrics(self):
        """
        获取性能指标
        
        Returns:
            性能指标数据
        """
        return self.performance_monitor.get_metrics()


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, operation):
        """开始计时特定操作"""
        self.start_times[operation] = time.time()
    
    def end_timer(self, operation):
        """结束计时特定操作"""
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            if operation not in self.metrics:
                self.metrics[operation] = []
            self.metrics[operation].append(duration)
            del self.start_times[operation]
    
    def get_metrics(self):
        """获取性能指标"""
        import statistics
        
        result = {}
        for op, durations in self.metrics.items():
            if durations:
                result[op] = {
                    "count": len(durations),
                    "avg_ms": statistics.mean(durations) * 1000,
                    "min_ms": min(durations) * 1000,
                    "max_ms": max(durations) * 1000
                }
                if len(durations) > 20:
                    result[op]["p95_ms"] = sorted(durations)[int(len(durations) * 0.95)] * 1000
        return result
    
    def reset(self):
        """重置所有指标"""
        self.metrics = {}
        self.start_times = {} 
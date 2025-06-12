"""
MuJoCo MCP 远程服务器实现 (v0.6.2)
通过Socket连接到外部MuJoCo Viewer进程，类似Blender/Figma MCP模式
"""
import uuid
import time
import json
from typing import Dict, Any, List, Optional
from .viewer_client import viewer_manager, ensure_viewer_connection
from .logging_config import get_logger, error_handler, MCPErrorCode

class MuJoCoRemoteServer:
    """MuJoCo远程MCP服务器，连接到外部MuJoCo Viewer"""
    
    def __init__(self):
        """初始化远程服务器"""
        self.name = "mujoco-mcp-remote"
        self.version = "0.6.2"
        self.description = "MuJoCo Model Context Protocol Server - Remote mode connecting to external MuJoCo Viewer GUI"
        self.logger = get_logger("mujoco_mcp.remote_server")
        
        # 工具注册表
        self._tools = {}
        self._resources = []
        self._running = False
        
        # 模型存储（模型ID -> 元数据）
        self._models = {}  # model_id -> model_metadata
        
        # 注册核心工具
        self._register_tools()
    
    def _register_tools(self):
        """注册MCP工具"""
        
        # 服务器信息工具
        self._tools["get_server_info"] = {
            "name": "get_server_info",
            "description": "Get detailed information about the MuJoCo MCP remote server",
            "parameters": {},
            "handler": self._handle_get_server_info
        }
        
        # 场景创建工具
        self._tools["create_scene"] = {
            "name": "create_scene",
            "description": "Create a physics scene and launch external MuJoCo Viewer",
            "parameters": {
                "scene_type": "Type of scene (pendulum, double_pendulum, cart_pole, robotic_arm)",
                "parameters": "(optional) Scene-specific parameters"
            },
            "handler": self._handle_create_scene
        }
        
        # 仿真控制工具
        self._tools["step_simulation"] = {
            "name": "step_simulation",
            "description": "Advance the physics simulation in external viewer",
            "parameters": {
                "model_id": "Model ID",
                "steps": "(optional) Number of simulation steps (default: 1)"
            },
            "handler": self._handle_step_simulation
        }
        
        # 状态查询工具
        self._tools["get_state"] = {
            "name": "get_state",
            "description": "Get current simulation state from external viewer",
            "parameters": {
                "model_id": "Model ID",
                "components": "(optional) List of state components to include"
            },
            "handler": self._handle_get_state
        }
        
        # 控制工具
        self._tools["set_joint_positions"] = {
            "name": "set_joint_positions",
            "description": "Set joint positions in external viewer",
            "parameters": {
                "model_id": "Model ID",
                "positions": "List of joint position values"
            },
            "handler": self._handle_set_joint_positions
        }
        
        # 重置工具
        self._tools["reset_simulation"] = {
            "name": "reset_simulation",
            "description": "Reset simulation in external viewer",
            "parameters": {
                "model_id": "Model ID"
            },
            "handler": self._handle_reset_simulation
        }
        
        # 自然语言接口
        self._tools["execute_command"] = {
            "name": "execute_command",
            "description": "Execute natural language command on external MuJoCo Viewer",
            "parameters": {
                "command": "Natural language command",
                "context": "(optional) Additional context"
            },
            "handler": self._handle_execute_command
        }
        
        # 获取已加载模型工具
        self._tools["get_loaded_models"] = {
            "name": "get_loaded_models",
            "description": "Get list of all loaded models in viewers",
            "parameters": {},
            "handler": self._handle_get_loaded_models
        }
    
    def _handle_get_server_info(self) -> Dict[str, Any]:
        """获取服务器信息"""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "mode": "remote_viewer",
            "capabilities": [
                "external_viewer_control",
                "socket_communication",
                "real_time_gui",
                "natural_language",
                "scene_creation"
            ],
            "connected_viewers": len(viewer_manager.clients),
            "supported_scenes": ["pendulum", "double_pendulum", "cart_pole", "robotic_arm"]
        }
    
    def _handle_create_scene(self, scene_type: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """创建场景并启动外部MuJoCo Viewer"""
        
        # 生成模型ID
        model_id = str(uuid.uuid4())
        
        # 创建模型XML
        model_xml = self._generate_scene_xml(scene_type, parameters or {})
        if not model_xml:
            return {
                "success": False,
                "error": f"Unsupported scene type: {scene_type}"
            }
        
        # 确保viewer连接
        if not ensure_viewer_connection(model_id):
            return {
                "success": False,
                "error": "Failed to connect to MuJoCo Viewer. Please start mujoco_viewer_server.py first."
            }
        
        # 获取viewer客户端
        client = viewer_manager.get_client(model_id)
        if not client:
            return {
                "success": False,
                "error": "Viewer client not available"
            }
        
        # 加载模型到viewer
        response = client.load_model(model_xml)
        if not response.get("success"):
            return {
                "success": False,
                "error": f"Failed to load model in viewer: {response.get('error')}"
            }
        
        # 启动viewer GUI
        viewer_response = client.start_viewer()
        if not viewer_response.get("success"):
            return {
                "success": False,
                "error": f"Failed to start viewer: {viewer_response.get('error')}"
            }
        
        # 保存模型元数据
        self._models[model_id] = {
            "scene_type": scene_type,
            "parameters": parameters,
            "model_info": response.get("model_info", {}),
            "created_time": time.time()
        }
        
        return {
            "success": True,
            "model_id": model_id,
            "scene_info": {
                "type": scene_type,
                "parameters": parameters
            },
            "viewer_info": viewer_response.get("viewer_info", {}),
            "model_info": response.get("model_info", {}),
            "message": f"Created {scene_type} scene and launched MuJoCo Viewer GUI"
        }
    
    def _handle_step_simulation(self, model_id: str, steps: int = 1) -> Dict[str, Any]:
        """步进仿真（外部viewer自动运行，这里只是状态同步）"""
        if model_id not in self._models:
            return {"success": False, "error": "Model not found"}
        
        client = viewer_manager.get_client(model_id)
        if not client:
            return {"success": False, "error": "Viewer not connected"}
        
        # 获取当前状态（viewer自动步进仿真）
        state_response = client.get_state()
        if not state_response.get("success"):
            return {"success": False, "error": "Failed to get simulation state"}
        
        return {
            "success": True,
            "steps_requested": steps,
            "current_state": state_response,
            "message": "Simulation running in external viewer"
        }
    
    def _handle_get_state(self, model_id: str, components: Optional[List[str]] = None) -> Dict[str, Any]:
        """获取仿真状态"""
        if model_id not in self._models:
            return {"success": False, "error": "Model not found"}
        
        client = viewer_manager.get_client(model_id)
        if not client:
            return {"success": False, "error": "Viewer not connected"}
        
        return client.get_state()
    
    def _handle_set_joint_positions(self, model_id: str, positions: List[float]) -> Dict[str, Any]:
        """设置关节位置"""
        if model_id not in self._models:
            return {"success": False, "error": "Model not found"}
        
        client = viewer_manager.get_client(model_id)
        if not client:
            return {"success": False, "error": "Viewer not connected"}
        
        return client.set_joint_positions(positions)
    
    def _handle_reset_simulation(self, model_id: str) -> Dict[str, Any]:
        """重置仿真"""
        if model_id not in self._models:
            return {"success": False, "error": "Model not found"}
        
        client = viewer_manager.get_client(model_id)
        if not client:
            return {"success": False, "error": "Viewer not connected"}
        
        return client.reset_simulation()
    
    def _handle_execute_command(self, command: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """执行自然语言命令"""
        context = context or {}
        
        # 简单的命令解析
        command_lower = command.lower()
        
        # 创建场景命令
        if "create" in command_lower and "pendulum" in command_lower:
            return self._handle_create_scene("pendulum", context)
        elif "create" in command_lower and "cart" in command_lower:
            return self._handle_create_scene("cart_pole", context)
        
        # 获取状态命令
        if "state" in command_lower or "status" in command_lower:
            model_id = context.get("model_id")
            if not model_id and self._models:
                model_id = list(self._models.keys())[0]  # 使用第一个模型
            
            if model_id:
                return self._handle_get_state(model_id)
        
        # 重置命令
        if "reset" in command_lower:
            model_id = context.get("model_id")
            if not model_id and self._models:
                model_id = list(self._models.keys())[0]
            
            if model_id:
                return self._handle_reset_simulation(model_id)
        
        return {
            "success": False,
            "error": f"Command not recognized: {command}",
            "suggestion": "Try commands like 'create pendulum', 'show state', or 'reset simulation'"
        }
    
    def _handle_get_loaded_models(self) -> Dict[str, Any]:
        """获取已加载的模型列表"""
        return self.get_loaded_models()
    
    def _generate_scene_xml(self, scene_type: str, parameters: Dict[str, Any]) -> Optional[str]:
        """生成场景的MuJoCo XML"""
        
        if scene_type == "pendulum":
            length = parameters.get("length", 0.5)
            mass = parameters.get("mass", 0.5)
            damping = parameters.get("damping", 0.1)
            
            return f"""
            <mujoco>
                <option gravity="0 0 -9.81" timestep="0.01"/>
                <default>
                    <joint damping="{damping}"/>
                </default>
                <worldbody>
                    <body name="pendulum" pos="0 0 0">
                        <joint name="hinge" type="hinge" axis="1 0 0"/>
                        <geom name="rod" type="cylinder" size="0.02 {length/2}" 
                              pos="0 0 -{length/2}" rgba="0.8 0.4 0.4 1"/>
                        <geom name="mass" type="sphere" size="0.08" 
                              pos="0 0 -{length}" mass="{mass}" rgba="0.4 0.4 0.8 1"/>
                    </body>
                </worldbody>
                <actuator>
                    <motor joint="hinge" gear="1"/>
                </actuator>
            </mujoco>
            """
        
        elif scene_type == "double_pendulum":
            l1 = parameters.get("length1", 0.4)
            l2 = parameters.get("length2", 0.4)
            m1 = parameters.get("mass1", 0.3)
            m2 = parameters.get("mass2", 0.3)
            
            return f"""
            <mujoco>
                <option gravity="0 0 -9.81" timestep="0.01"/>
                <default>
                    <joint damping="0.05"/>
                </default>
                <worldbody>
                    <body name="link1" pos="0 0 0">
                        <joint name="joint1" type="hinge" axis="1 0 0"/>
                        <geom name="rod1" type="cylinder" size="0.02 {l1/2}" 
                              pos="0 0 -{l1/2}" rgba="0.8 0.4 0.4 1"/>
                        <body name="link2" pos="0 0 -{l1}">
                            <joint name="joint2" type="hinge" axis="1 0 0"/>
                            <geom name="rod2" type="cylinder" size="0.02 {l2/2}" 
                                  pos="0 0 -{l2/2}" rgba="0.4 0.8 0.4 1"/>
                            <geom name="mass2" type="sphere" size="0.06" 
                                  pos="0 0 -{l2}" mass="{m2}" rgba="0.4 0.4 0.8 1"/>
                        </body>
                        <geom name="mass1" type="sphere" size="0.06" 
                              pos="0 0 -{l1}" mass="{m1}" rgba="0.8 0.4 0.8 1"/>
                    </body>
                </worldbody>
                <actuator>
                    <motor joint="joint1" gear="1"/>
                    <motor joint="joint2" gear="1"/>
                </actuator>
            </mujoco>
            """
        
        elif scene_type == "cart_pole":
            cart_mass = parameters.get("cart_mass", 1.0)
            pole_mass = parameters.get("pole_mass", 0.1)
            pole_length = parameters.get("pole_length", 0.5)
            
            return f"""
            <mujoco>
                <option gravity="0 0 -9.81" timestep="0.01"/>
                <default>
                    <joint damping="0.1"/>
                </default>
                <worldbody>
                    <body name="cart" pos="0 0 0.1">
                        <joint name="cart_slide" type="slide" axis="1 0 0" range="-2 2"/>
                        <geom name="cart_body" type="box" size="0.2 0.1 0.1" 
                              mass="{cart_mass}" rgba="0.4 0.4 0.8 1"/>
                        <body name="pole" pos="0 0 0.1">
                            <joint name="pole_hinge" type="hinge" axis="0 1 0"/>
                            <geom name="pole_rod" type="cylinder" size="0.02 {pole_length/2}" 
                                  pos="0 0 {pole_length/2}" mass="{pole_mass}" rgba="0.8 0.4 0.4 1"/>
                            <geom name="pole_tip" type="sphere" size="0.05" 
                                  pos="0 0 {pole_length}" mass="0.05" rgba="0.8 0.8 0.4 1"/>
                        </body>
                    </body>
                    <geom name="ground" type="plane" size="5 5 0.1" rgba="0.7 0.7 0.7 1"/>
                </worldbody>
                <actuator>
                    <motor joint="cart_slide" gear="100"/>
                </actuator>
            </mujoco>
            """
        
        elif scene_type == "robotic_arm":
            # 简单的2自由度机械臂
            link1_length = parameters.get("link1_length", 0.3)
            link2_length = parameters.get("link2_length", 0.3)
            base_mass = parameters.get("base_mass", 0.5)
            link1_mass = parameters.get("link1_mass", 0.3)
            link2_mass = parameters.get("link2_mass", 0.2)
            
            return f"""
            <mujoco>
                <option gravity="0 0 -9.81" timestep="0.01"/>
                <default>
                    <joint damping="0.3"/>
                    <geom rgba="0.6 0.6 0.8 1"/>
                </default>
                <worldbody>
                    <!-- 基座 -->
                    <body name="base" pos="0 0 0.1">
                        <geom name="base_geom" type="cylinder" size="0.08 0.05" 
                              mass="{base_mass}" rgba="0.3 0.3 0.3 1"/>
                        
                        <!-- 第一关节和连杆 -->
                        <body name="link1" pos="0 0 0.05">
                            <joint name="joint1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
                            <geom name="link1_geom" type="cylinder" size="0.03 {link1_length/2}" 
                                  pos="0 0 {link1_length/2}" mass="{link1_mass}" rgba="0.8 0.4 0.4 1"/>
                            
                            <!-- 第二关节和连杆 -->
                            <body name="link2" pos="0 0 {link1_length}">
                                <joint name="joint2" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
                                <geom name="link2_geom" type="cylinder" size="0.02 {link2_length/2}" 
                                      pos="0 0 {link2_length/2}" mass="{link2_mass}" rgba="0.4 0.8 0.4 1"/>
                                
                                <!-- 末端执行器 -->
                                <body name="end_effector" pos="0 0 {link2_length}">
                                    <geom name="ee_geom" type="sphere" size="0.04" 
                                          mass="0.05" rgba="0.8 0.8 0.4 1"/>
                                </body>
                            </body>
                        </body>
                    </body>
                    
                    <!-- 地面 -->
                    <geom name="ground" type="plane" size="2 2 0.1" rgba="0.7 0.7 0.7 1"/>
                    
                    <!-- 目标物体（可选） -->
                    <body name="target" pos="0.4 0.0 0.2">
                        <geom name="target_geom" type="box" size="0.05 0.05 0.05" 
                              rgba="1.0 0.2 0.2 0.5"/>
                    </body>
                </worldbody>
                
                <actuator>
                    <position joint="joint1" kp="100"/>
                    <position joint="joint2" kp="100"/>
                </actuator>
            </mujoco>
            """
        
        # 更多场景类型可以在这里添加
        return None
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """获取可用工具列表"""
        tools = []
        for tool_name, tool_info in self._tools.items():
            tools.append({
                "name": tool_info["name"],
                "description": tool_info["description"],
                "parameters": tool_info["parameters"]
            })
        return tools
    
    def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """调用工具"""
        if tool_name not in self._tools:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}
        
        tool = self._tools[tool_name]
        handler = tool["handler"]
        
        try:
            # 调用工具处理函数
            if tool_name == "get_server_info":
                return handler()
            elif tool_name == "create_scene":
                return handler(
                    parameters.get("scene_type", "pendulum"),
                    parameters.get("parameters")
                )
            elif tool_name == "step_simulation":
                return handler(
                    parameters.get("model_id"),
                    parameters.get("steps", 1)
                )
            elif tool_name == "get_state":
                return handler(
                    parameters.get("model_id"),
                    parameters.get("components")
                )
            elif tool_name == "set_joint_positions":
                return handler(
                    parameters.get("model_id"),
                    parameters.get("positions", [])
                )
            elif tool_name == "reset_simulation":
                return handler(parameters.get("model_id"))
            elif tool_name == "execute_command":
                return handler(
                    parameters.get("command", ""),
                    parameters.get("context")
                )
            elif tool_name == "get_loaded_models":
                return handler()
            else:
                return {"success": False, "error": f"Handler not implemented for tool: {tool_name}"}
        
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {"success": False, "error": str(e)}
    
    def get_loaded_models(self) -> Dict[str, Any]:
        """获取已加载的模型列表"""
        return {
            "success": True,
            "models": [
                {
                    "model_id": model_id,
                    "scene_type": info["scene_type"],
                    "parameters": info["parameters"],
                    "created_time": info["created_time"],
                    "viewer_connected": viewer_manager.get_client(model_id) is not None
                }
                for model_id, info in self._models.items()
            ],
            "total_count": len(self._models)
        }

__all__ = ['MuJoCoRemoteServer']
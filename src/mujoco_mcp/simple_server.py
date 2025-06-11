"""
MuJoCo MCP 简单服务器实现 (v0.2.1)
包含基本的MCP功能、模型加载、仿真控制和增强状态查询
"""
import logging
import uuid
from typing import Dict, List, Any, Optional
from .simulation import MuJoCoSimulation


class MuJoCoMCPServer:
    """简单的MuJoCo MCP服务器"""
    
    def __init__(self):
        """初始化服务器"""
        self.name = "mujoco-mcp"
        self.version = "0.2.1"
        self.description = "MuJoCo Model Context Protocol Server"
        self.logger = logging.getLogger("mujoco_mcp.simple_server")
        
        # 工具注册表
        self._tools = {}
        self._resources = []
        self._running = False
        
        # 模型存储
        self._models = {}  # model_id -> simulation instance
        self._model_names = {}  # model_id -> name
        
        # 注册标准MCP工具
        self._register_standard_tools()
        # 注册MuJoCo工具
        self._register_mujoco_tools()
        
    def _register_standard_tools(self):
        """注册标准的MCP工具"""
        # get_server_info 工具
        self._tools["get_server_info"] = {
            "name": "get_server_info",
            "description": "Get information about the MCP server",
            "parameters": {},
            "handler": self.get_server_info
        }
        
        # get_tools 工具
        self._tools["get_tools"] = {
            "name": "get_tools",
            "description": "Get list of available tools",
            "parameters": {},
            "handler": self._handle_get_tools
        }
    
    def _register_mujoco_tools(self):
        """注册MuJoCo相关工具"""
        # load_model 工具
        self._tools["load_model"] = {
            "name": "load_model",
            "description": "Load a MuJoCo model from XML string",
            "parameters": {
                "model_string": "XML string containing the MuJoCo model",
                "name": "(optional) Name for the loaded model"
            },
            "handler": self._handle_load_model
        }
        
        # get_loaded_models 工具
        self._tools["get_loaded_models"] = {
            "name": "get_loaded_models",
            "description": "Get list of loaded models",
            "parameters": {},
            "handler": self._handle_get_loaded_models
        }
        
        # step_simulation 工具
        self._tools["step_simulation"] = {
            "name": "step_simulation",
            "description": "Step the simulation forward",
            "parameters": {
                "model_id": "ID of the model to step",
                "steps": "(optional) Number of steps to advance (default: 1)"
            },
            "handler": self._handle_step_simulation
        }
        
        # reset_simulation 工具
        self._tools["reset_simulation"] = {
            "name": "reset_simulation",
            "description": "Reset simulation to initial state",
            "parameters": {
                "model_id": "ID of the model to reset"
            },
            "handler": self._handle_reset_simulation
        }
        
        # get_simulation_state 工具
        self._tools["get_simulation_state"] = {
            "name": "get_simulation_state",
            "description": "Get current simulation state",
            "parameters": {
                "model_id": "ID of the model",
                "include_positions": "(optional) Include joint positions",
                "include_velocities": "(optional) Include joint velocities"
            },
            "handler": self._handle_get_simulation_state
        }
        
        # set_joint_positions 工具
        self._tools["set_joint_positions"] = {
            "name": "set_joint_positions",
            "description": "Set joint positions",
            "parameters": {
                "model_id": "ID of the model",
                "positions": "List of joint positions to set"
            },
            "handler": self._handle_set_joint_positions
        }
        
        # get_joint_positions 工具
        self._tools["get_joint_positions"] = {
            "name": "get_joint_positions",
            "description": "Get current joint positions",
            "parameters": {
                "model_id": "ID of the model",
                "include_names": "(optional) Include joint names"
            },
            "handler": self._handle_get_joint_positions
        }
        
        # get_joint_velocities 工具
        self._tools["get_joint_velocities"] = {
            "name": "get_joint_velocities",
            "description": "Get current joint velocities",
            "parameters": {
                "model_id": "ID of the model",
                "include_names": "(optional) Include joint names"
            },
            "handler": self._handle_get_joint_velocities
        }
        
        # set_joint_velocities 工具
        self._tools["set_joint_velocities"] = {
            "name": "set_joint_velocities",
            "description": "Set joint velocities",
            "parameters": {
                "model_id": "ID of the model",
                "velocities": "List of joint velocities to set"
            },
            "handler": self._handle_set_joint_velocities
        }
        
        # get_body_states 工具
        self._tools["get_body_states"] = {
            "name": "get_body_states",
            "description": "Get rigid body states (positions and orientations)",
            "parameters": {
                "model_id": "ID of the model"
            },
            "handler": self._handle_get_body_states
        }
        
        # get_sensor_data 工具
        self._tools["get_sensor_data"] = {
            "name": "get_sensor_data",
            "description": "Get sensor readings",
            "parameters": {
                "model_id": "ID of the model"
            },
            "handler": self._handle_get_sensor_data
        }
        
    def get_server_info(self) -> Dict[str, Any]:
        """获取服务器信息"""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "capabilities": [
                "simulation",
                "control",
                "state_query"
            ]
        }
        
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
        
    def _handle_get_tools(self) -> Dict[str, Any]:
        """处理get_tools调用"""
        return {
            "tools": self.get_tools()
        }
        
    def get_resources(self) -> List[Dict[str, Any]]:
        """获取可用资源列表"""
        return self._resources
        
    def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """调用工具"""
        if tool_name not in self._tools:
            raise ValueError(f"Unknown tool: {tool_name}")
            
        tool = self._tools[tool_name]
        handler = tool["handler"]
        
        # 调用处理器
        if callable(handler):
            try:
                return handler(**parameters) if parameters else handler()
            except TypeError as e:
                # 转换TypeError为ValueError以提供更好的错误消息
                if "missing" in str(e) and "required" in str(e):
                    raise ValueError(f"Missing required parameter for tool '{tool_name}': {str(e)}")
                raise
        else:
            raise RuntimeError(f"Tool handler for {tool_name} is not callable")
            
    def is_running(self) -> bool:
        """检查服务器是否在运行"""
        return self._running
        
    def start(self):
        """启动服务器"""
        self._running = True
        self.logger.info(f"MuJoCo MCP Server v{self.version} started")
        
    def stop(self):
        """停止服务器"""
        self._running = False
        self.logger.info("MuJoCo MCP Server stopped")
        
    def _handle_load_model(self, model_string: str, name: Optional[str] = None) -> Dict[str, Any]:
        """处理load_model工具调用"""
        # 参数验证
        if not model_string:
            raise ValueError("model_string is required and cannot be empty")
            
        # 检查是否是有效的XML
        if not model_string.strip().startswith("<"):
            raise ValueError("model_string must be valid XML (should start with '<')")
            
        # 生成模型ID
        model_id = str(uuid.uuid4())
        
        # 创建仿真实例
        sim = MuJoCoSimulation()
        
        try:
            # 加载模型
            sim.load_from_xml_string(model_string)
            
            # 存储模型
            self._models[model_id] = sim
            self._model_names[model_id] = name or f"model_{model_id[:8]}"
            
            # 获取模型信息
            model_info = sim.get_model_info()
            
            result = {
                "success": True,
                "model_id": model_id,
                "message": f"Model loaded successfully with ID: {model_id}",
                "model_info": model_info
            }
            
            if name:
                result["name"] = name
                
            return result
            
        except Exception as e:
            # 清理失败的模型
            if model_id in self._models:
                del self._models[model_id]
            if model_id in self._model_names:
                del self._model_names[model_id]
                
            raise ValueError(f"Failed to load model: {str(e)}")
            
    def _handle_get_loaded_models(self) -> Dict[str, Any]:
        """处理get_loaded_models工具调用"""
        models = []
        
        for model_id, sim in self._models.items():
            model_info = {
                "model_id": model_id,
                "name": self._model_names.get(model_id, "unnamed")
            }
            
            # 添加基本模型信息
            try:
                info = sim.get_model_info()
                model_info.update({
                    "nq": info.get("nq", 0),
                    "nv": info.get("nv", 0),
                    "nbody": info.get("nbody", 0)
                })
            except Exception:
                pass
                
            models.append(model_info)
            
        return {"models": models}
    
    def _handle_step_simulation(self, model_id: str, steps: int = 1) -> Dict[str, Any]:
        """处理step_simulation工具调用"""
        # 参数验证
        if not model_id:
            raise ValueError("model_id is required")
            
        if model_id not in self._models:
            raise ValueError(f"Model not found: {model_id}")
            
        if steps < 1:
            raise ValueError("steps must be positive")
            
        # 获取仿真实例
        sim = self._models[model_id]
        
        # 步进仿真
        for _ in range(steps):
            sim.step()
            
        return {
            "success": True,
            "steps_completed": steps,
            "time": sim.get_time(),
            "message": f"Simulation stepped {steps} times"
        }
    
    def _handle_reset_simulation(self, model_id: str) -> Dict[str, Any]:
        """处理reset_simulation工具调用"""
        # 参数验证
        if not model_id:
            raise ValueError("model_id is required")
            
        if model_id not in self._models:
            raise ValueError(f"Model not found: {model_id}")
            
        # 获取仿真实例
        sim = self._models[model_id]
        
        # 重置仿真
        sim.reset()
        
        return {
            "success": True,
            "message": "Simulation reset to initial state",
            "time": sim.get_time()
        }
    
    def _handle_get_simulation_state(self, model_id: str, 
                                   include_positions: bool = False,
                                   include_velocities: bool = False) -> Dict[str, Any]:
        """处理get_simulation_state工具调用"""
        # 参数验证
        if not model_id:
            raise ValueError("model_id is required")
            
        if model_id not in self._models:
            raise ValueError(f"Model not found: {model_id}")
            
        # 获取仿真实例
        sim = self._models[model_id]
        
        # 构建状态信息
        state = {
            "time": sim.get_time(),
            "nq": sim.get_num_joints(),
            "nv": sim.model.nv if sim._initialized else 0
        }
        
        # 可选地包含位置和速度
        if include_positions:
            state["qpos"] = sim.get_joint_positions().tolist()
            
        if include_velocities:
            state["qvel"] = sim.get_joint_velocities().tolist()
            
        return state
    
    def _handle_set_joint_positions(self, model_id: str, positions: List[float]) -> Dict[str, Any]:
        """处理set_joint_positions工具调用"""
        # 参数验证
        if not model_id:
            raise ValueError("model_id is required")
            
        if model_id not in self._models:
            raise ValueError(f"Model not found: {model_id}")
            
        if not positions:
            raise ValueError("positions is required")
            
        # 获取仿真实例
        sim = self._models[model_id]
        
        # 检查位置数量是否匹配
        nq = sim.get_num_joints()
        if len(positions) != nq:
            raise ValueError(f"Expected {nq} positions, got {len(positions)}")
            
        # 设置关节位置
        sim.set_joint_positions(positions)
        
        return {
            "success": True,
            "message": f"Set {len(positions)} joint positions"
        }
    
    def _handle_get_joint_positions(self, model_id: str, include_names: bool = False) -> Dict[str, Any]:
        """处理get_joint_positions工具调用"""
        # 参数验证
        if not model_id:
            raise ValueError("model_id is required")
            
        if model_id not in self._models:
            raise ValueError(f"Model not found: {model_id}")
            
        # 获取仿真实例
        sim = self._models[model_id]
        
        # 获取关节位置
        positions = sim.get_joint_positions().tolist()
        
        result = {
            "positions": positions
        }
        
        # 可选地包含关节名称
        if include_names:
            result["names"] = sim.get_joint_names()
            
        return result
    
    def _handle_get_joint_velocities(self, model_id: str, include_names: bool = False) -> Dict[str, Any]:
        """处理get_joint_velocities工具调用"""
        # 参数验证
        if not model_id:
            raise ValueError("model_id is required")
            
        if model_id not in self._models:
            raise ValueError(f"Model not found: {model_id}")
            
        # 获取仿真实例
        sim = self._models[model_id]
        
        # 获取关节速度
        velocities = sim.get_joint_velocities().tolist()
        
        result = {
            "velocities": velocities
        }
        
        # 可选地包含关节名称
        if include_names:
            result["names"] = sim.get_joint_names()
            
        return result
    
    def _handle_set_joint_velocities(self, model_id: str, velocities: List[float]) -> Dict[str, Any]:
        """处理set_joint_velocities工具调用"""
        # 参数验证
        if not model_id:
            raise ValueError("model_id is required")
            
        if model_id not in self._models:
            raise ValueError(f"Model not found: {model_id}")
            
        if not velocities:
            raise ValueError("velocities is required")
            
        # 获取仿真实例
        sim = self._models[model_id]
        
        # 检查速度数量是否匹配
        nv = sim.model.nv if sim._initialized else 0
        if len(velocities) != nv:
            raise ValueError(f"Expected {nv} velocities, got {len(velocities)}")
            
        # 设置关节速度
        sim.set_joint_velocities(velocities)
        
        return {
            "success": True,
            "message": f"Set {len(velocities)} joint velocities"
        }
    
    def _handle_get_body_states(self, model_id: str) -> Dict[str, Any]:
        """处理get_body_states工具调用"""
        # 参数验证
        if not model_id:
            raise ValueError("model_id is required")
            
        if model_id not in self._models:
            raise ValueError(f"Model not found: {model_id}")
            
        # 获取仿真实例
        sim = self._models[model_id]
        
        # 获取刚体状态
        body_states = sim.get_rigid_body_states()
        
        return {
            "bodies": body_states
        }
    
    def _handle_get_sensor_data(self, model_id: str) -> Dict[str, Any]:
        """处理get_sensor_data工具调用"""
        # 参数验证
        if not model_id:
            raise ValueError("model_id is required")
            
        if model_id not in self._models:
            raise ValueError(f"Model not found: {model_id}")
            
        # 获取仿真实例
        sim = self._models[model_id]
        
        # 获取传感器数据
        sensor_data = sim.get_sensor_data()
        
        return {
            "sensors": sensor_data
        }
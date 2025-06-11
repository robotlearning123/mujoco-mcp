"""
MuJoCo MCP 简单服务器实现 (v0.1.2)
包含基本的MCP功能和load_model工具
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
        self.version = "0.1.2"
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
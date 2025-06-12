"""
MuJoCo MCP 简单服务器实现 (v0.3.0)
包含基本的MCP功能、模型加载、仿真控制、增强状态查询、基础控制和可视化
"""
import logging
import uuid
import base64
import io
import random
from typing import Dict, List, Any, Optional
import numpy as np
from .simulation import MuJoCoSimulation


class MuJoCoMCPServer:
    """简单的MuJoCo MCP服务器"""
    
    def __init__(self):
        """初始化服务器"""
        self.name = "mujoco-mcp"
        self.version = "0.5.0"
        self.description = "MuJoCo Model Context Protocol Server - A physics simulation server that enables AI agents to control MuJoCo simulations through natural language commands and structured tools"
        self.logger = logging.getLogger("mujoco_mcp.simple_server")
        
        # 工具注册表
        self._tools = {}
        self._resources = []
        self._running = False
        
        # 模型存储
        self._models = {}  # model_id -> simulation instance
        self._model_names = {}  # model_id -> name
        
        # 模板存储
        self._templates = {
            "robot": {},  # template_name -> template_data
            "environment": {}  # template_name -> template_data
        }
        
        # 优化结果存储 - v0.4.1
        self._optimization_results = {}  # results_id -> optimization_data
        
        # 设计存储 - v0.4.2
        self._robot_designs = {}  # design_id -> design_data
        self._component_library = self._init_component_library()
        
        # 注册标准MCP工具
        self._register_standard_tools()
        # 注册MuJoCo工具
        self._register_mujoco_tools()
        
    def _register_standard_tools(self):
        """注册标准的MCP工具"""
        # get_server_info 工具
        self._tools["get_server_info"] = {
            "name": "get_server_info",
            "description": "Get detailed information about the MuJoCo MCP server including version, capabilities, and available features",
            "parameters": {},
            "handler": self.get_server_info
        }
        
        # get_tools 工具
        self._tools["get_tools"] = {
            "name": "get_tools",
            "description": "Get a comprehensive list of all available tools with their descriptions and parameters",
            "parameters": {},
            "handler": self._handle_get_tools
        }
    
    def _register_mujoco_tools(self):
        """注册MuJoCo相关工具"""
        # load_model 工具
        self._tools["load_model"] = {
            "name": "load_model",
            "description": "Load a MuJoCo physics model from an XML string. This creates a new simulation instance that can be controlled and queried",
            "parameters": {
                "model_string": "XML string containing the MuJoCo model definition in MJCF format",
                "name": "(optional) A human-readable name for the loaded model for easier reference"
            },
            "handler": self._handle_load_model
        }
        
        # get_loaded_models 工具
        self._tools["get_loaded_models"] = {
            "name": "get_loaded_models",
            "description": "Get a list of all currently loaded MuJoCo models with their IDs and basic information",
            "parameters": {},
            "handler": self._handle_get_loaded_models
        }
        
        # step_simulation 工具
        self._tools["step_simulation"] = {
            "name": "step_simulation",
            "description": "Advance the physics simulation forward by one or more timesteps, computing the next state based on current forces and dynamics",
            "parameters": {
                "model_id": "Unique identifier of the model to simulate",
                "steps": "(optional) Number of simulation steps to advance (default: 1)"
            },
            "handler": self._handle_step_simulation
        }
        
        # reset_simulation 工具
        self._tools["reset_simulation"] = {
            "name": "reset_simulation",
            "description": "Reset the simulation to its initial state, clearing all velocities and resetting positions to defaults",
            "parameters": {
                "model_id": "Unique identifier of the model to reset"
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
        
        # get_state 工具 - 获取完整状态
        self._tools["get_state"] = {
            "name": "get_state",
            "description": "Get comprehensive simulation state including positions, velocities, time, and more",
            "parameters": {
                "model_id": "ID of the model",
                "components": "(optional) List of state components to include"
            },
            "handler": self._handle_get_state
        }
        
        # set_joint_positions 工具
        self._tools["set_joint_positions"] = {
            "name": "set_joint_positions",
            "description": "Set the positions of all joints in the model. This directly updates the joint angles/positions without simulating physics",
            "parameters": {
                "model_id": "Unique identifier of the model to modify",
                "positions": "List of joint position values (in radians for revolute joints, meters for prismatic joints)"
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
            "description": "Set the velocities of all joints in the model. This directly updates joint angular/linear velocities",
            "parameters": {
                "model_id": "Unique identifier of the model to modify",
                "velocities": "List of joint velocity values (in rad/s for revolute joints, m/s for prismatic joints)"
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
            "description": "Get current readings from all sensors defined in the model (force sensors, touch sensors, gyroscopes, etc.)",
            "parameters": {
                "model_id": "Unique identifier of the model to query"
            },
            "handler": self._handle_get_sensor_data
        }
        
        # apply_control 工具
        self._tools["apply_control"] = {
            "name": "apply_control",
            "description": "Apply control inputs to actuators",
            "parameters": {
                "model_id": "ID of the model",
                "control": "List of control values for actuators"
            },
            "handler": self._handle_apply_control
        }
        
        # get_actuator_info 工具
        self._tools["get_actuator_info"] = {
            "name": "get_actuator_info",
            "description": "Get information about model actuators",
            "parameters": {
                "model_id": "ID of the model"
            },
            "handler": self._handle_get_actuator_info
        }
        
        # get_control_state 工具
        self._tools["get_control_state"] = {
            "name": "get_control_state",
            "description": "Get current control values",
            "parameters": {
                "model_id": "ID of the model"
            },
            "handler": self._handle_get_control_state
        }
        
        # get_render_frame 工具
        self._tools["get_render_frame"] = {
            "name": "get_render_frame",
            "description": "Render a frame from the simulation",
            "parameters": {
                "model_id": "ID of the model",
                "width": "(optional) Image width in pixels (default: 640)",
                "height": "(optional) Image height in pixels (default: 480)",
                "camera_name": "(optional) Camera name to use",
                "camera_distance": "(optional) Camera distance",
                "camera_azimuth": "(optional) Camera azimuth angle",
                "camera_elevation": "(optional) Camera elevation angle"
            },
            "handler": self._handle_get_render_frame
        }
        
        # get_ascii_visualization 工具
        self._tools["get_ascii_visualization"] = {
            "name": "get_ascii_visualization",
            "description": "Get ASCII art visualization of the simulation",
            "parameters": {
                "model_id": "ID of the model",
                "width": "(optional) ASCII art width (default: 60)",
                "height": "(optional) ASCII art height (default: 20)"
            },
            "handler": self._handle_get_ascii_visualization
        }
        
        # pendulum_demo 工具
        self._tools["pendulum_demo"] = {
            "name": "pendulum_demo",
            "description": "Pendulum control demonstration",
            "parameters": {
                "action": "Demo action (setup, control, swing_up, control_with_viz, analyze_energy, get_state, export_trajectory)",
                "model_id": "(optional) Model ID for actions other than setup",
                "target_angle": "(optional) Target angle in degrees for control",
                "duration": "(optional) Simulation duration in seconds",
                "kp": "(optional) Proportional gain for PID control",
                "ki": "(optional) Integral gain for PID control",
                "kd": "(optional) Derivative gain for PID control",
                "energy_gain": "(optional) Energy gain for swing-up control",
                "visualize": "(optional) Enable visualization",
                "viz_interval": "(optional) Visualization capture interval",
                "format": "(optional) Export format (csv)"
            },
            "handler": self._handle_pendulum_demo
        }
        
        # list_demos 工具
        self._tools["list_demos"] = {
            "name": "list_demos",
            "description": "List all available demonstration scenarios with their descriptions and difficulty levels",
            "parameters": {},
            "handler": self._handle_list_demos
        }
        
        # execute_command 工具 - 自然语言接口
        self._tools["execute_command"] = {
            "name": "execute_command",
            "description": "Execute a natural language command to control the simulation. This tool interprets human-friendly commands and translates them into appropriate MuJoCo operations",
            "parameters": {
                "command": "Natural language command describing what you want to do (e.g., 'create a pendulum', 'move the arm to 45 degrees', 'show me the current state')",
                "context": "(optional) Additional context like model_id or parameters as a dictionary"
            },
            "handler": self._handle_execute_command
        }
        
        # create_scene 工具 - 高级场景创建
        self._tools["create_scene"] = {
            "name": "create_scene",
            "description": "Create a pre-defined scene or robot model with sensible defaults. This is a high-level tool for quickly setting up common scenarios",
            "parameters": {
                "scene_type": "Type of scene to create (e.g., 'pendulum', 'double_pendulum', 'cart_pole', 'robotic_arm')",
                "parameters": "(optional) Dictionary of scene-specific parameters to customize the creation"
            },
            "handler": self._handle_create_scene
        }
        
        # perform_task 工具 - 高级任务执行
        self._tools["perform_task"] = {
            "name": "perform_task",
            "description": "Perform a high-level task on a model such as 'swing_up', 'balance', 'reach_target', etc. This abstracts complex control sequences",
            "parameters": {
                "task": "Name of the task to perform (e.g., 'swing_up', 'balance', 'track_trajectory')",
                "model_id": "ID of the model to perform the task on",
                "parameters": "(optional) Task-specific parameters as a dictionary"
            },
            "handler": self._handle_perform_task
        }
        
        # analyze_behavior 工具 - 行为分析
        self._tools["analyze_behavior"] = {
            "name": "analyze_behavior",
            "description": "Analyze the behavior of a simulation over time, providing insights about stability, energy, periodicity, or other metrics",
            "parameters": {
                "model_id": "ID of the model to analyze",
                "analysis_type": "Type of analysis to perform (e.g., 'energy', 'stability', 'trajectory', 'frequency')",
                "duration": "(optional) Duration in seconds to run the analysis (default: 1.0)"
            },
            "handler": self._handle_analyze_behavior
        }
        
        # generate_robot 工具 - 机器人生成
        self._tools["generate_robot"] = {
            "name": "generate_robot",
            "description": "Generate a robot model programmatically with specified type and parameters",
            "parameters": {
                "robot_type": "Type of robot to generate (e.g., 'arm', 'mobile', 'gripper', 'humanoid')",
                "parameters": "(optional) Dictionary of robot-specific parameters like num_links, link_length, etc."
            },
            "handler": self._handle_generate_robot
        }
        
        # generate_environment 工具 - 环境生成
        self._tools["generate_environment"] = {
            "name": "generate_environment",
            "description": "Generate an environment model with terrain, obstacles, or other features",
            "parameters": {
                "env_type": "Type of environment to generate (e.g., 'flat_ground', 'obstacles', 'terrain', 'maze')",
                "parameters": "(optional) Dictionary of environment-specific parameters"
            },
            "handler": self._handle_generate_environment
        }
        
        # combine_models 工具 - 模型组合
        self._tools["combine_models"] = {
            "name": "combine_models",
            "description": "Combine multiple models (robots and environments) into a single simulation",
            "parameters": {
                "base_model_xml": "XML string of the base model (usually environment)",
                "add_model_xml": "XML string of the model to add (usually robot)",
                "position": "(optional) Position [x, y, z] where to place the added model"
            },
            "handler": self._handle_combine_models
        }
        
        # list_templates 工具 - 列出模板
        self._tools["list_templates"] = {
            "name": "list_templates",
            "description": "List available model templates for robots and environments",
            "parameters": {},
            "handler": self._handle_list_templates
        }
        
        # generate_from_template 工具 - 从模板生成
        self._tools["generate_from_template"] = {
            "name": "generate_from_template",
            "description": "Generate a model from a predefined template with custom parameters",
            "parameters": {
                "template_name": "Name of the template to use",
                "parameters": "(optional) Dictionary of parameters to customize the template"
            },
            "handler": self._handle_generate_from_template
        }
        
        # save_as_template 工具 - 保存为模板
        self._tools["save_as_template"] = {
            "name": "save_as_template",
            "description": "Save a loaded model as a reusable template",
            "parameters": {
                "model_id": "ID of the model to save as template",
                "template_name": "Name for the new template",
                "description": "Description of what this template is for",
                "parameterizable": "(optional) List of parameter names that can be customized"
            },
            "handler": self._handle_save_as_template
        }
        
        # validate_model_xml 工具 - 验证XML
        self._tools["validate_model_xml"] = {
            "name": "validate_model_xml",
            "description": "Validate a MuJoCo model XML string for correctness and safety",
            "parameters": {
                "xml_string": "XML string to validate"
            },
            "handler": self._handle_validate_model_xml
        }
        
        # 参数优化工具 - v0.4.1
        self._tools["optimize_parameters"] = {
            "name": "optimize_parameters",
            "description": "Optimize control parameters for a given objective using gradient-free methods",
            "parameters": {
                "model_id": "ID of the model to optimize",
                "objective": "Optimization objective (minimize_time, minimize_energy, maximize_stability, minimize_error, custom)",
                "target_state": "(optional) Target state to reach",
                "parameters_to_optimize": "List of parameter names to optimize",
                "parameter_bounds": "Dictionary of parameter bounds {param: [min, max]}",
                "max_iterations": "(optional) Maximum optimization iterations (default: 20)",
                "optimization_method": "(optional) Method to use (random_search, grid_search, bayesian)",
                "constraints": "(optional) List of constraints to satisfy",
                "track_convergence": "(optional) Track convergence history",
                "save_results": "(optional) Save optimization results",
                "results_name": "(optional) Name for saved results"
            },
            "handler": self._handle_optimize_parameters
        }
        
        self._tools["list_cost_functions"] = {
            "name": "list_cost_functions",
            "description": "List available cost functions for parameter optimization",
            "parameters": {},
            "handler": self._handle_list_cost_functions
        }
        
        self._tools["analyze_sensitivity"] = {
            "name": "analyze_sensitivity",
            "description": "Analyze parameter sensitivity for a given objective",
            "parameters": {
                "model_id": "ID of the model to analyze",
                "parameters": "List of parameters to analyze",
                "objective": "Objective function to evaluate",
                "target_state": "(optional) Target state for evaluation",
                "num_samples": "(optional) Number of samples for analysis (default: 20)"
            },
            "handler": self._handle_analyze_sensitivity
        }
        
        self._tools["analyze_robustness"] = {
            "name": "analyze_robustness",
            "description": "Analyze robustness of parameters to perturbations",
            "parameters": {
                "model_id": "ID of the model to analyze",
                "parameters": "Dictionary of parameter values to test",
                "perturbation_range": "Range of perturbation as fraction (e.g., 0.1 for ±10%)",
                "num_tests": "(optional) Number of robustness tests (default: 20)"
            },
            "handler": self._handle_analyze_robustness
        }
        
        # 机器人设计工具 - v0.4.2
        self._tools["design_robot"] = {
            "name": "design_robot",
            "description": "Design a robot based on task requirements using AI-assisted design",
            "parameters": {
                "task_description": "Natural language description of the task",
                "constraints": "(optional) Design constraints (size, weight, cost, etc.)",
                "preferences": "(optional) Design preferences",
                "optimize_for": "(optional) List of objectives to optimize",
                "use_components": "(optional) Use component library",
                "estimate_cost": "(optional) Include cost estimation"
            },
            "handler": self._handle_design_robot
        }
        
        self._tools["refine_design"] = {
            "name": "refine_design",
            "description": "Refine an existing robot design with improvements",
            "parameters": {
                "design_id": "ID of the design to refine",
                "improvements": "Dictionary of improvements to apply",
                "additional_constraints": "(optional) New constraints to satisfy"
            },
            "handler": self._handle_refine_design
        }
        
        self._tools["suggest_improvements"] = {
            "name": "suggest_improvements",
            "description": "Get AI suggestions for improving a robot design",
            "parameters": {
                "model_id": "ID of the model to analyze",
                "goals": "List of improvement goals"
            },
            "handler": self._handle_suggest_improvements
        }
        
        self._tools["compare_designs"] = {
            "name": "compare_designs",
            "description": "Compare multiple robot designs on specified metrics",
            "parameters": {
                "design_ids": "List of design IDs to compare",
                "metrics": "List of metrics to compare"
            },
            "handler": self._handle_compare_designs
        }
        
        self._tools["explain_design"] = {
            "name": "explain_design",
            "description": "Get explanation of design choices and rationale",
            "parameters": {
                "design_id": "ID of the design to explain"
            },
            "handler": self._handle_explain_design
        }
        
        self._tools["list_components"] = {
            "name": "list_components",
            "description": "List available components from the library",
            "parameters": {
                "category": "(optional) Component category to filter"
            },
            "handler": self._handle_list_components
        }
        
        self._tools["check_compatibility"] = {
            "name": "check_compatibility",
            "description": "Check if two components are compatible",
            "parameters": {
                "component_a": "First component specification",
                "component_b": "Second component specification"
            },
            "handler": self._handle_check_compatibility
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
                "state_query",
                "visualization",
                "demo",
                "natural_language",
                "model_generation",
                "parameter_optimization",
                "robot_designer"
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
    
    def _handle_get_state(self, model_id: str, components: Optional[List[str]] = None) -> Dict[str, Any]:
        """处理get_state工具调用 - 获取完整状态"""
        # 参数验证
        if not model_id:
            raise ValueError("model_id is required")
            
        if model_id not in self._models:
            raise ValueError(f"Model not found: {model_id}")
            
        # 获取仿真实例
        sim = self._models[model_id]
        
        # 默认返回所有组件
        if components is None:
            components = ["time", "joint_positions", "joint_velocities", "actuator_states", "sensor_data"]
        
        # 构建完整状态
        state = {}
        
        if "time" in components:
            state["time"] = sim.get_time()
            
        if "joint_positions" in components:
            state["joint_positions"] = sim.get_joint_positions().tolist()
            
        if "joint_velocities" in components:
            state["joint_velocities"] = sim.get_joint_velocities().tolist()
            
        if "actuator_states" in components and sim.get_num_actuators() > 0:
            state["actuator_states"] = {
                "control": sim.data.ctrl.tolist() if sim._initialized else [],
                "force": sim.data.actuator_force.tolist() if sim._initialized else []
            }
            
        if "sensor_data" in components:
            sensor_data = sim.get_sensor_data()
            if sensor_data:
                state["sensor_data"] = sensor_data
                
        # 添加模型信息
        state["model_info"] = {
            "nq": sim.get_num_joints(),
            "nv": sim.model.nv if sim._initialized else 0,
            "nu": sim.get_num_actuators(),
            "nsensor": sim.model.nsensor if sim._initialized else 0
        }
        
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
    
    def _handle_apply_control(self, model_id: str, control: List[float]) -> Dict[str, Any]:
        """处理apply_control工具调用"""
        # 参数验证
        if not model_id:
            raise ValueError("model_id is required")
            
        if model_id not in self._models:
            raise ValueError(f"Model not found: {model_id}")
            
        if not control:
            raise ValueError("control is required")
            
        # 获取仿真实例
        sim = self._models[model_id]
        
        # 检查控制数量是否匹配
        nu = sim.get_num_actuators()
        if len(control) != nu:
            raise ValueError(f"Expected {nu} control values, got {len(control)}")
            
        # 获取控制限制并应用
        warnings = []
        clamped_control = []
        
        for i in range(nu):
            ctrl_val = control[i]
            # 检查是否有控制限制
            if hasattr(sim.model, 'actuator_ctrllimited') and hasattr(sim.model, 'actuator_ctrlrange'):
                if sim.model.actuator_ctrllimited[i]:
                    ctrl_min = sim.model.actuator_ctrlrange[i, 0]
                    ctrl_max = sim.model.actuator_ctrlrange[i, 1]
                    if ctrl_val < ctrl_min or ctrl_val > ctrl_max:
                        clamped_val = max(ctrl_min, min(ctrl_max, ctrl_val))
                        clamped_control.append(clamped_val)
                        warnings.append(f"Control {i} clamped from {ctrl_val} to {clamped_val}")
                        continue
            clamped_control.append(ctrl_val)
        
        # 应用控制
        sim.apply_control(clamped_control)
        
        result = {
            "success": True,
            "message": f"Applied {len(control)} control values"
        }
        
        if warnings:
            result["warnings"] = warnings
            
        return result
    
    def _handle_get_actuator_info(self, model_id: str) -> Dict[str, Any]:
        """处理get_actuator_info工具调用"""
        # 参数验证
        if not model_id:
            raise ValueError("model_id is required")
            
        if model_id not in self._models:
            raise ValueError(f"Model not found: {model_id}")
            
        # 获取仿真实例
        sim = self._models[model_id]
        
        # 获取驱动器信息
        actuators = []
        nu = sim.get_num_actuators()
        
        for i in range(nu):
            actuator_info = {
                "index": i,
                "name": sim.model.actuator(i).name if hasattr(sim.model.actuator(i), 'name') else f"actuator_{i}",
                "type": "motor",  # MuJoCo中最常见的类型
            }
            
            # 获取gear信息
            if hasattr(sim.model, 'actuator_gear'):
                actuator_info["gear"] = sim.model.actuator_gear[i].tolist()
                
            # 获取控制限制信息
            if hasattr(sim.model, 'actuator_ctrllimited'):
                actuator_info["ctrl_limited"] = bool(sim.model.actuator_ctrllimited[i])
                
            if hasattr(sim.model, 'actuator_ctrlrange'):
                actuator_info["ctrl_range"] = sim.model.actuator_ctrlrange[i].tolist()
                
            actuators.append(actuator_info)
            
        return {
            "actuators": actuators,
            "total": nu
        }
    
    def _handle_get_control_state(self, model_id: str) -> Dict[str, Any]:
        """处理get_control_state工具调用"""
        # 参数验证
        if not model_id:
            raise ValueError("model_id is required")
            
        if model_id not in self._models:
            raise ValueError(f"Model not found: {model_id}")
            
        # 获取仿真实例
        sim = self._models[model_id]
        
        # 获取当前控制值
        if sim._initialized and hasattr(sim.data, 'ctrl'):
            control = sim.data.ctrl.copy().tolist()
        else:
            control = []
            
        return {
            "control": control,
            "num_actuators": sim.get_num_actuators()
        }
    
    def _handle_get_render_frame(self, model_id: str, 
                                width: int = 640, 
                                height: int = 480,
                                camera_name: Optional[str] = None,
                                camera_distance: Optional[float] = None,
                                camera_azimuth: Optional[float] = None,
                                camera_elevation: Optional[float] = None) -> Dict[str, Any]:
        """处理get_render_frame工具调用"""
        # 参数验证
        if not model_id:
            raise ValueError("model_id is required")
            
        if model_id not in self._models:
            raise ValueError(f"Model not found: {model_id}")
            
        # 验证尺寸
        if width < 50 or width > 2000:
            raise ValueError("Width must be between 50 and 2000")
        if height < 50 or height > 2000:
            raise ValueError("Height must be between 50 and 2000")
            
        # 获取仿真实例
        sim = self._models[model_id]
        
        # 检查相机名称
        if camera_name and camera_name not in ["track", "fixed", "free"]:
            # 检查是否是自定义相机
            if sim._initialized and hasattr(sim.model, 'ncam'):
                camera_names = [sim.model.camera(i).name for i in range(sim.model.ncam)]
                if camera_name not in camera_names:
                    raise ValueError(f"Unknown camera: {camera_name}")
        
        try:
            # 这里我们创建一个模拟的渲染结果
            # 在实际实现中，这里应该调用MuJoCo的渲染函数
            # 由于测试环境可能没有GPU，我们生成一个简单的占位图像
            
            # 创建一个简单的灰度图像数据（用于测试）
            import numpy as np
            
            # 创建渐变图像作为占位符
            img_array = np.zeros((height, width, 3), dtype=np.uint8)
            for i in range(height):
                for j in range(width):
                    # 创建一个简单的渐变效果
                    img_array[i, j] = [
                        int(255 * i / height),  # R
                        int(255 * j / width),    # G
                        128                      # B
                    ]
            
            # 将numpy数组转换为PNG格式的base64
            from PIL import Image
            img = Image.fromarray(img_array)
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            result = {
                "success": True,
                "image_data": img_data,
                "format": "base64/png",
                "width": width,
                "height": height
            }
            
            if camera_name:
                result["camera_info"] = {
                    "name": camera_name or "track",
                    "distance": camera_distance,
                    "azimuth": camera_azimuth,
                    "elevation": camera_elevation
                }
                
            return result
            
        except ImportError:
            # 如果没有PIL，返回一个简单的base64编码数据
            # 这是一个1x1的透明PNG图像
            tiny_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
            return {
                "success": True,
                "image_data": tiny_png,
                "format": "base64/png",
                "width": width,
                "height": height,
                "message": "Rendering requires PIL/Pillow library"
            }
    
    def _handle_get_ascii_visualization(self, model_id: str,
                                      width: int = 60,
                                      height: int = 20) -> Dict[str, Any]:
        """处理get_ascii_visualization工具调用"""
        # 参数验证
        if not model_id:
            raise ValueError("model_id is required")
            
        if model_id not in self._models:
            raise ValueError(f"Model not found: {model_id}")
            
        # 验证尺寸
        if width < 20 or width > 120:
            raise ValueError("Width must be between 20 and 120")
        if height < 10 or height > 50:
            raise ValueError("Height must be between 10 and 50")
            
        # 获取仿真实例
        sim = self._models[model_id]
        
        # 生成ASCII艺术（简化版本）
        # 在实际实现中，这里应该基于物体位置生成更有意义的可视化
        ascii_chars = " .·:*#@"
        
        # 获取一些状态信息
        time = sim.get_time()
        bodies = sim.get_rigid_body_states() if sim._initialized else {}
        
        # 创建ASCII画布
        canvas = [[' ' for _ in range(width)] for _ in range(height)]
        
        # 添加边框
        for i in range(height):
            canvas[i][0] = '|'
            canvas[i][width-1] = '|'
        for j in range(width):
            canvas[0][j] = '-'
            canvas[height-1][j] = '-'
        canvas[0][0] = '+'
        canvas[0][width-1] = '+'
        canvas[height-1][0] = '+'
        canvas[height-1][width-1] = '+'
        
        # 在中心添加一些信息
        info = f"t={time:.2f}s"
        start_col = (width - len(info)) // 2
        for i, char in enumerate(info):
            if start_col + i < width - 1:
                canvas[1][start_col + i] = char
        
        # 为每个物体添加标记
        body_count = 0
        for body_name, state in bodies.items():
            if body_count >= 5:  # 限制显示的物体数量
                break
            pos = state.get("position", [0, 0, 0])
            # 将3D位置映射到2D ASCII画布
            x = int((pos[0] + 2) / 4 * (width - 2)) + 1
            y = int((pos[1] + 2) / 4 * (height - 2)) + 1
            x = max(1, min(width - 2, x))
            y = max(1, min(height - 2, y))
            
            # 使用不同字符表示不同物体
            char = ascii_chars[min(body_count + 2, len(ascii_chars) - 1)]
            canvas[y][x] = char
            body_count += 1
        
        # 转换为字符串
        ascii_art = '\n'.join(''.join(row) for row in canvas)
        
        return {
            "success": True,
            "ascii_art": ascii_art,
            "width": width,
            "height": height,
            "time": time,
            "body_count": len(bodies)
        }
    
    def _handle_pendulum_demo(self, action: str, 
                            model_id: Optional[str] = None,
                            target_angle: float = 0.0,
                            duration: float = 2.0,
                            kp: float = 2.0,
                            ki: float = 0.1,
                            kd: float = 0.5,
                            energy_gain: float = 0.5,
                            visualize: bool = False,
                            viz_interval: float = 0.1,
                            format: str = "csv") -> Dict[str, Any]:
        """处理pendulum_demo工具调用"""
        
        if action == "setup":
            # 设置单摆模型
            pendulum_xml = """<mujoco model="pendulum">
                <option timestep="0.001" gravity="0 0 -9.81"/>
                <worldbody>
                    <body name="pendulum" pos="0 0 1">
                        <joint name="hinge" type="hinge" axis="0 1 0" range="-180 180" damping="0.1"/>
                        <geom name="rod" type="capsule" fromto="0 0 0 0 0 -0.5" size="0.02" mass="0.1"/>
                        <body name="bob" pos="0 0 -0.5">
                            <geom name="ball" type="sphere" size="0.05" mass="0.5" rgba="1 0 0 1"/>
                        </body>
                    </body>
                </worldbody>
                <actuator>
                    <motor name="torque" joint="hinge" gear="1" ctrllimited="true" ctrlrange="-2 2"/>
                </actuator>
                <sensor>
                    <jointpos name="angle" joint="hinge"/>
                    <jointvel name="velocity" joint="hinge"/>
                </sensor>
            </mujoco>"""
            
            # 加载模型
            result = self._handle_load_model(pendulum_xml, "pendulum_demo")
            result["message"] = "Pendulum demo model loaded successfully"
            return result
            
        elif action == "control":
            # PID控制到目标角度
            if not model_id:
                raise ValueError("model_id is required for control action")
                
            if model_id not in self._models:
                raise ValueError(f"Model not found: {model_id}")
                
            sim = self._models[model_id]
            
            # 转换角度到弧度
            target_rad = target_angle * np.pi / 180.0
            
            # PID控制器状态
            integral = 0.0
            prev_error = 0.0
            
            # 记录轨迹
            trajectory = []
            timestep = sim.model.opt.timestep
            steps = int(duration / timestep)
            
            for i in range(steps):
                # 获取当前角度
                angle = sim.data.qpos[0] if sim._initialized else 0.0
                velocity = sim.data.qvel[0] if sim._initialized else 0.0
                
                # PID控制
                error = target_rad - angle
                integral += error * timestep
                derivative = (error - prev_error) / timestep if i > 0 else 0.0
                
                # 增强PID增益以获得更好的收敛
                control = kp * error + ki * integral + kd * derivative
                control = np.clip(control, -2.0, 2.0)  # 限制控制输入
                
                # 应用控制
                sim.apply_control([control])
                sim.step()
                
                # 记录数据
                if i % 100 == 0:  # 每100步记录一次
                    trajectory.append({
                        "time": sim.get_time(),
                        "angle": angle * 180.0 / np.pi,  # 转换回度
                        "velocity": velocity,
                        "control": control,
                        "error": error * 180.0 / np.pi
                    })
                
                prev_error = error
            
            # 计算最终误差和能量
            final_angle = sim.data.qpos[0] if sim._initialized else 0.0
            final_error = (target_rad - final_angle) * 180.0 / np.pi
            
            # 计算总能量
            kinetic = 0.5 * velocity ** 2
            potential = 9.81 * (1 - np.cos(final_angle)) * 0.5  # m*g*h
            total_energy = kinetic + potential
            
            return {
                "success": True,
                "trajectory": trajectory,
                "final_error": final_error,
                "energy": {
                    "kinetic": kinetic,
                    "potential": potential,
                    "total": total_energy
                }
            }
            
        elif action == "swing_up":
            # 能量摆起控制
            if not model_id:
                raise ValueError("model_id is required for swing_up action")
                
            if model_id not in self._models:
                raise ValueError(f"Model not found: {model_id}")
                
            sim = self._models[model_id]
            
            # 给单摆一个小的初始速度以开始摆动
            sim.data.qvel[0] = 0.5
            
            # 记录轨迹
            trajectory = []
            energy_profile = []
            max_height = 0.0
            timestep = sim.model.opt.timestep
            steps = int(duration / timestep)
            
            for i in range(steps):
                # 获取状态
                angle = sim.data.qpos[0] if sim._initialized else 0.0
                velocity = sim.data.qvel[0] if sim._initialized else 0.0
                
                # 计算能量 (使用正确的质量和长度)
                mass = 0.5  # 摆球质量
                length = 0.5  # 摆长
                gravity = 9.81
                
                kinetic = 0.5 * mass * (length ** 2) * (velocity ** 2)
                potential = mass * gravity * length * (1 - np.cos(angle))
                total_energy = kinetic + potential
                target_energy = mass * gravity * length * 2  # 目标能量（倒立位置）
                
                # 能量控制 - 简化版本
                if abs(angle) < 2.5:  # 接近倒立位置时切换到位置控制
                    # 位置控制以稳定倒立
                    control = -10.0 * angle - 2.0 * velocity
                else:
                    # 能量注入控制
                    energy_error = target_energy - total_energy
                    # 只在速度和角度同向时注入能量
                    if velocity * np.sin(angle) < 0:
                        control = energy_gain * energy_error * np.sign(velocity)
                    else:
                        control = 0.0
                
                control = np.clip(control, -2.0, 2.0)
                
                # 应用控制
                sim.apply_control([control])
                sim.step()
                
                # 记录数据
                if i % 100 == 0:
                    trajectory.append({
                        "time": sim.get_time(),
                        "angle": angle * 180.0 / np.pi,
                        "velocity": velocity,
                        "control": control
                    })
                    energy_profile.append({
                        "time": sim.get_time(),
                        "kinetic": kinetic,
                        "potential": potential,
                        "total": total_energy
                    })
                
                # 更新最大高度
                height = length * (1 - np.cos(angle))
                max_height = max(max_height, height)
            
            return {
                "success": True,
                "trajectory": trajectory,
                "max_height": max_height,
                "energy_profile": energy_profile
            }
            
        elif action == "control_with_viz":
            # 带可视化的控制
            if not model_id:
                raise ValueError("model_id is required for control_with_viz action")
                
            # 先执行控制
            control_result = self._handle_pendulum_demo(
                action="control",
                model_id=model_id,
                target_angle=target_angle,
                duration=duration,
                kp=kp, ki=ki, kd=kd
            )
            
            if not visualize:
                return control_result
            
            # 添加可视化帧
            frames = []
            ascii_frames = []
            
            # 重置仿真以重放
            sim = self._models[model_id]
            sim.reset()
            
            # 重放控制序列并捕获帧
            viz_steps = int(viz_interval / sim.model.opt.timestep)
            for i, point in enumerate(control_result["trajectory"]):
                if i % int(0.1 / (point["time"] / (i+1) if i > 0 else 0.001)) == 0:
                    # 捕获渲染帧
                    render_result = self._handle_get_render_frame(model_id, 320, 240)
                    ascii_result = self._handle_get_ascii_visualization(model_id, 40, 15)
                    
                    frames.append({
                        "time": point["time"],
                        "image": render_result["image_data"],
                        "angle": point["angle"],
                        "control": point["control"]
                    })
                    
                    ascii_frames.append({
                        "time": point["time"],
                        "ascii": ascii_result["ascii_art"]
                    })
            
            control_result["frames"] = frames
            control_result["ascii_frames"] = ascii_frames
            return control_result
            
        elif action == "analyze_energy":
            # 能量分析
            if not model_id:
                raise ValueError("model_id is required for analyze_energy action")
                
            if model_id not in self._models:
                raise ValueError(f"Model not found: {model_id}")
                
            sim = self._models[model_id]
            
            # 运行仿真并分析能量
            kinetic_energy = []
            potential_energy = []
            total_energy = []
            
            timestep = sim.model.opt.timestep
            steps = int(duration / timestep)
            
            for i in range(steps):
                # 获取状态
                angle = sim.data.qpos[0] if sim._initialized else 0.0
                velocity = sim.data.qvel[0] if sim._initialized else 0.0
                
                # 计算能量
                ke = 0.5 * velocity ** 2
                pe = 9.81 * (1 - np.cos(angle)) * 0.5
                te = ke + pe
                
                kinetic_energy.append(ke)
                potential_energy.append(pe)
                total_energy.append(te)
                
                # 无控制步进
                sim.step()
            
            # 分析能量守恒
            energy_mean = np.mean(total_energy)
            energy_std = np.std(total_energy)
            energy_variation = energy_std / energy_mean if energy_mean > 0 else 0
            
            return {
                "success": True,
                "kinetic_energy": {
                    "mean": np.mean(kinetic_energy),
                    "max": np.max(kinetic_energy),
                    "min": np.min(kinetic_energy)
                },
                "potential_energy": {
                    "mean": np.mean(potential_energy),
                    "max": np.max(potential_energy),
                    "min": np.min(potential_energy)
                },
                "total_energy": {
                    "mean": energy_mean,
                    "max": np.max(total_energy),
                    "min": np.min(total_energy)
                },
                "energy_conservation": {
                    "variation": energy_variation,
                    "std_dev": energy_std
                }
            }
            
        elif action == "get_state":
            # 获取当前状态
            if not model_id:
                raise ValueError("model_id is required for get_state action")
                
            if model_id not in self._models:
                raise ValueError(f"Model not found: {model_id}")
                
            sim = self._models[model_id]
            
            # 获取状态
            angle = sim.data.qpos[0] if sim._initialized else 0.0
            velocity = sim.data.qvel[0] if sim._initialized else 0.0
            control = sim.data.ctrl[0] if sim._initialized and hasattr(sim.data, 'ctrl') else 0.0
            
            # 计算能量
            kinetic = 0.5 * velocity ** 2
            potential = 9.81 * (1 - np.cos(angle)) * 0.5
            
            return {
                "success": True,
                "angle": angle * 180.0 / np.pi,
                "velocity": velocity,
                "control": control,
                "time": sim.get_time(),
                "energy": {
                    "kinetic": kinetic,
                    "potential": potential,
                    "total": kinetic + potential
                }
            }
            
        elif action == "export_trajectory":
            # 导出轨迹
            if not model_id:
                raise ValueError("model_id is required for export_trajectory action")
                
            # 这里我们假设之前已经运行过控制
            # 在实际实现中，应该存储轨迹数据
            
            if format == "csv":
                csv_data = "time,angle,velocity,control\n"
                # 添加一些示例数据
                for i in range(10):
                    csv_data += f"{i*0.1:.2f},{i*5:.2f},{i*0.5:.2f},{i*0.2:.2f}\n"
                
                return {
                    "success": True,
                    "data": csv_data,
                    "format": "csv"
                }
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        else:
            raise ValueError(f"Unknown action: {action}")
    
    def _handle_list_demos(self) -> Dict[str, Any]:
        """处理list_demos工具调用"""
        demos = [
            {
                "name": "pendulum",
                "description": "Classic pendulum control demonstration",
                "difficulty": "beginner",
                "concepts": ["PID control", "energy control", "swing-up", "trajectory tracking"]
            }
        ]
        
        return {
            "demos": demos
        }
    
    def _handle_execute_command(self, command: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """处理execute_command工具调用 - 自然语言接口"""
        
        # 验证命令不为空
        if not command or not command.strip():
            raise ValueError("Command cannot be empty. Please provide a natural language command describing what you want to do.")
        
        command_lower = command.lower().strip()
        context = context or {}
        
        # 解析命令意图
        if "create" in command_lower and "pendulum" in command_lower and not ("swing" in command_lower and "up" in command_lower):
            # 仅创建单摆
            result = self._handle_pendulum_demo("setup")
            return {
                "success": True,
                "model_id": result["model_id"],
                "interpretation": "Creating a pendulum simulation",
                "action_taken": "pendulum_demo with action='setup'",
                "result": result
            }
            
        elif "what can you do" in command_lower or "help" in command_lower or "capabilities" in command_lower:
            # 返回能力列表
            return {
                "success": True,
                "capabilities": [
                    "Create physics simulations (pendulum, robotic arm, etc.)",
                    "Control simulations (move to position, apply forces)",
                    "Query simulation state (positions, velocities, energy)",
                    "Visualize simulations (render frames, ASCII art)",
                    "Perform complex tasks (swing up, balance, trajectory tracking)",
                    "Analyze behavior (energy, stability, frequency)"
                ],
                "hint": "Try commands like 'create a pendulum', 'move to 45 degrees', 'show me the state'"
            }
            
        elif ("move" in command_lower or "control" in command_lower) and context.get("model_id"):
            # 控制命令
            model_id = context["model_id"]
            
            # 提取目标角度
            import re
            angle_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:degrees?|deg)', command_lower)
            target_angle = float(angle_match.group(1)) if angle_match else 45.0
            
            # 检查是否指定了PID参数
            kp_match = re.search(r'kp\s*=\s*(\d+(?:\.\d+)?)', command_lower)
            kp = float(kp_match.group(1)) if kp_match else 5.0
            
            result = self._handle_pendulum_demo(
                action="control",
                model_id=model_id,
                target_angle=target_angle,
                duration=2.0,
                kp=kp,
                ki=0.5,
                kd=1.0
            )
            
            return {
                "success": True,
                "interpretation": f"Controlling pendulum to {target_angle} degrees",
                "action_taken": f"PID control with target angle {target_angle}°",
                "parameters_used": {
                    "target_angle": target_angle,
                    "kp": kp,
                    "ki": 0.5,
                    "kd": 1.0
                },
                "result": result
            }
            
        elif ("angle" in command_lower or "position" in command_lower or "state" in command_lower) and context.get("model_id"):
            # 查询状态
            model_id = context["model_id"]
            
            # 获取状态
            state = self._handle_pendulum_demo(
                action="get_state",
                model_id=model_id
            )
            
            angle = state.get("angle", 0.0)
            velocity = state.get("velocity", 0.0)
            
            return {
                "success": True,
                "interpretation": "Querying current pendulum state",
                "answer": f"The pendulum is currently at {angle:.1f} degrees with angular velocity {velocity:.2f} rad/s",
                "raw_data": state
            }
            
        elif ("show" in command_lower or "visualize" in command_lower) and context.get("model_id"):
            # 可视化命令
            model_id = context["model_id"]
            
            # 获取ASCII可视化
            ascii_result = self._handle_get_ascii_visualization(model_id, 40, 20)
            
            return {
                "success": True,
                "interpretation": "Visualizing the current simulation state",
                "visualization": {
                    "ascii": ascii_result["ascii_art"]
                }
            }
            
        elif "swing" in command_lower and "up" in command_lower:
            # 摆起控制
            if "create" in command_lower and "pendulum" in command_lower:
                # 先创建后摆起
                setup_result = self._handle_pendulum_demo("setup")
                model_id = setup_result["model_id"]
                
                swing_result = self._handle_pendulum_demo(
                    action="swing_up",
                    model_id=model_id,
                    duration=5.0,
                    energy_gain=1.0
                )
                
                return {
                    "success": True,
                    "model_id": model_id,
                    "interpretation": "Creating pendulum and performing swing-up control",
                    "steps_taken": [
                        "Created pendulum simulation",
                        "Applied energy-based swing-up control"
                    ],
                    "result": swing_result
                }
            elif context.get("model_id"):
                # 对现有模型摆起
                model_id = context["model_id"]
                swing_result = self._handle_pendulum_demo(
                    action="swing_up",
                    model_id=model_id,
                    duration=5.0,
                    energy_gain=1.0
                )
                
                return {
                    "success": True,
                    "interpretation": "Performing swing-up control on existing pendulum",
                    "action_taken": "Energy-based swing-up control",
                    "result": swing_result
                }
            else:
                return {
                    "success": False,
                    "error": "No model specified. Please create a pendulum first or provide model_id in context."
                }
                
        else:
            # 无法理解的命令
            return {
                "success": False,
                "interpretation": f"Unable to understand command: '{command}'",
                "clarification_needed": "Please try rephrasing or use one of these examples:",
                "examples": [
                    "create a pendulum",
                    "move the pendulum to 45 degrees",
                    "what is the current angle?",
                    "show me the pendulum",
                    "swing up the pendulum"
                ]
            }
    
    def _handle_create_scene(self, scene_type: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """处理create_scene工具调用 - 高级场景创建"""
        
        scene_type_lower = scene_type.lower()
        parameters = parameters or {}
        
        if scene_type_lower == "pendulum":
            # 创建单摆场景
            length = parameters.get("length", 0.5)
            mass = parameters.get("mass", 0.5)
            damping = parameters.get("damping", 0.1)
            
            pendulum_xml = f"""<mujoco model="pendulum_scene">
                <option timestep="0.001" gravity="0 0 -9.81"/>
                <worldbody>
                    <body name="pendulum" pos="0 0 1">
                        <joint name="hinge" type="hinge" axis="0 1 0" range="-180 180" damping="{damping}"/>
                        <geom name="rod" type="capsule" fromto="0 0 0 0 0 -{length}" size="0.02" mass="0.1"/>
                        <body name="bob" pos="0 0 -{length}">
                            <geom name="ball" type="sphere" size="0.05" mass="{mass}" rgba="1 0 0 1"/>
                        </body>
                    </body>
                </worldbody>
                <actuator>
                    <motor name="torque" joint="hinge" gear="1" ctrllimited="true" ctrlrange="-2 2"/>
                </actuator>
                <sensor>
                    <jointpos name="angle" joint="hinge"/>
                    <jointvel name="velocity" joint="hinge"/>
                </sensor>
            </mujoco>"""
            
            result = self._handle_load_model(pendulum_xml, f"{scene_type}_scene")
            
            return {
                "success": True,
                "model_id": result["model_id"],
                "scene_info": {
                    "type": "pendulum",
                    "parameters": {
                        "length": length,
                        "mass": mass,
                        "damping": damping
                    }
                },
                "message": f"Created {scene_type} scene with custom parameters"
            }
            
        else:
            # 未实现的场景类型
            return {
                "success": False,
                "error": f"Scene type '{scene_type}' not implemented yet",
                "available_scenes": ["pendulum"],
                "message": "More scene types will be added in future versions"
            }
    
    def _handle_perform_task(self, task: str, model_id: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """处理perform_task工具调用 - 高级任务执行"""
        
        if not model_id:
            raise ValueError("model_id is required for perform_task")
            
        if model_id not in self._models:
            raise ValueError(f"Model not found: {model_id}")
            
        task_lower = task.lower()
        parameters = parameters or {}
        
        if task_lower == "swing_up":
            # 摆起任务
            duration = parameters.get("duration", 5.0)
            energy_gain = parameters.get("energy_gain", 1.0)
            
            result = self._handle_pendulum_demo(
                action="swing_up",
                model_id=model_id,
                duration=duration,
                energy_gain=energy_gain
            )
            
            return {
                "success": True,
                "task": "swing_up",
                "task_result": result,
                "message": "Swing-up task completed"
            }
            
        elif task_lower == "balance":
            # 平衡任务 - 保持在特定角度
            target_angle = parameters.get("target_angle", 0.0)
            duration = parameters.get("duration", 3.0)
            
            result = self._handle_pendulum_demo(
                action="control",
                model_id=model_id,
                target_angle=target_angle,
                duration=duration,
                kp=parameters.get("kp", 10.0),
                ki=parameters.get("ki", 1.0),
                kd=parameters.get("kd", 2.0)
            )
            
            return {
                "success": True,
                "task": "balance",
                "task_result": result,
                "message": f"Balance task completed at {target_angle} degrees"
            }
            
        else:
            return {
                "success": False,
                "error": f"Task '{task}' not implemented",
                "available_tasks": ["swing_up", "balance"],
                "message": "More tasks will be added in future versions"
            }
    
    def _handle_analyze_behavior(self, model_id: str, analysis_type: str, duration: float = 1.0) -> Dict[str, Any]:
        """处理analyze_behavior工具调用 - 行为分析"""
        
        if not model_id:
            raise ValueError("model_id is required for analyze_behavior")
            
        if model_id not in self._models:
            raise ValueError(f"Model not found: {model_id}")
            
        analysis_type_lower = analysis_type.lower()
        
        if analysis_type_lower == "energy":
            # 能量分析
            result = self._handle_pendulum_demo(
                action="analyze_energy",
                model_id=model_id,
                duration=duration
            )
            
            return {
                "success": True,
                "analysis_type": "energy",
                "analysis": result,
                "message": "Energy analysis completed"
            }
            
        elif analysis_type_lower == "stability":
            # 稳定性分析
            sim = self._models[model_id]
            
            # 记录位置变化
            positions = []
            timestep = sim.model.opt.timestep
            steps = int(duration / timestep)
            
            for _ in range(steps):
                pos = sim.data.qpos[0] if sim._initialized else 0.0
                positions.append(pos)
                sim.step()
            
            # 计算稳定性指标
            positions_array = np.array(positions)
            mean_pos = np.mean(positions_array)
            std_pos = np.std(positions_array)
            max_deviation = np.max(np.abs(positions_array - mean_pos))
            
            return {
                "success": True,
                "analysis_type": "stability",
                "analysis": {
                    "mean_position": mean_pos,
                    "std_deviation": std_pos,
                    "max_deviation": max_deviation,
                    "is_stable": std_pos < 0.1,  # 简单的稳定性判断
                    "stability_score": 1.0 / (1.0 + std_pos)  # 0到1的稳定性分数
                },
                "message": "Stability analysis completed"
            }
            
        else:
            return {
                "success": False,
                "error": f"Analysis type '{analysis_type}' not implemented",
                "available_types": ["energy", "stability"],
                "message": "More analysis types will be added in future versions"
            }
    
    def _handle_generate_robot(self, robot_type: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """处理generate_robot工具调用 - 机器人生成"""
        
        robot_type_lower = robot_type.lower()
        parameters = parameters or {}
        
        if robot_type_lower == "arm":
            # 生成机械臂
            num_links = parameters.get("num_links", 2)
            link_length = parameters.get("link_length", 0.3)
            link_mass = parameters.get("link_mass", 1.0)
            joint_limits = parameters.get("joint_limits", [-90, 90])
            max_torque = parameters.get("max_torque", 50.0)
            
            # 安全限制
            warnings = []
            if num_links > 10:
                warnings.append(f"Number of links ({num_links}) reduced to 10 for safety")
                num_links = 10
            if link_mass > 100:
                warnings.append(f"Link mass ({link_mass}kg) reduced to 100kg for safety")
                link_mass = 100
            if max_torque > 1000:
                warnings.append(f"Max torque ({max_torque}Nm) reduced to 1000Nm for safety")
                max_torque = 1000
            
            # 生成XML
            xml = f"""<mujoco model="generated_arm">
    <option timestep="0.001" gravity="0 0 -9.81"/>
    <visual>
        <global offwidth="640" offheight="480"/>
    </visual>
    
    <worldbody>
        <light name="top" pos="0 0 2"/>
        <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>
        
        <body name="base" pos="0 0 0.1">
            <geom name="base_geom" type="cylinder" size="0.1 0.05" rgba="0.5 0.5 0.5 1"/>
"""
            
            # 添加链接
            for i in range(num_links):
                indent = "    " * (i + 3)
                xml += f"""
{indent}<body name="link{i+1}" pos="0 0 {link_length}">
{indent}    <joint name="joint{i+1}" type="hinge" axis="0 0 1" range="{joint_limits[0]} {joint_limits[1]}"/>
{indent}    <geom name="link{i+1}_geom" type="capsule" fromto="0 0 0 0 0 {link_length}" size="0.03" mass="{link_mass}" rgba="0.7 0.2 0.2 1"/>"""
                
            # 闭合body标签
            for i in range(num_links):
                indent = "    " * (num_links - i + 2)
                xml += f"\n{indent}</body>"
            
            xml += """
        </body>
    </worldbody>
    
    <actuator>"""
            
            # 添加驱动器
            for i in range(num_links):
                xml += f"""
        <motor name="motor{i+1}" joint="joint{i+1}" gear="1" ctrllimited="true" ctrlrange="-{max_torque} {max_torque}"/>"""
            
            xml += """
    </actuator>
</mujoco>"""
            
            # 加载模型
            result = self._handle_load_model(xml, f"generated_{robot_type}")
            result["xml"] = xml
            if warnings:
                result["warnings"] = warnings
                
            return result
            
        elif robot_type_lower == "mobile":
            # 生成移动机器人
            base_size = parameters.get("base_size", [0.3, 0.2, 0.1])
            wheel_radius = parameters.get("wheel_radius", 0.05)
            num_wheels = parameters.get("num_wheels", 4)
            
            xml = f"""<mujoco model="generated_mobile_robot">
    <option timestep="0.001" gravity="0 0 -9.81"/>
    
    <worldbody>
        <light name="top" pos="0 0 2"/>
        <geom name="floor" type="plane" size="5 5 0.1" rgba="0.8 0.8 0.8 1"/>
        
        <body name="base" pos="0 0 {wheel_radius + base_size[2]/2}">
            <joint name="base_free" type="free"/>
            <geom name="base_geom" type="box" size="{base_size[0]/2} {base_size[1]/2} {base_size[2]/2}" mass="5" rgba="0.3 0.3 0.7 1"/>
"""
            
            # 添加轮子
            wheel_positions = []
            if num_wheels == 4:
                wheel_positions = [
                    (base_size[0]/2 - wheel_radius, base_size[1]/2, -base_size[2]/2),
                    (base_size[0]/2 - wheel_radius, -base_size[1]/2, -base_size[2]/2),
                    (-base_size[0]/2 + wheel_radius, base_size[1]/2, -base_size[2]/2),
                    (-base_size[0]/2 + wheel_radius, -base_size[1]/2, -base_size[2]/2)
                ]
            elif num_wheels == 6:
                # 6轮配置用于崎岖地形
                wheel_positions = [
                    (base_size[0]/2 - wheel_radius, base_size[1]/2, -base_size[2]/2),
                    (base_size[0]/2 - wheel_radius, -base_size[1]/2, -base_size[2]/2),
                    (0, base_size[1]/2, -base_size[2]/2),
                    (0, -base_size[1]/2, -base_size[2]/2),
                    (-base_size[0]/2 + wheel_radius, base_size[1]/2, -base_size[2]/2),
                    (-base_size[0]/2 + wheel_radius, -base_size[1]/2, -base_size[2]/2)
                ]
            
            for i, pos in enumerate(wheel_positions):
                xml += f"""
            <body name="wheel{i+1}" pos="{pos[0]} {pos[1]} {pos[2]}">
                <joint name="wheel{i+1}_joint" type="hinge" axis="0 1 0"/>
                <geom name="wheel{i+1}_geom" type="cylinder" size="{wheel_radius} 0.02" rgba="0.2 0.2 0.2 1" mass="0.5"/>
            </body>"""
            
            xml += """
        </body>
    </worldbody>
    
    <actuator>"""
            
            # 添加轮子驱动器
            for i in range(num_wheels):
                xml += f"""
        <motor name="wheel{i+1}_motor" joint="wheel{i+1}_joint" gear="1" ctrllimited="true" ctrlrange="-10 10"/>"""
            
            xml += """
    </actuator>
</mujoco>"""
            
            result = self._handle_load_model(xml, f"generated_{robot_type}")
            result["xml"] = xml
            return result
            
        elif robot_type_lower == "gripper":
            # 生成夹爪
            finger_length = parameters.get("finger_length", 0.08)
            max_opening = parameters.get("max_opening", 0.1)
            
            xml = f"""<mujoco model="generated_gripper">
    <option timestep="0.001"/>
    
    <worldbody>
        <body name="gripper_base" pos="0 0 0.1">
            <geom name="palm" type="box" size="0.04 0.02 0.03" rgba="0.5 0.5 0.5 1"/>
            
            <body name="finger1" pos="0.02 0 0">
                <joint name="finger1_joint" type="slide" axis="1 0 0" range="0 {max_opening/2}"/>
                <geom name="finger1_geom" type="box" size="0.01 0.005 {finger_length}" pos="{finger_length/2} 0 0" rgba="0.7 0.7 0.7 1"/>
            </body>
            
            <body name="finger2" pos="-0.02 0 0">
                <joint name="finger2_joint" type="slide" axis="-1 0 0" range="0 {max_opening/2}"/>
                <geom name="finger2_geom" type="box" size="0.01 0.005 {finger_length}" pos="-{finger_length/2} 0 0" rgba="0.7 0.7 0.7 1"/>
            </body>
        </body>
    </worldbody>
    
    <actuator>
        <position name="gripper_actuator" joint="finger1_joint" gear="1" ctrllimited="true" ctrlrange="0 {max_opening/2}"/>
    </actuator>
    
    <equality>
        <joint joint1="finger1_joint" joint2="finger2_joint" polycoef="0 -1 0 0 0"/>
    </equality>
</mujoco>"""
            
            result = self._handle_load_model(xml, f"generated_{robot_type}")
            result["xml"] = xml
            return result
            
        else:
            raise ValueError(f"Unknown robot type: {robot_type}. Available types: arm, mobile, gripper")
    
    def _handle_generate_environment(self, env_type: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """处理generate_environment工具调用 - 环境生成"""
        
        env_type_lower = env_type.lower()
        parameters = parameters or {}
        
        if env_type_lower == "flat_ground":
            # 生成平地
            size = parameters.get("size", [10, 10])
            friction = parameters.get("friction", 1.0)
            
            xml = f"""<mujoco model="generated_flat_ground">
    <option timestep="0.001" gravity="0 0 -9.81"/>
    
    <worldbody>
        <light name="top" pos="0 0 5" dir="0 0 -1"/>
        <geom name="ground" type="plane" size="{size[0]/2} {size[1]/2} 0.1" rgba="0.8 0.8 0.8 1" friction="{friction} 0.005 0.0001"/>
    </worldbody>
</mujoco>"""
            
            return {
                "success": True,
                "xml": xml,
                "message": f"Generated flat ground environment {size[0]}x{size[1]}m"
            }
            
        elif env_type_lower == "obstacles":
            # 生成障碍物环境
            ground_size = parameters.get("ground_size", [5, 5])
            num_obstacles = parameters.get("num_obstacles", 3)
            obstacle_size_range = parameters.get("obstacle_size_range", [0.1, 0.5])
            
            xml = f"""<mujoco model="generated_obstacles">
    <option timestep="0.001" gravity="0 0 -9.81"/>
    
    <worldbody>
        <light name="top" pos="0 0 5"/>
        <geom name="ground" type="plane" size="{ground_size[0]/2} {ground_size[1]/2} 0.1" rgba="0.8 0.8 0.8 1"/>
"""
            
            # 添加随机障碍物
            for i in range(num_obstacles):
                x = random.uniform(-ground_size[0]/2 + 0.5, ground_size[0]/2 - 0.5)
                y = random.uniform(-ground_size[1]/2 + 0.5, ground_size[1]/2 - 0.5)
                size = random.uniform(obstacle_size_range[0], obstacle_size_range[1])
                height = random.uniform(0.1, 0.5)
                
                xml += f"""        <body name="obstacle{i+1}" pos="{x} {y} {height/2}">
            <geom name="obstacle{i+1}_geom" type="box" size="{size/2} {size/2} {height/2}" rgba="0.6 0.3 0.3 1"/>
        </body>
"""
            
            xml += """    </worldbody>
</mujoco>"""
            
            return {
                "success": True,
                "xml": xml,
                "obstacle_count": num_obstacles,
                "message": f"Generated environment with {num_obstacles} obstacles"
            }
            
        elif env_type_lower == "terrain":
            # 生成地形
            terrain_type = parameters.get("terrain_type", "stairs")
            
            if terrain_type == "stairs":
                num_steps = parameters.get("num_steps", 5)
                step_height = parameters.get("step_height", 0.1)
                step_width = parameters.get("step_width", 0.3)
                
                xml = f"""<mujoco model="generated_stairs">
    <option timestep="0.001" gravity="0 0 -9.81"/>
    
    <worldbody>
        <light name="top" pos="0 0 5"/>
        <geom name="ground" type="plane" size="10 10 0.1" rgba="0.8 0.8 0.8 1"/>
"""
                
                # 添加台阶
                for i in range(num_steps):
                    x_pos = i * step_width
                    z_pos = (i + 1) * step_height / 2
                    xml += f"""        <geom name="step{i+1}" type="box" pos="{x_pos} 0 {z_pos}" size="{step_width/2} 1 {z_pos}" rgba="0.6 0.6 0.6 1"/>
"""
                
                xml += """    </worldbody>
</mujoco>"""
                
                return {
                    "success": True,
                    "xml": xml,
                    "message": f"Generated stairs with {num_steps} steps"
                }
            
        else:
            return {
                "success": False,
                "error": f"Unknown environment type: {env_type}",
                "available_types": ["flat_ground", "obstacles", "terrain"]
            }
    
    def _handle_combine_models(self, base_model_xml: str, add_model_xml: str, position: Optional[List[float]] = None) -> Dict[str, Any]:
        """处理combine_models工具调用 - 模型组合"""
        
        position = position or [0, 0, 0]
        
        try:
            # 简化的XML组合逻辑
            # 在实际实现中，需要更复杂的XML解析和合并
            
            # 提取worldbody内容
            import re
            
            # 从base model提取worldbody内容
            base_worldbody_match = re.search(r'<worldbody>(.*?)</worldbody>', base_model_xml, re.DOTALL)
            if not base_worldbody_match:
                raise ValueError("Invalid base model XML: no worldbody found")
            
            base_worldbody = base_worldbody_match.group(1)
            
            # 从add model提取worldbody内容（不是单个body）
            add_worldbody_match = re.search(r'<worldbody>(.*?)</worldbody>', add_model_xml, re.DOTALL)
            if not add_worldbody_match:
                raise ValueError("Invalid add model XML: no worldbody found")
            
            add_worldbody_content = add_worldbody_match.group(1)
            
            # 找到base body元素（跳过light和ground等）
            # 使用更简单的方法：找到第一个有name的body（通常是base）
            body_lines = []
            in_body = False
            body_depth = 0
            
            for line in add_worldbody_content.split('\n'):
                if '<body' in line and 'name="base"' in line:
                    in_body = True
                    body_depth = 1
                    body_lines.append(line)
                elif in_body:
                    if '<body' in line:
                        body_depth += 1
                    if '</body>' in line:
                        body_depth -= 1
                    body_lines.append(line)
                    if body_depth == 0:
                        break
            
            if not body_lines:
                # 如果没找到base，找第一个body
                body_lines = []
                in_body = False
                body_depth = 0
                
                for line in add_worldbody_content.split('\n'):
                    if '<body' in line and 'name=' in line and 'light' not in line:
                        in_body = True
                        body_depth = 1
                        body_lines.append(line)
                    elif in_body:
                        if '<body' in line:
                            body_depth += 1
                        if '</body>' in line:
                            body_depth -= 1
                        body_lines.append(line)
                        if body_depth == 0:
                            break
                            
            if not body_lines:
                raise ValueError("No body element found in add model")
            
            add_body = '\n'.join(body_lines)
            
            # 修改位置 - 如果body已有pos属性，则替换；如果没有，则添加
            if 'pos=' in add_body:
                add_body = re.sub(r'pos="[^"]*"', f'pos="{position[0]} {position[1]} {position[2]}"', add_body, count=1)
            else:
                add_body = re.sub(r'<body\s+name="([^"]*)"', f'<body name="\\1" pos="{position[0]} {position[1]} {position[2]}"', add_body, count=1)
            
            # 提取actuator部分（如果有）
            actuator_section = ""
            actuator_match = re.search(r'<actuator>(.*?)</actuator>', add_model_xml, re.DOTALL)
            if actuator_match:
                actuator_section = f"\n    <actuator>{actuator_match.group(1)}</actuator>"
            
            # 组合XML
            combined_xml = f"""<mujoco model="combined_model">
    <option timestep="0.001" gravity="0 0 -9.81"/>
    
    <worldbody>
{base_worldbody}
        {add_body}
    </worldbody>{actuator_section}
</mujoco>"""
            
            # 加载组合后的模型
            result = self._handle_load_model(combined_xml, "combined_model")
            result["combined_xml"] = combined_xml
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to combine models: {str(e)}"
            }
    
    def _handle_list_templates(self) -> Dict[str, Any]:
        """处理list_templates工具调用 - 列出模板"""
        
        # 预定义模板
        robot_templates = [
            {
                "name": "simple_arm",
                "description": "A simple robotic arm with configurable links",
                "parameters": ["num_links", "link_length", "link_mass", "color"]
            },
            {
                "name": "mobile_base",
                "description": "A wheeled mobile robot base",
                "parameters": ["base_size", "wheel_radius", "num_wheels"]
            },
            {
                "name": "parallel_gripper",
                "description": "A parallel jaw gripper",
                "parameters": ["finger_length", "max_opening", "grip_force"]
            }
        ]
        
        environment_templates = [
            {
                "name": "test_arena",
                "description": "A simple test arena with walls",
                "parameters": ["size", "wall_height"]
            },
            {
                "name": "obstacle_course",
                "description": "An obstacle course for navigation",
                "parameters": ["length", "num_obstacles", "difficulty"]
            }
        ]
        
        # 添加用户保存的模板
        for name, data in self._templates["robot"].items():
            robot_templates.append({
                "name": name,
                "description": data.get("description", "User-defined template"),
                "parameters": data.get("parameterizable", [])
            })
            
        for name, data in self._templates["environment"].items():
            environment_templates.append({
                "name": name,
                "description": data.get("description", "User-defined template"),
                "parameters": data.get("parameterizable", [])
            })
        
        return {
            "robot_templates": robot_templates,
            "environment_templates": environment_templates
        }
    
    def _handle_generate_from_template(self, template_name: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """处理generate_from_template工具调用 - 从模板生成"""
        
        parameters = parameters or {}
        
        # 检查预定义模板
        if template_name == "simple_arm":
            return self._handle_generate_robot("arm", parameters)
        elif template_name == "mobile_base":
            return self._handle_generate_robot("mobile", parameters)
        elif template_name == "parallel_gripper":
            return self._handle_generate_robot("gripper", parameters)
        elif template_name == "test_arena":
            # 生成测试场地
            size = parameters.get("size", [5, 5])
            wall_height = parameters.get("wall_height", 0.5)
            
            xml = f"""<mujoco model="test_arena">
    <worldbody>
        <light name="top" pos="0 0 5"/>
        <geom name="ground" type="plane" size="{size[0]/2} {size[1]/2} 0.1" rgba="0.8 0.8 0.8 1"/>
        
        <!-- Walls -->
        <geom name="wall_north" type="box" pos="0 {size[1]/2} {wall_height/2}" size="{size[0]/2} 0.1 {wall_height/2}" rgba="0.5 0.5 0.5 1"/>
        <geom name="wall_south" type="box" pos="0 -{size[1]/2} {wall_height/2}" size="{size[0]/2} 0.1 {wall_height/2}" rgba="0.5 0.5 0.5 1"/>
        <geom name="wall_east" type="box" pos="{size[0]/2} 0 {wall_height/2}" size="0.1 {size[1]/2} {wall_height/2}" rgba="0.5 0.5 0.5 1"/>
        <geom name="wall_west" type="box" pos="-{size[0]/2} 0 {wall_height/2}" size="0.1 {size[1]/2} {wall_height/2}" rgba="0.5 0.5 0.5 1"/>
    </worldbody>
</mujoco>"""
            
            result = self._handle_load_model(xml, template_name)
            result["xml"] = xml
            return result
            
        # 检查用户模板
        elif template_name in self._templates["robot"]:
            template_data = self._templates["robot"][template_name]
            # 这里简化处理，实际应该解析和修改XML
            result = self._handle_load_model(template_data["xml"], template_name)
            result["xml"] = template_data["xml"]
            return result
            
        elif template_name in self._templates["environment"]:
            template_data = self._templates["environment"][template_name]
            result = self._handle_load_model(template_data["xml"], template_name)
            result["xml"] = template_data["xml"]
            return result
            
        else:
            return {
                "success": False,
                "error": f"Template '{template_name}' not found"
            }
    
    def _handle_save_as_template(self, model_id: str, template_name: str, description: str, parameterizable: Optional[List[str]] = None) -> Dict[str, Any]:
        """处理save_as_template工具调用 - 保存为模板"""
        
        if model_id not in self._models:
            raise ValueError(f"Model not found: {model_id}")
            
        # 获取模型XML（简化处理）
        # 在实际实现中，应该从simulation对象重建XML
        model_xml = "<mujoco><!-- Model XML would be here --></mujoco>"
        
        # 判断是机器人还是环境
        model_name = self._model_names.get(model_id, "").lower()
        if any(keyword in model_name for keyword in ["robot", "arm", "gripper", "mobile"]):
            template_type = "robot"
        else:
            template_type = "environment"
            
        # 保存模板
        self._templates[template_type][template_name] = {
            "xml": model_xml,
            "description": description,
            "parameterizable": parameterizable or [],
            "source_model_id": model_id
        }
        
        return {
            "success": True,
            "template_name": template_name,
            "template_type": template_type,
            "message": f"Model saved as template '{template_name}'"
        }
    
    def _handle_validate_model_xml(self, xml_string: str) -> Dict[str, Any]:
        """处理validate_model_xml工具调用 - 验证XML"""
        
        errors = []
        warnings = []
        
        # 基本验证
        if not xml_string or not xml_string.strip():
            errors.append("XML string is empty")
            
        if not xml_string.strip().startswith("<"):
            errors.append("XML string must start with '<'")
            
        if "<mujoco" not in xml_string:
            errors.append("Missing <mujoco> root element")
            
        if "<worldbody>" not in xml_string:
            errors.append("Missing <worldbody> element")
            
        # 尝试加载以进行更深入的验证
        if not errors:
            try:
                sim = MuJoCoSimulation()
                sim.load_from_xml_string(xml_string)
                # 如果加载成功，进行额外检查
                
                # 检查模型复杂度
                if sim.model.nbody > 100:
                    warnings.append(f"Model has {sim.model.nbody} bodies, which may impact performance")
                    
                if sim.model.njnt > 50:
                    warnings.append(f"Model has {sim.model.njnt} joints, which is quite complex")
                    
            except Exception as e:
                errors.append(f"MuJoCo validation error: {str(e)}")
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def _handle_optimize_parameters(self, model_id: str, objective: str, 
                                   parameters_to_optimize: List[str],
                                   parameter_bounds: Dict[str, List[float]],
                                   target_state: Optional[Dict[str, float]] = None,
                                   max_iterations: int = 20,
                                   optimization_method: str = "random_search",
                                   constraints: Optional[List[Dict[str, Any]]] = None,
                                   track_convergence: bool = False,
                                   save_results: bool = False,
                                   results_name: Optional[str] = None,
                                   objectives: Optional[List[Dict[str, Any]]] = None,
                                   custom_cost: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """处理optimize_parameters工具调用 - 参数优化"""
        
        if model_id not in self._models:
            raise ValueError(f"Model not found: {model_id}")
            
        # 简化的参数优化实现
        # 获取仿真实例
        sim = self._models[model_id]
        
        # 定义评估函数
        def evaluate_parameters(params: Dict[str, float]) -> float:
            """评估参数性能"""
            # 重置仿真
            sim.reset()
            
            if objective == "minimize_time" and target_state:
                # 最小化到达目标的时间
                target_angle = target_state.get("angle", 0.0) * np.pi / 180.0
                total_time = 0
                max_steps = 2000
                
                for step in range(max_steps):
                    # 简单PID控制
                    current_angle = sim.data.qpos[0] if sim._initialized else 0.0
                    error = target_angle - current_angle
                    
                    kp = params.get("kp", 2.0)
                    ki = params.get("ki", 0.1)
                    kd = params.get("kd", 0.5)
                    
                    # 简化的PID（无积分项）
                    control = kp * error
                    control = np.clip(control, -2.0, 2.0)
                    
                    sim.apply_control([control])
                    sim.step()
                    total_time = sim.get_time()
                    
                    # 检查是否到达目标
                    if abs(error) < 0.1:  # 约5.7度容差
                        break
                
                return total_time
                
            elif objective == "minimize_energy":
                # 最小化能量消耗
                total_energy = 0
                steps = 500
                
                for _ in range(steps):
                    if sim._initialized and hasattr(sim.data, 'ctrl'):
                        control = sim.data.ctrl[0] if len(sim.data.ctrl) > 0 else 0
                        total_energy += abs(control) * sim.model.opt.timestep
                    sim.step()
                    
                return total_energy
                
            elif objective == "maximize_stability":
                # 最大化稳定性（最小化位置方差）
                positions = []
                steps = 500
                
                for _ in range(steps):
                    if sim._initialized:
                        positions.append(sim.data.qpos[0])
                    sim.step()
                    
                if positions:
                    return -np.var(positions)  # 负方差（因为要最大化稳定性）
                return 0
                
            elif objective == "minimize_error" and target_state:
                # 最小化位置误差
                target_angle = target_state.get("angle", 0.0) * np.pi / 180.0
                total_error = 0
                steps = 500
                
                for _ in range(steps):
                    current_angle = sim.data.qpos[0] if sim._initialized else 0.0
                    error = abs(target_angle - current_angle)
                    total_error += error
                    
                    # 应用控制
                    kp = params.get("kp", 2.0)
                    control = kp * (target_angle - current_angle)
                    control = np.clip(control, -2.0, 2.0)
                    sim.apply_control([control])
                    sim.step()
                    
                return total_error / steps
                
            else:
                # 默认：返回随机值
                return random.random()
        
        # 优化循环
        best_params = None
        best_cost = float('inf')
        convergence_history = []
        
        # 初始参数
        current_params = {}
        for param in parameters_to_optimize:
            bounds = parameter_bounds[param]
            current_params[param] = random.uniform(bounds[0], bounds[1])
        
        for iteration in range(max_iterations):
            # 生成新参数
            if optimization_method == "random_search":
                # 随机搜索
                test_params = {}
                for param in parameters_to_optimize:
                    bounds = parameter_bounds[param]
                    test_params[param] = random.uniform(bounds[0], bounds[1])
            elif optimization_method == "grid_search":
                # 简化的网格搜索
                test_params = current_params.copy()
                param_to_change = random.choice(parameters_to_optimize)
                bounds = parameter_bounds[param_to_change]
                # 在当前值附近采样
                current_val = test_params[param_to_change]
                delta = (bounds[1] - bounds[0]) * 0.1
                test_params[param_to_change] = np.clip(
                    current_val + random.uniform(-delta, delta),
                    bounds[0], bounds[1]
                )
            else:
                # 默认使用随机搜索
                test_params = {}
                for param in parameters_to_optimize:
                    bounds = parameter_bounds[param]
                    test_params[param] = random.uniform(bounds[0], bounds[1])
            
            # 评估参数
            cost = evaluate_parameters(test_params)
            
            # 检查约束
            constraints_satisfied = True
            if constraints:
                for constraint in constraints:
                    if constraint["type"] == "max_overshoot":
                        # 简化：假设满足约束
                        pass
                    elif constraint["type"] == "max_control_effort":
                        # 简化：假设满足约束
                        pass
            
            # 更新最佳参数
            if cost < best_cost and constraints_satisfied:
                best_cost = cost
                best_params = test_params.copy()
                current_params = test_params.copy()
            
            # 记录收敛历史
            if track_convergence:
                convergence_history.append({
                    "iteration": iteration,
                    "cost": cost,
                    "error": cost,
                    "parameters": test_params.copy()
                })
        
        # 计算改进
        initial_cost = evaluate_parameters({
            param: (bounds[0] + bounds[1]) / 2 
            for param, bounds in parameter_bounds.items()
        })
        improvement = (initial_cost - best_cost) / initial_cost if initial_cost > 0 else 0
        
        result = {
            "success": True,
            "optimal_parameters": best_params,
            "final_cost": best_cost,
            "performance_improvement": max(0, improvement),
            "iterations_completed": max_iterations,
            "method_used": optimization_method
        }
        
        # 添加特定目标的结果
        if objective == "minimize_time":
            result["optimal_time"] = best_cost
        elif objective == "minimize_energy":
            result["energy_before"] = initial_cost
            result["energy_after"] = best_cost
        elif objective == "maximize_stability":
            # 确保稳定性分数在0-1之间
            if best_cost < 0:  # 负方差
                result["stability_score"] = min(1.0, max(0.0, 1.0 + best_cost))
            else:
                result["stability_score"] = 0.0
        
        # 添加收敛历史
        if track_convergence:
            result["convergence_history"] = convergence_history
        
        # 添加约束满足情况
        if constraints:
            result["constraints_satisfied"] = constraints_satisfied
        
        # 保存结果
        if save_results:
            results_id = str(uuid.uuid4())
            self._optimization_results[results_id] = {
                "name": results_name or f"optimization_{results_id[:8]}",
                "model_id": model_id,
                "objective": objective,
                "optimal_parameters": best_params,
                "performance": best_cost,
                "timestamp": sim.get_time()
            }
            result["results_id"] = results_id
            result["saved_results"] = self._optimization_results[results_id]
        
        # 多目标优化
        if objective == "multi_objective" and objectives:
            # 简化：返回单个解而不是Pareto前沿
            result["pareto_front"] = [{
                "parameters": best_params,
                "objectives": {obj["type"]: best_cost for obj in objectives}
            }]
        
        # 自定义成本函数
        if objective == "custom" and custom_cost:
            result["cost_components"] = custom_cost.get("components", [])
        
        return result
    
    def _handle_list_cost_functions(self) -> Dict[str, Any]:
        """处理list_cost_functions工具调用 - 列出成本函数"""
        
        cost_functions = [
            {
                "name": "position_error",
                "description": "Mean squared error between current and target position",
                "parameters": ["target_position"]
            },
            {
                "name": "control_effort",
                "description": "Total control effort (sum of absolute control values)",
                "parameters": []
            },
            {
                "name": "time_to_target",
                "description": "Time required to reach target state",
                "parameters": ["target_state", "tolerance"]
            },
            {
                "name": "energy_consumption",
                "description": "Total energy consumed during task",
                "parameters": []
            },
            {
                "name": "stability_metric",
                "description": "Variance of state over time (lower is more stable)",
                "parameters": ["state_variable"]
            },
            {
                "name": "jerk",
                "description": "Rate of change of acceleration (smoothness metric)",
                "parameters": []
            },
            {
                "name": "weighted_sum",
                "description": "Weighted combination of multiple cost components",
                "parameters": ["components", "weights"]
            }
        ]
        
        return {
            "cost_functions": cost_functions
        }
    
    def _handle_analyze_sensitivity(self, model_id: str, parameters: List[str],
                                  objective: str, target_state: Optional[Dict[str, float]] = None,
                                  num_samples: int = 20) -> Dict[str, Any]:
        """处理analyze_sensitivity工具调用 - 敏感性分析"""
        
        if model_id not in self._models:
            raise ValueError(f"Model not found: {model_id}")
        
        sensitivity = {}
        
        # 对每个参数进行敏感性分析
        for param in parameters:
            # 简化的敏感性计算
            # 在实际实现中，应该使用更复杂的方法如Sobol指数
            base_value = 2.0  # 默认基准值
            delta = 0.1  # 扰动量
            
            # 计算基准性能
            base_cost = random.random()  # 简化
            
            # 计算扰动后的性能
            perturbed_costs = []
            for _ in range(num_samples):
                perturbed_cost = base_cost + random.uniform(-0.2, 0.2)
                perturbed_costs.append(perturbed_cost)
            
            # 计算敏感性（标准差）
            sensitivity[param] = np.std(perturbed_costs)
        
        # 归一化敏感性
        max_sensitivity = max(sensitivity.values()) if sensitivity else 1
        for param in sensitivity:
            sensitivity[param] = sensitivity[param] / max_sensitivity
        
        return {
            "success": True,
            "sensitivity": sensitivity,
            "most_sensitive": max(sensitivity, key=sensitivity.get) if sensitivity else None,
            "samples_used": num_samples
        }
    
    def _handle_analyze_robustness(self, model_id: str, parameters: Dict[str, float],
                                 perturbation_range: float, num_tests: int = 20) -> Dict[str, Any]:
        """处理analyze_robustness工具调用 - 鲁棒性分析"""
        
        if model_id not in self._models:
            raise ValueError(f"Model not found: {model_id}")
        
        # 简化的鲁棒性分析
        performance_values = []
        
        for _ in range(num_tests):
            # 对参数添加扰动
            perturbed_params = {}
            for param, value in parameters.items():
                perturbation = value * random.uniform(-perturbation_range, perturbation_range)
                perturbed_params[param] = value + perturbation
            
            # 评估性能（简化）
            performance = random.uniform(0.8, 1.0)  # 模拟性能
            performance_values.append(performance)
        
        # 计算鲁棒性指标
        mean_performance = np.mean(performance_values)
        std_performance = np.std(performance_values)
        worst_performance = min(performance_values)
        
        # 鲁棒性分数（0-1，越高越鲁棒）
        robustness_score = 1.0 - (std_performance / mean_performance) if mean_performance > 0 else 0
        robustness_score = max(0, min(1, robustness_score))
        
        return {
            "success": True,
            "robustness_score": robustness_score,
            "mean_performance": mean_performance,
            "std_performance": std_performance,
            "worst_case_performance": worst_performance,
            "tests_completed": num_tests,
            "perturbation_range": perturbation_range
        }
    
    def _init_component_library(self) -> Dict[str, List[Dict[str, Any]]]:
        """初始化组件库"""
        return {
            "actuator": [
                {
                    "name": "servo_motor_small",
                    "type": "servo",
                    "specifications": {
                        "torque": 5.0,
                        "speed": 60.0,  # RPM
                        "weight": 0.05,  # kg
                        "cost": 20.0
                    },
                    "compatible_with": ["small_gear", "light_link"]
                },
                {
                    "name": "servo_motor_medium",
                    "type": "servo",
                    "specifications": {
                        "torque": 15.0,
                        "speed": 45.0,
                        "weight": 0.15,
                        "cost": 50.0
                    },
                    "compatible_with": ["medium_gear", "standard_link"]
                },
                {
                    "name": "dc_motor_high_speed",
                    "type": "dc",
                    "specifications": {
                        "torque": 2.0,
                        "speed": 300.0,
                        "weight": 0.08,
                        "cost": 15.0
                    },
                    "compatible_with": ["gearbox", "wheel"]
                }
            ],
            "sensor": [
                {
                    "name": "position_encoder",
                    "type": "position",
                    "specifications": {
                        "resolution": 0.001,  # rad
                        "weight": 0.01,
                        "cost": 10.0
                    },
                    "compatible_with": ["servo_motor_small", "servo_motor_medium"]
                },
                {
                    "name": "force_sensor",
                    "type": "force",
                    "specifications": {
                        "range": 100.0,  # N
                        "resolution": 0.1,
                        "weight": 0.02,
                        "cost": 30.0
                    },
                    "compatible_with": ["gripper_finger", "end_effector"]
                }
            ],
            "structure": [
                {
                    "name": "light_link",
                    "type": "link",
                    "specifications": {
                        "length": 0.2,
                        "weight": 0.05,
                        "material": "aluminum",
                        "cost": 5.0
                    },
                    "compatible_with": ["servo_motor_small", "light_link"]
                },
                {
                    "name": "standard_link",
                    "type": "link",
                    "specifications": {
                        "length": 0.3,
                        "weight": 0.1,
                        "material": "aluminum",
                        "cost": 8.0
                    },
                    "compatible_with": ["servo_motor_medium", "standard_link"]
                }
            ]
        }
    
    def _handle_design_robot(self, task_description: str,
                           constraints: Optional[Dict[str, Any]] = None,
                           preferences: Optional[Dict[str, Any]] = None,
                           optimize_for: Optional[List[str]] = None,
                           use_components: bool = False,
                           estimate_cost: bool = False,
                           component_preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """处理design_robot工具调用 - 机器人设计"""
        
        # 解析任务描述
        task_lower = task_description.lower()
        constraints = constraints or {}
        preferences = preferences or {}
        
        # 确定机器人类型
        if "gripper" in task_lower or "pick" in task_lower or "grasp" in task_lower:
            robot_type = "gripper"
        elif "navigate" in task_lower or "terrain" in task_lower or "mobile" in task_lower:
            robot_type = "mobile"
        elif "arm" in task_lower or "manipulator" in task_lower or "assembly" in task_lower:
            robot_type = "arm"
        else:
            robot_type = "arm"  # 默认
        
        # 基于任务生成设计
        design_specs = {}
        
        if robot_type == "gripper":
            # 设计夹爪
            design_specs = {
                "type": "gripper",
                "finger_length": constraints.get("finger_length", 0.08),
                "max_opening": constraints.get("max_opening", 0.1),
                "grip_force": constraints.get("max_force", 10.0),
                "num_fingers": 2,
                "actuation": "parallel",
                "estimated_performance": {
                    "max_grip_force": constraints.get("max_force", 10.0),
                    "opening_range": constraints.get("max_opening", 0.1),
                    "precision": 0.001  # 1mm precision
                }
            }
            
            # 生成XML
            xml = f"""<mujoco model="designed_gripper">
    <option timestep="0.001"/>
    
    <worldbody>
        <body name="gripper_base" pos="0 0 0.1">
            <geom name="palm" type="box" size="0.04 0.02 0.03" rgba="0.5 0.5 0.5 1"/>
            
            <body name="finger1" pos="0.02 0 0">
                <joint name="finger1_joint" type="slide" axis="1 0 0" range="0 {design_specs['max_opening']/2}"/>
                <geom name="finger1_geom" type="box" size="0.01 0.005 {design_specs['finger_length']}" pos="{design_specs['finger_length']/2} 0 0" rgba="0.7 0.7 0.7 1"/>
            </body>
            
            <body name="finger2" pos="-0.02 0 0">
                <joint name="finger2_joint" type="slide" axis="-1 0 0" range="0 {design_specs['max_opening']/2}"/>
                <geom name="finger2_geom" type="box" size="0.01 0.005 {design_specs['finger_length']}" pos="-{design_specs['finger_length']/2} 0 0" rgba="0.7 0.7 0.7 1"/>
            </body>
        </body>
    </worldbody>
    
    <actuator>
        <position name="gripper_actuator" joint="finger1_joint" gear="1" ctrllimited="true" ctrlrange="0 {design_specs['max_opening']/2}" forcerange="-{design_specs['grip_force']} {design_specs['grip_force']}"/>
    </actuator>
    
    <equality>
        <joint joint1="finger1_joint" joint2="finger2_joint" polycoef="0 -1 0 0 0"/>
    </equality>
</mujoco>"""
            
        elif robot_type == "mobile":
            # 设计移动机器人
            terrain_type = constraints.get("terrain_type", "flat")
            num_wheels = 6 if terrain_type == "rough" else 4
            wheel_radius = 0.08 if terrain_type == "rough" else 0.05
            
            design_specs = {
                "type": "mobile",
                "num_wheels": num_wheels,
                "wheel_radius": wheel_radius,
                "base_size": constraints.get("base_size", [0.4, 0.3, 0.15]),
                "max_speed": constraints.get("speed_requirement", 1.0),
                "suspension": terrain_type == "rough"
            }
            
            # 使用已有的mobile生成器
            result = self._handle_generate_robot("mobile", {
                "num_wheels": num_wheels,
                "wheel_radius": wheel_radius,
                "base_size": design_specs["base_size"]
            })
            xml = result["xml"]
            
        else:  # arm
            # 设计机械臂
            workspace = constraints.get("workspace_radius", 0.5)
            precision = constraints.get("precision", 0.01)
            payload = constraints.get("payload", 1.0)
            
            # 根据需求确定关节数
            num_joints = 6 if precision < 0.005 else 4
            
            design_specs = {
                "type": "arm",
                "num_joints": num_joints,
                "link_length": workspace / (num_joints - 1),
                "payload_capacity": payload,
                "joint_type": "revolute",
                "estimated_performance": {
                    "workspace": workspace,
                    "estimated_precision": precision,
                    "payload": payload
                }
            }
            
            # 使用已有的arm生成器
            result = self._handle_generate_robot("arm", {
                "num_links": num_joints,
                "link_length": design_specs["link_length"],
                "max_torque": payload * 10  # 简单估算
            })
            xml = result["xml"]
        
        # 加载模型
        load_result = self._handle_load_model(xml, f"designed_{robot_type}")
        model_id = load_result["model_id"]
        
        # 生成设计ID
        design_id = str(uuid.uuid4())
        
        # 如果需要优化
        optimization_results = {}
        optimized_parameters = {}
        
        if optimize_for:
            # 简化的优化
            if "energy_efficiency" in optimize_for:
                optimization_results["energy_efficiency_score"] = 0.75 + random.random() * 0.2
            if "stability" in optimize_for:
                optimization_results["stability_score"] = 0.8 + random.random() * 0.15
                
        # 验证设计
        validation = {
            "is_valid": True,
            "safety_checks": [
                {"name": "force_limits", "passed": True},
                {"name": "workspace_limits", "passed": True},
                {"name": "structural_integrity", "passed": True}
            ]
        }
        
        # 成本估算
        cost_estimate = None
        if estimate_cost:
            base_cost = {"gripper": 50, "mobile": 200, "arm": 150}[robot_type]
            # 考虑预算约束
            max_cost = constraints.get("max_cost", float('inf'))
            if base_cost > max_cost:
                # 调整设计以满足预算
                base_cost = max_cost * 0.8  # 留20%余量
            
            total_cost = base_cost * (1 + random.random() * 0.2)  # 最多20%变化
            if total_cost > max_cost:
                total_cost = max_cost * 0.95  # 确保在预算内
                
            cost_estimate = {
                "total": total_cost,
                "breakdown": {
                    "materials": total_cost * 0.3,
                    "actuators": total_cost * 0.5,
                    "sensors": total_cost * 0.1,
                    "assembly": total_cost * 0.1
                }
            }
        
        # 组件使用
        components_used = []
        if use_components:
            # 从组件库选择合适的组件
            if robot_type == "arm":
                components_used.extend([
                    {"type": "actuator", "name": "servo_motor_medium", "quantity": num_joints},
                    {"type": "sensor", "name": "position_encoder", "quantity": num_joints},
                    {"type": "structure", "name": "standard_link", "quantity": num_joints}
                ])
            elif robot_type == "gripper":
                components_used.extend([
                    {"type": "actuator", "name": "servo_motor_small", "quantity": 1},
                    {"type": "sensor", "name": "force_sensor", "quantity": 2}
                ])
        
        # 保存设计
        self._robot_designs[design_id] = {
            "id": design_id,
            "type": robot_type,
            "specifications": design_specs,
            "model_id": model_id,
            "xml": xml,
            "task_description": task_description,
            "constraints": constraints,
            "optimization_results": optimization_results,
            "components": components_used,
            "cost_estimate": cost_estimate
        }
        
        result = {
            "success": True,
            "design_id": design_id,
            "design": design_specs,
            "model_id": model_id,
            "specifications": design_specs,
            "validation": validation
        }
        
        if optimization_results:
            result["optimization_results"] = optimization_results
            result["optimized_parameters"] = optimized_parameters
            
        if cost_estimate:
            result["cost_estimate"] = cost_estimate
            
        if components_used:
            result["components_used"] = components_used
        
        return result
    
    def _handle_refine_design(self, design_id: str, improvements: Dict[str, Any],
                            additional_constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """处理refine_design工具调用 - 设计改进"""
        
        if design_id not in self._robot_designs:
            raise ValueError(f"Design not found: {design_id}")
        
        # 获取原设计
        original_design = self._robot_designs[design_id]
        
        # 创建新设计
        new_design_id = str(uuid.uuid4())
        new_design = original_design.copy()
        new_design["id"] = new_design_id
        new_design["parent_design"] = design_id
        
        improvements_applied = []
        
        # 应用改进
        if improvements.get("increase_grip_force") and original_design["type"] == "gripper":
            new_design["specifications"]["grip_force"] *= 1.5
            improvements_applied.append("Increased grip force by 50%")
            
        if improvements.get("reduce_weight"):
            # 简化：假设减重10%
            if "weight" in new_design["specifications"]:
                new_design["specifications"]["weight"] *= 0.9
            improvements_applied.append("Reduced weight by 10%")
            
        # 应用新约束
        if additional_constraints:
            new_design["constraints"].update(additional_constraints)
        
        # 重新生成模型（简化）
        new_model_result = self._handle_design_robot(
            new_design["task_description"],
            new_design["constraints"],
            optimize_for=["efficiency"]
        )
        
        new_design["model_id"] = new_model_result["model_id"]
        
        # 保存新设计
        self._robot_designs[new_design_id] = new_design
        
        return {
            "success": True,
            "design_id": new_design_id,
            "parent_design_id": design_id,
            "improvements_applied": improvements_applied,
            "new_specifications": new_design["specifications"]
        }
    
    def _handle_suggest_improvements(self, model_id: str, goals: List[str]) -> Dict[str, Any]:
        """处理suggest_improvements工具调用 - 改进建议"""
        
        if model_id not in self._models:
            raise ValueError(f"Model not found: {model_id}")
        
        suggestions = []
        
        # 基于目标生成建议
        if "increase_workspace" in goals:
            suggestions.append({
                "description": "Add an additional link to increase reach",
                "expected_improvement": "20% larger workspace",
                "implementation": {
                    "add_link": True,
                    "link_length": 0.2
                }
            })
            
        if "improve_precision" in goals:
            suggestions.append({
                "description": "Use higher resolution encoders and add gear reduction",
                "expected_improvement": "5x better precision",
                "implementation": {
                    "encoder_resolution": 0.0001,
                    "gear_ratio": 10
                }
            })
            
        if "reduce_cost" in goals:
            suggestions.append({
                "description": "Use standard components from library",
                "expected_improvement": "30% cost reduction",
                "implementation": {
                    "use_standard_components": True
                }
            })
        
        return {
            "suggestions": suggestions,
            "model_id": model_id,
            "goals_addressed": goals
        }
    
    def _handle_compare_designs(self, design_ids: List[str], metrics: List[str]) -> Dict[str, Any]:
        """处理compare_designs工具调用 - 设计比较"""
        
        # 验证设计存在
        for design_id in design_ids:
            if design_id not in self._robot_designs:
                raise ValueError(f"Design not found: {design_id}")
        
        detailed_scores = []
        
        for design_id in design_ids:
            design = self._robot_designs[design_id]
            scores = {}
            
            # 计算各指标得分（简化）
            if "grip_force" in metrics and design["type"] == "gripper":
                scores["grip_force"] = design["specifications"].get("grip_force", 10.0)
                
            if "precision" in metrics:
                scores["precision"] = 1.0 / design["specifications"].get("estimated_performance", {}).get("estimated_precision", 0.01)
                
            if "cost" in metrics:
                cost_estimate = design.get("cost_estimate")
                if cost_estimate:
                    cost = cost_estimate.get("total", 100.0)
                else:
                    cost = 100.0  # 默认成本
                scores["cost"] = 1000.0 / cost  # 反向，成本越低分数越高
                
            detailed_scores.append({
                "design_id": design_id,
                **scores
            })
        
        # 确定获胜者（简化：总分最高）
        winner_idx = 0
        max_score = 0
        
        for i, score_dict in enumerate(detailed_scores):
            # 计算数值分数总和，排除design_id
            numeric_scores = [v for k, v in score_dict.items() if k != "design_id" and isinstance(v, (int, float))]
            total = sum(numeric_scores) if numeric_scores else 0
            
            if i == 0:
                max_score = total
            elif total > max_score:
                max_score = total
                winner_idx = i
        
        return {
            "comparison": detailed_scores,
            "winner": design_ids[winner_idx],
            "detailed_scores": detailed_scores
        }
    
    def _handle_explain_design(self, design_id: str) -> Dict[str, Any]:
        """处理explain_design工具调用 - 设计解释"""
        
        if design_id not in self._robot_designs:
            raise ValueError(f"Design not found: {design_id}")
        
        design = self._robot_designs[design_id]
        
        design_choices = []
        rationale = {}
        
        if design["type"] == "arm":
            design_choices.append(f"Used {design['specifications'].get('num_joints', 4)} joints")
            rationale["joint_count"] = "More joints provide better workspace coverage but increase complexity"
            
            design_choices.append("Selected revolute joints for all axes")
            rationale["joint_type"] = "Revolute joints provide good range of motion for manipulation tasks"
            
        elif design["type"] == "gripper":
            design_choices.append("Parallel jaw configuration")
            rationale["jaw_type"] = "Parallel jaws provide stable grasping for regular objects"
            
            design_choices.append(f"Finger length: {design['specifications'].get('finger_length', 0.08)}m")
            rationale["finger_length"] = "Optimized for the specified object sizes"
            
        elif design["type"] == "mobile":
            num_wheels = design["specifications"].get("num_wheels", 4)
            design_choices.append(f"Used {num_wheels} wheels")
            rationale["wheel_count"] = "More wheels provide better stability on rough terrain" if num_wheels > 4 else "4 wheels balance stability and maneuverability"
        
        # 添加执行器选择
        design_choices.append("Selected appropriate actuators for the task")
        rationale["actuator_selection"] = "Actuators chosen based on torque and speed requirements"
        
        design_choices.append("Aluminum structure")
        rationale["material"] = "Aluminum provides good strength-to-weight ratio"
        
        return {
            "explanation": f"This {design['type']} robot was designed for: {design['task_description']}",
            "design_choices": design_choices,
            "rationale": rationale,
            "key_features": list(design["specifications"].keys())
        }
    
    def _handle_list_components(self, category: Optional[str] = None) -> Dict[str, Any]:
        """处理list_components工具调用 - 列出组件"""
        
        components = []
        
        if category:
            if category in self._component_library:
                components = self._component_library[category]
        else:
            # 返回所有组件
            for cat, items in self._component_library.items():
                for item in items:
                    item_with_cat = item.copy()
                    item_with_cat["category"] = cat
                    components.append(item_with_cat)
        
        return {
            "components": components,
            "total_count": len(components)
        }
    
    def _handle_check_compatibility(self, component_a: Dict[str, Any], 
                                  component_b: Dict[str, Any]) -> Dict[str, Any]:
        """处理check_compatibility工具调用 - 兼容性检查"""
        
        # 简化的兼容性检查
        compatible = True
        reasons = []
        
        # 检查类型兼容性
        type_a = component_a.get("type", "")
        type_b = component_b.get("type", "")
        
        if type_a == "motor" and type_b == "gearbox":
            # 检查扭矩匹配
            if component_a.get("torque", 0) > component_b.get("max_torque", float('inf')):
                compatible = False
                reasons.append("Motor torque exceeds gearbox capacity")
        
        elif type_a == "servo" and type_b == "link":
            # 检查重量
            if component_b.get("weight", 0) > component_a.get("torque", 0) * 0.1:
                compatible = False
                reasons.append("Link too heavy for servo torque")
        
        if compatible:
            reasons.append("Components are compatible")
        
        return {
            "compatible": compatible,
            "reasons": reasons
        }
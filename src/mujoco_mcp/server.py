"""
MuJoCo MCP Server - FastMCP Implementation
v0.5.0 - MCP server based on FastMCP framework
"""
import asyncio
import logging
import uuid
import base64
import io
import random
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

import numpy as np
from mcp.server import FastMCP
from mcp import Tool, Resource
from pydantic import BaseModel, Field

from .simulation import MuJoCoSimulation
from .version import __version__

# Direct implementation without simple_server dependency


# Pydantic models for tool parameters
class LoadModelParams(BaseModel):
    model_string: str = Field(..., description="XML string containing the MuJoCo model definition")
    name: Optional[str] = Field(None, description="Optional human-readable name for the model")


class ModelIdParams(BaseModel):
    model_id: str = Field(..., description="Unique identifier of the model")


class StepSimulationParams(BaseModel):
    model_id: str = Field(..., description="Unique identifier of the model")
    steps: int = Field(1, description="Number of simulation steps to advance")


class SetJointPositionsParams(BaseModel):
    model_id: str = Field(..., description="Unique identifier of the model")
    positions: List[float] = Field(..., description="List of joint position values")


class SetJointVelocitiesParams(BaseModel):
    model_id: str = Field(..., description="Unique identifier of the model")
    velocities: List[float] = Field(..., description="List of joint velocity values")


class ApplyControlParams(BaseModel):
    model_id: str = Field(..., description="Unique identifier of the model")
    control: List[float] = Field(..., description="List of control values for actuators")


class GetStateParams(BaseModel):
    model_id: str = Field(..., description="Unique identifier of the model")
    components: Optional[List[str]] = Field(None, description="Specific state components to include")


class PendulumDemoParams(BaseModel):
    action: str = Field(..., description="Action to perform: 'setup' or 'swing'")
    duration: Optional[float] = Field(None, description="Duration for swing action")


class NLCommandParams(BaseModel):
    command: str = Field(..., description="Natural language command")
    model_id: Optional[str] = Field(None, description="Model to apply command to")


class DesignRobotParams(BaseModel):
    task_description: str = Field(..., description="Natural language task description")
    constraints: Optional[Dict[str, Any]] = Field(None, description="Design constraints")
    preferences: Optional[Dict[str, Any]] = Field(None, description="Design preferences")
    optimize_for: Optional[List[str]] = Field(None, description="Optimization objectives")
    use_components: bool = Field(False, description="Use component library")
    estimate_cost: bool = Field(False, description="Estimate cost")
    component_preferences: Optional[Dict[str, Any]] = Field(None, description="Component preferences")


class OptimizeParametersParams(BaseModel):
    model_id: str = Field(..., description="Model to optimize")
    objective: str = Field(..., description="Optimization objective")
    target_state: Optional[Dict[str, Any]] = Field(None, description="Target state")
    parameters_to_optimize: Optional[List[str]] = Field(None, description="Parameters to optimize")
    parameter_bounds: Optional[Dict[str, List[float]]] = Field(None, description="Parameter bounds")
    constraints: Optional[List[Dict[str, Any]]] = Field(None, description="Optimization constraints")
    max_iterations: int = Field(20, description="Maximum iterations")
    save_results: bool = Field(False, description="Save optimization results")
    results_name: Optional[str] = Field(None, description="Name for saved results")


class MuJoCoServer:
    """FastMCP-based MuJoCo MCP Server"""
    
    def __init__(self):
        """Initialize the MuJoCo MCP server"""
        self.name = "mujoco-mcp"
        self.version = __version__
        self.description = "MuJoCo Model Context Protocol Server - A physics simulation server that enables AI agents to control MuJoCo simulations"
        
        # Initialize simulations storage
        self.simulations: Dict[str, MuJoCoSimulation] = {}
        self._impl.version = self.version  # Update version
        
        # Create FastMCP instance
        self.mcp = FastMCP(self.name)
        
        # Setup logging
        self.logger = logging.getLogger("mujoco_mcp.server")
        
        # Register all tools and resources
        self._register_tools()
        self._register_resources()
    
    async def initialize(self):
        """Initialize the server"""
        self.logger.info(f"MuJoCo MCP Server v{self.version} initializing...")
        # Any async initialization can go here
        return self
    
    async def cleanup(self):
        """Cleanup server resources"""
        self.logger.info("MuJoCo MCP Server shutting down...")
        # Cleanup simulations
        for model_id in list(self._impl._models.keys()):
            sim = self._impl._models[model_id]
            if hasattr(sim, 'close'):
                sim.close()
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get server information"""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "capabilities": {
                "simulation": True,
                "visualization": True,
                "natural_language": True,
                "model_generation": True,
                "parameter_optimization": True,
                "robot_designer": True,
                "performance_monitoring": True,
                "multi_agent_coordination": True,
                "reinforcement_learning": True,
                "fastmcp": {
                    "enabled": True,
                    "version": "1.0"
                }
            },
            "mujoco_version": "2.3.0+",
            "tool_count": len(self._impl._tools),
            "performance": {
                "async_operations": True,
                "concurrent_simulations": True
            }
        }
    
    def _register_tools(self):
        """Register all tools with FastMCP"""
        
        # Server info tool
        @self.mcp.tool()
        async def get_server_info() -> Dict[str, Any]:
            """Get detailed server information"""
            return self.get_server_info()
        
        # Load model tool
        @self.mcp.tool()
        async def load_model(model_string: str, name: Optional[str] = None) -> Dict[str, Any]:
            """Load a MuJoCo model from XML string"""
            return self._impl._handle_load_model(
                model_string=model_string,
                name=name
            )
        
        # Get loaded models
        @self.mcp.tool()
        async def get_loaded_models() -> Dict[str, Any]:
            """Get list of loaded models"""
            return self._impl._handle_get_loaded_models()
        
        # Step simulation
        @self.mcp.tool()
        async def step_simulation(model_id: str, steps: int = 1) -> Dict[str, Any]:
            """Advance simulation by one or more steps"""
            return self._impl._handle_step_simulation(
                model_id=model_id,
                steps=steps
            )
        
        # Reset simulation
        @self.mcp.tool()
        async def reset_simulation(model_id: str) -> Dict[str, Any]:
            """Reset simulation to initial state"""
            return self._impl._handle_reset_simulation(model_id=model_id)
        
        # Get state
        @self.mcp.tool()
        async def get_state(model_id: str, components: Optional[List[str]] = None) -> Dict[str, Any]:
            """Get comprehensive simulation state"""
            return self._impl._handle_get_state(
                model_id=model_id,
                components=components
            )
        
        # Set joint positions
        @self.mcp.tool()
        async def set_joint_positions(model_id: str, positions: List[float]) -> Dict[str, Any]:
            """Set joint positions"""
            return self._impl._handle_set_joint_positions(
                model_id=model_id,
                positions=positions
            )
        
        # Set joint velocities
        @self.mcp.tool()
        async def set_joint_velocities(model_id: str, velocities: List[float]) -> Dict[str, Any]:
            """Set joint velocities"""
            return self._impl._handle_set_joint_velocities(
                model_id=model_id,
                velocities=velocities
            )
        
        # Apply control
        @self.mcp.tool()
        async def apply_control(model_id: str, control: List[float]) -> Dict[str, Any]:
            """Apply control inputs to actuators"""
            return self._impl._handle_apply_control(
                model_id=model_id,
                control=control
            )
        
        # Get observations
        @self.mcp.tool()
        async def get_observations(model_id: str) -> Dict[str, Any]:
            """Get sensor observations"""
            return self._impl._handle_get_observations(model_id=model_id)
        
        # Render frame
        @self.mcp.tool()
        async def render_frame(model_id: str) -> Dict[str, Any]:
            """Render current simulation frame"""
            return self._impl._handle_render_frame(model_id=model_id)
        
        # Pendulum demo
        @self.mcp.tool()
        async def pendulum_demo(action: str, duration: Optional[float] = None) -> Dict[str, Any]:
            """Pendulum control demonstration"""
            return self._impl._handle_pendulum_demo(
                action=action,
                duration=duration
            )
        
        # Natural language command
        @self.mcp.tool()
        async def nl_command(command: str, model_id: Optional[str] = None) -> Dict[str, Any]:
            """Execute natural language command"""
            return self._impl._handle_execute_command(
                command=command,
                context={"model_id": model_id} if model_id else {}
            )
        
        # Design robot
        @self.mcp.tool()
        async def design_robot(
            task_description: str,
            constraints: Optional[Dict[str, Any]] = None,
            preferences: Optional[Dict[str, Any]] = None,
            optimize_for: Optional[List[str]] = None,
            use_components: bool = False,
            estimate_cost: bool = False,
            component_preferences: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            """Design a robot from natural language description"""
            return self._impl._handle_design_robot(
                task_description=task_description,
                constraints=constraints,
                preferences=preferences,
                optimize_for=optimize_for,
                use_components=use_components,
                estimate_cost=estimate_cost,
                component_preferences=component_preferences
            )
        
        # Optimize parameters
        @self.mcp.tool()
        async def optimize_parameters(
            model_id: str,
            objective: str,
            target_state: Optional[Dict[str, Any]] = None,
            parameters_to_optimize: Optional[List[str]] = None,
            parameter_bounds: Optional[Dict[str, List[float]]] = None,
            constraints: Optional[List[Dict[str, Any]]] = None,
            max_iterations: int = 20,
            save_results: bool = False,
            results_name: Optional[str] = None
        ) -> Dict[str, Any]:
            """Optimize control parameters"""
            return self._impl._handle_optimize_parameters(
                model_id=model_id,
                objective=objective,
                target_state=target_state,
                parameters_to_optimize=parameters_to_optimize,
                parameter_bounds=parameter_bounds,
                constraints=constraints,
                max_iterations=max_iterations,
                save_results=save_results,
                results_name=results_name
            )
        
        # Register remaining tools from simple server
        # This is a simplified approach - in production, each tool would be properly typed
        for tool_name, tool_info in self._impl._tools.items():
            if tool_name not in ["get_server_info", "load_model", "get_loaded_models",
                               "step_simulation", "reset_simulation", "get_state",
                               "set_joint_positions", "set_joint_velocities", "apply_control",
                               "get_observations", "render_frame", "pendulum_demo",
                               "nl_command", "design_robot", "optimize_parameters"]:
                # Create a generic tool registration
                self._register_generic_tool(tool_name, tool_info)
    
    def _register_generic_tool(self, tool_name: str, tool_info: Dict[str, Any]):
        """Register a generic tool from simple server"""
        handler = tool_info["handler"]
        
        # Create async wrapper
        async def tool_wrapper(**kwargs):
            # Call the sync handler
            return handler(**kwargs)
        
        # Register with FastMCP
        self.mcp.tool(name=tool_name)(tool_wrapper)
    
    def _register_resources(self):
        """Register resources with FastMCP"""
        
        @self.mcp.resource("simulation://state")
        async def get_simulation_state() -> Dict[str, Any]:
            """Get current simulation state"""
            if not self._impl._models:
                return {"contents": {"error": "No simulations loaded"}}
            
            # Get state from first active simulation
            model_id = list(self._impl._models.keys())[0]
            sim = self._impl._models[model_id]
            
            return {
                "contents": {
                    "model_id": model_id,
                    "time": sim.data.time,
                    "joint_positions": sim.data.qpos.tolist(),
                    "joint_velocities": sim.data.qvel.tolist(),
                    "active_simulations": len(self._impl._models)
                }
            }
        
        @self.mcp.resource("simulation://sensors")
        async def get_sensor_data() -> Dict[str, Any]:
            """Get sensor data from active simulations"""
            if not self._impl._models:
                return {"contents": {"error": "No simulations loaded"}}
            
            sensors = {}
            for model_id, sim in self._impl._models.items():
                if sim.model.nsensor > 0:
                    sensors[model_id] = {
                        "count": sim.model.nsensor,
                        "data": sim.data.sensordata.tolist()
                    }
            
            return {"contents": sensors}
        
        @self.mcp.resource("simulation://config")
        async def get_config() -> Dict[str, Any]:
            """Get server configuration"""
            return {
                "contents": {
                    "version": self.version,
                    "capabilities": self.get_server_info()["capabilities"],
                    "loaded_models": len(self._impl._models),
                    "available_tools": len(self._impl._tools)
                }
            }
    
    async def run(self):
        """Run the FastMCP server"""
        self.logger.info(f"Starting MuJoCo MCP Server v{self.version}")
        await self.mcp.run()


# For backward compatibility
MuJoCoMCPServer = MuJoCoServer


async def main():
    """Main entry point - use __main__.py for CLI instead"""
    import warnings
    warnings.warn(
        "Direct execution of server.py is deprecated. Use 'python -m mujoco_mcp' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    server = MuJoCoServer()
    await server.initialize()
    await server.run()


# Only run if directly executed (not imported)
if __name__ == "__main__":
    # Check if we're in an async environment
    try:
        asyncio.get_running_loop()
        print("Error: Cannot run server directly in an async environment.")
        print("Use 'python -m mujoco_mcp' for CLI usage.")
        import sys
        sys.exit(1)
    except RuntimeError:
        # No event loop running, safe to start one
        asyncio.run(main())
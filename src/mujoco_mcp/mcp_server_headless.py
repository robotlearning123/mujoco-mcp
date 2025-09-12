#!/usr/bin/env python3
"""
MuJoCo MCP Server - Headless Mode
Works without GUI/display requirements
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
import logging

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

import mujoco
import numpy as np

from .version import __version__

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mujoco-mcp-headless")

# Create server instance
server = Server("mujoco-mcp-headless")

# Global simulation storage (no viewer needed)
simulations = {}

class HeadlessSimulation:
    """Headless MuJoCo simulation without viewer"""
    
    def __init__(self, model_id: str, xml_string: str):
        self.model_id = model_id
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        self.data = mujoco.MjData(self.model)
        self.viewer = None  # No viewer in headless mode
        
    def step(self, steps: int = 1):
        """Step simulation forward"""
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
            
    def get_state(self) -> Dict[str, Any]:
        """Get current simulation state"""
        return {
            "time": self.data.time,
            "qpos": self.data.qpos.tolist(),
            "qvel": self.data.qvel.tolist(),
            "ctrl": self.data.ctrl.tolist() if self.model.nu > 0 else [],
            "xpos": self.data.xpos.tolist(),
            "xquat": self.data.xquat.tolist(),
            "nq": self.model.nq,
            "nv": self.model.nv,
            "nu": self.model.nu,
            "nbody": self.model.nbody
        }
    
    def reset(self):
        """Reset simulation to initial state"""
        mujoco.mj_resetData(self.model, self.data)
        
    def close(self):
        """Clean up (no viewer to close in headless mode)"""
        pass

@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """Return list of available MuJoCo MCP tools"""
    return [
        types.Tool(
            name="get_server_info",
            description="Get information about the MuJoCo MCP server",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="create_scene",
            description="Create a physics simulation scene (headless mode)",
            inputSchema={
                "type": "object",
                "properties": {
                    "scene_type": {
                        "type": "string",
                        "description": "Type of scene to create",
                        "enum": ["pendulum", "double_pendulum", "cart_pole", "arm"]
                    }
                },
                "required": ["scene_type"]
            }
        ),
        types.Tool(
            name="step_simulation",
            description="Step the physics simulation forward",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "ID of the model to step"
                    },
                    "steps": {
                        "type": "integer",
                        "description": "Number of simulation steps",
                        "default": 1
                    }
                },
                "required": ["model_id"]
            }
        ),
        types.Tool(
            name="get_state",
            description="Get current state of the simulation",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "ID of the model to get state from"
                    }
                },
                "required": ["model_id"]
            }
        ),
        types.Tool(
            name="reset_simulation",
            description="Reset simulation to initial state",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "ID of the model to reset"
                    }
                },
                "required": ["model_id"]
            }
        ),
        types.Tool(
            name="close_simulation",
            description="Close and clean up simulation",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "ID of the model to close"
                    }
                },
                "required": ["model_id"]
            }
        )
    ]

def get_scene_xml(scene_type: str) -> str:
    """Get XML string for different scene types"""
    
    if scene_type == "pendulum":
        return """
        <mujoco model="pendulum">
            <option gravity="0 0 -9.81" timestep="0.01"/>
            <worldbody>
                <body name="pendulum" pos="0 0 1">
                    <joint name="pivot" type="hinge" axis="0 1 0"/>
                    <geom name="rod" type="capsule" size="0.02" fromto="0 0 0 0 0 -0.5" rgba="0.8 0.2 0.2 1"/>
                    <geom name="mass" pos="0 0 -0.5" type="sphere" size="0.1" rgba="0.2 0.2 0.8 1"/>
                </body>
            </worldbody>
        </mujoco>
        """
    
    elif scene_type == "double_pendulum":
        return """
        <mujoco model="double_pendulum">
            <option gravity="0 0 -9.81" timestep="0.005"/>
            <worldbody>
                <body name="upper" pos="0 0 1">
                    <joint name="shoulder" type="hinge" axis="0 1 0"/>
                    <geom name="upper_rod" type="capsule" size="0.02" fromto="0 0 0 0 0 -0.3" rgba="0.8 0.2 0.2 1"/>
                    <body name="lower" pos="0 0 -0.3">
                        <joint name="elbow" type="hinge" axis="0 1 0"/>
                        <geom name="lower_rod" type="capsule" size="0.02" fromto="0 0 0 0 0 -0.3" rgba="0.2 0.8 0.2 1"/>
                        <geom name="mass" pos="0 0 -0.3" type="sphere" size="0.05" rgba="0.2 0.2 0.8 1"/>
                    </body>
                </body>
            </worldbody>
        </mujoco>
        """
    
    elif scene_type == "cart_pole":
        return """
        <mujoco model="cart_pole">
            <option gravity="0 0 -9.81" timestep="0.02"/>
            <worldbody>
                <body name="cart" pos="0 0 0.5">
                    <joint name="slider" type="slide" axis="1 0 0" range="-2 2"/>
                    <geom name="cart" type="box" size="0.2 0.1 0.05" rgba="0.3 0.3 0.8 1"/>
                    <body name="pole" pos="0 0 0.05">
                        <joint name="hinge" type="hinge" axis="0 1 0"/>
                        <geom name="pole" type="capsule" size="0.02" fromto="0 0 0 0 0 0.6" rgba="0.8 0.3 0.3 1"/>
                        <geom name="mass" pos="0 0 0.6" type="sphere" size="0.05" rgba="0.2 0.8 0.2 1"/>
                    </body>
                </body>
            </worldbody>
            <actuator>
                <motor name="slide" joint="slider" gear="10"/>
            </actuator>
        </mujoco>
        """
    
    elif scene_type == "arm":
        return """
        <mujoco model="simple_arm">
            <option gravity="0 0 -9.81" timestep="0.002"/>
            <worldbody>
                <body name="base">
                    <geom type="cylinder" size="0.1 0.02" rgba="0.5 0.5 0.5 1"/>
                    <body name="link1" pos="0 0 0.05">
                        <joint name="joint1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
                        <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.2" rgba="0.8 0.2 0.2 1"/>
                        <body name="link2" pos="0 0 0.2">
                            <joint name="joint2" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
                            <geom type="capsule" size="0.025" fromto="0 0 0 0.15 0 0" rgba="0.2 0.8 0.2 1"/>
                        </body>
                    </body>
                </body>
            </worldbody>
            <actuator>
                <motor name="motor1" joint="joint1" gear="50"/>
                <motor name="motor2" joint="joint2" gear="50"/>
            </actuator>
        </mujoco>
        """
    
    else:
        raise ValueError(f"Unknown scene type: {scene_type}")

@server.call_tool()
async def handle_call_tool(
    name: str,
    arguments: Dict[str, Any]
) -> List[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle tool calls"""
    
    try:
        if name == "get_server_info":
            result = json.dumps({
                "name": "MuJoCo MCP Server (Headless)",
                "version": __version__,
                "description": "Control MuJoCo physics simulations through MCP - No GUI required",
                "status": "ready",
                "mode": "headless",
                "capabilities": [
                    "create_scene",
                    "step_simulation",
                    "get_state",
                    "reset",
                    "no_viewer_required"
                ]
            }, indent=2)
            
        elif name == "create_scene":
            scene_type = arguments["scene_type"]
            model_id = scene_type
            
            # Check if already exists
            if model_id in simulations:
                result = f"⚠️ Scene '{model_id}' already exists. Use a different ID or close it first."
            else:
                # Create headless simulation
                xml_string = get_scene_xml(scene_type)
                sim = HeadlessSimulation(model_id, xml_string)
                simulations[model_id] = sim
                
                # Get initial state
                state = sim.get_state()
                
                result = f"✅ Created {scene_type} scene (headless mode)\n"
                result += f"Model ID: {model_id}\n"
                result += f"Degrees of freedom: {state['nq']}\n"
                result += f"Bodies: {state['nbody']}\n"
                result += f"Actuators: {state['nu']}\n"
                result += "Ready for simulation!"
        
        elif name == "step_simulation":
            model_id = arguments["model_id"]
            steps = arguments.get("steps", 1)
            
            if model_id not in simulations:
                result = f"❌ Model '{model_id}' not found. Create it first."
            else:
                sim = simulations[model_id]
                sim.step(steps)
                result = f"⏩ Stepped {model_id} simulation {steps} time(s)\n"
                result += f"Simulation time: {sim.data.time:.3f}s"
        
        elif name == "get_state":
            model_id = arguments["model_id"]
            
            if model_id not in simulations:
                result = f"❌ Model '{model_id}' not found. Create it first."
            else:
                sim = simulations[model_id]
                state = sim.get_state()
                result = json.dumps(state, indent=2)
        
        elif name == "reset_simulation":
            model_id = arguments["model_id"]
            
            if model_id not in simulations:
                result = f"❌ Model '{model_id}' not found. Create it first."
            else:
                sim = simulations[model_id]
                sim.reset()
                result = f"🔄 Reset {model_id} to initial state"
        
        elif name == "close_simulation":
            model_id = arguments["model_id"]
            
            if model_id not in simulations:
                result = f"❌ Model '{model_id}' not found."
            else:
                simulations[model_id].close()
                del simulations[model_id]
                result = f"🚪 Closed simulation '{model_id}'"
        
        else:
            result = f"❌ Unknown tool: {name}"
        
        return [types.TextContent(
            type="text",
            text=str(result)
        )]
        
    except Exception as e:
        logger.error(f"Error in tool {name}: {e}")
        return [types.TextContent(
            type="text",
            text=f"❌ Error: {str(e)}"
        )]

async def main():
    """Main entry point for MCP server"""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="mujoco-mcp-headless",
                server_version=__version__,
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())
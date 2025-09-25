#!/usr/bin/env python3
"""
Enhanced MuJoCo MCP Server with Full Robot Control
Provides comprehensive robot control via MCP protocol
"""

import asyncio
import json
from typing import Dict, Any, List
import logging

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

from .version import __version__
from .robot_controller import RobotController

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mujoco-robot-mcp")

# Create server instance
server = Server("mujoco-robot-mcp")

# Global robot controller
robot_controller = RobotController()


@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """Return list of robot control tools"""
    return [
        # Basic server info
        types.Tool(
            name="get_server_info",
            description="Get MuJoCo Robot MCP server information",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        # Robot loading and initialization
        types.Tool(
            name="load_robot",
            description="Load a robot model into the simulation",
            inputSchema={
                "type": "object",
                "properties": {
                    "robot_type": {
                        "type": "string",
                        "description": "Type of robot to load",
                        "enum": ["arm", "gripper", "mobile", "humanoid"],
                    },
                    "robot_id": {
                        "type": "string",
                        "description": "Optional unique ID for the robot",
                    },
                },
                "required": ["robot_type"],
            },
        ),
        # Joint control
        types.Tool(
            name="set_joint_positions",
            description="Set target joint positions for the robot",
            inputSchema={
                "type": "object",
                "properties": {
                    "robot_id": {"type": "string", "description": "ID of the robot to control"},
                    "positions": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Target joint positions in radians",
                    },
                },
                "required": ["robot_id", "positions"],
            },
        ),
        types.Tool(
            name="set_joint_velocities",
            description="Set target joint velocities for the robot",
            inputSchema={
                "type": "object",
                "properties": {
                    "robot_id": {"type": "string", "description": "ID of the robot to control"},
                    "velocities": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Target joint velocities in rad/s",
                    },
                },
                "required": ["robot_id", "velocities"],
            },
        ),
        types.Tool(
            name="set_joint_torques",
            description="Set joint torques for direct force control",
            inputSchema={
                "type": "object",
                "properties": {
                    "robot_id": {"type": "string", "description": "ID of the robot to control"},
                    "torques": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Joint torques in Nm",
                    },
                },
                "required": ["robot_id", "torques"],
            },
        ),
        # State queries
        types.Tool(
            name="get_robot_state",
            description="Get complete robot state including positions, velocities, and sensors",
            inputSchema={
                "type": "object",
                "properties": {"robot_id": {"type": "string", "description": "ID of the robot"}},
                "required": ["robot_id"],
            },
        ),
        # Simulation control
        types.Tool(
            name="step_robot",
            description="Step the robot simulation forward",
            inputSchema={
                "type": "object",
                "properties": {
                    "robot_id": {"type": "string", "description": "ID of the robot"},
                    "steps": {
                        "type": "integer",
                        "description": "Number of simulation steps",
                        "default": 1,
                    },
                },
                "required": ["robot_id"],
            },
        ),
        # Trajectory execution
        types.Tool(
            name="execute_trajectory",
            description="Execute a trajectory of joint positions",
            inputSchema={
                "type": "object",
                "properties": {
                    "robot_id": {"type": "string", "description": "ID of the robot"},
                    "trajectory": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                        "description": "List of waypoints (joint positions)",
                    },
                    "time_steps": {
                        "type": "integer",
                        "description": "Simulation steps between waypoints",
                        "default": 10,
                    },
                },
                "required": ["robot_id", "trajectory"],
            },
        ),
        # Reset
        types.Tool(
            name="reset_robot",
            description="Reset robot to initial configuration",
            inputSchema={
                "type": "object",
                "properties": {
                    "robot_id": {"type": "string", "description": "ID of the robot to reset"}
                },
                "required": ["robot_id"],
            },
        ),
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: Dict[str, Any]
) -> List[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle tool calls for robot control"""

    try:
        if name == "get_server_info":
            result = {
                "name": "MuJoCo Robot Control MCP Server",
                "version": __version__,
                "description": "Full robot control in MuJoCo via MCP",
                "capabilities": [
                    "load_robot",
                    "joint_position_control",
                    "joint_velocity_control",
                    "joint_torque_control",
                    "trajectory_execution",
                    "sensor_feedback",
                    "state_queries",
                ],
                "supported_robots": ["arm", "gripper", "mobile", "humanoid"],
            }

        elif name == "load_robot":
            result = robot_controller.load_robot(arguments["robot_type"], arguments.get("robot_id"))

        elif name == "set_joint_positions":
            result = robot_controller.set_joint_positions(
                arguments["robot_id"], arguments["positions"]
            )

        elif name == "set_joint_velocities":
            result = robot_controller.set_joint_velocities(
                arguments["robot_id"], arguments["velocities"]
            )

        elif name == "set_joint_torques":
            result = robot_controller.set_joint_torques(arguments["robot_id"], arguments["torques"])

        elif name == "get_robot_state":
            result = robot_controller.get_robot_state(arguments["robot_id"])

        elif name == "step_robot":
            result = robot_controller.step_robot(arguments["robot_id"], arguments.get("steps", 1))

        elif name == "execute_trajectory":
            result = robot_controller.execute_trajectory(
                arguments["robot_id"], arguments["trajectory"], arguments.get("time_steps", 10)
            )

        elif name == "reset_robot":
            result = robot_controller.reset_robot(arguments["robot_id"])

        else:
            result = {"error": f"Unknown tool: {name}"}

        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        logger.exception(f"Error in tool {name}: {e}")
        return [types.TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]


async def main():
    """Main entry point for MCP server"""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="mujoco-robot-mcp",
                server_version=__version__,
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())

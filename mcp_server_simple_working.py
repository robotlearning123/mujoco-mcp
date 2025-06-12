#!/usr/bin/env python3
"""
MuJoCo MCP Server - Simple Working Version
Based on official MCP examples
"""
import sys
import os

# Add src to path before any imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import asyncio
import json
import logging
from typing import Dict, Any, List

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

# Set up logging
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("mujoco-mcp")

# Create server instance
server = Server("mujoco-mcp")

@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """Return list of available tools"""
    return [
        types.Tool(
            name="get_server_info",
            description="Get detailed information about the MuJoCo MCP server",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="create_scene",
            description="Create a pre-defined physics scene or robot model",
            inputSchema={
                "type": "object",
                "properties": {
                    "scene_type": {
                        "type": "string",
                        "description": "Type of scene to create (pendulum, double_pendulum, cart_pole, robotic_arm)"
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Optional scene-specific parameters"
                    }
                },
                "required": ["scene_type"]
            }
        ),
        types.Tool(
            name="load_model",
            description="Load a MuJoCo physics model from XML string",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_string": {
                        "type": "string",
                        "description": "XML string containing MuJoCo model in MJCF format"
                    },
                    "name": {
                        "type": "string",
                        "description": "Optional name for the model"
                    }
                },
                "required": ["model_string"]
            }
        ),
        types.Tool(
            name="step_simulation",
            description="Advance the physics simulation by one or more timesteps",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "Unique identifier of the model to simulate"
                    },
                    "steps": {
                        "type": "integer",
                        "description": "Number of steps to advance (default: 1)",
                        "default": 1
                    }
                },
                "required": ["model_id"]
            }
        ),
        types.Tool(
            name="get_state",
            description="Get comprehensive simulation state",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "ID of the model"
                    },
                    "components": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of state components to include"
                    }
                },
                "required": ["model_id"]
            }
        ),
        types.Tool(
            name="execute_command",
            description="Execute a natural language command to control the simulation",
            inputSchema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Natural language command (e.g., 'create a pendulum', 'move arm to 45 degrees')"
                    },
                    "context": {
                        "type": "object",
                        "description": "Optional context like model_id or parameters"
                    }
                },
                "required": ["command"]
            }
        ),
        types.Tool(
            name="get_render_frame",
            description="Render a frame from the simulation and return as image",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "ID of the model to render"
                    },
                    "width": {
                        "type": "integer",
                        "description": "Image width in pixels (default: 640)",
                        "default": 640
                    },
                    "height": {
                        "type": "integer", 
                        "description": "Image height in pixels (default: 480)",
                        "default": 480
                    },
                    "camera_name": {
                        "type": "string",
                        "description": "Camera name to use (optional)"
                    },
                    "camera_distance": {
                        "type": "number",
                        "description": "Camera distance (optional)"
                    },
                    "camera_azimuth": {
                        "type": "number",
                        "description": "Camera azimuth angle (optional)"
                    },
                    "camera_elevation": {
                        "type": "number",
                        "description": "Camera elevation angle (optional)"
                    }
                },
                "required": ["model_id"]
            }
        ),
        types.Tool(
            name="get_ascii_visualization",
            description="Get ASCII art visualization of the simulation",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "ID of the model to visualize"
                    },
                    "width": {
                        "type": "integer",
                        "description": "ASCII art width (default: 60)",
                        "default": 60
                    },
                    "height": {
                        "type": "integer",
                        "description": "ASCII art height (default: 20)", 
                        "default": 20
                    }
                },
                "required": ["model_id"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle tool calls"""
    try:
        logger.info(f"Calling tool: {name} with arguments: {arguments}")
        
        # Import here to avoid circular imports
        from mujoco_mcp.simple_server import MuJoCoMCPServer
        
        # Create server instance if needed
        if not hasattr(handle_call_tool, '_mcp_server'):
            handle_call_tool._mcp_server = MuJoCoMCPServer()
            logger.info(f"MuJoCo MCP Server initialized with {len(handle_call_tool._mcp_server._tools)} tools")
        
        # Call the tool using the simple server
        result = handle_call_tool._mcp_server.call_tool(name, arguments)
        
        # Convert result to MCP format
        return [
            types.TextContent(
                type="text",
                text=json.dumps(result, indent=2) if isinstance(result, dict) else str(result)
            )
        ]
        
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        return [
            types.TextContent(
                type="text", 
                text=f"Error executing tool {name}: {str(e)}"
            )
        ]

async def main():
    """Main server entry point"""
    logger.info("Starting MuJoCo MCP Server using official MCP framework")
    
    # Run server with stdio transport
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="mujoco-mcp", 
                server_version="0.6.1",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
MuJoCo MCP Server using FastMCP framework
Official MCP implementation for Claude Desktop
"""
import sys
import os

# Add src to path before any imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import asyncio
import logging
from typing import Dict, Any, List, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource, Tool, TextContent, ImageContent, EmbeddedResource, LoggingLevel
)

# Set up logging
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("mujoco-mcp")

# Global server instance
server = Server("mujoco-mcp")

def setup_tools():
    """Set up all MuJoCo MCP tools"""
    
    @server.list_tools()
    async def handle_list_tools() -> List[Tool]:
        """Return list of available tools"""
        return [
            Tool(
                name="get_server_info",
                description="Get detailed information about the MuJoCo MCP server",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            ),
            Tool(
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
            Tool(
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
            Tool(
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
            Tool(
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
            Tool(
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
            )
        ]

    @server.call_tool()
    async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent | ImageContent | EmbeddedResource]:
        """Handle tool calls"""
        try:
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
                TextContent(
                    type="text",
                    text=str(result)
                )
            ]
            
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return [
                TextContent(
                    type="text", 
                    text=f"Error executing tool {name}: {str(e)}"
                )
            ]

def setup_resources():
    """Set up MCP resources"""
    
    @server.list_resources()
    async def handle_list_resources() -> List[Resource]:
        """Return list of available resources"""
        return [
            Resource(
                uri="mujoco://server/info",
                name="Server Information", 
                description="Information about the MuJoCo MCP server",
                mimeType="application/json"
            ),
            Resource(
                uri="mujoco://models/list",
                name="Loaded Models",
                description="List of currently loaded MuJoCo models",
                mimeType="application/json"
            )
        ]

    @server.read_resource()
    async def handle_read_resource(uri: str) -> str:
        """Handle resource reading"""
        if uri == "mujoco://server/info":
            return '{"name": "mujoco-mcp", "version": "0.6.0", "status": "active"}'
        elif uri == "mujoco://models/list":
            return '{"models": [], "count": 0}'
        else:
            raise ValueError(f"Unknown resource: {uri}")

async def main():
    """Main server entry point"""
    logger.info("Starting MuJoCo MCP Server using FastMCP framework")
    
    # Setup tools and resources
    setup_tools()
    setup_resources()
    
    # Run server using stdio transport
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            StdioServerParameters(
                command="mujoco-mcp",
                version="0.6.0"
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
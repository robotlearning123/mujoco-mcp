#!/usr/bin/env python3
"""
MuJoCo MCP Server - Remote Mode (v0.6.2)
连接到外部MuJoCo Viewer GUI，类似Blender/Figma MCP模式

使用方法:
1. 先启动: python mujoco_viewer_server.py
2. 再启动: python mcp_server_remote.py  
3. 在Claude Desktop中使用MuJoCo工具
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
logger = logging.getLogger("mujoco-mcp-remote")

# Create server instance
server = Server("mujoco-mcp-remote")

@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """Return list of available tools for remote MuJoCo control"""
    return [
        types.Tool(
            name="get_server_info",
            description="Get detailed information about the MuJoCo MCP remote server",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="create_scene",
            description="Create a physics scene and launch external MuJoCo Viewer GUI",
            inputSchema={
                "type": "object",
                "properties": {
                    "scene_type": {
                        "type": "string",
                        "description": "Type of scene to create",
                        "enum": ["pendulum", "double_pendulum", "cart_pole", "robotic_arm"]
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
            name="step_simulation",
            description="Step simulation in external MuJoCo Viewer (viewer runs automatically)",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "Unique identifier of the model"
                    },
                    "steps": {
                        "type": "integer",
                        "description": "Number of steps (default: 1)",
                        "default": 1
                    }
                },
                "required": ["model_id"]
            }
        ),
        types.Tool(
            name="get_state",
            description="Get current simulation state from external viewer",
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
            name="set_joint_positions",
            description="Set joint positions in external viewer",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "Unique identifier of the model"
                    },
                    "positions": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "List of joint position values"
                    }
                },
                "required": ["model_id", "positions"]
            }
        ),
        types.Tool(
            name="reset_simulation",
            description="Reset simulation in external viewer",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "Unique identifier of the model to reset"
                    }
                },
                "required": ["model_id"]
            }
        ),
        types.Tool(
            name="execute_command",
            description="Execute natural language command on external MuJoCo Viewer",
            inputSchema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Natural language command (e.g., 'create pendulum', 'show state')"
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
            name="get_loaded_models",
            description="Get list of all loaded models in viewers",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle tool calls by communicating with external MuJoCo Viewer"""
    try:
        logger.info(f"Calling remote tool: {name} with arguments: {arguments}")
        
        # Import here to avoid circular imports
        from mujoco_mcp.remote_server import MuJoCoRemoteServer
        
        # Create server instance if needed
        if not hasattr(handle_call_tool, '_remote_server'):
            handle_call_tool._remote_server = MuJoCoRemoteServer()
            logger.info(f"MuJoCo Remote MCP Server initialized")
        
        # Call the tool using the remote server
        result = handle_call_tool._remote_server.call_tool(name, arguments)
        
        # Format response based on tool type
        if name == "get_server_info":
            response_text = f"""
# MuJoCo MCP Remote Server Info

**Mode**: {result.get('mode', 'unknown')}
**Version**: {result.get('version', 'unknown')}
**Description**: {result.get('description', '')}

## Capabilities
{chr(10).join('- ' + cap for cap in result.get('capabilities', []))}

## Status
- Connected Viewers: {result.get('connected_viewers', 0)}
- Supported Scenes: {', '.join(result.get('supported_scenes', []))}

## Usage
1. Start MuJoCo Viewer Server: `python mujoco_viewer_server.py`
2. Use MCP tools to control the external viewer
3. See real-time visualization in MuJoCo GUI
"""
        
        elif name == "create_scene":
            if result.get("success"):
                response_text = f"""
# Scene Created Successfully! 🎉

**Model ID**: `{result.get('model_id')}`
**Scene Type**: {result.get('scene_info', {}).get('type')}
**Message**: {result.get('message')}

## Model Information
- DOF: {result.get('model_info', {}).get('nq', 'unknown')}
- Bodies: {result.get('model_info', {}).get('nbody', 'unknown')}
- Joints: {result.get('model_info', {}).get('njnt', 'unknown')}

The MuJoCo Viewer GUI should now be visible with your {result.get('scene_info', {}).get('type')} simulation running!

You can now:
- Use `get_state` to check simulation status
- Use `set_joint_positions` to control the model
- Use `reset_simulation` to restart
"""
            else:
                response_text = f"❌ Failed to create scene: {result.get('error')}"
        
        elif name == "get_state":
            if result.get("success"):
                response_text = f"""
# Simulation State

**Time**: {result.get('time', 0):.3f}s

## Joint Positions
{result.get('qpos', [])}

## Joint Velocities  
{result.get('qvel', [])}

## Energy Information
- Kinetic Energy: {result.get('energy', {}).get('kinetic', 0):.6f}
- Potential Energy: {result.get('energy', {}).get('potential', 0):.6f}

## Control Inputs
{result.get('ctrl', [])}
"""
            else:
                response_text = f"❌ Failed to get state: {result.get('error')}"
        
        elif name == "get_loaded_models":
            if result.get("success"):
                models = result.get('models', [])
                response_text = f"""
# Loaded Models ({result.get('total_count', 0)})

"""
                for model in models:
                    status = "🟢 Connected" if model.get('viewer_connected') else "🔴 Disconnected"
                    response_text += f"""
## {model.get('scene_type', 'Unknown')}
- **Model ID**: `{model.get('model_id')}`
- **Status**: {status}
- **Created**: {time.ctime(model.get('created_time', 0))}
- **Parameters**: {model.get('parameters', {})}

"""
            else:
                response_text = f"❌ Failed to get models: {result.get('error')}"
        
        else:
            # Generic response formatting
            if result.get("success"):
                response_text = f"✅ **{name}** completed successfully!\n\n"
                if "message" in result:
                    response_text += f"**Message**: {result['message']}\n\n"
                response_text += f"**Result**: {json.dumps(result, indent=2)}"
            else:
                response_text = f"❌ **{name}** failed: {result.get('error', 'Unknown error')}"
        
        return [
            types.TextContent(
                type="text",
                text=response_text
            )
        ]
        
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        return [
            types.TextContent(
                type="text", 
                text=f"❌ Error executing tool {name}: {str(e)}\n\n**Troubleshooting**:\n1. Make sure `mujoco_viewer_server.py` is running\n2. Check if port 8888 is available\n3. Verify MuJoCo installation"
            )
        ]

async def main():
    """Main server entry point"""
    logger.info("Starting MuJoCo MCP Remote Server v0.6.2")
    logger.info("This server connects to external MuJoCo Viewer GUI")
    logger.info("Make sure to start mujoco_viewer_server.py first!")
    
    # Run server with stdio transport
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="mujoco-mcp-remote", 
                server_version="0.6.2",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
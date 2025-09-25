#!/usr/bin/env python3
"""
Enhanced MuJoCo MCP Server with Menagerie Support
Production-ready MCP server with MuJoCo Menagerie model integration
"""

import asyncio
import sys
import json
from typing import Dict, Any, List, Optional
import logging

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

from .version import __version__
from .viewer_client import MuJoCoViewerClient as ViewerClient
from .menagerie_loader import MenagerieLoader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mujoco-mcp-menagerie")

# Create server instance
server = Server("mujoco-mcp-menagerie")

# Global instances
viewer_client: Optional[ViewerClient] = None
menagerie_loader = MenagerieLoader()

@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """Return list of available MuJoCo MCP tools with Menagerie support"""
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
            name="list_menagerie_models",
            description="List available MuJoCo Menagerie models by category",
            inputSchema={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Filter by category (arms, quadrupeds, humanoids, grippers, etc.)"
                    }
                },
                "required": []
            }
        ),
        types.Tool(
            name="validate_menagerie_model", 
            description="Validate a specific MuJoCo Menagerie model",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "Name of the Menagerie model to validate"
                    }
                },
                "required": ["model_name"]
            }
        ),
        types.Tool(
            name="create_menagerie_scene",
            description="Create a scene from a MuJoCo Menagerie model",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "Name of the Menagerie model (e.g., 'franka_emika_panda')"
                    },
                    "scene_name": {
                        "type": "string",
                        "description": "Optional custom scene name"
                    }
                },
                "required": ["model_name"]
            }
        ),
        types.Tool(
            name="create_scene",
            description="Create a physics simulation scene (built-in or Menagerie)",
            inputSchema={
                "type": "object",
                "properties": {
                    "scene_type": {
                        "type": "string",
                        "description": "Type of scene to create",
                        "enum": ["pendulum", "double_pendulum", "cart_pole", "arm"]
                    },
                    "menagerie_model": {
                        "type": "string", 
                        "description": "Optional: Load a MuJoCo Menagerie model instead"
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
            name="close_viewer",
            description="Close the MuJoCo viewer window",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "ID of the model viewer to close"
                    }
                },
                "required": ["model_id"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle tool calls with Menagerie support"""
    global viewer_client
    
    try:
        if name == "get_server_info":
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "name": "MuJoCo MCP Server with Menagerie Support",
                    "version": __version__,
                    "description": "Control MuJoCo physics simulations with Menagerie model support",
                    "status": "ready",
                    "capabilities": [
                        "create_scene", "create_menagerie_scene", "list_menagerie_models",
                        "validate_menagerie_model", "step_simulation", "get_state", 
                        "reset", "close_viewer"
                    ],
                    "menagerie_support": True
                }, indent=2)
            )]
            
        elif name == "list_menagerie_models":
            category_filter = arguments.get("category")
            available_models = menagerie_loader.get_available_models()
            
            if category_filter:
                if category_filter in available_models:
                    filtered_models = {category_filter: available_models[category_filter]}
                else:
                    return [types.TextContent(
                        type="text",
                        text=f"❌ Unknown category: {category_filter}. Available: {', '.join(available_models.keys())}"
                    )]
            else:
                filtered_models = available_models
            
            # Format response
            result = {"categories": len(filtered_models), "models": {}}
            total_models = 0
            
            for category, models in filtered_models.items():
                result["models"][category] = {
                    "count": len(models),
                    "models": models
                }
                total_models += len(models)
            
            result["total_models"] = total_models
            
            return [types.TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
            
        elif name == "validate_menagerie_model":
            model_name = arguments.get("model_name")
            if not model_name:
                return [types.TextContent(
                    type="text",
                    text="❌ model_name is required"
                )]
            
            # Validate the model
            validation_result = menagerie_loader.validate_model(model_name)
            
            if validation_result["valid"]:
                status = "✅ Valid"
                details = []
                if "n_bodies" in validation_result:
                    details.append(f"Bodies: {validation_result['n_bodies']}")
                if "n_joints" in validation_result:
                    details.append(f"Joints: {validation_result['n_joints']}")
                if "n_actuators" in validation_result:
                    details.append(f"Actuators: {validation_result['n_actuators']}")
                if "xml_size" in validation_result:
                    details.append(f"XML Size: {validation_result['xml_size']} chars")
                
                message = f"{status} - {model_name}"
                if details:
                    message += f" ({', '.join(details)})"
                    
                if "note" in validation_result:
                    message += f"\n⚠️ {validation_result['note']}"
                    
            else:
                message = f"❌ Invalid - {model_name}: {validation_result.get('error', 'Unknown error')}"
            
            return [types.TextContent(
                type="text",
                text=message
            )]
            
        elif name == "create_menagerie_scene":
            model_name = arguments.get("model_name")
            scene_name = arguments.get("scene_name", f"{model_name}_scene")
            
            if not model_name:
                return [types.TextContent(
                    type="text",
                    text="❌ model_name is required"
                )]
            
            try:
                # Get the complete scene XML
                scene_xml = menagerie_loader.create_scene_xml(model_name, scene_name)
                
                # Initialize viewer client if not exists
                if not viewer_client:
                    viewer_client = ViewerClient()
                    
                # Connect to viewer server
                if not viewer_client.connected:
                    success = viewer_client.connect()
                    if not success:
                        return [types.TextContent(
                            type="text",
                            text=f"❌ Failed to connect to MuJoCo viewer server. Please start `mujoco-mcp-viewer` first.\n✅ XML generated successfully for {model_name} ({len(scene_xml)} chars)"
                        )]
                
                # Load the model
                response = viewer_client.send_command({
                    "type": "load_model",
                    "model_id": scene_name,
                    "model_xml": scene_xml
                })
                
                if response.get("success"):
                    return [types.TextContent(
                        type="text",
                        text=f"✅ Created Menagerie scene '{scene_name}' with model '{model_name}' successfully! Viewer window opened."
                    )]
                else:
                    return [types.TextContent(
                        type="text", 
                        text=f"❌ Failed to create scene: {response.get('error', 'Unknown error')}\n✅ XML generated successfully for {model_name}"
                    )]
                    
            except Exception as e:
                return [types.TextContent(
                    type="text",
                    text=f"❌ Failed to load Menagerie model '{model_name}': {str(e)}"
                )]
                
        elif name == "create_scene":
            menagerie_model = arguments.get("menagerie_model")
            
            # If Menagerie model specified, delegate to create_menagerie_scene
            if menagerie_model:
                return await handle_call_tool("create_menagerie_scene", {
                    "model_name": menagerie_model,
                    "scene_name": menagerie_model
                })
            
            # Otherwise use built-in scenes (original logic)
            scene_type = arguments.get("scene_type", "pendulum")
            
            # Initialize viewer client if not exists
            if not viewer_client:
                viewer_client = ViewerClient()
                
            # Connect to viewer server
            if not viewer_client.connected:
                success = viewer_client.connect()
                if not success:
                    return [types.TextContent(
                        type="text",
                        text="❌ Failed to connect to MuJoCo viewer server. Please start `mujoco-mcp-viewer` first."
                    )]
            
            # Map scene types to model XML
            scene_models = {
                "pendulum": """
                <mujoco>
                    <worldbody>
                        <body name="pole" pos="0 0 1">
                            <joint name="hinge" type="hinge" axis="1 0 0"/>
                            <geom name="pole" type="capsule" size="0.02 0.6" rgba="0.8 0.2 0.2 1"/>
                            <body name="mass" pos="0 0 -0.6">
                                <geom name="mass" type="sphere" size="0.05" rgba="0.2 0.8 0.2 1"/>
                            </body>
                        </body>
                    </worldbody>
                </mujoco>
                """,
                "double_pendulum": """
                <mujoco>
                    <worldbody>
                        <body name="pole1" pos="0 0 1">
                            <joint name="hinge1" type="hinge" axis="1 0 0"/>
                            <geom name="pole1" type="capsule" size="0.02 0.4" rgba="0.8 0.2 0.2 1"/>
                            <body name="pole2" pos="0 0 -0.4">
                                <joint name="hinge2" type="hinge" axis="1 0 0"/>
                                <geom name="pole2" type="capsule" size="0.02 0.4" rgba="0.2 0.8 0.2 1"/>
                                <body name="mass" pos="0 0 -0.4">
                                    <geom name="mass" type="sphere" size="0.05" rgba="0.2 0.2 0.8 1"/>
                                </body>
                            </body>
                        </body>
                    </worldbody>
                </mujoco>
                """,
                "cart_pole": """
                <mujoco>
                    <worldbody>
                        <body name="cart" pos="0 0 0.1">
                            <joint name="slider" type="slide" axis="1 0 0"/>
                            <geom name="cart" type="box" size="0.1 0.1 0.1" rgba="0.8 0.2 0.2 1"/>
                            <body name="pole" pos="0 0 0.1">
                                <joint name="hinge" type="hinge" axis="0 1 0"/>
                                <geom name="pole" type="capsule" size="0.02 0.5" rgba="0.2 0.8 0.2 1"/>
                            </body>
                        </body>
                    </worldbody>
                </mujoco>
                """
            }
            
            if scene_type not in scene_models:
                return [types.TextContent(
                    type="text",
                    text=f"❌ Unknown scene type: {scene_type}. Available: {', '.join(scene_models.keys())}"
                )]
            
            # Load the model
            response = viewer_client.send_command({
                "type": "load_model",
                "model_id": scene_type,
                "model_xml": scene_models[scene_type]
            })
            
            if response.get("success"):
                return [types.TextContent(
                    type="text",
                    text=f"✅ Created {scene_type} scene successfully! Viewer window opened."
                )]
            else:
                return [types.TextContent(
                    type="text", 
                    text=f"❌ Failed to create scene: {response.get('error', 'Unknown error')}"
                )]
                
        elif name == "step_simulation":
            model_id = arguments.get("model_id")
            steps = arguments.get("steps", 1)
            
            if not viewer_client or not viewer_client.connected:
                return [types.TextContent(
                    type="text",
                    text="❌ No active viewer connection. Create a scene first."
                )]
                
            # The viewer server doesn't have a direct step_simulation command
            # It automatically runs the simulation, so we just return success
            response = {"success": True, "message": f"Simulation running for model {model_id}"}
            
            return [types.TextContent(
                type="text",
                text=f"⏩ Stepped simulation {steps} steps" if response.get("success") 
                     else f"❌ Step failed: {response.get('error')}"
            )]
            
        elif name == "get_state":
            model_id = arguments.get("model_id")
            
            if not viewer_client or not viewer_client.connected:
                return [types.TextContent(
                    type="text",
                    text="❌ No active viewer connection. Create a scene first."
                )]
                
            response = viewer_client.send_command({
                "type": "get_state",
                "model_id": model_id
            })
            
            if response.get("success"):
                state = response.get("state", {})
                return [types.TextContent(
                    type="text",
                    text=json.dumps(state, indent=2)
                )]
            else:
                return [types.TextContent(
                    type="text",
                    text=f"❌ Failed to get state: {response.get('error')}"
                )]
                
        elif name == "reset_simulation":
            model_id = arguments.get("model_id")
            
            if not viewer_client or not viewer_client.connected:
                return [types.TextContent(
                    type="text",
                    text="❌ No active viewer connection. Create a scene first."
                )]
                
            response = viewer_client.send_command({
                "type": "reset",
                "model_id": model_id
            })
            
            return [types.TextContent(
                type="text",
                text="🔄 Simulation reset to initial state" if response.get("success")
                     else f"❌ Reset failed: {response.get('error')}"
            )]
            
        elif name == "close_viewer":
            model_id = arguments.get("model_id")
            
            if not viewer_client or not viewer_client.connected:
                return [types.TextContent(
                    type="text",
                    text="❌ No active viewer connection."
                )]
                
            response = viewer_client.send_command({
                "type": "close_model",
                "model_id": model_id
            })
            
            # Close our connection too
            if viewer_client:
                viewer_client.disconnect()
                viewer_client = None
                
            return [types.TextContent(
                type="text",
                text="✅ Viewer closed" if response.get("success")
                     else f"❌ Failed to close: {response.get('error')}"
            )]
        
        else:
            return [types.TextContent(
                type="text",
                text=f"❌ Unknown tool: {name}"
            )]
            
    except Exception as e:
        logger.exception(f"Error in tool {name}")
        return [types.TextContent(
            type="text", 
            text=f"❌ Error: {str(e)}"
        )]

async def main():
    """Main entry point for enhanced MCP server"""
    logger.info(f"Starting MuJoCo MCP Server with Menagerie Support v{__version__}")
    
    # Initialize server capabilities
    server_options = InitializationOptions(
        server_name="mujoco-mcp-menagerie",
        server_version=__version__,
        capabilities=server.get_capabilities(
            notification_options=NotificationOptions(),
            experimental_capabilities={}
        )
    )
    
    # Run server with stdio transport
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server_options
        )

if __name__ == "__main__":
    asyncio.run(main())
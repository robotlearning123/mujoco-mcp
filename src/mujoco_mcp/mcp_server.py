#!/usr/bin/env python3
"""
MuJoCo MCP Server for stdio transport
Production-ready MCP server that works with Claude Desktop and other MCP clients
MCP Protocol Version: 2024-11-05
"""

import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

from .version import __version__
from .viewer_client import MuJoCoViewerClient as ViewerClient

# MCP Protocol constants
MCP_PROTOCOL_VERSION = "2024-11-05"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mujoco-mcp")

# Create server instance
server = Server("mujoco-mcp")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@dataclass
class _ResourcePayload:
    content: str
    mime_type: str | None = None


viewer_client: ViewerClient | None = None


def _json_content(payload: Dict[str, Any]) -> List[types.TextContent]:
    """Serialize payload as indented JSON for MCP text responses."""
    return [
        types.TextContent(
            type="text",
            text=json.dumps(payload, indent=2, ensure_ascii=False)
        )
    ]


def _success(message: str, data: Dict[str, Any] | None = None) -> List[types.TextContent]:
    """Create a standard success payload."""
    payload: Dict[str, Any] = {"status": "ok", "message": message}
    if data is not None:
        payload["data"] = data
    return _json_content(payload)


def _error(
    code: str,
    message: str,
    remediation: str | None = None,
    details: Dict[str, Any] | None = None,
) -> List[types.TextContent]:
    """Create a standard error payload following MCP guidance."""
    error_body: Dict[str, Any] = {
        "status": "error",
        "error": {
            "code": code,
            "message": message,
        },
    }

    if remediation:
        error_body["error"]["remediation"] = remediation
    if details:
        error_body["error"]["details"] = details

    return _json_content(error_body)


def _redact_arguments(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Redact large or sensitive payloads before logging tool calls."""
    return {
        key: "<redacted>" if isinstance(value, str) and len(value) > 256 else value
        for key, value in arguments.items()
    }

@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """Return list of available MuJoCo MCP tools."""

    return [
        types.Tool(
            name="get_server_info",
            description="Get information about the MuJoCo MCP server",
            inputSchema={
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
                "examples": [{}],
            },
        ),
        types.Tool(
            name="create_scene",
            description="Create a physics simulation scene",
            inputSchema={
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "scene_type": {
                        "type": "string",
                        "description": "Type of scene to create",
                        "enum": ["pendulum", "double_pendulum", "cart_pole", "arm"],
                    }
                },
                "required": ["scene_type"],
                "additionalProperties": False,
                "examples": [
                    {"scene_type": "pendulum"},
                    {"scene_type": "double_pendulum"},
                ],
            },
        ),
        types.Tool(
            name="step_simulation",
            description="Step the physics simulation forward",
            inputSchema={
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "ID of the model to step",
                    },
                    "steps": {
                        "type": "integer",
                        "description": "Number of simulation steps",
                        "default": 1,
                        "minimum": 1,
                    },
                },
                "required": ["model_id"],
                "additionalProperties": False,
                "examples": [
                    {"model_id": "pendulum", "steps": 5},
                ],
            },
        ),
        types.Tool(
            name="get_state",
            description="Get current state of the simulation",
            inputSchema={
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "ID of the model to get state from",
                    }
                },
                "required": ["model_id"],
                "additionalProperties": False,
                "examples": [{"model_id": "pendulum"}],
            },
        ),
        types.Tool(
            name="reset_simulation",
            description="Reset simulation to initial state",
            inputSchema={
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "ID of the model to reset",
                    }
                },
                "required": ["model_id"],
                "additionalProperties": False,
                "examples": [{"model_id": "pendulum"}],
            },
        ),
        types.Tool(
            name="close_viewer",
            description="Close the MuJoCo viewer window",
            inputSchema={
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "ID of the model viewer to close",
                    }
                },
                "required": ["model_id"],
                "additionalProperties": False,
                "examples": [{"model_id": "pendulum"}],
            },
        ),
    ]


def _state_snapshot() -> Dict[str, Any]:
    """Return a lightweight snapshot of the first active simulation."""

    if not viewer_client or not viewer_client.connected:
        return {
            "status": "ok",
            "data": {
                "active": False,
                "message": "Viewer connection inactive",
            },
        }

    try:
        response = viewer_client.send_command({"type": "get_state"})
        if not response.get("success"):
            return {
                "status": "error",
                "error": {
                    "code": "viewer_error",
                    "message": response.get("error", "Unable to fetch state."),
                },
            }

        state = response.get("state")
        if state is None:
            state_keys = ["time", "qpos", "qvel", "ctrl", "xpos"]
            state = {key: response[key] for key in state_keys if key in response}

        return {
            "status": "ok",
            "data": {
                "active": True,
                "state": state,
            },
        }
    except Exception as exc:  # pragma: no cover - defensive path
        logger.exception("Failed to build state snapshot")
        return {
            "status": "error",
            "error": {
                "code": "internal_error",
                "message": "Failed to gather simulation state.",
                "details": {"exception": str(exc)},
            },
        }


@server.list_resources()
async def handle_list_resources() -> List[types.Resource]:
    """Advertise available resources following MCP guidelines."""

    return [
        types.Resource(
            name="simulation_state",
            title="Simulation State Snapshot",
            uri="simulation://state",
            description="Latest state snapshot of the first active simulation (if any).",
            mimeType="application/json",
        ),
        types.Resource(
            name="server_config",
            title="Server Configuration",
            uri="simulation://config",
            description="Server capabilities, version, and protocol metadata.",
            mimeType="application/json",
        ),
    ]


@server.read_resource()
async def handle_read_resource(uri: str):
    """Provide resource contents for the advertised URIs."""

    if uri == "simulation://state":
        payload = _state_snapshot()
    elif uri == "simulation://config":
        tools = await handle_list_tools()
        payload = {
            "status": "ok",
            "data": {
                "version": __version__,
                "protocol_version": MCP_PROTOCOL_VERSION,
                "tools": [tool.name for tool in tools],
            },
        }
    else:
        payload = {
            "status": "error",
            "error": {
                "code": "unknown_resource",
                "message": f"Resource '{uri}' is not available.",
            },
        }

    return [_ResourcePayload(content=json.dumps(payload), mime_type="application/json")]

@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle tool calls with MCP-compliant responses."""

    global viewer_client
    start = time.perf_counter()
    redacted_args = _redact_arguments(arguments)

    try:
        if name == "get_server_info":
            logger.info("get_server_info invoked")
            return _success(
                "Server information",
                {
                    "name": "MuJoCo MCP Server",
                    "version": __version__,
                    "description": "Control MuJoCo physics simulations through MCP",
                    "protocol_version": MCP_PROTOCOL_VERSION,
                    "capabilities": [
                        "create_scene",
                        "step_simulation",
                        "get_state",
                        "reset_simulation",
                        "close_viewer",
                    ],
                },
            )

        if name not in {"create_scene", "step_simulation", "get_state", "reset_simulation", "close_viewer"}:
            logger.warning("Unknown tool requested", extra={"tool": name})
            return _error(
                code="unknown_tool",
                message=f"Tool '{name}' is not available.",
                remediation="Call list_tools to discover supported tools.",
            )

        if not viewer_client:
            viewer_client = ViewerClient()

        if not viewer_client.connected and not viewer_client.connect():
            return _error(
                code="viewer_unavailable",
                message="Failed to connect to the MuJoCo viewer server.",
                remediation="Start 'mujoco-mcp-viewer' and retry the tool call.",
            )

        if name == "create_scene":
            scene_type = arguments.get("scene_type", "pendulum")
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
                """,
                "arm": """
                <mujoco>
                    <worldbody>
                        <body name="base">
                            <joint name="hinge" type="hinge" axis="0 0 1"/>
                            <geom name="link" type="capsule" size="0.02 0.4" rgba="0.8 0.2 0.2 1"/>
                        </body>
                    </worldbody>
                </mujoco>
                """,
            }

            if scene_type not in scene_models:
                return _error(
                    code="invalid_scene",
                    message=f"Scene type '{scene_type}' is not supported.",
                    remediation=f"Use one of: {', '.join(scene_models)}.",
                )

            response = viewer_client.send_command(
                {
                    "type": "load_model",
                    "model_id": scene_type,
                    "model_xml": scene_models[scene_type],
                }
            )

            if not response.get("success"):
                return _error(
                    code="viewer_error",
                    message=response.get("error", "Unknown viewer error"),
                )

            return _success(
                "Scene created",
                {
                    "model_id": scene_type,
                    "viewer_response": response,
                },
            )

        model_id = arguments.get("model_id")
        if not model_id:
            return _error(
                code="missing_argument",
                message="The 'model_id' argument is required.",
                remediation="Pass the target model identifier in the tool arguments.",
            )

        if name == "step_simulation":
            steps = max(1, int(arguments.get("steps", 1)))
            # Simulation runs continuously; acknowledge the request.
            return _success(
                "Simulation step acknowledged",
                {"model_id": model_id, "steps": steps},
            )

        if name == "get_state":
            response = viewer_client.send_command({"type": "get_state", "model_id": model_id})
            if not response.get("success"):
                return _error(
                    code="viewer_error",
                    message=response.get("error", "Failed to retrieve state."),
                )

            state = response.get("state")
            if state is None:
                state_keys = ["time", "qpos", "qvel", "qacc", "ctrl", "xpos"]
                state = {key: response[key] for key in state_keys if key in response}

            return _success("Simulation state", {"model_id": model_id, "state": state})

        if name == "reset_simulation":
            response = viewer_client.send_command({"type": "reset", "model_id": model_id})
            if not response.get("success"):
                return _error(
                    code="viewer_error",
                    message=response.get("error", "Reset failed."),
                )
            return _success("Simulation reset", {"model_id": model_id})

        if name == "close_viewer":
            response = viewer_client.send_command({"type": "close_model", "model_id": model_id})
            if viewer_client:
                viewer_client.disconnect()
                viewer_client = None

            if not response.get("success"):
                return _error(
                    code="viewer_error",
                    message=response.get("error", "Failed to close viewer."),
                )

            return _success("Viewer closed", {"model_id": model_id})

        return _error(
            code="unknown_tool",
            message=f"Tool '{name}' is not available.",
        )

    except Exception as exc:
        logger.exception("Error in tool handler", extra={"tool": name, "arguments": redacted_args})
        return _error(
            code="internal_error",
            message="Unexpected server error.",
            details={"exception": str(exc)},
        )
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "Tool handled",
            extra={"tool": name, "duration_ms": round(duration_ms, 2), "arguments": redacted_args},
        )

async def main():
    """Main entry point for MCP server"""
    logger.info(f"Starting MuJoCo MCP Server v{__version__}")
    logger.info(f"MCP Protocol Version: {MCP_PROTOCOL_VERSION}")

    # Initialize server capabilities with enhanced configuration
    capabilities = server.get_capabilities(
        notification_options=NotificationOptions(),
        experimental_capabilities={}
    )

    server_options = InitializationOptions(
        server_name="mujoco-mcp",
        server_version=__version__,
        capabilities=capabilities,
        protocol_versions=[MCP_PROTOCOL_VERSION],
        instructions="MuJoCo physics simulation server with viewer support. "
                    f"Implements MCP Protocol {MCP_PROTOCOL_VERSION}. "
                    "Provides tools for creating scenes, controlling simulation, and managing state."
    )

    logger.info(f"Server capabilities: {capabilities}")
    logger.info("MCP server initialization complete")

    # Run server with stdio transport
    try:
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            logger.info("Starting MCP server stdio transport")
            await server.run(
                read_stream,
                write_stream,
                server_options
            )
    except Exception as exc:
        logger.exception("MCP server error", extra={"exception": str(exc)})
        raise

if __name__ == "__main__":
    asyncio.run(main())

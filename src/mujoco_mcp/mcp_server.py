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
from pathlib import Path
from typing import Dict, Any, List

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

from .version import __version__
from .viewer_client import MuJoCoViewerClient as ViewerClient
from .simulation import MuJoCoSimulation

# MCP Protocol constants
MCP_PROTOCOL_VERSION = "2024-11-05"

# MCP Best Practices: Character limit for responses (25K tokens ≈ 100K chars)
CHARACTER_LIMIT = 100000

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

# Managed headless simulations (fallback when viewer is unavailable)
headless_simulations: Dict[str, MuJoCoSimulation] = {}

# MCP Best Practice: Externalize domain data (scene models) to separate files
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
BUILTIN_SCENE_TYPES = ["pendulum", "double_pendulum", "cart_pole", "arm"]
ALLOWED_SCENE_MODES = {"auto", "viewer", "headless"}


def _load_scene_model(scene_type: str) -> str:
    """Load scene model XML from external file.

    MCP Best Practice: Separate domain data from code for easier
    maintenance and testing.
    """
    model_file = MODELS_DIR / f"{scene_type}.xml"
    if not model_file.exists():
        raise FileNotFoundError(f"Scene model not found: {model_file}")
    return model_file.read_text()


def _json_content(payload: Dict[str, Any]) -> List[types.TextContent]:
    """Serialize payload as indented JSON for MCP text responses with character limit."""
    text = json.dumps(payload, indent=2, ensure_ascii=False)

    # MCP Best Practice: Implement character limits to prevent token overflow
    if len(text) > CHARACTER_LIMIT:
        truncated_payload = payload.copy()
        if "data" in truncated_payload:
            truncated_payload["data"] = "[TRUNCATED - Response exceeded 100K character limit]"
        truncated_payload["warning"] = f"Response truncated from {len(text)} to {CHARACTER_LIMIT} characters"
        text = json.dumps(truncated_payload, indent=2, ensure_ascii=False)

    return [
        types.TextContent(
            type="text",
            text=text
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


def _format_state(state: Dict[str, Any], format_type: str = "detailed") -> Dict[str, Any]:
    """Format simulation state based on requested detail level.

    MCP Best Practice: Provide concise vs detailed response format options
    to optimize token usage and LLM context.
    """
    if format_type == "concise":
        # Return minimal state for quick checks
        return {
            "time": state.get("time", 0.0),
            "nq": len(state.get("qpos", [])),
            "nv": len(state.get("qvel", [])),
            "summary": f"{len(state.get('qpos', []))} DOF at t={state.get('time', 0.0):.3f}s"
        }
    else:
        # Return full detailed state (default)
        return state

@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """Return list of available MuJoCo MCP tools."""

    return [
        types.Tool(
            name="get_server_info",
            description="Get information about the MuJoCo MCP server including version, capabilities, and protocol details",
            inputSchema={
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
                "examples": [{}],
            },
            # MCP Best Practice: Tool annotations for LLM optimization
            readOnlyHint=True,  # Read-only operation
            destructiveHint=False,  # Non-destructive
            idempotentHint=True,  # Same result on repeated calls
            openWorldHint=False,  # Purely informational, no external system interaction
        ),
        types.Tool(
            name="create_scene",
            description="Create a physics simulation scene. Supports pendulum, double_pendulum, cart_pole, and arm scenes. Can run in viewer mode (with GUI) or headless mode.",
            inputSchema={
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "scene_type": {
                        "type": "string",
                        "description": "Type of scene to create: pendulum (simple), double_pendulum (chaotic), cart_pole (balancing), arm (robot)",
                        "enum": BUILTIN_SCENE_TYPES,
                    },
                    "model_id": {
                        "type": "string",
                        "description": "Optional custom identifier for the simulation (default: same as scene_type)",
                    },
                    "mode": {
                        "type": "string",
                        "description": "Execution mode: 'auto' (try viewer, fallback headless), 'viewer' (require GUI), 'headless' (no GUI)",
                        "enum": sorted(ALLOWED_SCENE_MODES),
                        "default": "auto",
                    },
                },
                "required": ["scene_type"],
                "additionalProperties": False,
                "examples": [
                    {"scene_type": "pendulum"},
                    {"scene_type": "cart_pole", "mode": "headless", "model_id": "cart"},
                    {"scene_type": "double_pendulum", "mode": "viewer"},
                    {"scene_type": "arm", "model_id": "robot_arm_1", "mode": "auto"},
                ],
            },
            readOnlyHint=False,  # Creates simulation state
            destructiveHint=False,  # Non-destructive (creates new, doesn't destroy existing)
            idempotentHint=False,  # Multiple calls create different simulations
            openWorldHint=True,  # Interacts with MuJoCo physics engine
        ),
        types.Tool(
            name="step_simulation",
            description="Step the physics simulation forward in time. Each step advances the simulation by one timestep (typically 0.002 seconds). Use this to animate the simulation.",
            inputSchema={
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "ID of the model to step (from create_scene)",
                    },
                    "steps": {
                        "type": "integer",
                        "description": "Number of simulation steps to execute (default: 1, typical: 100-500 for visible motion)",
                        "default": 1,
                        "minimum": 1,
                    },
                },
                "required": ["model_id"],
                "additionalProperties": False,
                "examples": [
                    {"model_id": "pendulum", "steps": 1},
                    {"model_id": "pendulum", "steps": 100},
                    {"model_id": "cart_pole", "steps": 500},
                ],
            },
            readOnlyHint=False,  # Modifies simulation state
            destructiveHint=False,  # Non-destructive (advances time, doesn't destroy)
            idempotentHint=False,  # Each call advances time differently
            openWorldHint=True,  # Interacts with MuJoCo physics engine
        ),
        types.Tool(
            name="get_state",
            description="Get current state of the simulation including joint positions, velocities, and time. Returns detailed physics state for analysis.",
            inputSchema={
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "ID of the model to get state from (from create_scene)",
                    },
                    "format": {
                        "type": "string",
                        "description": "Response format: 'concise' (time + summary) or 'detailed' (full state arrays)",
                        "enum": ["concise", "detailed"],
                        "default": "detailed",
                    }
                },
                "required": ["model_id"],
                "additionalProperties": False,
                "examples": [
                    {"model_id": "pendulum"},
                    {"model_id": "pendulum", "format": "concise"},
                    {"model_id": "cart_pole", "format": "detailed"},
                ],
            },
            readOnlyHint=True,  # Read-only operation
            destructiveHint=False,  # Non-destructive
            idempotentHint=True,  # Same state if simulation hasn't advanced
            openWorldHint=True,  # Reads from MuJoCo physics engine
        ),
        types.Tool(
            name="reset_simulation",
            description="Reset simulation to initial state. Resets joint positions, velocities, and time back to t=0. Useful for rerunning experiments.",
            inputSchema={
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "ID of the model to reset (from create_scene)",
                    }
                },
                "required": ["model_id"],
                "additionalProperties": False,
                "examples": [
                    {"model_id": "pendulum"},
                    {"model_id": "cart_pole"},
                    {"model_id": "robot_arm_1"},
                ],
            },
            readOnlyHint=False,  # Modifies simulation state
            destructiveHint=False,  # Non-destructive (resets, doesn't delete)
            idempotentHint=True,  # Multiple resets produce same initial state
            openWorldHint=True,  # Interacts with MuJoCo physics engine
        ),
        types.Tool(
            name="close_viewer",
            description="Close the MuJoCo viewer window and clean up simulation resources. Use this when done with a simulation to free memory.",
            inputSchema={
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "ID of the model viewer to close (from create_scene)",
                    }
                },
                "required": ["model_id"],
                "additionalProperties": False,
                "examples": [
                    {"model_id": "pendulum"},
                    {"model_id": "cart_pole"},
                    {"model_id": "robot_arm_1"},
                ],
            },
            readOnlyHint=False,  # Modifies system state
            destructiveHint=True,  # Destroys the simulation
            idempotentHint=False,  # Can't close twice (will error)
            openWorldHint=True,  # Closes MuJoCo viewer/simulation
        ),
    ]


def _state_snapshot() -> Dict[str, Any]:
    """Return a lightweight snapshot of the first active simulation."""

    if headless_simulations:
        model_id, simulation = next(iter(headless_simulations.items()))
        return {
            "status": "ok",
            "data": {
                "active": True,
                "mode": "headless",
                "model_id": model_id,
                "state": simulation.get_state_snapshot(),
            },
        }

    try:
        if not viewer_client or not viewer_client.connected:
            return {
                "status": "ok",
                "data": {
                    "active": False,
                    "mode": "viewer",
                    "message": "Viewer connection inactive",
                },
            }

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
                "mode": "viewer",
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
                        "headless_mode",
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

        if name == "create_scene":
            requested_mode = arguments.get("mode", "auto")
            if requested_mode not in ALLOWED_SCENE_MODES:
                return _error(
                    code="invalid_mode",
                    message=f"Unsupported execution mode '{requested_mode}'.",
                    remediation=f"Use one of: {', '.join(sorted(ALLOWED_SCENE_MODES))}.",
                )

            scene_type = arguments.get("scene_type")
            if scene_type not in BUILTIN_SCENE_TYPES:
                return _error(
                    code="invalid_scene",
                    message=f"Scene type '{scene_type}' is not supported.",
                    remediation=f"Use one of: {', '.join(sorted(BUILTIN_SCENE_TYPES))}.",
                )

            model_id = arguments.get("model_id") or scene_type

            if requested_mode != "viewer" and model_id in headless_simulations:
                return _error(
                    code="duplicate_model",
                    message=f"Simulation '{model_id}' already exists in headless mode.",
                    remediation="Close the existing simulation or provide a different model_id.",
                )

            # Load scene model XML from external file
            try:
                model_xml = _load_scene_model(scene_type)
            except FileNotFoundError as exc:
                return _error(
                    code="model_not_found",
                    message=f"Scene model file not found: {scene_type}",
                    details={"error": str(exc)},
                )

            if requested_mode in {"auto", "viewer"}:
                if viewer_client is None:
                    viewer_client = ViewerClient()
                if viewer_client.connected or viewer_client.connect():
                    viewer_response = viewer_client.send_command(
                        {
                            "type": "load_model",
                            "model_id": model_id,
                            "model_xml": model_xml,
                        }
                    )
                    if not viewer_response.get("success"):
                        return _error(
                            code="viewer_error",
                            message=viewer_response.get("error", "Unknown viewer error"),
                        )

                    headless_simulations.pop(model_id, None)
                    return _success(
                        "Scene created (viewer mode)",
                        {
                            "model_id": model_id,
                            "mode": "viewer",
                            "viewer_response": viewer_response,
                        },
                    )

                if requested_mode == "viewer":
                    return _error(
                        code="viewer_unavailable",
                        message="Failed to connect to the MuJoCo viewer server.",
                        remediation="Start 'mujoco-mcp-viewer' or request mode='headless'.",
                    )

            try:
                simulation = MuJoCoSimulation(model_xml=model_xml)
            except Exception as exc:
                logger.exception("Failed to create headless simulation")
                return _error(
                    code="headless_error",
                    message="Failed to create headless simulation.",
                    remediation="Ensure MuJoCo is installed and accessible.",
                    details={"exception": str(exc)},
                )

            headless_simulations[model_id] = simulation
            return _success(
                "Scene created (headless mode)",
                {
                    "model_id": model_id,
                    "mode": "headless",
                    "state": simulation.get_state_snapshot(),
                },
            )

        model_id = arguments.get("model_id")
        if not model_id:
            return _error(
                code="missing_argument",
                message="The 'model_id' argument is required.",
                remediation="Pass the target model identifier in the tool arguments.",
            )

        headless_simulation = headless_simulations.get(model_id)

        if name == "step_simulation":
            steps = max(1, int(arguments.get("steps", 1)))

            if headless_simulation:
                headless_simulation.step(steps)
                return _success(
                    "Simulation step completed",
                    {
                        "model_id": model_id,
                        "mode": "headless",
                        "steps": steps,
                        "time": headless_simulation.get_time(),
                    },
                )

            if viewer_client is None:
                viewer_client = ViewerClient()
            if not viewer_client.connected and not viewer_client.connect():
                return _error(
                    code="viewer_unavailable",
                    message="Failed to connect to the MuJoCo viewer server.",
                    remediation="Start 'mujoco-mcp-viewer' or recreate the scene in headless mode.",
                )

            return _success(
                "Simulation step acknowledged",
                {"model_id": model_id, "mode": "viewer", "steps": steps},
            )

        if name == "get_state":
            # MCP Best Practice: Support concise/detailed format parameter
            format_type = arguments.get("format", "detailed")

            if headless_simulation:
                full_state = headless_simulation.get_state_snapshot()
                formatted_state = _format_state(full_state, format_type)
                return _success(
                    "Simulation state",
                    {
                        "model_id": model_id,
                        "mode": "headless",
                        "format": format_type,
                        "state": formatted_state,
                    },
                )

            if viewer_client is None:
                viewer_client = ViewerClient()
            if not viewer_client.connected and not viewer_client.connect():
                return _error(
                    code="viewer_unavailable",
                    message="Failed to connect to the MuJoCo viewer server.",
                    remediation="Start 'mujoco-mcp-viewer' or create the scene in headless mode.",
                )

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

            formatted_state = _format_state(state, format_type)
            return _success(
                "Simulation state",
                {"model_id": model_id, "mode": "viewer", "format": format_type, "state": formatted_state},
            )

        if name == "reset_simulation":
            if headless_simulation:
                headless_simulation.reset()
                return _success(
                    "Simulation reset",
                    {"model_id": model_id, "mode": "headless"},
                )

            if viewer_client is None:
                viewer_client = ViewerClient()
            if not viewer_client.connected and not viewer_client.connect():
                return _error(
                    code="viewer_unavailable",
                    message="Failed to connect to the MuJoCo viewer server.",
                    remediation="Start 'mujoco-mcp-viewer' or recreate the scene in headless mode.",
                )

            response = viewer_client.send_command({"type": "reset", "model_id": model_id})
            if not response.get("success"):
                return _error(
                    code="viewer_error",
                    message=response.get("error", "Reset failed."),
                )

            return _success(
                "Simulation reset",
                {"model_id": model_id, "mode": "viewer"},
            )

        if name == "close_viewer":
            if headless_simulation:
                headless_simulation.close()
                del headless_simulations[model_id]
                return _success(
                    "Simulation closed",
                    {"model_id": model_id, "mode": "headless"},
                )

            if viewer_client is None:
                viewer_client = ViewerClient()
            if not viewer_client.connected and not viewer_client.connect():
                return _error(
                    code="viewer_unavailable",
                    message="Failed to connect to the MuJoCo viewer server.",
                    remediation="Start 'mujoco-mcp-viewer' before attempting to close the viewer.",
                )

            response = viewer_client.send_command({"type": "close_model", "model_id": model_id})
            if viewer_client:
                viewer_client.disconnect()
                viewer_client = None

            if not response.get("success"):
                return _error(
                    code="viewer_error",
                    message=response.get("error", "Failed to close viewer."),
                )

            return _success(
                "Viewer closed",
                {"model_id": model_id, "mode": "viewer"},
            )

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

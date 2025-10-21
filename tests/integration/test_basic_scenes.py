#!/usr/bin/env python3
"""
Integration tests validating the basic MCP tool workflow without relying on the GUI viewer.
"""

from __future__ import annotations

import json
import uuid
import sys
from pathlib import Path

import pytest

# Add project to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from mujoco_mcp import mcp_server as mcp_module
from mujoco_mcp.mcp_server import handle_list_tools, handle_call_tool


async def _call_tool(name: str, arguments: dict[str, object]) -> dict[str, object]:
    """Invoke a tool and return the parsed JSON payload."""
    result = await handle_call_tool(name, arguments)
    assert result, "Tool returned no content"
    payload = json.loads(result[0].text)
    assert payload["status"] == "ok", payload
    return payload


@pytest.mark.asyncio
async def test_headless_basic_scenes():
    """Headless mode should support create → step → get_state → reset → close operations."""
    scenes = ["pendulum", "double_pendulum", "cart_pole"]

    for scene_type in scenes:
        model_id = f"{scene_type}_{uuid.uuid4().hex[:8]}"

        create_payload = await _call_tool(
            "create_scene",
            {"scene_type": scene_type, "mode": "headless", "model_id": model_id},
        )
        assert create_payload["data"]["mode"] == "headless"
        initial_state = create_payload["data"]["state"]
        assert initial_state["statistics"]["nq"] > 0

        step_payload = await _call_tool(
            "step_simulation", {"model_id": model_id, "steps": 25}
        )
        assert step_payload["data"]["mode"] == "headless"
        assert step_payload["data"]["steps"] == 25
        assert step_payload["data"]["time"] > initial_state["time"]

        state_payload = await _call_tool("get_state", {"model_id": model_id})
        assert state_payload["data"]["mode"] == "headless"
        assert state_payload["data"]["state"]["time"] >= step_payload["data"]["time"]

        await _call_tool("reset_simulation", {"model_id": model_id})
        reset_state = await _call_tool("get_state", {"model_id": model_id})
        assert reset_state["data"]["state"]["time"] == 0.0

        close_payload = await _call_tool("close_viewer", {"model_id": model_id})
        assert close_payload["data"]["mode"] == "headless"


@pytest.mark.asyncio
async def test_auto_mode_fallback_without_viewer(monkeypatch: pytest.MonkeyPatch):
    """Auto mode should gracefully fall back to headless execution when the viewer is unavailable."""

    model_id = f"auto_mode_{uuid.uuid4().hex[:8]}"

    original_viewer_client = mcp_module.ViewerClient

    class AlwaysFailViewer(original_viewer_client):
        """Viewer client stub that always fails to connect."""

        def connect(self) -> bool:  # type: ignore[override]
            return False

    monkeypatch.setattr(mcp_module, "viewer_client", None)
    monkeypatch.setattr(mcp_module, "ViewerClient", AlwaysFailViewer)

    create_payload = await _call_tool(
        "create_scene",
        {"scene_type": "pendulum", "model_id": model_id},
    )
    assert create_payload["data"]["mode"] == "headless"

    tools = await handle_list_tools()
    tool_names = {tool.name for tool in tools}
    assert {"get_server_info", "create_scene", "step_simulation", "get_state", "reset_simulation", "close_viewer"} <= tool_names

    server_info = await _call_tool("get_server_info", {})
    assert "headless_mode" in server_info["data"]["capabilities"]

    await _call_tool("close_viewer", {"model_id": model_id})

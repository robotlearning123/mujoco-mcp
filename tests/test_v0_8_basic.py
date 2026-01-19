"""
v0.8 basic tests - simplified version, focused on core functionality
"""

import pytest


def test_package_import():
    """Test package import"""
    try:
        import mujoco_mcp
        from mujoco_mcp.version import __version__

        assert __version__.startswith("0.8."), f"Expected version 0.8.x, got {__version__}"
    except ImportError as e:
        pytest.fail(f"Package import failed: {e}")


def test_mcp_server_import():
    """Test MCP server import"""
    try:
        from mujoco_mcp.mcp_server import handle_list_tools, handle_call_tool

        assert callable(handle_list_tools)
        assert callable(handle_call_tool)
    except ImportError as e:
        pytest.fail(f"MCP server import failed: {e}")


@pytest.mark.asyncio
async def test_tools_listing():
    """Test tools listing functionality"""
    from mujoco_mcp.mcp_server import handle_list_tools

    tools = await handle_list_tools()
    assert len(tools) == 6

    tool_names = [tool.name for tool in tools]
    expected_tools = [
        "get_server_info",
        "create_scene",
        "step_simulation",
        "get_state",
        "reset_simulation",
        "close_viewer",
    ]

    for tool_name in expected_tools:
        assert tool_name in tool_names


@pytest.mark.asyncio
async def test_server_info_tool():
    """Test server info tool"""
    from mujoco_mcp.mcp_server import handle_call_tool

    result = await handle_call_tool("get_server_info", {})
    assert result is not None
    assert len(result) > 0
    assert "MuJoCo MCP Server" in result[0].text

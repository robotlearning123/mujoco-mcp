"""
v0.8 基础测试 - 简化版本，专注于核心功能
"""

import pytest


def test_package_import():
    """测试包导入"""
    try:
        import mujoco_mcp
        from mujoco_mcp.version import __version__
        assert __version__ == "0.8.0"
    except ImportError as e:
        pytest.fail(f"Package import failed: {e}")


def test_mcp_server_import():
    """测试MCP服务器导入"""
    try:
        from mujoco_mcp.mcp_server import handle_list_tools, handle_call_tool
        assert callable(handle_list_tools)
        assert callable(handle_call_tool)
    except ImportError as e:
        pytest.fail(f"MCP server import failed: {e}")


@pytest.mark.asyncio
async def test_tools_listing():
    """测试工具列表功能"""
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
        "close_viewer"
    ]
    
    for tool_name in expected_tools:
        assert tool_name in tool_names


@pytest.mark.asyncio
async def test_server_info_tool():
    """测试服务器信息工具"""
    from mujoco_mcp.mcp_server import handle_call_tool
    
    result = await handle_call_tool("get_server_info", {})
    assert result is not None
    assert len(result) > 0
    assert "MuJoCo MCP Server" in result[0].text
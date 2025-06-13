"""
Test MCP server functionality
"""
import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch

from mujoco_mcp.mcp_server import handle_list_tools, handle_call_tool

@pytest.mark.asyncio
@pytest.mark.mcp
async def test_list_tools():
    """Test that tools are listed correctly"""
    tools = await handle_list_tools()
    
    assert len(tools) > 0
    
    tool_names = [tool.name for tool in tools]
    expected_tools = [
        "get_server_info",
        "create_scene", 
        "step_simulation",
        "get_state",
        "reset_simulation",
        "close_viewer"
    ]
    
    for expected_tool in expected_tools:
        assert expected_tool in tool_names, f"Missing tool: {expected_tool}"

@pytest.mark.asyncio 
@pytest.mark.mcp
async def test_get_server_info():
    """Test get_server_info tool"""
    result = await handle_call_tool("get_server_info", {})
    
    assert len(result) == 1
    assert result[0].type == "text"
    
    # Parse the JSON response
    server_info = json.loads(result[0].text)
    
    assert "name" in server_info
    assert "version" in server_info
    assert "description" in server_info
    assert "status" in server_info
    assert "capabilities" in server_info
    
    assert server_info["name"] == "MuJoCo MCP Server"
    assert server_info["status"] == "ready"

@pytest.mark.asyncio
@pytest.mark.mcp
async def test_create_scene_no_viewer():
    """Test create_scene when viewer is not available"""
    # Reset global viewer client
    import mujoco_mcp.mcp_server as server_module
    server_module.viewer_client = None
    
    # Mock viewer client to simulate connection failure
    with patch('mujoco_mcp.mcp_server.ViewerClient') as mock_client_class:
        mock_client = Mock()
        mock_client.connected = False
        mock_client.connect = Mock(return_value=False)
        mock_client_class.return_value = mock_client
        
        result = await handle_call_tool("create_scene", {"scene_type": "pendulum"})
        
        assert len(result) == 1
        assert result[0].type == "text"
        assert "Failed to connect to MuJoCo viewer server" in result[0].text

@pytest.mark.asyncio
@pytest.mark.mcp
async def test_invalid_tool():
    """Test calling an invalid tool"""
    result = await handle_call_tool("invalid_tool", {})
    
    assert len(result) == 1
    assert result[0].type == "text"
    assert "Unknown tool" in result[0].text

@pytest.mark.asyncio
@pytest.mark.mcp
async def test_tool_error_handling():
    """Test error handling in tools"""
    # Reset global viewer client
    import mujoco_mcp.mcp_server as server_module
    server_module.viewer_client = None
    
    # Test with invalid arguments that should cause an error
    with patch('mujoco_mcp.mcp_server.ViewerClient') as mock_client_class:
        # Make the ViewerClient constructor raise an exception
        mock_client_class.side_effect = Exception("Test error")
        
        result = await handle_call_tool("create_scene", {"scene_type": "pendulum"})
        
        assert len(result) == 1
        assert result[0].type == "text"
        assert "Error:" in result[0].text
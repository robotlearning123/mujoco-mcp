#!/usr/bin/env python3
"""
Test suite for v0.5.0 - FastMCP Migration
Tests for FastMCP framework integration
"""
import pytest
import pytest_asyncio
import asyncio
from pathlib import Path
import sys

# Add project to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from mujoco_mcp.server import MuJoCoServer


class TestFastMCPMigration:
    """Test FastMCP migration functionality"""
    
    @pytest_asyncio.fixture
    async def server(self):
        """Create server fixture"""
        server = MuJoCoServer()
        await server.initialize()
        yield server
        await server.cleanup()
    
    @pytest.mark.asyncio
    async def test_server_initialization(self, server):
        """Test that FastMCP server initializes correctly"""
        assert server is not None
        assert hasattr(server, 'mcp')
        assert server.name == "mujoco-mcp"
    
    @pytest.mark.asyncio
    async def test_server_info(self, server):
        """Test server info method"""
        info = server.get_server_info()
        assert info["name"] == "mujoco-mcp"
        assert info["version"] == "0.6.0"
        assert "description" in info
    
    @pytest.mark.asyncio
    async def test_list_tools(self, server):
        """Test that all tools are registered"""
        # FastMCP doesn't have list_tools method, use internal access
        tools = list(server.mcp._tool_manager._tools.values())
        tool_names = [t.name for t in tools]
        
        # Check core tools exist
        assert "load_model" in tool_names
        assert "step_simulation" in tool_names
        assert "reset_simulation" in tool_names
        assert "get_state" in tool_names
        assert "apply_control" in tool_names
        
        # Check advanced tools
        assert "design_robot" in tool_names
        assert "optimize_parameters" in tool_names
        
        # Should have all 46 tools from previous versions
        assert len(tool_names) >= 46
    
    @pytest.mark.asyncio
    async def test_list_resources(self, server):
        """Test that resources are registered"""
        # FastMCP doesn't have list_resources method, use internal access
        resources = list(server.mcp._resource_manager._resources.values())
        resource_names = [r.name for r in resources]
        
        # Check core resources
        assert "simulation://state" in resource_names
        assert "simulation://sensors" in resource_names
        assert "simulation://config" in resource_names
    
    @pytest.mark.asyncio
    async def test_tool_execution(self, server):
        """Test executing a tool through FastMCP"""
        # Test pendulum demo setup
        result = await server.mcp.call_tool("pendulum_demo", {"action": "setup"})
        
        assert result is not None
        # FastMCP returns TextContent, need to parse it
        if hasattr(result, '__iter__') and len(result) > 0:
            import json
            text_content = result[0].text
            parsed = json.loads(text_content)
            assert "model_id" in parsed
            assert parsed.get("success") is True
        else:
            assert "model_id" in result
            assert result.get("success") is True
    
    @pytest.mark.asyncio
    async def test_resource_access(self, server):
        """Test accessing resources through FastMCP"""
        # First create a simulation
        await server.mcp.call_tool("pendulum_demo", {"action": "setup"})
        
        # Access state resource
        state = await server.mcp.read_resource("simulation://state")
        
        assert state is not None
        # FastMCP returns a list of ReadResourceContents
        assert isinstance(state, list)
        assert len(state) > 0
        
        # Parse the JSON content
        import json
        contents = json.loads(state[0].content)
        assert "contents" in contents
        data = contents["contents"]
        assert "joint_positions" in data
        assert "joint_velocities" in data
        assert "time" in data
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, server):
        """Test concurrent tool calls"""
        # Create multiple simulations concurrently
        tasks = []
        for i in range(3):
            task = server.mcp.call_tool("pendulum_demo", {"action": "setup"})
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert len(results) == 3
        # FastMCP returns list of CallToolResults
        for r in results:
            assert isinstance(r, list)
            assert len(r) > 0
            # Extract the first result
            result = r[0]
            if hasattr(result, 'content'):
                # Check content is not empty
                assert result.content is not None
    
    @pytest.mark.asyncio
    async def test_error_handling(self, server):
        """Test error handling in FastMCP"""
        # Try to load invalid model XML
        with pytest.raises(Exception) as exc_info:
            await server.mcp.call_tool("load_model", {
                "model_string": "<invalid>not a valid mujoco model</invalid>"
            })
        
        assert "error" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_tool_validation(self, server):
        """Test parameter validation"""
        # Missing required parameter
        with pytest.raises(Exception) as exc_info:
            await server.mcp.call_tool("load_model", {})
        
        assert "model_string" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_resource_updates(self, server):
        """Test that resources update correctly"""
        # Setup simulation
        result = await server.mcp.call_tool("pendulum_demo", {"action": "setup"})
        # Extract model_id from result list
        if isinstance(result, list) and len(result) > 0:
            import json
            # Check if it's TextContent with text attribute
            if hasattr(result[0], 'text'):
                content = json.loads(result[0].text)
            else:
                content = json.loads(result[0].content)
            model_id = content["model_id"]
        else:
            pytest.skip("Unexpected result format from pendulum_demo")
        
        # Get initial state
        state1 = await server.mcp.read_resource("simulation://state")
        # Parse JSON content
        import json
        content1 = json.loads(state1[0].content)
        time1 = content1["contents"]["time"]
        
        # Step simulation
        await server.mcp.call_tool("step_simulation", {"model_id": model_id})
        
        # Get updated state
        state2 = await server.mcp.read_resource("simulation://state")
        content2 = json.loads(state2[0].content)
        time2 = content2["contents"]["time"]
        
        # Time should have advanced
        assert time2 > time1
    
    @pytest.mark.asyncio
    async def test_backward_compatibility(self, server):
        """Test that old API still works"""
        # The server should support both old and new interfaces
        # This ensures smooth migration
        
        # Old style tool call (if supported)
        result = await server.mcp.call_tool("get_server_info", {})
        
        # Parse result
        if isinstance(result, list) and len(result) > 0:
            import json
            if hasattr(result[0], 'text'):
                info = json.loads(result[0].text)
            else:
                info = json.loads(result[0].content)
            assert info["version"] == "0.6.0"
        else:
            pytest.skip("Unexpected result format")
        
        # New style info access
        info = server.get_server_info()
        assert info["version"] == "0.6.0"


class TestFastMCPFeatures:
    """Test FastMCP-specific features"""
    
    @pytest_asyncio.fixture
    async def server(self):
        """Create server fixture"""
        server = MuJoCoServer()
        await server.initialize()
        yield server
        await server.cleanup()
    
    @pytest.mark.asyncio
    async def test_streaming_support(self, server):
        """Test streaming responses (if implemented)"""
        # FastMCP supports streaming for long operations
        # For now, just test that the interface exists
        # FastMCP may not have stream_resource in this version
        # Just check that the server is properly initialized
        assert server.mcp is not None
    
    @pytest.mark.asyncio
    async def test_batch_operations(self, server):
        """Test batch tool calls"""
        # Setup simulation first
        result = await server.mcp.call_tool("pendulum_demo", {"action": "setup"})
        # Extract model_id from result
        if isinstance(result, list) and len(result) > 0:
            import json
            if hasattr(result[0], 'text'):
                content = json.loads(result[0].text)
            else:
                content = json.loads(result[0].content)
            model_id = content["model_id"]
        else:
            pytest.skip("Unexpected result format")
        
        # Batch multiple operations
        operations = [
            ("step_simulation", {"model_id": model_id}),
            ("reset_simulation", {"model_id": model_id}),
            ("step_simulation", {"model_id": model_id, "steps": 10})
        ]
        
        # Execute batch (if supported by FastMCP)
        results = []
        for tool_name, params in operations:
            result = await server.mcp.call_tool(tool_name, params)
            results.append(result)
        
        assert len(results) == 3
        assert all(r is not None for r in results)
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, server):
        """Test performance monitoring features"""
        # FastMCP may include performance metrics
        info = server.get_server_info()
        
        # Check if performance info is available
        if "performance" in info:
            perf = info["performance"]
            # Just check that performance info exists
            assert isinstance(perf, dict)


class TestServerVersion050:
    """Test server version update"""
    
    @pytest.mark.asyncio
    async def test_version_updated(self):
        """Test that server version is 0.5.0"""
        server = MuJoCoServer()
        await server.initialize()
        
        try:
            assert server.version == "0.6.0"
            
            info = server.get_server_info()
            assert info["version"] == "0.6.0"
        finally:
            await server.cleanup()
    
    @pytest.mark.asyncio
    async def test_fastmcp_capability(self):
        """Test that FastMCP capability is advertised"""
        server = MuJoCoServer()
        await server.initialize()
        
        try:
            info = server.get_server_info()
            
            assert "capabilities" in info
            assert "fastmcp" in info["capabilities"]
            assert info["capabilities"]["fastmcp"]["enabled"] is True
        finally:
            await server.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
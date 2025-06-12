#!/usr/bin/env python3
"""
Test suite for v0.5.0 - FastMCP Migration (Simplified)
Focus on core functionality rather than FastMCP internals
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


class TestFastMCPBasics:
    """Test basic FastMCP integration"""
    
    @pytest_asyncio.fixture
    async def server(self):
        """Create server fixture"""
        server = MuJoCoServer()
        await server.initialize()
        yield server
        await server.cleanup()
    
    @pytest.mark.asyncio
    async def test_server_creation(self, server):
        """Test that server can be created"""
        assert server is not None
        assert server.name == "mujoco-mcp"
        assert server.version == "0.5.0"
        assert hasattr(server, 'mcp')
        assert server.mcp is not None
    
    @pytest.mark.asyncio
    async def test_server_info(self, server):
        """Test server info"""
        info = server.get_server_info()
        assert info["name"] == "mujoco-mcp"
        assert info["version"] == "0.5.0"
        assert info["capabilities"]["fastmcp"]["enabled"] is True
    
    @pytest.mark.asyncio
    async def test_tools_registered(self, server):
        """Test that tools are registered with FastMCP"""
        # Check that tool manager exists and has tools
        assert hasattr(server.mcp, '_tool_manager')
        assert len(server.mcp._tool_manager._tools) > 0
        
        # Check some core tools
        tool_names = list(server.mcp._tool_manager._tools.keys())
        assert "load_model" in tool_names
        assert "step_simulation" in tool_names
        assert "pendulum_demo" in tool_names
    
    @pytest.mark.asyncio
    async def test_resources_registered(self, server):
        """Test that resources are registered with FastMCP"""
        # Check that resource manager exists and has resources
        assert hasattr(server.mcp, '_resource_manager')
        assert len(server.mcp._resource_manager._resources) > 0
        
        # Check core resources
        resource_names = list(server.mcp._resource_manager._resources.keys())
        assert "simulation://state" in resource_names
        assert "simulation://sensors" in resource_names
        assert "simulation://config" in resource_names


class TestSimpleFunctionality:
    """Test basic functionality through simple server interface"""
    
    @pytest_asyncio.fixture
    async def server(self):
        """Create server fixture"""
        server = MuJoCoServer()
        await server.initialize()
        yield server
        await server.cleanup()
    
    @pytest.mark.asyncio
    async def test_pendulum_demo_through_impl(self, server):
        """Test pendulum demo through implementation"""
        # Use the implementation directly to avoid FastMCP complexities
        result = server._impl._handle_pendulum_demo(action="setup")
        
        assert result["success"] is True
        assert "model_id" in result
        assert "model_info" in result
        assert result["model_info"]["nq"] == 1  # Single joint pendulum
    
    @pytest.mark.asyncio
    async def test_load_model_through_impl(self, server):
        """Test loading a simple model"""
        simple_xml = """
        <mujoco>
            <worldbody>
                <body name="box">
                    <geom type="box" size="0.1 0.1 0.1"/>
                </body>
            </worldbody>
        </mujoco>
        """
        
        result = server._impl._handle_load_model(model_string=simple_xml, name="test_box")
        
        assert result["success"] is True
        assert "model_id" in result
        assert result["name"] == "test_box"
    
    @pytest.mark.asyncio
    async def test_simulation_control(self, server):
        """Test basic simulation control"""
        # Setup pendulum
        setup_result = server._impl._handle_pendulum_demo(action="setup")
        model_id = setup_result["model_id"]
        
        # Step simulation
        step_result = server._impl._handle_step_simulation(model_id=model_id, steps=10)
        assert step_result["success"] is True
        assert step_result["steps_completed"] == 10
        
        # Get state
        state_result = server._impl._handle_get_simulation_state(model_id=model_id, include_positions=True, include_velocities=True)
        assert "time" in state_result
        assert "qpos" in state_result  # get_simulation_state returns qpos, not joint_positions
        assert state_result["time"] > 0  # Time should have advanced
        
        # Reset simulation
        reset_result = server._impl._handle_reset_simulation(model_id=model_id)
        assert reset_result["success"] is True
        
        # Check time is back to 0
        state_after_reset = server._impl._handle_get_simulation_state(model_id=model_id, include_positions=True, include_velocities=True)
        assert state_after_reset["time"] == 0.0


class TestAdvancedFeatures:
    """Test advanced features availability"""
    
    @pytest_asyncio.fixture
    async def server(self):
        """Create server fixture"""
        server = MuJoCoServer()
        await server.initialize()
        yield server
        await server.cleanup()
    
    @pytest.mark.asyncio
    async def test_nl_command_available(self, server):
        """Test natural language command tool is available"""
        tool_names = list(server.mcp._tool_manager._tools.keys())
        assert "nl_command" in tool_names
    
    @pytest.mark.asyncio
    async def test_design_robot_available(self, server):
        """Test robot designer tool is available"""
        tool_names = list(server.mcp._tool_manager._tools.keys())
        assert "design_robot" in tool_names
    
    @pytest.mark.asyncio
    async def test_optimize_parameters_available(self, server):
        """Test parameter optimization tool is available"""
        tool_names = list(server.mcp._tool_manager._tools.keys())
        assert "optimize_parameters" in tool_names
    
    @pytest.mark.asyncio
    async def test_all_previous_tools_migrated(self, server):
        """Test that all 46 tools from previous versions are available"""
        tool_names = list(server.mcp._tool_manager._tools.keys())
        
        # We should have at least 46 tools as per previous versions
        assert len(tool_names) >= 46
        
        # Check a sample of important tools
        important_tools = [
            "load_model", "step_simulation", "reset_simulation",
            "get_state", "apply_control", "render_frame",
            "pendulum_demo", "nl_command", "design_robot",
            "optimize_parameters", "generate_robot", "generate_environment"
        ]
        
        for tool in important_tools:
            assert tool in tool_names, f"Tool {tool} not found in migrated tools"


class TestBackwardCompatibility:
    """Test backward compatibility"""
    
    @pytest.mark.asyncio
    async def test_server_can_be_imported_as_before(self):
        """Test that MuJoCoMCPServer alias works"""
        from mujoco_mcp.server import MuJoCoMCPServer
        
        server = MuJoCoMCPServer()
        assert server is not None
        assert server.version == "0.5.0"
    
    @pytest.mark.asyncio
    async def test_simple_server_still_works(self):
        """Test that simple server can still be used directly"""
        from mujoco_mcp.simple_server import MuJoCoMCPServer
        
        server = MuJoCoMCPServer()
        assert server is not None
        # Simple server has been updated to 0.5.0
        assert server.version == "0.5.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
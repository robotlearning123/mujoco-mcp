#!/usr/bin/env python3
"""
Test suite for v0.2.1 - Enhanced State Query
Tests for get_joint_positions, get_joint_velocities, and enhanced state queries
"""
import pytest
import json
from pathlib import Path
import sys

# Add project to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from mujoco_mcp.simple_server import MuJoCoMCPServer


class TestEnhancedStateQuery:
    """Test enhanced state query tools"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.server = MuJoCoMCPServer()
        
        # Two-link arm model for testing
        self.arm_xml = """<mujoco model="two_link_arm">
            <option timestep="0.01"/>
            <worldbody>
                <body name="link1" pos="0 0 0">
                    <joint name="shoulder" type="hinge" axis="0 0 1"/>
                    <geom name="link1_geom" type="capsule" size="0.05 0.3" pos="0.15 0 0" euler="0 90 0"/>
                    <body name="link2" pos="0.3 0 0">
                        <joint name="elbow" type="hinge" axis="0 0 1"/>
                        <geom name="link2_geom" type="capsule" size="0.04 0.25" pos="0.125 0 0" euler="0 90 0"/>
                    </body>
                </body>
            </worldbody>
        </mujoco>"""
        
        # Load model
        result = self.server.call_tool("load_model", {
            "model_string": self.arm_xml,
            "name": "test_arm"
        })
        self.model_id = result["model_id"]
    
    def test_get_joint_positions_tool_exists(self):
        """Test that get_joint_positions tool is registered"""
        tools = self.server.get_tools()
        tool_names = [t["name"] for t in tools]
        assert "get_joint_positions" in tool_names
        
        # Check tool metadata
        tool = next(t for t in tools if t["name"] == "get_joint_positions")
        assert "description" in tool
        assert "parameters" in tool
        assert "model_id" in tool["parameters"]
    
    def test_get_joint_positions(self):
        """Test getting joint positions"""
        result = self.server.call_tool("get_joint_positions", {
            "model_id": self.model_id
        })
        
        assert "positions" in result
        assert isinstance(result["positions"], list)
        assert len(result["positions"]) == 2  # Two-link arm has 2 joints
        assert all(p == 0.0 for p in result["positions"])  # Initial positions
    
    def test_get_joint_positions_with_names(self):
        """Test getting joint positions with names"""
        result = self.server.call_tool("get_joint_positions", {
            "model_id": self.model_id,
            "include_names": True
        })
        
        assert "positions" in result
        assert "names" in result
        assert len(result["names"]) == 2
        assert result["names"] == ["shoulder", "elbow"]
    
    def test_get_joint_velocities_tool_exists(self):
        """Test that get_joint_velocities tool is registered"""
        tools = self.server.get_tools()
        tool_names = [t["name"] for t in tools]
        assert "get_joint_velocities" in tool_names
    
    def test_get_joint_velocities(self):
        """Test getting joint velocities"""
        result = self.server.call_tool("get_joint_velocities", {
            "model_id": self.model_id
        })
        
        assert "velocities" in result
        assert isinstance(result["velocities"], list)
        assert len(result["velocities"]) == 2
        assert all(v == 0.0 for v in result["velocities"])  # Initial velocities
    
    def test_get_joint_velocities_after_motion(self):
        """Test getting velocities after setting them"""
        # Set some velocities
        self.server.call_tool("set_joint_velocities", {
            "model_id": self.model_id,
            "velocities": [1.0, -0.5]
        })
        
        # Get velocities
        result = self.server.call_tool("get_joint_velocities", {
            "model_id": self.model_id
        })
        
        assert abs(result["velocities"][0] - 1.0) < 1e-6
        assert abs(result["velocities"][1] - (-0.5)) < 1e-6
    
    def test_set_joint_velocities_tool_exists(self):
        """Test that set_joint_velocities tool exists"""
        tools = self.server.get_tools()
        tool_names = [t["name"] for t in tools]
        assert "set_joint_velocities" in tool_names
    
    def test_set_joint_velocities(self):
        """Test setting joint velocities"""
        result = self.server.call_tool("set_joint_velocities", {
            "model_id": self.model_id,
            "velocities": [0.5, 0.5]
        })
        
        assert result["success"] is True
        
        # Verify velocities were set
        state = self.server.call_tool("get_joint_velocities", {
            "model_id": self.model_id
        })
        assert all(abs(v - 0.5) < 1e-6 for v in state["velocities"])
    
    def test_get_body_states_tool_exists(self):
        """Test that get_body_states tool exists"""
        tools = self.server.get_tools()
        tool_names = [t["name"] for t in tools]
        assert "get_body_states" in tool_names
    
    def test_get_body_states(self):
        """Test getting rigid body states"""
        result = self.server.call_tool("get_body_states", {
            "model_id": self.model_id
        })
        
        assert "bodies" in result
        assert isinstance(result["bodies"], dict)
        assert "link1" in result["bodies"]
        assert "link2" in result["bodies"]
        
        # Check body state structure
        for body_name, state in result["bodies"].items():
            assert "position" in state
            assert "orientation" in state
            assert len(state["position"]) == 3
            assert len(state["orientation"]) == 4  # quaternion
    
    def test_get_sensor_data_tool(self):
        """Test sensor data retrieval (even if no sensors)"""
        # This model has no sensors, but tool should still work
        result = self.server.call_tool("get_sensor_data", {
            "model_id": self.model_id
        })
        
        assert "sensors" in result
        assert isinstance(result["sensors"], dict)
        assert len(result["sensors"]) == 0  # No sensors in this model
    
    def test_state_consistency(self):
        """Test that different state query methods are consistent"""
        # Set some positions and velocities
        self.server.call_tool("set_joint_positions", {
            "model_id": self.model_id,
            "positions": [0.5, -0.3]
        })
        self.server.call_tool("set_joint_velocities", {
            "model_id": self.model_id,
            "velocities": [0.1, -0.2]
        })
        
        # Get state via different methods
        full_state = self.server.call_tool("get_simulation_state", {
            "model_id": self.model_id,
            "include_positions": True,
            "include_velocities": True
        })
        
        pos_state = self.server.call_tool("get_joint_positions", {
            "model_id": self.model_id
        })
        
        vel_state = self.server.call_tool("get_joint_velocities", {
            "model_id": self.model_id
        })
        
        # Verify consistency
        assert full_state["qpos"] == pos_state["positions"]
        assert full_state["qvel"] == vel_state["velocities"]


class TestServerEnhancements:
    """Test server enhancements for v0.2.1"""
    
    def test_server_version(self):
        """Test that server version is updated"""
        server = MuJoCoMCPServer()
        assert server.version == "0.2.1"
        
        info = server.get_server_info()
        assert info["version"] == "0.2.1"
    
    def test_new_tools_available(self):
        """Test that all new tools are available"""
        server = MuJoCoMCPServer()
        tools = server.get_tools()
        tool_names = [t["name"] for t in tools]
        
        new_tools = [
            "get_joint_positions", 
            "get_joint_velocities",
            "set_joint_velocities",
            "get_body_states",
            "get_sensor_data"
        ]
        
        for tool in new_tools:
            assert tool in tool_names, f"Missing tool: {tool}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
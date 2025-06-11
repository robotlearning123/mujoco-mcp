#!/usr/bin/env python3
"""
Test suite for v0.2.0 - Simulation Control
Tests for step_simulation, reset_simulation, and get_simulation_state
"""
import pytest
import json
from pathlib import Path
import sys

# Add project to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from mujoco_mcp.simple_server import MuJoCoMCPServer


class TestSimulationControl:
    """Test simulation control tools"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.server = MuJoCoMCPServer()
        
        # Simple pendulum model for testing
        self.pendulum_xml = """<mujoco model="pendulum">
            <option timestep="0.01"/>
            <worldbody>
                <body name="pendulum" pos="0 0 1">
                    <joint name="hinge" type="hinge" axis="0 1 0"/>
                    <geom name="mass" type="sphere" size="0.1" pos="0 0 -0.5" mass="1"/>
                </body>
            </worldbody>
        </mujoco>"""
        
        # Load a test model
        result = self.server.call_tool("load_model", {
            "model_string": self.pendulum_xml,
            "name": "test_pendulum"
        })
        self.model_id = result["model_id"]
    
    def test_step_simulation_tool_exists(self):
        """Test that step_simulation tool is registered"""
        tools = self.server.get_tools()
        tool_names = [t["name"] for t in tools]
        assert "step_simulation" in tool_names
        
        # Check tool metadata
        step_tool = next(t for t in tools if t["name"] == "step_simulation")
        assert "description" in step_tool
        assert "parameters" in step_tool
        assert "model_id" in step_tool["parameters"]
    
    def test_step_simulation_single_step(self):
        """Test stepping simulation by one step"""
        # Get initial state
        initial_state = self.server.call_tool("get_simulation_state", {
            "model_id": self.model_id
        })
        initial_time = initial_state["time"]
        
        # Step simulation
        result = self.server.call_tool("step_simulation", {
            "model_id": self.model_id
        })
        
        assert result["success"] is True
        assert result["steps_completed"] == 1
        assert "time" in result
        assert result["time"] > initial_time
    
    def test_step_simulation_multiple_steps(self):
        """Test stepping simulation by multiple steps"""
        # Step simulation by 10 steps
        result = self.server.call_tool("step_simulation", {
            "model_id": self.model_id,
            "steps": 10
        })
        
        assert result["success"] is True
        assert result["steps_completed"] == 10
        assert "time" in result
        
        # Check that time advanced correctly (10 steps * 0.01 timestep)
        expected_time = 10 * 0.01
        assert abs(result["time"] - expected_time) < 1e-6
    
    def test_reset_simulation_tool_exists(self):
        """Test that reset_simulation tool is registered"""
        tools = self.server.get_tools()
        tool_names = [t["name"] for t in tools]
        assert "reset_simulation" in tool_names
    
    def test_reset_simulation(self):
        """Test resetting simulation to initial state"""
        # Step simulation first
        self.server.call_tool("step_simulation", {
            "model_id": self.model_id,
            "steps": 50
        })
        
        # Get state after stepping
        state_after_step = self.server.call_tool("get_simulation_state", {
            "model_id": self.model_id
        })
        assert state_after_step["time"] > 0
        
        # Reset simulation
        result = self.server.call_tool("reset_simulation", {
            "model_id": self.model_id
        })
        
        assert result["success"] is True
        assert "message" in result
        
        # Get state after reset
        state_after_reset = self.server.call_tool("get_simulation_state", {
            "model_id": self.model_id
        })
        assert state_after_reset["time"] == 0.0
    
    def test_get_simulation_state_tool_exists(self):
        """Test that get_simulation_state tool is registered"""
        tools = self.server.get_tools()
        tool_names = [t["name"] for t in tools]
        assert "get_simulation_state" in tool_names
    
    def test_get_simulation_state(self):
        """Test getting simulation state"""
        result = self.server.call_tool("get_simulation_state", {
            "model_id": self.model_id
        })
        
        # Check basic state information
        assert "time" in result
        assert "nq" in result  # number of positions
        assert "nv" in result  # number of velocities
        assert result["time"] == 0.0  # Initial time
        assert result["nq"] == 1  # Pendulum has 1 joint
        assert result["nv"] == 1
    
    def test_get_simulation_state_with_positions(self):
        """Test getting simulation state includes positions and velocities"""
        result = self.server.call_tool("get_simulation_state", {
            "model_id": self.model_id,
            "include_positions": True,
            "include_velocities": True
        })
        
        assert "qpos" in result  # joint positions
        assert "qvel" in result  # joint velocities
        assert len(result["qpos"]) == 1  # Pendulum has 1 joint
        assert len(result["qvel"]) == 1
        assert result["qpos"][0] == 0.0  # Initial position
        assert result["qvel"][0] == 0.0  # Initial velocity
    
    def test_simulation_persistence(self):
        """Test that simulation state persists between calls"""
        # Set pendulum to 45 degrees
        self.server.call_tool("set_joint_positions", {
            "model_id": self.model_id,
            "positions": [0.785]  # 45 degrees in radians
        })
        
        # Step simulation
        self.server.call_tool("step_simulation", {
            "model_id": self.model_id,
            "steps": 100
        })
        
        # Get state
        state1 = self.server.call_tool("get_simulation_state", {
            "model_id": self.model_id,
            "include_positions": True
        })
        
        # Step again
        self.server.call_tool("step_simulation", {
            "model_id": self.model_id,
            "steps": 100
        })
        
        # Get state again
        state2 = self.server.call_tool("get_simulation_state", {
            "model_id": self.model_id,
            "include_positions": True
        })
        
        # Position should have changed (pendulum swinging)
        assert state2["qpos"][0] != state1["qpos"][0]
        assert state2["time"] > state1["time"]
    
    def test_invalid_model_id(self):
        """Test error handling for invalid model ID"""
        with pytest.raises(ValueError) as exc_info:
            self.server.call_tool("step_simulation", {
                "model_id": "invalid_id"
            })
        assert "model not found" in str(exc_info.value).lower()
    
    def test_missing_model_id(self):
        """Test error handling for missing model ID"""
        with pytest.raises(ValueError) as exc_info:
            self.server.call_tool("step_simulation", {})
        assert "model_id" in str(exc_info.value).lower()
    
    def test_negative_steps(self):
        """Test error handling for negative steps"""
        with pytest.raises(ValueError) as exc_info:
            self.server.call_tool("step_simulation", {
                "model_id": self.model_id,
                "steps": -1
            })
        assert "positive" in str(exc_info.value).lower()
    
    def test_set_joint_positions_tool_exists(self):
        """Test that set_joint_positions tool exists"""
        tools = self.server.get_tools()
        tool_names = [t["name"] for t in tools]
        assert "set_joint_positions" in tool_names
    
    def test_set_joint_positions(self):
        """Test setting joint positions"""
        # Set position to 90 degrees
        result = self.server.call_tool("set_joint_positions", {
            "model_id": self.model_id,
            "positions": [1.57]  # 90 degrees in radians
        })
        
        assert result["success"] is True
        
        # Verify position was set
        state = self.server.call_tool("get_simulation_state", {
            "model_id": self.model_id,
            "include_positions": True
        })
        assert abs(state["qpos"][0] - 1.57) < 1e-6


class TestServerEnhancements:
    """Test server enhancements for v0.2.0"""
    
    def test_server_version(self):
        """Test that server version is updated"""
        server = MuJoCoMCPServer()
        assert server.version == "0.2.0"
        
        info = server.get_server_info()
        assert info["version"] == "0.2.0"
    
    def test_tool_count(self):
        """Test that we have the expected number of tools"""
        server = MuJoCoMCPServer()
        tools = server.get_tools()
        
        # Should have at least all previous tools plus new ones
        tool_names = [t["name"] for t in tools]
        required_tools = [
            "get_server_info", "get_tools", 
            "load_model", "get_loaded_models",
            "step_simulation", "reset_simulation", 
            "get_simulation_state", "set_joint_positions"
        ]
        for tool in required_tools:
            assert tool in tool_names, f"Missing tool: {tool}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
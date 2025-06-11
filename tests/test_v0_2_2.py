#!/usr/bin/env python3
"""
Test suite for v0.2.2 - Basic Control
Tests for apply_control and actuator information
"""
import pytest
import json
import math
from pathlib import Path
import sys

# Add project to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from mujoco_mcp.simple_server import MuJoCoMCPServer


class TestBasicControl:
    """Test basic control functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.server = MuJoCoMCPServer()
        
        # Cart-pole model with actuator
        self.cartpole_xml = """<mujoco model="cartpole">
            <option timestep="0.01"/>
            <worldbody>
                <body name="cart" pos="0 0 0">
                    <joint name="slider" type="slide" axis="1 0 0" limited="true" range="-2 2"/>
                    <geom name="cart" type="box" size="0.2 0.2 0.1" rgba="0.7 0.7 0 1"/>
                    <body name="pole" pos="0 0 0">
                        <joint name="hinge" type="hinge" axis="0 1 0"/>
                        <geom name="pole" type="cylinder" size="0.05 0.5" rgba="0 0.7 0.7 1" pos="0 0 0.5"/>
                    </body>
                </body>
            </worldbody>
            <actuator>
                <motor name="cart_motor" joint="slider" gear="10" ctrllimited="true" ctrlrange="-1 1"/>
            </actuator>
        </mujoco>"""
        
        # Load model
        result = self.server.call_tool("load_model", {
            "model_string": self.cartpole_xml,
            "name": "test_cartpole"
        })
        self.model_id = result["model_id"]
    
    def test_apply_control_tool_exists(self):
        """Test that apply_control tool is registered"""
        tools = self.server.get_tools()
        tool_names = [t["name"] for t in tools]
        assert "apply_control" in tool_names
        
        # Check tool metadata
        tool = next(t for t in tools if t["name"] == "apply_control")
        assert "description" in tool
        assert "parameters" in tool
        assert "model_id" in tool["parameters"]
        assert "control" in tool["parameters"]
    
    def test_apply_control_single_actuator(self):
        """Test applying control to single actuator"""
        # Apply control
        result = self.server.call_tool("apply_control", {
            "model_id": self.model_id,
            "control": [0.5]  # 50% forward
        })
        
        assert result["success"] is True
        assert "message" in result
        
        # Step simulation to see effect
        self.server.call_tool("step_simulation", {
            "model_id": self.model_id,
            "steps": 10
        })
        
        # Check that cart moved
        state = self.server.call_tool("get_simulation_state", {
            "model_id": self.model_id,
            "include_positions": True
        })
        assert state["qpos"][0] > 0  # Cart moved forward
    
    def test_apply_control_validation(self):
        """Test control input validation"""
        # Test wrong number of controls
        with pytest.raises(ValueError) as exc_info:
            self.server.call_tool("apply_control", {
                "model_id": self.model_id,
                "control": [0.5, 0.5]  # Too many
            })
        assert "expected 1" in str(exc_info.value).lower()
        
        # Test empty control
        with pytest.raises(ValueError) as exc_info:
            self.server.call_tool("apply_control", {
                "model_id": self.model_id,
                "control": []
            })
        assert "control" in str(exc_info.value).lower()
    
    def test_get_actuator_info_tool_exists(self):
        """Test that get_actuator_info tool exists"""
        tools = self.server.get_tools()
        tool_names = [t["name"] for t in tools]
        assert "get_actuator_info" in tool_names
    
    def test_get_actuator_info(self):
        """Test getting actuator information"""
        result = self.server.call_tool("get_actuator_info", {
            "model_id": self.model_id
        })
        
        assert "actuators" in result
        assert len(result["actuators"]) == 1
        
        actuator = result["actuators"][0]
        assert actuator["name"] == "cart_motor"
        assert actuator["type"] == "motor"
        assert "gear" in actuator
        assert actuator["gear"][0] == 10.0  # First gear component
        assert "ctrl_limited" in actuator
        assert actuator["ctrl_limited"] is True
        assert "ctrl_range" in actuator
        assert actuator["ctrl_range"] == [-1.0, 1.0]
    
    def test_control_limits_enforcement(self):
        """Test that control limits are enforced"""
        # Try to apply control outside limits
        result = self.server.call_tool("apply_control", {
            "model_id": self.model_id,
            "control": [2.0]  # Above limit of 1.0
        })
        
        # Should succeed but clamp the value
        assert result["success"] is True
        assert any("clamped" in w for w in result.get("warnings", []))
    
    def test_get_control_state(self):
        """Test getting current control state"""
        # Set a control value
        self.server.call_tool("apply_control", {
            "model_id": self.model_id,
            "control": [0.75]
        })
        
        # Get control state
        result = self.server.call_tool("get_control_state", {
            "model_id": self.model_id
        })
        
        assert "control" in result
        assert len(result["control"]) == 1
        assert result["control"][0] == 0.75
    
    def test_multi_actuator_model(self):
        """Test model with multiple actuators"""
        # Two-link arm with two motors
        arm_xml = """<mujoco model="two_link_arm">
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
            <actuator>
                <motor name="shoulder_motor" joint="shoulder" gear="5"/>
                <motor name="elbow_motor" joint="elbow" gear="3"/>
            </actuator>
        </mujoco>"""
        
        result = self.server.call_tool("load_model", {
            "model_string": arm_xml,
            "name": "test_arm"
        })
        arm_id = result["model_id"]
        
        # Apply control to both motors
        result = self.server.call_tool("apply_control", {
            "model_id": arm_id,
            "control": [0.5, -0.3]
        })
        assert result["success"] is True
        
        # Get actuator info
        info = self.server.call_tool("get_actuator_info", {
            "model_id": arm_id
        })
        assert len(info["actuators"]) == 2
        assert info["actuators"][0]["name"] == "shoulder_motor"
        assert info["actuators"][1]["name"] == "elbow_motor"


class TestServerEnhancements:
    """Test server enhancements for v0.2.2"""
    
    def test_server_version(self):
        """Test that server version is updated"""
        server = MuJoCoMCPServer()
        assert server.version == "0.2.2"
        
        info = server.get_server_info()
        assert info["version"] == "0.2.2"
    
    def test_new_tools_available(self):
        """Test that all new tools are available"""
        server = MuJoCoMCPServer()
        tools = server.get_tools()
        tool_names = [t["name"] for t in tools]
        
        new_tools = [
            "apply_control",
            "get_actuator_info", 
            "get_control_state"
        ]
        
        for tool in new_tools:
            assert tool in tool_names, f"Missing tool: {tool}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
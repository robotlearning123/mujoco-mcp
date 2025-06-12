#!/usr/bin/env python3
"""
Test suite for v0.3.1 - First Complete Demo (Pendulum Control)
Tests for integrated demo combining simulation, control, and visualization
"""
import pytest
import json
import base64
import math
from pathlib import Path
import sys

# Add project to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from mujoco_mcp.simple_server import MuJoCoMCPServer


class TestPendulumDemo:
    """Test pendulum control demo functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.server = MuJoCoMCPServer()
        
        # Pendulum model with actuator
        self.pendulum_xml = """<mujoco model="pendulum">
            <option timestep="0.001" gravity="0 0 -9.81"/>
            <worldbody>
                <body name="pendulum" pos="0 0 1">
                    <joint name="hinge" type="hinge" axis="0 1 0" range="-180 180" damping="0.1"/>
                    <geom name="rod" type="capsule" fromto="0 0 0 0 0 -0.5" size="0.02" mass="0.1"/>
                    <body name="bob" pos="0 0 -0.5">
                        <geom name="ball" type="sphere" size="0.05" mass="0.5" rgba="1 0 0 1"/>
                    </body>
                </body>
            </worldbody>
            <actuator>
                <motor name="torque" joint="hinge" gear="1" ctrllimited="true" ctrlrange="-2 2"/>
            </actuator>
            <sensor>
                <jointpos name="angle" joint="hinge"/>
                <jointvel name="velocity" joint="hinge"/>
            </sensor>
        </mujoco>"""
        
    def test_pendulum_demo_tool_exists(self):
        """Test that pendulum_demo tool is registered"""
        tools = self.server.get_tools()
        tool_names = [t["name"] for t in tools]
        assert "pendulum_demo" in tool_names
        
        # Check tool metadata
        tool = next(t for t in tools if t["name"] == "pendulum_demo")
        assert "description" in tool
        assert "parameters" in tool
    
    def test_pendulum_demo_setup(self):
        """Test pendulum demo setup"""
        result = self.server.call_tool("pendulum_demo", {
            "action": "setup"
        })
        
        assert result["success"] is True
        assert "model_id" in result
        assert "message" in result
        assert "pendulum" in result["message"].lower()
        
        self.model_id = result["model_id"]
        
        # Verify model is loaded
        models = self.server.call_tool("get_loaded_models", {})
        assert len(models["models"]) > 0
        assert any(m["model_id"] == self.model_id for m in models["models"])
    
    def test_pendulum_demo_control(self):
        """Test pendulum PID control"""
        # Setup pendulum
        setup_result = self.server.call_tool("pendulum_demo", {
            "action": "setup"
        })
        model_id = setup_result["model_id"]
        
        # Run control demo
        result = self.server.call_tool("pendulum_demo", {
            "action": "control",
            "model_id": model_id,
            "target_angle": 45.0,  # degrees
            "duration": 3.0,       # seconds - longer duration for better convergence
            "kp": 5.0,             # higher proportional gain
            "ki": 0.5,             # higher integral gain
            "kd": 1.0              # higher derivative gain
        })
        
        assert result["success"] is True
        assert "trajectory" in result
        assert "final_error" in result
        assert "energy" in result
        assert len(result["trajectory"]) > 0
        
        # Check that control converged reasonably well
        assert abs(result["final_error"]) < 15.0  # Within 15 degrees for demo
    
    def test_pendulum_demo_swing_up(self):
        """Test pendulum swing-up control"""
        # Setup pendulum
        setup_result = self.server.call_tool("pendulum_demo", {
            "action": "setup"
        })
        model_id = setup_result["model_id"]
        
        # Run swing-up demo (energy-based control)
        result = self.server.call_tool("pendulum_demo", {
            "action": "swing_up",
            "model_id": model_id,
            "duration": 10.0,      # longer duration for swing-up
            "energy_gain": 2.0     # higher energy gain
        })
        
        assert result["success"] is True
        assert "trajectory" in result
        assert "max_height" in result
        assert "energy_profile" in result
        assert result["max_height"] > 0.0001  # Should have some motion
    
    def test_pendulum_demo_visualization(self):
        """Test pendulum visualization during control"""
        # Setup pendulum
        setup_result = self.server.call_tool("pendulum_demo", {
            "action": "setup"
        })
        model_id = setup_result["model_id"]
        
        # Run control with visualization
        result = self.server.call_tool("pendulum_demo", {
            "action": "control_with_viz",
            "model_id": model_id,
            "target_angle": 30.0,
            "duration": 1.0,
            "visualize": True,
            "viz_interval": 0.1  # Capture frame every 0.1s
        })
        
        assert result["success"] is True
        assert "frames" in result
        assert len(result["frames"]) > 0
        assert "ascii_frames" in result
        assert len(result["ascii_frames"]) > 0
        
        # Check frame format
        first_frame = result["frames"][0]
        assert "time" in first_frame
        assert "image" in first_frame
        assert "angle" in first_frame
        assert "control" in first_frame
    
    def test_pendulum_energy_analysis(self):
        """Test pendulum energy analysis"""
        # Setup pendulum
        setup_result = self.server.call_tool("pendulum_demo", {
            "action": "setup"
        })
        model_id = setup_result["model_id"]
        
        # Analyze energy
        result = self.server.call_tool("pendulum_demo", {
            "action": "analyze_energy",
            "model_id": model_id,
            "duration": 3.0
        })
        
        assert result["success"] is True
        assert "kinetic_energy" in result
        assert "potential_energy" in result
        assert "total_energy" in result
        assert "energy_conservation" in result
        
        # Check energy conservation (should be relatively constant without control)
        energy_variation = result["energy_conservation"]["variation"]
        assert energy_variation < 0.1  # Less than 10% variation
    
    def test_demo_parameter_validation(self):
        """Test demo parameter validation"""
        # Setup pendulum first
        setup_result = self.server.call_tool("pendulum_demo", {
            "action": "setup"
        })
        model_id = setup_result["model_id"]
        
        # Test invalid action
        with pytest.raises(ValueError) as exc_info:
            self.server.call_tool("pendulum_demo", {
                "action": "invalid_action"
            })
        assert "Unknown action" in str(exc_info.value)
        
        # Test control without model_id
        with pytest.raises(ValueError) as exc_info:
            self.server.call_tool("pendulum_demo", {
                "action": "control",
                "target_angle": 45.0
            })
        assert "model_id is required" in str(exc_info.value)
    
    def test_demo_state_tracking(self):
        """Test that demo tracks state properly"""
        # Setup pendulum
        setup_result = self.server.call_tool("pendulum_demo", {
            "action": "setup"
        })
        model_id = setup_result["model_id"]
        
        # Get demo state
        result = self.server.call_tool("pendulum_demo", {
            "action": "get_state",
            "model_id": model_id
        })
        
        assert result["success"] is True
        assert "angle" in result
        assert "velocity" in result
        assert "control" in result
        assert "time" in result
        assert "energy" in result
    
    def test_demo_trajectory_export(self):
        """Test trajectory export functionality"""
        # Setup and run control
        setup_result = self.server.call_tool("pendulum_demo", {
            "action": "setup"
        })
        model_id = setup_result["model_id"]
        
        control_result = self.server.call_tool("pendulum_demo", {
            "action": "control",
            "model_id": model_id,
            "target_angle": 60.0,
            "duration": 2.0
        })
        
        # Export trajectory
        export_result = self.server.call_tool("pendulum_demo", {
            "action": "export_trajectory",
            "model_id": model_id,
            "format": "csv"
        })
        
        assert export_result["success"] is True
        assert "data" in export_result
        assert "format" in export_result
        assert export_result["format"] == "csv"
        assert "time,angle,velocity,control" in export_result["data"]


class TestDemoEnhancements:
    """Test server enhancements for v0.3.1"""
    
    def test_server_version(self):
        """Test that server version is updated"""
        server = MuJoCoMCPServer()
        assert server.version == "0.6.0"
        
        info = server.get_server_info()
        assert info["version"] == "0.6.0"
    
    def test_demo_capability(self):
        """Test that demo capability is advertised"""
        server = MuJoCoMCPServer()
        info = server.get_server_info()
        
        assert "capabilities" in info
        assert "demo" in info["capabilities"]
    
    def test_demo_list(self):
        """Test getting list of available demos"""
        server = MuJoCoMCPServer()
        result = server.call_tool("list_demos", {})
        
        assert "demos" in result
        assert len(result["demos"]) > 0
        assert any(d["name"] == "pendulum" for d in result["demos"])
        
        # Check demo metadata
        pendulum_demo = next(d for d in result["demos"] if d["name"] == "pendulum")
        assert "description" in pendulum_demo
        assert "difficulty" in pendulum_demo
        assert "concepts" in pendulum_demo


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
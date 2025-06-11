#!/usr/bin/env python3
"""
Test suite for v0.3.0 - Simple Visualization
Tests for get_render_frame and camera control
"""
import pytest
import json
import base64
from pathlib import Path
import sys

# Add project to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from mujoco_mcp.simple_server import MuJoCoMCPServer


class TestVisualization:
    """Test visualization functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.server = MuJoCoMCPServer()
        
        # Simple box model for testing
        self.box_xml = """<mujoco model="test_box">
            <option timestep="0.01"/>
            <visual>
                <global offwidth="640" offheight="480"/>
            </visual>
            <worldbody>
                <light name="top" pos="0 0 2"/>
                <body name="box" pos="0 0 0.5">
                    <geom name="box_geom" type="box" size="0.2 0.2 0.2" rgba="1 0 0 1"/>
                </body>
                <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>
            </worldbody>
        </mujoco>"""
        
        # Load model
        result = self.server.call_tool("load_model", {
            "model_string": self.box_xml,
            "name": "test_box"
        })
        self.model_id = result["model_id"]
    
    def test_get_render_frame_tool_exists(self):
        """Test that get_render_frame tool is registered"""
        tools = self.server.get_tools()
        tool_names = [t["name"] for t in tools]
        assert "get_render_frame" in tool_names
        
        # Check tool metadata
        tool = next(t for t in tools if t["name"] == "get_render_frame")
        assert "description" in tool
        assert "parameters" in tool
        assert "model_id" in tool["parameters"]
    
    def test_get_render_frame_default(self):
        """Test getting render frame with default parameters"""
        result = self.server.call_tool("get_render_frame", {
            "model_id": self.model_id
        })
        
        assert "success" in result
        assert result["success"] is True
        assert "image_data" in result
        assert "format" in result
        assert result["format"] == "base64/png"
        assert "width" in result
        assert "height" in result
        
        # Verify base64 data
        try:
            decoded = base64.b64decode(result["image_data"])
            assert len(decoded) > 0
        except Exception as e:
            pytest.fail(f"Invalid base64 data: {e}")
    
    def test_get_render_frame_custom_size(self):
        """Test rendering with custom size"""
        result = self.server.call_tool("get_render_frame", {
            "model_id": self.model_id,
            "width": 800,
            "height": 600
        })
        
        assert result["success"] is True
        assert result["width"] == 800
        assert result["height"] == 600
    
    def test_get_render_frame_with_camera(self):
        """Test rendering with camera parameters"""
        result = self.server.call_tool("get_render_frame", {
            "model_id": self.model_id,
            "camera_name": "track",  # MuJoCo's tracking camera
            "camera_distance": 3.0,
            "camera_azimuth": 45.0,
            "camera_elevation": -20.0
        })
        
        assert result["success"] is True
        assert "camera_info" in result
        assert result["camera_info"]["name"] == "track"
    
    def test_get_ascii_visualization_tool(self):
        """Test ASCII visualization tool"""
        tools = self.server.get_tools()
        tool_names = [t["name"] for t in tools]
        assert "get_ascii_visualization" in tool_names
    
    def test_get_ascii_visualization(self):
        """Test getting ASCII visualization"""
        result = self.server.call_tool("get_ascii_visualization", {
            "model_id": self.model_id,
            "width": 60,
            "height": 20
        })
        
        assert "success" in result
        assert result["success"] is True
        assert "ascii_art" in result
        assert isinstance(result["ascii_art"], str)
        assert len(result["ascii_art"]) > 0
        
        # Check it contains some expected characters
        ascii_art = result["ascii_art"]
        assert any(c in ascii_art for c in ['#', '*', '.', ' ', '\n'])
    
    def test_invalid_camera_name(self):
        """Test error handling for invalid camera name"""
        with pytest.raises(ValueError) as exc_info:
            self.server.call_tool("get_render_frame", {
                "model_id": self.model_id,
                "camera_name": "nonexistent_camera"
            })
        assert "camera" in str(exc_info.value).lower()
    
    def test_render_size_limits(self):
        """Test size limits for rendering"""
        # Test too large
        with pytest.raises(ValueError) as exc_info:
            self.server.call_tool("get_render_frame", {
                "model_id": self.model_id,
                "width": 5000,
                "height": 5000
            })
        assert "between" in str(exc_info.value).lower() or "must be" in str(exc_info.value).lower()
        
        # Test too small
        with pytest.raises(ValueError) as exc_info:
            self.server.call_tool("get_render_frame", {
                "model_id": self.model_id,
                "width": 10,
                "height": 10
            })
        assert "between" in str(exc_info.value).lower() or "must be" in str(exc_info.value).lower()
    
    def test_visualization_state_consistency(self):
        """Test that visualization doesn't affect simulation state"""
        # Get initial state
        initial_state = self.server.call_tool("get_simulation_state", {
            "model_id": self.model_id,
            "include_positions": True
        })
        
        # Render frame
        self.server.call_tool("get_render_frame", {
            "model_id": self.model_id
        })
        
        # Get state again
        final_state = self.server.call_tool("get_simulation_state", {
            "model_id": self.model_id,
            "include_positions": True
        })
        
        # State should be unchanged
        assert initial_state["time"] == final_state["time"]
        assert initial_state["qpos"] == final_state["qpos"]


class TestServerEnhancements:
    """Test server enhancements for v0.3.0"""
    
    def test_server_version(self):
        """Test that server version is updated"""
        server = MuJoCoMCPServer()
        assert server.version == "0.3.0"
        
        info = server.get_server_info()
        assert info["version"] == "0.3.0"
    
    def test_new_capabilities(self):
        """Test that visualization capability is advertised"""
        server = MuJoCoMCPServer()
        info = server.get_server_info()
        
        assert "capabilities" in info
        assert "visualization" in info["capabilities"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
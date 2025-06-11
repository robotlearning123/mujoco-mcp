#!/usr/bin/env python3
"""
Test suite for v0.1.2 - First MCP Tool
Tests for load_model tool functionality
"""
import pytest
import json
from pathlib import Path
import sys

# Add project to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from mujoco_mcp.simple_server import MuJoCoMCPServer


class TestLoadModelTool:
    """Test the load_model MCP tool"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.server = MuJoCoMCPServer()
        self.simple_xml = """<mujoco>
            <worldbody>
                <body name="box">
                    <geom type="box" size="0.1 0.1 0.1"/>
                </body>
            </worldbody>
        </mujoco>"""
        
        self.pendulum_xml = """<mujoco model="pendulum">
            <option timestep="0.01"/>
            <worldbody>
                <body name="cart" pos="0 0 0">
                    <joint name="slider" type="slide" axis="1 0 0"/>
                    <geom name="cart" type="box" size="0.2 0.2 0.2" rgba="0.7 0.7 0 1"/>
                    <body name="pole" pos="0 0 0">
                        <joint name="hinge" type="hinge" axis="0 1 0"/>
                        <geom name="pole" type="cylinder" size="0.05 0.5" rgba="0 0.7 0.7 1" pos="0 0 0.5"/>
                    </body>
                </body>
            </worldbody>
        </mujoco>"""
    
    def test_load_model_tool_exists(self):
        """Test that load_model tool is registered"""
        tools = self.server.get_tools()
        tool_names = [t["name"] for t in tools]
        assert "load_model" in tool_names, f"Available tools: {tool_names}"
        
        # Check tool metadata
        load_model_tool = next(t for t in tools if t["name"] == "load_model")
        assert "description" in load_model_tool
        assert "parameters" in load_model_tool
        assert "model_string" in load_model_tool["parameters"]
    
    def test_load_model_from_string(self):
        """Test loading a model from XML string"""
        result = self.server.call_tool("load_model", {
            "model_string": self.simple_xml
        })
        
        assert result["success"] is True
        assert "model_id" in result
        assert isinstance(result["model_id"], str)
        assert len(result["model_id"]) > 0
        assert "message" in result
    
    def test_load_model_with_name(self):
        """Test loading a model with a custom name"""
        result = self.server.call_tool("load_model", {
            "model_string": self.simple_xml,
            "name": "test_model"
        })
        
        assert result["success"] is True
        assert "model_id" in result
        assert result.get("name") == "test_model"
    
    def test_load_model_validation(self):
        """Test model loading parameter validation"""
        # Test missing required parameter
        with pytest.raises(ValueError) as exc_info:
            self.server.call_tool("load_model", {})
        assert "model_string" in str(exc_info.value)
        
        # Test empty model string
        with pytest.raises(ValueError) as exc_info:
            self.server.call_tool("load_model", {"model_string": ""})
        assert "empty" in str(exc_info.value).lower()
        
        # Test invalid model string
        with pytest.raises(ValueError) as exc_info:
            self.server.call_tool("load_model", {"model_string": "not xml"})
        assert "valid xml" in str(exc_info.value).lower()
    
    def test_load_multiple_models(self):
        """Test loading multiple models"""
        # Load first model
        result1 = self.server.call_tool("load_model", {
            "model_string": self.simple_xml,
            "name": "model1"
        })
        
        # Load second model
        result2 = self.server.call_tool("load_model", {
            "model_string": self.pendulum_xml,
            "name": "model2"
        })
        
        assert result1["success"] is True
        assert result2["success"] is True
        assert result1["model_id"] != result2["model_id"]
    
    def test_model_persistence(self):
        """Test that loaded models persist in server"""
        # Load a model
        result = self.server.call_tool("load_model", {
            "model_string": self.simple_xml
        })
        model_id = result["model_id"]
        
        # Verify model exists in server
        assert hasattr(self.server, "_models")
        assert model_id in self.server._models
        assert self.server._models[model_id] is not None
    
    def test_load_complex_model(self):
        """Test loading a more complex model"""
        result = self.server.call_tool("load_model", {
            "model_string": self.pendulum_xml
        })
        
        assert result["success"] is True
        assert "model_id" in result
        
        # Model should be stored
        model_id = result["model_id"]
        assert model_id in self.server._models
    
    def test_model_info_in_response(self):
        """Test that model info is returned in response"""
        result = self.server.call_tool("load_model", {
            "model_string": self.pendulum_xml
        })
        
        assert "model_info" in result
        info = result["model_info"]
        
        # Check basic model information
        assert "nq" in info  # number of generalized coordinates
        assert "nv" in info  # number of degrees of freedom
        assert "nbody" in info  # number of bodies
        assert info["nbody"] > 0
    
    def test_error_handling_invalid_xml(self):
        """Test error handling for invalid XML"""
        invalid_xml = "<mujoco><invalid></mujoco>"
        
        with pytest.raises(ValueError) as exc_info:
            self.server.call_tool("load_model", {
                "model_string": invalid_xml
            })
        assert "failed to load" in str(exc_info.value).lower()
    
    def test_get_loaded_models_tool(self):
        """Test get_loaded_models tool"""
        # Check tool exists
        tools = self.server.get_tools()
        tool_names = [t["name"] for t in tools]
        assert "get_loaded_models" in tool_names
        
        # Get initial models (should be empty)
        result = self.server.call_tool("get_loaded_models", {})
        assert "models" in result
        assert len(result["models"]) == 0
        
        # Load some models
        self.server.call_tool("load_model", {
            "model_string": self.simple_xml,
            "name": "test1"
        })
        self.server.call_tool("load_model", {
            "model_string": self.pendulum_xml,
            "name": "test2"
        })
        
        # Get loaded models
        result = self.server.call_tool("get_loaded_models", {})
        assert len(result["models"]) == 2
        
        # Check model information
        model_names = [m["name"] for m in result["models"]]
        assert "test1" in model_names
        assert "test2" in model_names


class TestServerEnhancements:
    """Test server enhancements for v0.1.2"""
    
    def test_server_version(self):
        """Test that server version is updated"""
        server = MuJoCoMCPServer()
        assert server.version == "0.1.2"
        
        info = server.get_server_info()
        assert info["version"] == "0.1.2"
    
    def test_tool_count(self):
        """Test that we have the expected number of tools"""
        server = MuJoCoMCPServer()
        tools = server.get_tools()
        
        # Should have at least: get_server_info, get_tools, load_model, get_loaded_models
        assert len(tools) >= 4
        
        tool_names = [t["name"] for t in tools]
        required_tools = ["get_server_info", "get_tools", "load_model", "get_loaded_models"]
        for tool in required_tools:
            assert tool in tool_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
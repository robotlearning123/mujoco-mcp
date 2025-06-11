#!/usr/bin/env python3
"""
Test suite for v0.4.0 - Model Generator
Tests for programmatic model generation tools
"""
import pytest
import json
from pathlib import Path
import sys

# Add project to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from mujoco_mcp.simple_server import MuJoCoMCPServer


class TestModelGenerator:
    """Test model generation functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.server = MuJoCoMCPServer()
    
    def test_generate_robot_tool_exists(self):
        """Test that generate_robot tool is registered"""
        tools = self.server.get_tools()
        tool_names = [t["name"] for t in tools]
        assert "generate_robot" in tool_names
        
        # Check tool metadata
        tool = next(t for t in tools if t["name"] == "generate_robot")
        assert "description" in tool
        assert "parameters" in tool
        assert "robot_type" in tool["parameters"]
    
    def test_generate_simple_arm(self):
        """Test generating a simple robotic arm"""
        result = self.server.call_tool("generate_robot", {
            "robot_type": "arm",
            "parameters": {
                "num_links": 3,
                "link_length": 0.3,
                "link_mass": 0.5
            }
        })
        
        assert result["success"] is True
        assert "model_id" in result
        assert "xml" in result
        assert "<mujoco" in result["xml"]
        assert "arm" in result["xml"].lower()
        
        # Verify model can be loaded
        models = self.server.call_tool("get_loaded_models", {})
        assert len(models["models"]) > 0
    
    def test_generate_mobile_robot(self):
        """Test generating a mobile robot"""
        result = self.server.call_tool("generate_robot", {
            "robot_type": "mobile",
            "parameters": {
                "base_size": [0.3, 0.2, 0.1],
                "wheel_radius": 0.05,
                "num_wheels": 4
            }
        })
        
        assert result["success"] is True
        assert "model_id" in result
        assert "wheel" in result["xml"].lower()
    
    def test_generate_gripper(self):
        """Test generating a gripper"""
        result = self.server.call_tool("generate_robot", {
            "robot_type": "gripper",
            "parameters": {
                "finger_length": 0.08,
                "max_opening": 0.1
            }
        })
        
        assert result["success"] is True
        assert "finger" in result["xml"].lower() or "gripper" in result["xml"].lower()
    
    def test_generate_with_constraints(self):
        """Test generation with physical constraints"""
        result = self.server.call_tool("generate_robot", {
            "robot_type": "arm",
            "parameters": {
                "num_links": 2,
                "link_length": 0.5,
                "joint_limits": [-90, 90],
                "max_torque": 10.0
            }
        })
        
        assert result["success"] is True
        assert "range" in result["xml"]  # Joint limits
        assert "ctrlrange" in result["xml"] or "forcerange" in result["xml"]  # Torque limits
    
    def test_invalid_robot_type(self):
        """Test error handling for invalid robot type"""
        with pytest.raises(ValueError) as exc_info:
            self.server.call_tool("generate_robot", {
                "robot_type": "invalid_type"
            })
        assert "robot type" in str(exc_info.value).lower()


class TestEnvironmentGenerator:
    """Test environment generation functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.server = MuJoCoMCPServer()
    
    def test_generate_environment_tool_exists(self):
        """Test that generate_environment tool is registered"""
        tools = self.server.get_tools()
        tool_names = [t["name"] for t in tools]
        assert "generate_environment" in tool_names
    
    def test_generate_flat_ground(self):
        """Test generating flat ground environment"""
        result = self.server.call_tool("generate_environment", {
            "env_type": "flat_ground",
            "parameters": {
                "size": [10, 10],
                "friction": 1.0
            }
        })
        
        assert result["success"] is True
        assert "xml" in result
        assert "plane" in result["xml"] or "box" in result["xml"]
        assert "ground" in result["xml"].lower()
    
    def test_generate_obstacles(self):
        """Test generating environment with obstacles"""
        result = self.server.call_tool("generate_environment", {
            "env_type": "obstacles",
            "parameters": {
                "ground_size": [5, 5],
                "num_obstacles": 3,
                "obstacle_size_range": [0.1, 0.5]
            }
        })
        
        assert result["success"] is True
        assert result["obstacle_count"] == 3
        assert "geom" in result["xml"]
    
    def test_generate_terrain(self):
        """Test generating terrain environment"""
        result = self.server.call_tool("generate_environment", {
            "env_type": "terrain",
            "parameters": {
                "terrain_type": "stairs",
                "num_steps": 5,
                "step_height": 0.1,
                "step_width": 0.3
            }
        })
        
        assert result["success"] is True
        assert "success" in result
        # Should have multiple geoms for stairs
        assert result["xml"].count("geom") >= 5
    
    def test_combine_robot_and_environment(self):
        """Test combining generated robot with environment"""
        # Generate environment
        env_result = self.server.call_tool("generate_environment", {
            "env_type": "flat_ground",
            "parameters": {"size": [5, 5]}
        })
        
        # Generate robot
        robot_result = self.server.call_tool("generate_robot", {
            "robot_type": "arm",
            "parameters": {"num_links": 2}
        })
        
        # Combine them
        result = self.server.call_tool("combine_models", {
            "base_model_xml": env_result["xml"],
            "add_model_xml": robot_result["xml"],
            "position": [0, 0, 0.5]
        })
        
        assert result["success"] is True
        assert "model_id" in result
        assert "combined_xml" in result


class TestModelTemplates:
    """Test model template system"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.server = MuJoCoMCPServer()
    
    def test_list_templates_tool(self):
        """Test list_templates tool"""
        result = self.server.call_tool("list_templates", {})
        
        assert "robot_templates" in result
        assert "environment_templates" in result
        assert len(result["robot_templates"]) > 0
        
        # Check template structure
        template = result["robot_templates"][0]
        assert "name" in template
        assert "description" in template
        assert "parameters" in template
    
    def test_generate_from_template(self):
        """Test generating from template"""
        result = self.server.call_tool("generate_from_template", {
            "template_name": "simple_arm",
            "parameters": {
                "num_links": 4,
                "color": "blue"
            }
        })
        
        assert result["success"] is True
        assert "model_id" in result
        assert "rgba" in result["xml"]  # Color applied
    
    def test_save_as_template(self):
        """Test saving generated model as template"""
        # First generate a model
        gen_result = self.server.call_tool("generate_robot", {
            "robot_type": "arm",
            "parameters": {"num_links": 3}
        })
        
        # Save as template
        result = self.server.call_tool("save_as_template", {
            "model_id": gen_result["model_id"],
            "template_name": "my_custom_arm",
            "description": "Custom 3-link arm",
            "parameterizable": ["link_length", "link_mass", "color"]
        })
        
        assert result["success"] is True
        assert result["template_name"] == "my_custom_arm"
        
        # Verify it appears in list
        templates = self.server.call_tool("list_templates", {})
        template_names = [t["name"] for t in templates["robot_templates"]]
        assert "my_custom_arm" in template_names


class TestXMLValidation:
    """Test XML validation and safety checks"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.server = MuJoCoMCPServer()
    
    def test_validate_generated_xml(self):
        """Test that generated XML is valid MuJoCo format"""
        result = self.server.call_tool("generate_robot", {
            "robot_type": "arm",
            "parameters": {"num_links": 2}
        })
        
        # Validate XML
        validation = self.server.call_tool("validate_model_xml", {
            "xml_string": result["xml"]
        })
        
        assert validation["is_valid"] is True
        assert "errors" not in validation or len(validation["errors"]) == 0
    
    def test_safety_constraints(self):
        """Test safety constraints in generation"""
        # Try to generate with unrealistic parameters
        result = self.server.call_tool("generate_robot", {
            "robot_type": "arm",
            "parameters": {
                "num_links": 50,  # Too many
                "link_mass": 1000,  # Too heavy
                "max_torque": 10000  # Too strong
            }
        })
        
        # Should either fail or apply safety limits
        if result["success"]:
            assert "warnings" in result
            assert len(result["warnings"]) > 0
        else:
            assert "error" in result
            assert "safety" in result["error"].lower() or "limit" in result["error"].lower()


class TestServerVersion:
    """Test server version update"""
    
    def test_version_updated(self):
        """Test that server version is 0.4.0"""
        server = MuJoCoMCPServer()
        assert server.version == "0.4.0"
        
        info = server.get_server_info()
        assert info["version"] == "0.4.0"
    
    def test_model_generation_capability(self):
        """Test that model generation capability is advertised"""
        server = MuJoCoMCPServer()
        info = server.get_server_info()
        
        assert "capabilities" in info
        assert "model_generation" in info["capabilities"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
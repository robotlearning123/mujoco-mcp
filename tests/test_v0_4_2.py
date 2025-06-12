#!/usr/bin/env python3
"""
Test suite for v0.4.2 - Robot Designer MVP
Tests for intelligent robot design functionality
"""
import pytest
import json
from pathlib import Path
import sys

# Add project to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from mujoco_mcp.simple_server import MuJoCoMCPServer


class TestRobotDesigner:
    """Test robot designer functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.server = MuJoCoMCPServer()
    
    def test_design_robot_tool_exists(self):
        """Test that design_robot tool is registered"""
        tools = self.server.get_tools()
        tool_names = [t["name"] for t in tools]
        assert "design_robot" in tool_names
        
        # Check tool metadata
        tool = next(t for t in tools if t["name"] == "design_robot")
        assert "description" in tool
        assert "parameters" in tool
        assert "task_description" in tool["parameters"]
        assert "constraints" in tool["parameters"]
    
    def test_design_simple_gripper(self):
        """Test designing a simple gripper"""
        result = self.server.call_tool("design_robot", {
            "task_description": "I need a gripper that can pick up small objects",
            "constraints": {
                "max_size": [0.2, 0.2, 0.2],  # 20cm cube
                "max_force": 10.0  # 10N
            }
        })
        
        assert result["success"] is True
        assert "design" in result
        assert "model_id" in result
        assert "specifications" in result
        
        # Check specifications
        specs = result["specifications"]
        assert "type" in specs
        assert specs["type"] == "gripper"
        assert "estimated_performance" in specs
    
    def test_design_mobile_robot(self):
        """Test designing a mobile robot"""
        result = self.server.call_tool("design_robot", {
            "task_description": "Design a robot that can navigate rough terrain",
            "constraints": {
                "max_weight": 50.0,  # 50kg
                "terrain_type": "rough",
                "speed_requirement": 1.0  # 1m/s
            },
            "preferences": {
                "wheel_type": "rugged",
                "stability": "high"
            }
        })
        
        assert result["success"] is True
        assert "design" in result
        specs = result["specifications"]
        assert specs["type"] == "mobile"
        assert "num_wheels" in specs
        assert specs["num_wheels"] >= 4  # For stability on rough terrain
    
    def test_design_manipulator(self):
        """Test designing a manipulator arm"""
        result = self.server.call_tool("design_robot", {
            "task_description": "I need a robotic arm for precise assembly tasks",
            "constraints": {
                "workspace_radius": 0.8,  # 80cm reach
                "precision": 0.001,  # 1mm precision
                "payload": 5.0  # 5kg payload
            }
        })
        
        assert result["success"] is True
        specs = result["specifications"]
        assert specs["type"] == "arm"
        assert "num_joints" in specs
        assert specs["num_joints"] >= 6  # For full workspace coverage
        assert "estimated_precision" in specs["estimated_performance"]
    
    def test_design_with_optimization(self):
        """Test design with automatic parameter optimization"""
        result = self.server.call_tool("design_robot", {
            "task_description": "Design an energy-efficient walking robot",
            "constraints": {
                "max_power": 100.0,  # 100W
                "min_speed": 0.5  # 0.5m/s
            },
            "optimize_for": ["energy_efficiency", "stability"]
        })
        
        assert result["success"] is True
        assert "optimization_results" in result
        assert "optimized_parameters" in result
        
        # Check that optimization was performed
        opt_results = result["optimization_results"]
        assert "energy_efficiency_score" in opt_results
        assert opt_results["energy_efficiency_score"] > 0.7
    
    def test_design_validation(self):
        """Test that designed robots are validated"""
        result = self.server.call_tool("design_robot", {
            "task_description": "Simple arm for testing",
            "constraints": {"num_links": 3}
        })
        
        assert result["success"] is True
        assert "validation" in result
        
        validation = result["validation"]
        assert "is_valid" in validation
        assert validation["is_valid"] is True
        assert "safety_checks" in validation
        assert all(check["passed"] for check in validation["safety_checks"])
    
    def test_design_iteration(self):
        """Test iterative design refinement"""
        # First design
        result1 = self.server.call_tool("design_robot", {
            "task_description": "Basic gripper",
            "constraints": {"max_size": [0.1, 0.1, 0.1]}
        })
        
        design_id = result1["design_id"]
        
        # Refine design
        result2 = self.server.call_tool("refine_design", {
            "design_id": design_id,
            "improvements": {
                "increase_grip_force": True,
                "reduce_weight": True
            },
            "additional_constraints": {
                "max_weight": 0.5  # 500g
            }
        })
        
        assert result2["success"] is True
        assert result2["design_id"] != design_id  # New version
        assert "improvements_applied" in result2
        assert len(result2["improvements_applied"]) > 0
    
    def test_component_library(self):
        """Test using component library in design"""
        result = self.server.call_tool("design_robot", {
            "task_description": "Modular robot arm",
            "use_components": True,
            "component_preferences": {
                "actuator_type": "servo",
                "sensor_types": ["position", "force"]
            }
        })
        
        assert result["success"] is True
        assert "components_used" in result
        
        components = result["components_used"]
        assert len(components) > 0
        assert any(c["type"] == "actuator" for c in components)
        assert any(c["type"] == "sensor" for c in components)
    
    def test_design_cost_estimation(self):
        """Test cost estimation for designs"""
        result = self.server.call_tool("design_robot", {
            "task_description": "Low-cost educational robot",
            "constraints": {
                "max_cost": 100.0,  # $100 budget
                "complexity": "simple"
            },
            "estimate_cost": True
        })
        
        assert result["success"] is True
        assert "cost_estimate" in result
        
        cost = result["cost_estimate"]
        assert "total" in cost
        assert cost["total"] <= 100.0  # Within budget
        assert "breakdown" in cost
        assert "materials" in cost["breakdown"]
        assert "actuators" in cost["breakdown"]


class TestDesignAssistant:
    """Test AI design assistant features"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.server = MuJoCoMCPServer()
    
    def test_suggest_improvements(self):
        """Test design improvement suggestions"""
        # Create a basic design first
        design_result = self.server.call_tool("design_robot", {
            "task_description": "Simple arm",
            "constraints": {"num_links": 2}
        })
        
        model_id = design_result["model_id"]
        
        # Get improvement suggestions
        result = self.server.call_tool("suggest_improvements", {
            "model_id": model_id,
            "goals": ["increase_workspace", "improve_precision"]
        })
        
        assert "suggestions" in result
        assert len(result["suggestions"]) > 0
        
        suggestion = result["suggestions"][0]
        assert "description" in suggestion
        assert "expected_improvement" in suggestion
        assert "implementation" in suggestion
    
    def test_compare_designs(self):
        """Test comparing multiple robot designs"""
        # Create two designs
        design1 = self.server.call_tool("design_robot", {
            "task_description": "Gripper A",
            "constraints": {"finger_length": 0.08}
        })
        
        design2 = self.server.call_tool("design_robot", {
            "task_description": "Gripper B",
            "constraints": {"finger_length": 0.10}
        })
        
        # Compare them
        result = self.server.call_tool("compare_designs", {
            "design_ids": [design1["design_id"], design2["design_id"]],
            "metrics": ["grip_force", "precision", "cost"]
        })
        
        assert "comparison" in result
        assert "winner" in result
        assert "detailed_scores" in result
        
        # Check detailed scores
        scores = result["detailed_scores"]
        assert len(scores) == 2
        assert all(metric in scores[0] for metric in ["grip_force", "precision", "cost"])
    
    def test_explain_design_choices(self):
        """Test design explanation feature"""
        design_result = self.server.call_tool("design_robot", {
            "task_description": "High-speed pick and place robot",
            "constraints": {"cycle_time": 0.5}  # 0.5 seconds
        })
        
        result = self.server.call_tool("explain_design", {
            "design_id": design_result["design_id"]
        })
        
        assert "explanation" in result
        assert "design_choices" in result
        assert "rationale" in result
        
        # Check that key choices are explained
        choices = result["design_choices"]
        assert any("actuator" in choice.lower() for choice in choices)
        assert any("structure" in choice.lower() for choice in choices)


class TestComponentLibrary:
    """Test component library functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.server = MuJoCoMCPServer()
    
    def test_list_components(self):
        """Test listing available components"""
        result = self.server.call_tool("list_components", {
            "category": "actuator"
        })
        
        assert "components" in result
        assert len(result["components"]) > 0
        
        component = result["components"][0]
        assert "name" in component
        assert "type" in component
        assert "specifications" in component
        assert "compatible_with" in component
    
    def test_component_compatibility(self):
        """Test checking component compatibility"""
        result = self.server.call_tool("check_compatibility", {
            "component_a": {"type": "motor", "torque": 10.0},
            "component_b": {"type": "gearbox", "ratio": 10.0}
        })
        
        assert "compatible" in result
        assert "reasons" in result
        if not result["compatible"]:
            assert len(result["reasons"]) > 0


class TestServerVersion042:
    """Test server version update"""
    
    def test_version_updated(self):
        """Test that server version is 0.4.2"""
        server = MuJoCoMCPServer()
        assert server.version == "0.6.0"
        
        info = server.get_server_info()
        assert info["version"] == "0.6.0"
    
    def test_robot_designer_capability(self):
        """Test that robot designer capability is advertised"""
        server = MuJoCoMCPServer()
        info = server.get_server_info()
        
        assert "capabilities" in info
        assert "robot_designer" in info["capabilities"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
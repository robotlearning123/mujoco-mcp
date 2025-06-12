#!/usr/bin/env python3
"""
Test suite for v0.3.2 - Natural Language Interface
Tests for LLM-friendly interface and high-level commands
"""
import pytest
import json
from pathlib import Path
import sys

# Add project to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from mujoco_mcp.simple_server import MuJoCoMCPServer


class TestNaturalLanguageInterface:
    """Test natural language interface functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.server = MuJoCoMCPServer()
    
    def test_execute_command_tool_exists(self):
        """Test that execute_command tool is registered"""
        tools = self.server.get_tools()
        tool_names = [t["name"] for t in tools]
        assert "execute_command" in tool_names
        
        # Check tool metadata
        tool = next(t for t in tools if t["name"] == "execute_command")
        assert "description" in tool
        assert "natural language" in tool["description"].lower()
        assert "parameters" in tool
        assert "command" in tool["parameters"]
    
    def test_simple_pendulum_command(self):
        """Test creating a pendulum with natural language"""
        result = self.server.call_tool("execute_command", {
            "command": "create a pendulum"
        })
        
        assert result["success"] is True
        assert "model_id" in result
        assert "interpretation" in result
        assert "pendulum" in result["interpretation"].lower()
        
        # Verify model was created
        models = self.server.call_tool("get_loaded_models", {})
        assert len(models["models"]) > 0
    
    def test_control_command(self):
        """Test controlling pendulum with natural language"""
        # First create a pendulum
        setup_result = self.server.call_tool("execute_command", {
            "command": "create a pendulum"
        })
        model_id = setup_result["model_id"]
        
        # Control it
        result = self.server.call_tool("execute_command", {
            "command": "move the pendulum to 45 degrees",
            "context": {"model_id": model_id}
        })
        
        assert result["success"] is True
        assert "action_taken" in result
        assert "control" in result["action_taken"].lower()
    
    def test_query_command(self):
        """Test querying state with natural language"""
        # Create and setup
        setup_result = self.server.call_tool("execute_command", {
            "command": "create a simple pendulum"
        })
        model_id = setup_result["model_id"]
        
        # Query state
        result = self.server.call_tool("execute_command", {
            "command": "what is the current angle of the pendulum?",
            "context": {"model_id": model_id}
        })
        
        assert result["success"] is True
        assert "answer" in result
        assert any(word in result["answer"].lower() for word in ["angle", "degree", "position"])
    
    def test_visualization_command(self):
        """Test visualization with natural language"""
        # Create pendulum
        setup_result = self.server.call_tool("execute_command", {
            "command": "create a pendulum"
        })
        model_id = setup_result["model_id"]
        
        # Request visualization
        result = self.server.call_tool("execute_command", {
            "command": "show me the pendulum",
            "context": {"model_id": model_id}
        })
        
        assert result["success"] is True
        assert "visualization" in result
        assert any(key in result["visualization"] for key in ["ascii", "image"])
    
    def test_complex_command(self):
        """Test complex multi-step command"""
        result = self.server.call_tool("execute_command", {
            "command": "create a pendulum and swing it up to the top position"
        })
        
        assert result["success"] is True
        assert "steps_taken" in result
        assert len(result["steps_taken"]) > 1
        assert any("create" in step.lower() for step in result["steps_taken"])
        assert any("swing" in step.lower() or "control" in step.lower() for step in result["steps_taken"])
    
    def test_help_command(self):
        """Test help and capability queries"""
        result = self.server.call_tool("execute_command", {
            "command": "what can you do?"
        })
        
        assert result["success"] is True
        assert "capabilities" in result
        assert isinstance(result["capabilities"], list)
        assert len(result["capabilities"]) > 0
    
    def test_invalid_command(self):
        """Test handling of unclear commands"""
        result = self.server.call_tool("execute_command", {
            "command": "do something random"
        })
        
        assert "success" in result
        if not result["success"]:
            assert "error" in result or "clarification_needed" in result
        else:
            assert "interpretation" in result
    
    def test_command_with_parameters(self):
        """Test commands with specific parameters"""
        # Create pendulum
        setup_result = self.server.call_tool("execute_command", {
            "command": "create a pendulum"
        })
        model_id = setup_result["model_id"]
        
        # Command with specific parameters
        result = self.server.call_tool("execute_command", {
            "command": "control the pendulum to 30 degrees using PID with kp=5.0",
            "context": {"model_id": model_id}
        })
        
        assert result["success"] is True
        assert "parameters_used" in result
        assert result["parameters_used"].get("kp") == 5.0
        assert result["parameters_used"].get("target_angle") == 30.0


class TestHighLevelActions:
    """Test high-level action tools"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.server = MuJoCoMCPServer()
    
    def test_create_scene_tool(self):
        """Test create_scene high-level tool"""
        tools = self.server.get_tools()
        tool_names = [t["name"] for t in tools]
        assert "create_scene" in tool_names
        
        # Create a scene
        result = self.server.call_tool("create_scene", {
            "scene_type": "pendulum",
            "parameters": {
                "length": 0.5,
                "mass": 0.5
            }
        })
        
        assert result["success"] is True
        assert "model_id" in result
        assert "scene_info" in result
    
    def test_perform_task_tool(self):
        """Test perform_task high-level tool"""
        # Create a pendulum scene
        scene_result = self.server.call_tool("create_scene", {
            "scene_type": "pendulum"
        })
        model_id = scene_result["model_id"]
        
        # Perform a task
        result = self.server.call_tool("perform_task", {
            "task": "swing_up",
            "model_id": model_id,
            "parameters": {
                "duration": 5.0
            }
        })
        
        assert result["success"] is True
        assert "task_result" in result
        assert "trajectory" in result["task_result"]
    
    def test_analyze_behavior_tool(self):
        """Test analyze_behavior tool"""
        # Create and run simulation
        scene_result = self.server.call_tool("create_scene", {
            "scene_type": "pendulum"
        })
        model_id = scene_result["model_id"]
        
        # Run for a bit
        self.server.call_tool("step_simulation", {
            "model_id": model_id,
            "steps": 100
        })
        
        # Analyze behavior
        result = self.server.call_tool("analyze_behavior", {
            "model_id": model_id,
            "analysis_type": "energy",
            "duration": 1.0
        })
        
        assert result["success"] is True
        assert "analysis" in result
        # Check for energy-related keys in the analysis
        analysis = result["analysis"]
        assert any(key in analysis for key in ["kinetic_energy", "potential_energy", "total_energy", "energy_conservation"])


class TestEnhancedDescriptions:
    """Test enhanced tool descriptions for LLM understanding"""
    
    def test_tool_descriptions_enhanced(self):
        """Test that all tools have LLM-friendly descriptions"""
        server = MuJoCoMCPServer()
        tools = server.get_tools()
        
        for tool in tools:
            # Check description exists and is detailed
            assert "description" in tool
            assert len(tool["description"]) > 10  # Has a meaningful description
            
            # Check parameters have descriptions
            if "parameters" in tool and tool["parameters"]:
                for param_name, param_desc in tool["parameters"].items():
                    if isinstance(param_desc, str):
                        assert len(param_desc) > 5  # Has actual description
    
    def test_server_info_enhanced(self):
        """Test server info includes natural language capabilities"""
        server = MuJoCoMCPServer()
        info = server.get_server_info()
        
        assert "capabilities" in info
        assert "natural_language" in info["capabilities"]
        
        # Check for examples or usage hints
        assert "description" in info
        assert len(info["description"]) > 50  # Detailed description
    
    def test_error_messages_helpful(self):
        """Test error messages are helpful for LLM interpretation"""
        server = MuJoCoMCPServer()
        
        # Try invalid command
        try:
            server.call_tool("execute_command", {
                "command": ""  # Empty command
            })
        except ValueError as e:
            error_msg = str(e)
            assert len(error_msg) > 20  # Detailed error
            assert any(word in error_msg.lower() for word in ["command", "empty", "required"])


class TestServerVersion:
    """Test server version update"""
    
    def test_version_updated(self):
        """Test that server version is 0.3.2"""
        server = MuJoCoMCPServer()
        assert server.version == "0.6.0"
        
        info = server.get_server_info()
        assert info["version"] == "0.6.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
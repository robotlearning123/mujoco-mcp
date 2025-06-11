#!/usr/bin/env python3
"""
Test suite for v0.4.1 - Parameter Optimization
Tests for parameter optimization functionality
"""
import pytest
import json
from pathlib import Path
import sys

# Add project to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from mujoco_mcp.simple_server import MuJoCoMCPServer


class TestParameterOptimization:
    """Test parameter optimization functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.server = MuJoCoMCPServer()
    
    def test_optimize_parameters_tool_exists(self):
        """Test that optimize_parameters tool is registered"""
        tools = self.server.get_tools()
        tool_names = [t["name"] for t in tools]
        assert "optimize_parameters" in tool_names
        
        # Check tool metadata
        tool = next(t for t in tools if t["name"] == "optimize_parameters")
        assert "description" in tool
        assert "parameters" in tool
        assert "model_id" in tool["parameters"]
        assert "objective" in tool["parameters"]
    
    def test_optimize_pendulum_swing_time(self):
        """Test optimizing pendulum parameters for fastest swing"""
        # Create pendulum
        result = self.server.call_tool("pendulum_demo", {"action": "setup"})
        model_id = result["model_id"]
        
        # Optimize for fastest swing to 90 degrees
        result = self.server.call_tool("optimize_parameters", {
            "model_id": model_id,
            "objective": "minimize_time",
            "target_state": {"angle": 90.0},
            "parameters_to_optimize": ["kp", "ki", "kd"],
            "parameter_bounds": {
                "kp": [0.1, 20.0],
                "ki": [0.0, 5.0],
                "kd": [0.0, 10.0]
            },
            "max_iterations": 10
        })
        
        assert result["success"] is True
        assert "optimal_parameters" in result
        assert "kp" in result["optimal_parameters"]
        assert "performance_improvement" in result
        assert result["performance_improvement"] > 0  # Should improve
    
    def test_optimize_energy_efficiency(self):
        """Test optimizing for energy efficiency"""
        # Create pendulum
        result = self.server.call_tool("pendulum_demo", {"action": "setup"})
        model_id = result["model_id"]
        
        # Optimize for energy efficiency
        result = self.server.call_tool("optimize_parameters", {
            "model_id": model_id,
            "objective": "minimize_energy",
            "target_state": {"angle": 45.0},
            "parameters_to_optimize": ["damping"],
            "parameter_bounds": {
                "damping": [0.01, 2.0]
            }
        })
        
        assert result["success"] is True
        assert "optimal_parameters" in result
        assert "energy_before" in result
        assert "energy_after" in result
        assert result["energy_after"] <= result["energy_before"]
    
    def test_optimize_stability(self):
        """Test optimizing for stability"""
        # Create pendulum
        result = self.server.call_tool("pendulum_demo", {"action": "setup"})
        model_id = result["model_id"]
        
        # Optimize for stability at upright position
        result = self.server.call_tool("optimize_parameters", {
            "model_id": model_id,
            "objective": "maximize_stability",
            "target_state": {"angle": 0.0},
            "parameters_to_optimize": ["kp", "kd"],
            "parameter_bounds": {
                "kp": [5.0, 50.0],
                "kd": [1.0, 20.0]
            }
        })
        
        assert result["success"] is True
        assert "stability_score" in result
        assert 0 <= result["stability_score"] <= 1  # Should be normalized score
    
    def test_multi_objective_optimization(self):
        """Test multi-objective optimization"""
        # Create pendulum
        result = self.server.call_tool("pendulum_demo", {"action": "setup"})
        model_id = result["model_id"]
        
        # Optimize for both speed and energy
        result = self.server.call_tool("optimize_parameters", {
            "model_id": model_id,
            "objective": "multi_objective",
            "objectives": [
                {"type": "minimize_time", "weight": 0.5},
                {"type": "minimize_energy", "weight": 0.5}
            ],
            "target_state": {"angle": 90.0},
            "parameters_to_optimize": ["kp", "ki", "kd"],
            "parameter_bounds": {
                "kp": [0.1, 20.0],
                "ki": [0.0, 5.0],
                "kd": [0.0, 10.0]
            }
        })
        
        assert result["success"] is True
        assert "pareto_front" in result
        assert len(result["pareto_front"]) > 0
    
    def test_gradient_free_optimization(self):
        """Test gradient-free optimization methods"""
        # Create pendulum
        result = self.server.call_tool("pendulum_demo", {"action": "setup"})
        model_id = result["model_id"]
        
        # Test different optimization methods
        methods = ["random_search", "grid_search", "bayesian"]
        
        for method in methods[:1]:  # Test at least one method
            result = self.server.call_tool("optimize_parameters", {
                "model_id": model_id,
                "objective": "minimize_error",
                "target_state": {"angle": 45.0},
                "parameters_to_optimize": ["kp"],
                "parameter_bounds": {"kp": [1.0, 10.0]},
                "optimization_method": method,
                "max_iterations": 5
            })
            
            assert result["success"] is True
            assert result["method_used"] == method
    
    def test_constraint_satisfaction(self):
        """Test optimization with constraints"""
        # Create pendulum
        result = self.server.call_tool("pendulum_demo", {"action": "setup"})
        model_id = result["model_id"]
        
        # Optimize with constraints
        result = self.server.call_tool("optimize_parameters", {
            "model_id": model_id,
            "objective": "minimize_time",
            "target_state": {"angle": 90.0},
            "parameters_to_optimize": ["kp", "kd"],
            "parameter_bounds": {
                "kp": [0.1, 50.0],
                "kd": [0.0, 20.0]
            },
            "constraints": [
                {"type": "max_overshoot", "value": 10.0},  # Max 10 degree overshoot
                {"type": "max_control_effort", "value": 1.5}  # Max control torque
            ]
        })
        
        assert result["success"] is True
        assert "constraints_satisfied" in result
        assert result["constraints_satisfied"] is True
    
    def test_optimization_convergence(self):
        """Test optimization convergence tracking"""
        # Create pendulum
        result = self.server.call_tool("pendulum_demo", {"action": "setup"})
        model_id = result["model_id"]
        
        # Run optimization with convergence tracking
        result = self.server.call_tool("optimize_parameters", {
            "model_id": model_id,
            "objective": "minimize_error",
            "target_state": {"angle": 45.0},
            "parameters_to_optimize": ["kp"],
            "parameter_bounds": {"kp": [0.1, 20.0]},
            "track_convergence": True,
            "max_iterations": 10
        })
        
        assert result["success"] is True
        assert "convergence_history" in result
        assert len(result["convergence_history"]) > 0
        
        # Check that error decreases
        history = result["convergence_history"]
        assert history[-1]["error"] <= history[0]["error"]
    
    def test_save_optimization_results(self):
        """Test saving optimization results"""
        # Create pendulum
        result = self.server.call_tool("pendulum_demo", {"action": "setup"})
        model_id = result["model_id"]
        
        # Optimize and save
        result = self.server.call_tool("optimize_parameters", {
            "model_id": model_id,
            "objective": "minimize_time",
            "target_state": {"angle": 90.0},
            "parameters_to_optimize": ["kp"],
            "parameter_bounds": {"kp": [1.0, 10.0]},
            "save_results": True,
            "results_name": "optimal_pendulum_params"
        })
        
        assert result["success"] is True
        assert "results_id" in result
        
        # Verify we can retrieve saved results
        assert "saved_results" in result
        assert result["saved_results"]["name"] == "optimal_pendulum_params"


class TestCostFunctions:
    """Test various cost functions"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.server = MuJoCoMCPServer()
    
    def test_list_cost_functions(self):
        """Test listing available cost functions"""
        result = self.server.call_tool("list_cost_functions", {})
        
        assert "cost_functions" in result
        assert len(result["cost_functions"]) > 0
        
        # Check structure
        func = result["cost_functions"][0]
        assert "name" in func
        assert "description" in func
        assert "parameters" in func
    
    def test_custom_cost_function(self):
        """Test using custom cost function"""
        # Create pendulum
        result = self.server.call_tool("pendulum_demo", {"action": "setup"})
        model_id = result["model_id"]
        
        # Define custom cost
        result = self.server.call_tool("optimize_parameters", {
            "model_id": model_id,
            "objective": "custom",
            "custom_cost": {
                "type": "weighted_sum",
                "components": [
                    {"type": "position_error", "weight": 1.0},
                    {"type": "control_effort", "weight": 0.1},
                    {"type": "jerk", "weight": 0.01}
                ]
            },
            "target_state": {"angle": 45.0},
            "parameters_to_optimize": ["kp", "kd"],
            "parameter_bounds": {
                "kp": [1.0, 20.0],
                "kd": [0.1, 5.0]
            }
        })
        
        assert result["success"] is True
        assert "cost_components" in result
        assert len(result["cost_components"]) == 3


class TestOptimizationAnalysis:
    """Test optimization analysis tools"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.server = MuJoCoMCPServer()
    
    def test_sensitivity_analysis(self):
        """Test parameter sensitivity analysis"""
        # Create pendulum
        result = self.server.call_tool("pendulum_demo", {"action": "setup"})
        model_id = result["model_id"]
        
        # Run sensitivity analysis
        result = self.server.call_tool("analyze_sensitivity", {
            "model_id": model_id,
            "parameters": ["kp", "ki", "kd"],
            "objective": "position_error",
            "target_state": {"angle": 45.0},
            "num_samples": 10
        })
        
        assert result["success"] is True
        assert "sensitivity" in result
        assert "kp" in result["sensitivity"]
        
        # Most sensitive parameter should have highest value
        sensitivities = result["sensitivity"]
        assert max(sensitivities.values()) > 0
    
    def test_robustness_analysis(self):
        """Test robustness of optimized parameters"""
        # Create pendulum
        result = self.server.call_tool("pendulum_demo", {"action": "setup"})
        model_id = result["model_id"]
        
        # First optimize
        opt_result = self.server.call_tool("optimize_parameters", {
            "model_id": model_id,
            "objective": "minimize_error",
            "target_state": {"angle": 45.0},
            "parameters_to_optimize": ["kp"],
            "parameter_bounds": {"kp": [1.0, 10.0]}
        })
        
        optimal_kp = opt_result["optimal_parameters"]["kp"]
        
        # Test robustness
        result = self.server.call_tool("analyze_robustness", {
            "model_id": model_id,
            "parameters": {"kp": optimal_kp},
            "perturbation_range": 0.1,  # ±10% perturbation
            "num_tests": 10
        })
        
        assert result["success"] is True
        assert "robustness_score" in result
        assert 0 <= result["robustness_score"] <= 1


class TestServerVersion041:
    """Test server version update"""
    
    def test_version_updated(self):
        """Test that server version is 0.4.1"""
        server = MuJoCoMCPServer()
        assert server.version == "0.4.1"
        
        info = server.get_server_info()
        assert info["version"] == "0.4.1"
    
    def test_optimization_capability(self):
        """Test that optimization capability is advertised"""
        server = MuJoCoMCPServer()
        info = server.get_server_info()
        
        assert "capabilities" in info
        assert "parameter_optimization" in info["capabilities"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
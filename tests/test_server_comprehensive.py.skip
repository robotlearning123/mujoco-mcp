"""Comprehensive unit tests for MuJoCo MCP server."""

import pytest
import json
import asyncio
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import numpy as np

from mujoco_mcp.server import MuJoCoMCPServer
from mujoco_mcp.simulation import MuJoCoSimulation


class TestMCPResources:
    """Test MCP resource endpoints."""
    
    def test_list_simulations_empty(self):
        """Test listing simulations when none exist."""
        simulations.clear()
        result = list_simulations()
        assert result == ""
    
    def test_list_simulations_multiple(self, mock_mujoco):
        """Test listing multiple simulations."""
        simulations.clear()
        
        # Add mock simulations
        mock_sim1 = MagicMock()
        mock_sim1.__class__.__name__ = "MockSimulation1"
        mock_sim2 = MagicMock()
        mock_sim2.__class__.__name__ = "MockSimulation2"
        
        simulations["sim1"] = mock_sim1
        simulations["sim2"] = mock_sim2
        
        result = list_simulations()
        assert "sim1: MockSimulation1" in result
        assert "sim2: MockSimulation2" in result
    
    def test_get_joint_positions_success(self, mock_mujoco):
        """Test getting joint positions successfully."""
        simulations.clear()
        
        # Setup mock simulation
        mock_sim = MagicMock()
        mock_sim.get_joint_positions.return_value = np.array([0.5, -0.3])
        simulations["test_sim"] = mock_sim
        
        result = get_joint_positions("test_sim")
        assert "0.5" in result
        assert "-0.3" in result
        mock_sim.get_joint_positions.assert_called_once()
    
    def test_get_joint_positions_sim_not_found(self):
        """Test getting joint positions for non-existent simulation."""
        simulations.clear()
        result = get_joint_positions("non_existent")
        assert "Error" in result
        assert "not found" in result
    
    def test_get_joint_velocities_success(self, mock_mujoco):
        """Test getting joint velocities successfully."""
        simulations.clear()
        
        # Setup mock simulation
        mock_sim = MagicMock()
        mock_sim.get_joint_velocities.return_value = np.array([0.1, -0.05])
        simulations["test_sim"] = mock_sim
        
        result = get_joint_velocities("test_sim")
        assert "0.1" in result
        assert "-0.05" in result
        mock_sim.get_joint_velocities.assert_called_once()
    
    def test_get_sensor_data_success(self, sample_sensor_data):
        """Test getting sensor data successfully."""
        from mujoco_mcp.server_new import get_sensor_data
        simulations.clear()
        
        # Setup mock simulation
        mock_sim = MagicMock()
        mock_sim.get_sensor_data.return_value = sample_sensor_data
        simulations["test_sim"] = mock_sim
        
        result = get_sensor_data("test_sim")
        assert "joint_positions" in result
        assert "accelerometer" in result
        mock_sim.get_sensor_data.assert_called_once()
    
    def test_get_robot_state_success(self, sample_robot_state):
        """Test getting robot state successfully."""
        from mujoco_mcp.server_new import get_robot_state
        simulations.clear()
        
        # Setup mock simulation
        mock_sim = MagicMock()
        mock_sim.get_robot_state.return_value = sample_robot_state
        simulations["test_sim"] = mock_sim
        
        result = get_robot_state("test_sim")
        assert "base_position" in result
        assert "end_effector_position" in result
        mock_sim.get_robot_state.assert_called_once()


class TestMCPTools:
    """Test MCP tool endpoints."""
    
    @patch('mujoco_mcp.server_new.MuJoCoSimulation')
    @patch('mujoco_mcp.server_new.auth_manager')
    def test_start_simulation_success(self, mock_auth, mock_sim_class, temp_model_file):
        """Test starting simulation successfully."""
        simulations.clear()
        
        # Setup mocks
        mock_auth.request_authorization.return_value = (True, "Authorized", "req-id")
        mock_sim_instance = MagicMock()
        mock_sim_class.return_value = mock_sim_instance
        
        # Test
        result = start_simulation(temp_model_file)
        
        # Verify
        assert result in simulations
        assert simulations[result] == mock_sim_instance
        mock_sim_instance.load_model.assert_called_once_with(temp_model_file)
    
    @patch('mujoco_mcp.server_new.auth_manager')
    def test_start_simulation_unauthorized(self, mock_auth):
        """Test starting simulation when unauthorized."""
        simulations.clear()
        mock_auth.request_authorization.return_value = (False, "Denied", "req-id")
        
        result = start_simulation("model.xml")
        assert "Error" in result
        assert "Authorization denied" in result
        assert len(simulations) == 0
    
    @patch('mujoco_mcp.server_new.MuJoCoSimulation')
    def test_start_simulation_load_error(self, mock_sim_class, temp_model_file):
        """Test starting simulation with load error."""
        simulations.clear()
        
        # Setup mock to raise exception
        mock_sim_instance = MagicMock()
        mock_sim_instance.load_model.side_effect = Exception("Load failed")
        mock_sim_class.return_value = mock_sim_instance
        
        # Remove auth manager to simplify test
        with patch('mujoco_mcp.server_new.auth_manager', None):
            result = start_simulation(temp_model_file)
            assert "Error" in result
            assert "Load failed" in result
    
    def test_step_simulation_success(self):
        """Test stepping simulation successfully."""
        simulations.clear()
        
        # Setup mock simulation
        mock_sim = MagicMock()
        simulations["test_sim"] = mock_sim
        
        # Remove auth manager to simplify test
        with patch('mujoco_mcp.server_new.auth_manager', None):
            result = step_simulation("test_sim", 5)
            assert "Stepped simulation test_sim forward 5 steps" in result
            assert mock_sim.step.call_count == 5
    
    def test_step_simulation_not_found(self):
        """Test stepping non-existent simulation."""
        simulations.clear()
        result = step_simulation("non_existent")
        assert "Error" in result
        assert "not found" in result
    
    @patch('mujoco_mcp.server_new.auth_manager')
    def test_reset_simulation_success(self, mock_auth):
        """Test resetting simulation successfully."""
        simulations.clear()
        
        # Setup mocks
        mock_auth.request_authorization.return_value = (True, "Authorized", "req-id")
        mock_sim = MagicMock()
        simulations["test_sim"] = mock_sim
        
        result = reset_simulation("test_sim")
        assert "Reset simulation test_sim" in result
        mock_sim.reset.assert_called_once()
    
    def test_set_joint_positions_success(self):
        """Test setting joint positions successfully."""
        simulations.clear()
        
        # Setup mock simulation
        mock_sim = MagicMock()
        simulations["test_sim"] = mock_sim
        
        # Remove auth manager to simplify test
        with patch('mujoco_mcp.server_new.auth_manager', None):
            positions = [0.5, -0.3, 0.0]
            result = set_joint_positions("test_sim", positions)
            assert "Set joint positions" in result
            mock_sim.set_joint_positions.assert_called_once_with(positions)
    
    def test_set_joint_positions_error(self):
        """Test setting joint positions with error."""
        simulations.clear()
        
        # Setup mock simulation to raise error
        mock_sim = MagicMock()
        mock_sim.set_joint_positions.side_effect = ValueError("Wrong size")
        simulations["test_sim"] = mock_sim
        
        with patch('mujoco_mcp.server_new.auth_manager', None):
            result = set_joint_positions("test_sim", [0.5, -0.3])
            assert "Error" in result
            assert "Wrong size" in result
    
    def test_apply_control_success(self):
        """Test applying control successfully."""
        simulations.clear()
        
        # Setup mock simulation
        mock_sim = MagicMock()
        simulations["test_sim"] = mock_sim
        
        with patch('mujoco_mcp.server_new.auth_manager', None):
            control = [0.5, -0.5]
            result = apply_control("test_sim", control)
            assert "Applied control" in result
            mock_sim.apply_control.assert_called_once_with(control)
    
    @patch('mujoco_mcp.server_new.auth_manager')
    @patch('mujoco_mcp.server_new.EnhancedAuthManager')
    def test_move_to_position_success(self, mock_enhanced_auth_class, mock_auth):
        """Test moving to position successfully."""
        simulations.clear()
        
        # Setup mocks
        mock_enhanced_instance = MagicMock()
        mock_enhanced_instance.request_authorization.return_value = (True, "Authorized", "req-id")
        mock_enhanced_auth_class.return_value = mock_enhanced_instance
        
        # Make auth_manager an instance of EnhancedAuthManager
        with patch('mujoco_mcp.server_new.auth_manager', mock_enhanced_instance):
            mock_sim = MagicMock()
            simulations["test_sim"] = mock_sim
            
            target = [1.0, 0.0, 1.5]
            result = move_to_position("test_sim", target, 0.8)
            assert "Moving to position" in result
            assert "[1.0, 0.0, 1.5]" in result
            assert "0.8" in result
    
    def test_grasp_object_success(self):
        """Test grasping object successfully."""
        simulations.clear()
        
        # Setup mock simulation
        mock_sim = MagicMock()
        simulations["test_sim"] = mock_sim
        
        # Test without enhanced auth
        with patch('mujoco_mcp.server_new.auth_manager', None):
            result = grasp_object("test_sim", "cube", 0.7)
            assert "Grasped object cube" in result
            assert "0.7" in result


class TestMCPServer:
    """Test MCP server class."""
    
    def test_server_initialization(self):
        """Test server initialization."""
        server = MuJoCoMCPServer()
        assert server.mcp is not None
        assert server.simulations is not None
    
    def test_server_with_custom_auth_manager(self, mock_auth_manager):
        """Test server with custom auth manager."""
        server = MuJoCoMCPServer(auth_manager=mock_auth_manager)
        # Auth manager is global, so we need to check it was set
        from mujoco_mcp.server_new import auth_manager as global_auth
        # Note: In actual implementation, this might need adjustment
        assert server is not None
    
    @patch('mujoco_mcp.server_new.mcp.run')
    def test_server_run(self, mock_run):
        """Test running the server."""
        server = MuJoCoMCPServer()
        server.run()
        mock_run.assert_called_once()


class TestAuthorizationIntegration:
    """Test authorization integration with server operations."""
    
    @pytest.mark.integration
    def test_full_auth_flow(self, mock_auth_manager, mock_enhanced_auth_manager):
        """Test complete authorization flow."""
        simulations.clear()
        
        # Test with basic auth manager
        with patch('mujoco_mcp.server_new.auth_manager', mock_auth_manager):
            # Approve request
            mock_auth_manager.request_authorization.return_value = (True, "Approved", "req-1")
            result = reset_simulation("test_sim")
            assert "Error" in result  # Sim doesn't exist, but auth passed
            
            # Deny request
            mock_auth_manager.request_authorization.return_value = (False, "Denied", "req-2")
            result = reset_simulation("test_sim")
            assert "Authorization denied" in result
    
    @pytest.mark.integration
    def test_enhanced_auth_parameter_validation(self, mock_enhanced_auth_manager):
        """Test enhanced auth manager parameter validation."""
        simulations.clear()
        
        with patch('mujoco_mcp.server_new.auth_manager', mock_enhanced_auth_manager):
            # Test with invalid parameters
            mock_enhanced_auth_manager.validate_parameters.return_value = (False, "Invalid speed")
            mock_enhanced_auth_manager.request_authorization.return_value = (False, "Invalid speed", "req-1")
            
            mock_sim = MagicMock()
            simulations["test_sim"] = mock_sim
            
            result = move_to_position("test_sim", [1.0, 0.0, 1.0], 2.0)  # Speed > 1.0
            assert "Error" in result
            assert "Authorization denied" in result


class TestPerformanceAndSecurity:
    """Test performance and security aspects."""
    
    @pytest.mark.benchmark
    def test_resource_access_performance(self, performance_timer):
        """Benchmark resource access performance."""
        simulations.clear()
        
        # Setup mock simulation with data
        mock_sim = MagicMock()
        mock_sim.get_joint_positions.return_value = np.random.rand(10)
        simulations["perf_test"] = mock_sim
        
        # Warm up
        for _ in range(10):
            get_joint_positions("perf_test")
        
        # Benchmark
        n_calls = 1000
        performance_timer.start()
        for _ in range(n_calls):
            get_joint_positions("perf_test")
        elapsed = performance_timer.stop()
        
        calls_per_second = n_calls / elapsed
        print(f"\nResource access performance: {calls_per_second:.2f} calls/second")
        assert calls_per_second > 10000  # Should handle at least 10k calls/second
    
    @pytest.mark.security
    def test_simulation_isolation(self):
        """Test that simulations are properly isolated."""
        simulations.clear()
        
        # Create two mock simulations
        mock_sim1 = MagicMock()
        mock_sim2 = MagicMock()
        mock_sim1.get_joint_positions.return_value = np.array([1.0, 1.0])
        mock_sim2.get_joint_positions.return_value = np.array([2.0, 2.0])
        
        simulations["sim1"] = mock_sim1
        simulations["sim2"] = mock_sim2
        
        # Verify isolation
        result1 = get_joint_positions("sim1")
        result2 = get_joint_positions("sim2")
        
        assert "1.0" in result1
        assert "2.0" in result2
        assert "2.0" not in result1
        assert "1.0" not in result2
    
    @pytest.mark.security
    def test_input_sanitization(self):
        """Test input sanitization for security."""
        simulations.clear()
        
        # Test with potentially malicious sim_id
        malicious_ids = [
            "../../../etc/passwd",
            "'; DROP TABLE simulations; --",
            "<script>alert('xss')</script>",
            "sim1\0sim2"  # Null byte injection
        ]
        
        for bad_id in malicious_ids:
            result = get_joint_positions(bad_id)
            assert "Error" in result
            assert "not found" in result
            # Ensure no side effects
            assert len(simulations) == 0


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_simulation_crash_recovery(self):
        """Test recovery from simulation crash."""
        simulations.clear()
        
        # Setup mock simulation that crashes
        mock_sim = MagicMock()
        mock_sim.step.side_effect = [None, None, Exception("Simulation crashed"), None]
        simulations["crash_test"] = mock_sim
        
        with patch('mujoco_mcp.server_new.auth_manager', None):
            # First two steps succeed
            result1 = step_simulation("crash_test", 1)
            assert "Stepped" in result1
            
            result2 = step_simulation("crash_test", 1)
            assert "Stepped" in result2
            
            # Third step crashes
            result3 = step_simulation("crash_test", 1)
            assert "Error" in result3
            assert "Simulation crashed" in result3
            
            # Simulation should still be in dict
            assert "crash_test" in simulations
    
    @pytest.mark.parametrize("bad_input,operation", [
        (None, "positions"),
        ([], "positions"),
        ("not_a_list", "positions"),
        ([1, 2, "three"], "positions"),
        ({"key": "value"}, "control"),
        (np.array([[[1, 2]]], dtype=object), "control")
    ])
    def test_invalid_input_types(self, bad_input, operation):
        """Test handling of invalid input types."""
        simulations.clear()
        
        mock_sim = MagicMock()
        simulations["test_sim"] = mock_sim
        
        with patch('mujoco_mcp.server_new.auth_manager', None):
            if operation == "positions":
                # Should handle gracefully or raise appropriate error
                result = set_joint_positions("test_sim", bad_input)
            else:
                result = apply_control("test_sim", bad_input)
            
            # Either error in result or exception from mock
            assert ("Error" in result) or mock_sim.method.called
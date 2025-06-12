"""Comprehensive unit tests for MuJoCo simulation module."""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import tempfile
import os

from mujoco_mcp.simulation import MuJoCoSimulation


class TestMuJoCoSimulation:
    """Test cases for basic MuJoCo simulation functionality."""
    
    def test_initialization(self):
        """Test simulation initialization."""
        sim = MuJoCoSimulation()
        assert sim is not None
        assert hasattr(sim, 'model')
        assert hasattr(sim, 'data')
    
    @patch('mujoco_mcp.simulation.mujoco')
    def test_load_model_success(self, mock_mujoco, temp_model_file):
        """Test successful model loading."""
        # Setup mock
        mock_model = MagicMock()
        mock_data = MagicMock()
        mock_mujoco.MjModel.from_xml_path.return_value = mock_model
        mock_mujoco.MjData.return_value = mock_data
        
        # Test
        sim = MuJoCoSimulation()
        sim.load_from_file(temp_model_file)
        
        # Verify
        mock_mujoco.MjModel.from_xml_path.assert_called_once_with(temp_model_file)
        mock_mujoco.MjData.assert_called_once_with(mock_model)
        assert sim.model == mock_model
        assert sim.data == mock_data
    
    @patch('mujoco_mcp.simulation.mujoco')
    def test_load_model_file_not_found(self, mock_mujoco):
        """Test model loading with non-existent file."""
        mock_mujoco.MjModel.from_xml_path.side_effect = FileNotFoundError()
        
        sim = MuJoCoSimulation()
        with pytest.raises(FileNotFoundError):
            sim.load_from_file("non_existent_file.xml")
    
    @patch('mujoco_mcp.simulation.mujoco')
    def test_load_model_invalid_xml(self, mock_mujoco):
        """Test model loading with invalid XML."""
        mock_mujoco.MjModel.from_xml_path.side_effect = Exception("Invalid XML")
        
        sim = MuJoCoSimulation()
        with pytest.raises(Exception) as exc_info:
            sim.load_from_file("invalid.xml")
        assert "Invalid XML" in str(exc_info.value)
    
    @patch('mujoco_mcp.simulation.mujoco')
    def test_step_simulation(self, mock_mujoco):
        """Test simulation stepping."""
        # Setup
        mock_model = MagicMock()
        mock_data = MagicMock()
        sim = MuJoCoSimulation()
        sim.model = mock_model
        sim.data = mock_data
        sim._initialized = True
        
        # Test single step
        sim.step()
        mock_mujoco.mj_step.assert_called_once_with(mock_model, mock_data)
        
        # Test multiple steps
        mock_mujoco.mj_step.reset_mock()
        sim.step(5)
        assert mock_mujoco.mj_step.call_count == 5
    
    @patch('mujoco_mcp.simulation.mujoco')
    def test_reset_simulation(self, mock_mujoco):
        """Test simulation reset."""
        # Setup
        mock_model = MagicMock()
        mock_data = MagicMock()
        sim = MuJoCoSimulation()
        sim.model = mock_model
        sim.data = mock_data
        sim._initialized = True
        
        # Test
        sim.reset()
        mock_mujoco.mj_resetData.assert_called_once_with(mock_model, mock_data)
    
    def test_get_joint_positions(self, mock_mujoco):
        """Test getting joint positions."""
        _, mock_model, mock_data = mock_mujoco
        mock_data.qpos = np.array([0.5, -0.3, 0.0])
        
        sim = MuJoCoSimulation()
        sim.model = mock_model
        sim.data = mock_data
        sim._initialized = True
        
        positions = sim.get_joint_positions()
        np.testing.assert_array_equal(positions, np.array([0.5, -0.3, 0.0]))
    
    @patch('mujoco_mcp.simulation.mujoco')
    def test_set_joint_positions(self, mock_mujoco_module, mock_mujoco):
        """Test setting joint positions."""
        _, mock_model, mock_data = mock_mujoco
        mock_data.qpos = np.zeros(3)
        
        sim = MuJoCoSimulation()
        sim.model = mock_model
        sim.data = mock_data
        sim._initialized = True
        
        new_positions = np.array([1.0, -0.5, 0.2])
        sim.set_joint_positions(new_positions)
        np.testing.assert_array_equal(mock_data.qpos, new_positions)
    
    def test_set_joint_positions_wrong_size(self, mock_mujoco):
        """Test setting joint positions with wrong array size."""
        _, mock_model, mock_data = mock_mujoco
        mock_model.nq = 3
        mock_data.qpos = np.zeros(3)
        
        sim = MuJoCoSimulation()
        sim.model = mock_model
        sim.data = mock_data
        sim._initialized = True
        
        with pytest.raises(ValueError) as exc_info:
            sim.set_joint_positions(np.array([1.0, -0.5]))  # Wrong size
        assert "could not broadcast" in str(exc_info.value)
    
    def test_get_joint_velocities(self, mock_mujoco):
        """Test getting joint velocities."""
        _, mock_model, mock_data = mock_mujoco
        mock_data.qvel = np.array([0.1, -0.05, 0.0])
        
        sim = MuJoCoSimulation()
        sim.model = mock_model
        sim.data = mock_data
        sim._initialized = True
        
        velocities = sim.get_joint_velocities()
        np.testing.assert_array_equal(velocities, np.array([0.1, -0.05, 0.0]))
    
    def test_apply_control(self, mock_mujoco):
        """Test applying control inputs."""
        _, mock_model, mock_data = mock_mujoco
        mock_model.nu = 2
        mock_data.ctrl = np.zeros(2)
        
        sim = MuJoCoSimulation()
        sim.model = mock_model
        sim.data = mock_data
        sim._initialized = True
        
        control = np.array([0.5, -0.5])
        sim.apply_control(control)
        np.testing.assert_array_equal(mock_data.ctrl, control)
    
    def test_apply_control_wrong_size(self, mock_mujoco):
        """Test applying control with wrong array size."""
        _, mock_model, mock_data = mock_mujoco
        mock_model.nu = 2
        mock_data.ctrl = np.zeros(2)
        
        sim = MuJoCoSimulation()
        sim.model = mock_model
        sim.data = mock_data
        sim._initialized = True
        
        with pytest.raises(ValueError) as exc_info:
            sim.apply_control(np.array([0.5, -0.5, 0.0]))  # Wrong size
        assert "could not broadcast" in str(exc_info.value)
    
    def test_get_sensor_data(self, mock_mujoco):
        """Test getting sensor data."""
        _, mock_model, mock_data = mock_mujoco
        mock_data.sensordata = np.array([0.1, 0.2, 0.3, 0.4])
        
        # Mock sensors
        mock_sensor = MagicMock()
        mock_sensor.name = "sensor_0"
        mock_model.sensor.return_value = mock_sensor
        mock_model.nsensor = 1
        
        sim = MuJoCoSimulation()
        sim.model = mock_model
        sim.data = mock_data
        sim._initialized = True
        
        sensor_data = sim.get_sensor_data()
        assert isinstance(sensor_data, dict)
        assert 'sensor_0' in sensor_data
        assert sensor_data['sensor_0'] == [0.1]


# # NOTE: EnhancedMuJoCoSimulation tests commented out - class not implemented in v0.6.0
# # class TestEnhancedMuJoCoSimulation:
#     """Test cases for enhanced MuJoCo simulation functionality."""
#     
#     @patch('mujoco_mcp.simulation.mujoco')
#     def test_get_body_positions(self, mock_mujoco):
#         """Test getting body positions."""
#         _, mock_model, mock_data = mock_mujoco
#         mock_data.xpos = np.array([
#             [0.0, 0.0, 1.0],
#             [0.0, 0.0, 1.2],
#             [0.5, 0.0, 0.5]
#         ])
#         
#         sim = EnhancedMuJoCoSimulation()
#         sim.model = mock_model
        sim._initialized = True#         sim.data = mock_data
#         
#         positions = sim.get_body_positions()
#         assert positions.shape == (3, 3)
#         np.testing.assert_array_equal(positions[0], [0.0, 0.0, 1.0])
#     
#     @patch('mujoco_mcp.simulation.mujoco')
#     def test_get_body_orientations(self, mock_mujoco):
#         """Test getting body orientations."""
#         _, mock_model, mock_data = mock_mujoco
#         mock_data.xquat = np.array([
#             [1.0, 0.0, 0.0, 0.0],
#             [0.707, 0.707, 0.0, 0.0],
#             [1.0, 0.0, 0.0, 0.0]
#         ])
#         
#         sim = EnhancedMuJoCoSimulation()
#         sim.model = mock_model
        sim._initialized = True#         sim.data = mock_data
#         
#         orientations = sim.get_body_orientations()
#         assert orientations.shape == (3, 4)
#         np.testing.assert_array_almost_equal(orientations[0], [1.0, 0.0, 0.0, 0.0])
#     
#     @patch('mujoco_mcp.simulation.mujoco')
#     def test_get_contact_forces(self, mock_mujoco):
#         """Test getting contact forces."""
#         mock_mj, mock_model, mock_data = mock_mujoco
#         
#         # Mock contact data
#         mock_contact = MagicMock()
#         mock_contact.geom1 = 0
#         mock_contact.geom2 = 1
#         mock_data.contact = [mock_contact]
#         mock_data.ncon = 1
#         
#         # Mock force computation
#         mock_force = np.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0])
#         mock_mj.mj_contactForce.return_value = None  # Modifies array in-place
#         
#         sim = EnhancedMuJoCoSimulation()
#         sim.model = mock_model
        sim._initialized = True#         sim.data = mock_data
#         
#         # Patch the force array modification
#         with patch('numpy.zeros', return_value=mock_force):
#             forces = sim.get_contact_forces()
#             assert len(forces) == 1
#             assert forces[0]['geom1'] == 0
#             assert forces[0]['geom2'] == 1
#             assert 'force' in forces[0]
#     
#     def test_get_robot_state(self, mock_mujoco):
#         """Test getting complete robot state."""
#         _, mock_model, mock_data = mock_mujoco
#         
#         sim = EnhancedMuJoCoSimulation()
#         sim.model = mock_model
        sim._initialized = True#         sim.data = mock_data
#         
#         state = sim.get_robot_state()
#         assert isinstance(state, dict)
#         assert 'joint_positions' in state
#         assert 'joint_velocities' in state
#         assert 'body_positions' in state
#         assert 'body_orientations' in state
#         assert 'timestamp' in state
#     
#     @patch('mujoco_mcp.simulation.mujoco')
#     def test_apply_external_force(self, mock_mujoco):
#         """Test applying external force to a body."""
#         mock_mj, mock_model, mock_data = mock_mujoco
#         mock_model.body_name2id.return_value = 1
#         mock_data.xfrc_applied = np.zeros((3, 6))
#         
#         sim = EnhancedMuJoCoSimulation()
#         sim.model = mock_model
        sim._initialized = True#         sim.data = mock_data
#         
#         force = np.array([10.0, 0.0, 0.0])
#         torque = np.array([0.0, 1.0, 0.0])
#         sim.apply_external_force("link1", force, torque)
#         
#         np.testing.assert_array_equal(mock_data.xfrc_applied[1, :3], force)
#         np.testing.assert_array_equal(mock_data.xfrc_applied[1, 3:], torque)
#     
#     def test_apply_external_force_invalid_body(self, mock_mujoco):
#         """Test applying force to non-existent body."""
#         _, mock_model, mock_data = mock_mujoco
#         mock_model.body_name2id.side_effect = KeyError("Body not found")
#         
#         sim = EnhancedMuJoCoSimulation()
#         sim.model = mock_model
        sim._initialized = True#         sim.data = mock_data
#         
#         with pytest.raises(ValueError) as exc_info:
#             sim.apply_external_force("invalid_body", np.zeros(3))
#         assert "Body 'invalid_body' not found" in str(exc_info.value)
#     
#     @pytest.mark.parametrize("position,expected_valid", [
#         ([0.0, 0.0, 1.0], True),
#         ([10.0, 10.0, 10.0], False),  # Out of workspace
#         ([-5.0, -5.0, -5.0], False),  # Out of workspace
#     ])
#     def test_is_position_reachable(self, position, expected_valid, mock_mujoco):
#         """Test checking if position is reachable."""
#         _, mock_model, mock_data = mock_mujoco
#         
#         sim = EnhancedMuJoCoSimulation()
#         sim.model = mock_model
        sim._initialized = True#         sim.data = mock_data
#         
#         # Mock workspace bounds
#         sim.workspace_bounds = {
#             'x': [-2.0, 2.0],
#             'y': [-2.0, 2.0],
#             'z': [0.0, 3.0]
#         }
#         
#         is_reachable = sim.is_position_reachable(position)
#         assert is_reachable == expected_valid
#     
#     @pytest.mark.slow
#     def test_performance_step_simulation(self, mock_mujoco, performance_timer):
#         """Benchmark simulation stepping performance."""
#         _, mock_model, mock_data = mock_mujoco
#         
#         sim = EnhancedMuJoCoSimulation()
#         sim.model = mock_model
        sim._initialized = True#         sim.data = mock_data
#         
#         # Warm up
#         for _ in range(10):
#             sim.step()
#         
#         # Benchmark
#         n_steps = 1000
#         performance_timer.start()
#         for _ in range(n_steps):
#             sim.step()
#         elapsed = performance_timer.stop()
#         
#         steps_per_second = n_steps / elapsed
#         print(f"\nPerformance: {steps_per_second:.2f} steps/second")
#         
#         # Assert reasonable performance (adjust threshold as needed)
#         assert steps_per_second > 1000  # At least 1000 steps/second
# 
# 
# class TestSimulationIntegration:
#     """Integration tests for simulation module."""
#     
#     @pytest.mark.integration
#     @patch('mujoco_mcp.simulation.mujoco')
#     def test_full_simulation_cycle(self, mock_mujoco, temp_model_file):
#         """Test complete simulation lifecycle."""
#         mock_mj, mock_model, mock_data = mock_mujoco
#         
#         # Create simulation
#         sim = EnhancedMuJoCoSimulation()
#         
#         # Load model
#         sim.load_from_file(temp_model_file)
#         
#         # Reset
#         sim.reset()
#         
#         # Apply control and step
#         control = np.array([0.5, -0.5])
#         sim.apply_control(control)
#         sim.step(10)
#         
#         # Get state
#         state = sim.get_robot_state()
#         
#         # Verify
#         assert mock_mj.mj_resetData.called
#         assert mock_mj.mj_step.call_count == 10
#         assert isinstance(state, dict)
#     
#     @pytest.mark.integration
#     def test_trajectory_execution(self, mock_mujoco, sample_trajectory):
#         """Test executing a trajectory."""
#         _, mock_model, mock_data = mock_mujoco
#         
#         sim = EnhancedMuJoCoSimulation()
#         sim.model = mock_model
        sim._initialized = True#         sim.data = mock_data
#         
#         # Execute trajectory
#         for waypoint in sample_trajectory['joint_trajectory']:
#             sim.set_joint_positions(np.array(waypoint))
#             sim.step(50)  # 50 steps per waypoint
#         
#         # Verify final position
#         final_positions = sim.get_joint_positions()
#         expected_final = sample_trajectory['joint_trajectory'][-1]
#         np.testing.assert_array_almost_equal(final_positions[:2], expected_final)
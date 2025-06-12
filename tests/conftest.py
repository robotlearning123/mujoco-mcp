"""pytest configuration and fixtures for MuJoCo MCP tests."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock
import numpy as np


@pytest.fixture
def temp_model_file():
    """Create a temporary MuJoCo XML model file for testing."""
    xml_content = """<?xml version="1.0" ?>
<mujoco model="test_robot">
    <option timestep="0.002" gravity="0 0 -9.81"/>
    
    <worldbody>
        <body name="base" pos="0 0 1">
            <joint name="joint1" type="hinge" axis="1 0 0" range="-2 2"/>
            <geom name="base_geom" type="box" size="0.1 0.1 0.1" mass="1"/>
            
            <body name="link1" pos="0 0 0.2">
                <joint name="joint2" type="hinge" axis="0 1 0" range="-2 2"/>
                <geom name="link1_geom" type="cylinder" size="0.05 0.2" mass="0.5"/>
            </body>
        </body>
        
        <body name="object1" pos="0.5 0 0.5">
            <freejoint name="object1_joint"/>
            <geom name="object1_geom" type="sphere" size="0.1" mass="0.1"/>
        </body>
    </worldbody>
    
    <actuator>
        <motor name="motor1" joint="joint1" ctrlrange="-1 1"/>
        <motor name="motor2" joint="joint2" ctrlrange="-1 1"/>
    </actuator>
    
    <sensor>
        <jointpos name="joint1_pos" joint="joint1"/>
        <jointvel name="joint1_vel" joint="joint1"/>
        <jointpos name="joint2_pos" joint="joint2"/>
        <jointvel name="joint2_vel" joint="joint2"/>
    </sensor>
</mujoco>
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(xml_content)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    os.unlink(temp_path)


@pytest.fixture
def mock_mujoco():
    """Mock MuJoCo module for testing without actual MuJoCo installation."""
    mock = MagicMock()
    
    # Mock MjModel
    mock_model = MagicMock()
    mock_model.nq = 2  # Number of joint positions
    mock_model.nv = 2  # Number of joint velocities
    mock_model.nu = 2  # Number of actuators
    mock_model.nsensordata = 4  # Number of sensor data points
    mock_model.nbody = 3  # Number of bodies
    mock_model.njnt = 3  # Number of joints
    
    # Mock MjData
    mock_data = MagicMock()
    mock_data.qpos = np.array([0.0, 0.0])
    mock_data.qvel = np.array([0.0, 0.0])
    mock_data.ctrl = np.array([0.0, 0.0])
    mock_data.sensordata = np.array([0.0, 0.0, 0.0, 0.0])
    mock_data.xpos = np.array([[0, 0, 1], [0, 0, 1.2], [0.5, 0, 0.5]])
    mock_data.xquat = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])
    
    mock.MjModel.from_xml_path.return_value = mock_model
    mock.MjData.return_value = mock_data
    mock.mj_step = MagicMock()
    mock.mj_resetData = MagicMock()
    mock.mj_forward = MagicMock()
    
    return mock, mock_model, mock_data


@pytest.fixture
def mock_auth_manager():
    """Mock authorization manager for testing."""
    manager = MagicMock()
    manager.request_authorization.return_value = (True, "Authorized", "test-request-id")
    manager.approve_request.return_value = True
    manager.reject_request.return_value = True
    return manager


@pytest.fixture
def mock_enhanced_auth_manager():
    """Mock enhanced authorization manager for testing."""
    from mujoco_mcp.enhanced_auth_manager import EnhancedAuthManager
    
    manager = MagicMock(spec=EnhancedAuthManager)
    manager.request_authorization.return_value = (True, "Authorized", "test-request-id")
    manager.validate_parameters.return_value = (True, "Valid")
    manager.check_rate_limit.return_value = (True, "Within limits")
    return manager


@pytest.fixture
def sample_joint_positions():
    """Sample joint position data."""
    return np.array([0.5, -0.3, 0.0, 1.2])


@pytest.fixture
def sample_joint_velocities():
    """Sample joint velocity data."""
    return np.array([0.1, -0.05, 0.0, 0.2])


@pytest.fixture
def sample_control_inputs():
    """Sample control input data."""
    return np.array([0.5, -0.5])


@pytest.fixture
def sample_sensor_data():
    """Sample sensor data."""
    return {
        "joint_positions": [0.5, -0.3],
        "joint_velocities": [0.1, -0.05],
        "contact_forces": [0.0, 0.0, 9.81],
        "accelerometer": [0.0, 0.0, -9.81]
    }


@pytest.fixture
def sample_robot_state():
    """Sample complete robot state."""
    return {
        "joint_positions": [0.5, -0.3],
        "joint_velocities": [0.1, -0.05],
        "base_position": [0.0, 0.0, 1.0],
        "base_orientation": [1.0, 0.0, 0.0, 0.0],
        "end_effector_position": [0.0, 0.0, 1.4],
        "gripper_state": "open"
    }


@pytest.fixture
def sample_trajectory():
    """Sample trajectory data."""
    return {
        "waypoints": [
            {"position": [0.0, 0.0, 1.0], "time": 0.0},
            {"position": [0.5, 0.0, 1.0], "time": 1.0},
            {"position": [0.5, 0.5, 1.0], "time": 2.0},
            {"position": [0.0, 0.5, 1.0], "time": 3.0},
            {"position": [0.0, 0.0, 1.0], "time": 4.0}
        ],
        "joint_trajectory": [
            [0.0, 0.0],
            [0.5, 0.0],
            [0.5, 0.5],
            [0.0, 0.5],
            [0.0, 0.0]
        ]
    }


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests."""
    # Reset any global state
    from mujoco_mcp import server_manager
    if hasattr(server_manager, '_server_instance'):
        server_manager._server_instance = None
    yield


@pytest.fixture
def performance_timer():
    """Simple performance timer for benchmarking."""
    import time
    
    class Timer:
        def __init__(self):
            self.times = []
            self.start_time = None
        
        def start(self):
            self.start_time = time.perf_counter()
        
        def stop(self):
            if self.start_time is not None:
                elapsed = time.perf_counter() - self.start_time
                self.times.append(elapsed)
                self.start_time = None
                return elapsed
            return 0.0
        
        def average(self):
            return sum(self.times) / len(self.times) if self.times else 0.0
        
        def reset(self):
            self.times = []
            self.start_time = None
    
    return Timer()


# Configure pytest
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "benchmark: marks tests as performance benchmarks")
    config.addinivalue_line("markers", "security: marks tests as security-related")
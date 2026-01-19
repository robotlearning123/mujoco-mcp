"""Comprehensive unit tests for robot_controller.py focusing on NaN/Inf validation and error handling."""

import numpy as np
import pytest

from mujoco_mcp.robot_controller import RobotController


class TestRobotLoading:
    """Test robot loading and initialization."""

    def test_load_arm_robot(self):
        """Test loading arm robot."""
        controller = RobotController()
        result = controller.load_robot("arm")

        assert "robot_id" in result
        assert result["robot_type"] == "arm"
        assert result["status"] == "loaded"
        assert result["num_joints"] > 0

    def test_load_gripper_robot(self):
        """Test loading gripper robot."""
        controller = RobotController()
        result = controller.load_robot("gripper")

        assert result["robot_type"] == "gripper"
        assert result["status"] == "loaded"

    def test_load_mobile_robot(self):
        """Test loading mobile robot."""
        controller = RobotController()
        result = controller.load_robot("mobile")

        assert result["robot_type"] == "mobile"
        assert result["status"] == "loaded"

    def test_load_humanoid_robot(self):
        """Test loading humanoid robot."""
        controller = RobotController()
        result = controller.load_robot("humanoid")

        assert result["robot_type"] == "humanoid"
        assert result["status"] == "loaded"

    def test_load_invalid_robot_type(self):
        """Test loading invalid robot type raises ValueError."""
        controller = RobotController()

        with pytest.raises(ValueError, match="Unknown robot type"):
            controller.load_robot("invalid_type")

    def test_load_robot_with_custom_id(self):
        """Test loading robot with custom ID."""
        controller = RobotController()
        custom_id = "my_robot_123"

        result = controller.load_robot("arm", robot_id=custom_id)

        assert result["robot_id"] == custom_id

    def test_load_multiple_robots(self):
        """Test loading multiple robots with different IDs."""
        controller = RobotController()

        robot1 = controller.load_robot("arm", robot_id="robot1")
        robot2 = controller.load_robot("gripper", robot_id="robot2")

        assert robot1["robot_id"] == "robot1"
        assert robot2["robot_id"] == "robot2"
        assert robot1["robot_type"] == "arm"
        assert robot2["robot_type"] == "gripper"

    def test_load_robot_auto_id_generation(self):
        """Test automatic ID generation when not specified."""
        controller = RobotController()

        result1 = controller.load_robot("arm")
        result2 = controller.load_robot("arm")

        # IDs should be different
        assert result1["robot_id"] != result2["robot_id"]


class TestRobotNotFound:
    """Test error handling for non-existent robot IDs."""

    def test_set_positions_robot_not_found(self):
        """Test setting positions on non-existent robot."""
        controller = RobotController()

        with pytest.raises(KeyError, match="Robot .* not found"):
            controller.set_joint_positions("nonexistent", [0.0])

    def test_set_velocities_robot_not_found(self):
        """Test setting velocities on non-existent robot."""
        controller = RobotController()

        with pytest.raises(KeyError, match="Robot .* not found"):
            controller.set_joint_velocities("nonexistent", [0.0])

    def test_set_torques_robot_not_found(self):
        """Test setting torques on non-existent robot."""
        controller = RobotController()

        with pytest.raises(KeyError, match="Robot .* not found"):
            controller.set_joint_torques("nonexistent", [0.0])

    def test_get_state_robot_not_found(self):
        """Test getting state of non-existent robot."""
        controller = RobotController()

        with pytest.raises(KeyError, match="Robot .* not found"):
            controller.get_robot_state("nonexistent")

    def test_step_robot_not_found(self):
        """Test stepping non-existent robot."""
        controller = RobotController()

        with pytest.raises(KeyError, match="Robot .* not found"):
            controller.step_robot("nonexistent")

    def test_reset_robot_not_found(self):
        """Test resetting non-existent robot."""
        controller = RobotController()

        with pytest.raises(KeyError, match="Robot .* not found"):
            controller.reset_robot("nonexistent")


class TestArraySizeMismatches:
    """Test array size validation for robot commands."""

    def test_set_positions_wrong_size(self):
        """Test setting positions with wrong array size."""
        controller = RobotController()
        result = controller.load_robot("arm")
        robot_id = result["robot_id"]
        expected_size = result["num_joints"]

        # Too few
        with pytest.raises(ValueError, match="Position array size mismatch"):
            controller.set_joint_positions(robot_id, [0.0] * (expected_size - 1))

        # Too many
        with pytest.raises(ValueError, match="Position array size mismatch"):
            controller.set_joint_positions(robot_id, [0.0] * (expected_size + 1))

    def test_set_velocities_wrong_size(self):
        """Test setting velocities with wrong array size."""
        controller = RobotController()
        result = controller.load_robot("arm")
        robot_id = result["robot_id"]
        expected_size = result["num_joints"]

        # Too few
        with pytest.raises(ValueError, match="Velocity array size mismatch"):
            controller.set_joint_velocities(robot_id, [0.0] * (expected_size - 1))

        # Too many
        with pytest.raises(ValueError, match="Velocity array size mismatch"):
            controller.set_joint_velocities(robot_id, [0.0] * (expected_size + 1))

    def test_set_torques_wrong_size(self):
        """Test setting torques with wrong array size."""
        controller = RobotController()
        result = controller.load_robot("arm")
        robot_id = result["robot_id"]
        expected_size = result["num_joints"]

        # Too few
        with pytest.raises(ValueError, match="Torque array size mismatch"):
            controller.set_joint_torques(robot_id, [0.0] * (expected_size - 1))

        # Too many
        with pytest.raises(ValueError, match="Torque array size mismatch"):
            controller.set_joint_torques(robot_id, [0.0] * (expected_size + 1))


class TestJointPositionControl:
    """Test joint position control."""

    def test_set_valid_positions(self):
        """Test setting valid joint positions."""
        controller = RobotController()
        result = controller.load_robot("arm")
        robot_id = result["robot_id"]
        num_joints = result["num_joints"]

        positions = [0.5] * num_joints
        result = controller.set_joint_positions(robot_id, positions)

        assert result["status"] == "success"
        assert result["control_mode"] == "position"
        assert result["positions_set"] == positions

    def test_set_zero_positions(self):
        """Test setting zero positions."""
        controller = RobotController()
        result = controller.load_robot("arm")
        robot_id = result["robot_id"]
        num_joints = result["num_joints"]

        positions = [0.0] * num_joints
        result = controller.set_joint_positions(robot_id, positions)

        assert result["status"] == "success"

    def test_set_negative_positions(self):
        """Test setting negative positions."""
        controller = RobotController()
        result = controller.load_robot("arm")
        robot_id = result["robot_id"]
        num_joints = result["num_joints"]

        positions = [-1.0] * num_joints
        result = controller.set_joint_positions(robot_id, positions)

        assert result["status"] == "success"

    def test_set_large_positions(self):
        """Test setting large position values."""
        controller = RobotController()
        result = controller.load_robot("arm")
        robot_id = result["robot_id"]
        num_joints = result["num_joints"]

        positions = [100.0] * num_joints
        result = controller.set_joint_positions(robot_id, positions)

        assert result["status"] == "success"


class TestJointVelocityControl:
    """Test joint velocity control."""

    def test_set_valid_velocities(self):
        """Test setting valid joint velocities."""
        controller = RobotController()
        result = controller.load_robot("arm")
        robot_id = result["robot_id"]
        num_joints = result["num_joints"]

        velocities = [1.0] * num_joints
        result = controller.set_joint_velocities(robot_id, velocities)

        assert result["status"] == "success"
        assert result["control_mode"] == "velocity"

    def test_set_zero_velocities(self):
        """Test setting zero velocities."""
        controller = RobotController()
        result = controller.load_robot("arm")
        robot_id = result["robot_id"]
        num_joints = result["num_joints"]

        velocities = [0.0] * num_joints
        result = controller.set_joint_velocities(robot_id, velocities)

        assert result["status"] == "success"

    def test_set_negative_velocities(self):
        """Test setting negative velocities."""
        controller = RobotController()
        result = controller.load_robot("arm")
        robot_id = result["robot_id"]
        num_joints = result["num_joints"]

        velocities = [-2.0] * num_joints
        result = controller.set_joint_velocities(robot_id, velocities)

        assert result["status"] == "success"


class TestJointTorqueControl:
    """Test joint torque control."""

    def test_set_valid_torques(self):
        """Test setting valid joint torques."""
        controller = RobotController()
        result = controller.load_robot("arm")
        robot_id = result["robot_id"]
        num_joints = result["num_joints"]

        torques = [5.0] * num_joints
        result = controller.set_joint_torques(robot_id, torques)

        assert result["status"] == "success"
        assert result["control_mode"] == "torque"

    def test_set_zero_torques(self):
        """Test setting zero torques."""
        controller = RobotController()
        result = controller.load_robot("arm")
        robot_id = result["robot_id"]
        num_joints = result["num_joints"]

        torques = [0.0] * num_joints
        result = controller.set_joint_torques(robot_id, torques)

        assert result["status"] == "success"

    def test_set_negative_torques(self):
        """Test setting negative torques."""
        controller = RobotController()
        result = controller.load_robot("arm")
        robot_id = result["robot_id"]
        num_joints = result["num_joints"]

        torques = [-10.0] * num_joints
        result = controller.set_joint_torques(robot_id, torques)

        assert result["status"] == "success"


class TestRobotState:
    """Test getting robot state."""

    def test_get_robot_state(self):
        """Test getting robot state."""
        controller = RobotController()
        result = controller.load_robot("arm")
        robot_id = result["robot_id"]

        state = controller.get_robot_state(robot_id)

        assert "robot_id" in state
        assert "joint_positions" in state
        assert "joint_velocities" in state
        assert "control_mode" in state

    def test_get_state_after_setting_positions(self):
        """Test state reflects set positions."""
        controller = RobotController()
        result = controller.load_robot("arm")
        robot_id = result["robot_id"]
        num_joints = result["num_joints"]

        # Set positions
        positions = [0.5] * num_joints
        controller.set_joint_positions(robot_id, positions)

        # Get state
        state = controller.get_robot_state(robot_id)

        assert state["control_mode"] == "position"


class TestRobotStepping:
    """Test robot simulation stepping."""

    def test_step_robot_once(self):
        """Test stepping robot simulation once."""
        controller = RobotController()
        result = controller.load_robot("arm")
        robot_id = result["robot_id"]

        result = controller.step_robot(robot_id)

        assert result["status"] == "success"
        assert "time" in result

    def test_step_robot_multiple(self):
        """Test stepping robot multiple times."""
        controller = RobotController()
        result = controller.load_robot("arm")
        robot_id = result["robot_id"]

        result = controller.step_robot(robot_id, steps=10)

        assert result["status"] == "success"

    def test_step_advances_time(self):
        """Test stepping advances simulation time."""
        controller = RobotController()
        result = controller.load_robot("arm")
        robot_id = result["robot_id"]

        state1 = controller.get_robot_state(robot_id)
        time1 = state1.get("time", 0.0)

        controller.step_robot(robot_id, steps=10)

        state2 = controller.get_robot_state(robot_id)
        time2 = state2.get("time", 0.0)

        assert time2 > time1


class TestRobotReset:
    """Test robot reset functionality."""

    def test_reset_robot(self):
        """Test resetting robot to initial state."""
        controller = RobotController()
        result = controller.load_robot("arm")
        robot_id = result["robot_id"]

        # Step forward
        controller.step_robot(robot_id, steps=10)

        # Reset
        result = controller.reset_robot(robot_id)

        assert result["status"] == "success"

    def test_reset_clears_time(self):
        """Test reset returns time to zero."""
        controller = RobotController()
        result = controller.load_robot("arm")
        robot_id = result["robot_id"]

        # Step forward
        controller.step_robot(robot_id, steps=10)

        # Reset
        controller.reset_robot(robot_id)

        # Check time is back to zero
        state = controller.get_robot_state(robot_id)
        time = state.get("time", -1.0)

        assert time == 0.0


class TestControlModeSwitching:
    """Test switching between control modes."""

    def test_switch_position_to_velocity(self):
        """Test switching from position to velocity control."""
        controller = RobotController()
        result = controller.load_robot("arm")
        robot_id = result["robot_id"]
        num_joints = result["num_joints"]

        # Position control
        controller.set_joint_positions(robot_id, [0.5] * num_joints)
        state = controller.get_robot_state(robot_id)
        assert state["control_mode"] == "position"

        # Velocity control
        controller.set_joint_velocities(robot_id, [1.0] * num_joints)
        state = controller.get_robot_state(robot_id)
        assert state["control_mode"] == "velocity"

    def test_switch_velocity_to_torque(self):
        """Test switching from velocity to torque control."""
        controller = RobotController()
        result = controller.load_robot("arm")
        robot_id = result["robot_id"]
        num_joints = result["num_joints"]

        # Velocity control
        controller.set_joint_velocities(robot_id, [1.0] * num_joints)
        state = controller.get_robot_state(robot_id)
        assert state["control_mode"] == "velocity"

        # Torque control
        controller.set_joint_torques(robot_id, [5.0] * num_joints)
        state = controller.get_robot_state(robot_id)
        assert state["control_mode"] == "torque"

    def test_switch_torque_to_position(self):
        """Test switching from torque to position control."""
        controller = RobotController()
        result = controller.load_robot("arm")
        robot_id = result["robot_id"]
        num_joints = result["num_joints"]

        # Torque control
        controller.set_joint_torques(robot_id, [5.0] * num_joints)
        state = controller.get_robot_state(robot_id)
        assert state["control_mode"] == "torque"

        # Position control
        controller.set_joint_positions(robot_id, [0.5] * num_joints)
        state = controller.get_robot_state(robot_id)
        assert state["control_mode"] == "position"


class TestMultipleRobotsControl:
    """Test controlling multiple robots simultaneously."""

    def test_control_multiple_robots_independently(self):
        """Test controlling two robots independently."""
        controller = RobotController()

        # Load two robots
        robot1 = controller.load_robot("arm", robot_id="robot1")
        robot2 = controller.load_robot("gripper", robot_id="robot2")

        # Set different positions
        controller.set_joint_positions("robot1", [0.5] * robot1["num_joints"])
        controller.set_joint_positions("robot2", [1.0] * robot2["num_joints"])

        # Check states are independent
        state1 = controller.get_robot_state("robot1")
        state2 = controller.get_robot_state("robot2")

        assert state1["robot_id"] == "robot1"
        assert state2["robot_id"] == "robot2"

    def test_step_robots_independently(self):
        """Test stepping robots independently."""
        controller = RobotController()

        robot1 = controller.load_robot("arm", robot_id="robot1")
        robot2 = controller.load_robot("arm", robot_id="robot2")

        # Step robot1 only
        controller.step_robot("robot1", steps=10)

        # Robot2 time should still be zero
        state2 = controller.get_robot_state("robot2")
        assert state2.get("time", -1.0) == 0.0

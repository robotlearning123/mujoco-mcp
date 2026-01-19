"""Comprehensive unit tests for multi_robot_coordinator.py focusing on dataclass validation."""

import numpy as np
import pytest

from mujoco_mcp.multi_robot_coordinator import (
    RobotState,
    CoordinatedTask,
    TaskType,
    RobotStatus,
    TaskStatus,
)


class TestRobotState:
    """Test RobotState dataclass validation."""

    def test_valid_robot_state(self):
        """Test creating valid robot state."""
        state = RobotState(
            robot_id="robot1",
            model_type="arm",
            joint_positions=np.array([0.0, 0.5, 1.0]),
            joint_velocities=np.array([0.1, 0.2, 0.3]),
        )
        assert state.robot_id == "robot1"
        assert len(state.joint_positions) == 3
        assert len(state.joint_velocities) == 3

    def test_position_velocity_dimension_mismatch(self):
        """Test that mismatched position/velocity dimensions are rejected."""
        with pytest.raises(
            ValueError,
            match="joint_positions length .* must match joint_velocities length",
        ):
            RobotState(
                robot_id="robot1",
                model_type="arm",
                joint_positions=np.array([0.0, 0.5, 1.0]),
                joint_velocities=np.array([0.1, 0.2]),  # Wrong size
            )

    def test_different_sizes_both_directions(self):
        """Test dimension mismatch works both ways."""
        # Velocities longer than positions
        with pytest.raises(
            ValueError,
            match="joint_positions length .* must match joint_velocities length",
        ):
            RobotState(
                robot_id="robot1",
                model_type="arm",
                joint_positions=np.array([0.0, 0.5]),
                joint_velocities=np.array([0.1, 0.2, 0.3, 0.4]),
            )

        # Positions longer than velocities
        with pytest.raises(
            ValueError,
            match="joint_positions length .* must match joint_velocities length",
        ):
            RobotState(
                robot_id="robot1",
                model_type="arm",
                joint_positions=np.array([0.0, 0.5, 1.0, 1.5]),
                joint_velocities=np.array([0.1, 0.2]),
            )

    def test_zero_dimension_arrays(self):
        """Test empty position/velocity arrays (valid for models with no joints)."""
        state = RobotState(
            robot_id="robot1",
            model_type="static",
            joint_positions=np.array([]),
            joint_velocities=np.array([]),
        )
        assert len(state.joint_positions) == 0
        assert len(state.joint_velocities) == 0

    def test_single_joint_robot(self):
        """Test robot with single joint."""
        state = RobotState(
            robot_id="robot1",
            model_type="pendulum",
            joint_positions=np.array([0.5]),
            joint_velocities=np.array([0.1]),
        )
        assert len(state.joint_positions) == 1
        assert len(state.joint_velocities) == 1

    def test_many_joints_robot(self):
        """Test robot with many joints."""
        num_joints = 100
        state = RobotState(
            robot_id="robot1",
            model_type="complex",
            joint_positions=np.zeros(num_joints),
            joint_velocities=np.zeros(num_joints),
        )
        assert len(state.joint_positions) == num_joints
        assert len(state.joint_velocities) == num_joints

    def test_optional_end_effector_fields(self):
        """Test optional end effector position and velocity."""
        state = RobotState(
            robot_id="robot1",
            model_type="arm",
            joint_positions=np.array([0.0, 0.5]),
            joint_velocities=np.array([0.1, 0.2]),
            end_effector_pos=np.array([1.0, 2.0, 3.0]),
            end_effector_vel=np.array([0.1, 0.2, 0.3]),
        )
        assert state.end_effector_pos is not None
        assert state.end_effector_vel is not None

    def test_default_status(self):
        """Test default status is 'idle'."""
        state = RobotState(
            robot_id="robot1",
            model_type="arm",
            joint_positions=np.array([0.0]),
            joint_velocities=np.array([0.0]),
        )
        assert state.status == RobotStatus.IDLE

    def test_custom_status(self):
        """Test setting custom status."""
        state = RobotState(
            robot_id="robot1",
            model_type="arm",
            joint_positions=np.array([0.0]),
            joint_velocities=np.array([0.0]),
            status=RobotStatus.EXECUTING,
        )
        assert state.status == RobotStatus.EXECUTING

    def test_frozen_dataclass(self):
        """Test that RobotState is immutable."""
        state = RobotState(
            robot_id="robot1",
            model_type="arm",
            joint_positions=np.array([0.0]),
            joint_velocities=np.array([0.0]),
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            state.status = "new_status"


class TestCoordinatedTask:
    """Test CoordinatedTask dataclass validation."""

    def test_valid_coordinated_task(self):
        """Test creating valid coordinated task."""
        task = CoordinatedTask(
            task_id="task1",
            task_type=TaskType.PICK_AND_PLACE,
            robots=["robot1", "robot2"],
            parameters={"target": "object1"},
        )
        assert task.task_id == "task1"
        assert len(task.robots) == 2

    def test_empty_robots_list(self):
        """Test that empty robots list is rejected."""
        with pytest.raises(ValueError, match="robots list cannot be empty"):
            CoordinatedTask(
                task_id="task1",
                task_type=TaskType.PICK_AND_PLACE,
                robots=[],  # Empty list
                parameters={},
            )

    def test_single_robot_task(self):
        """Test task with single robot is valid."""
        task = CoordinatedTask(
            task_id="task1",
            task_type=TaskType.PICK_AND_PLACE,
            robots=["robot1"],
            parameters={},
        )
        assert len(task.robots) == 1

    def test_many_robots_task(self):
        """Test task with many robots."""
        robots = [f"robot{i}" for i in range(10)]
        task = CoordinatedTask(
            task_id="task1",
            task_type=TaskType.PICK_AND_PLACE,
            robots=robots,
            parameters={},
        )
        assert len(task.robots) == 10

    def test_negative_timeout(self):
        """Test that negative timeout is rejected."""
        with pytest.raises(ValueError, match="timeout must be positive"):
            CoordinatedTask(
                task_id="task1",
                task_type=TaskType.PICK_AND_PLACE,
                robots=["robot1"],
                parameters={},
                timeout=-1.0,
            )

    def test_zero_timeout(self):
        """Test that zero timeout is rejected."""
        with pytest.raises(ValueError, match="timeout must be positive"):
            CoordinatedTask(
                task_id="task1",
                task_type=TaskType.PICK_AND_PLACE,
                robots=["robot1"],
                parameters={},
                timeout=0.0,
            )

    def test_very_small_positive_timeout(self):
        """Test very small positive timeout is valid."""
        task = CoordinatedTask(
            task_id="task1",
            task_type=TaskType.PICK_AND_PLACE,
            robots=["robot1"],
            parameters={},
            timeout=0.001,
        )
        assert task.timeout == 0.001

    def test_very_large_timeout(self):
        """Test very large timeout is valid."""
        task = CoordinatedTask(
            task_id="task1",
            task_type=TaskType.PICK_AND_PLACE,
            robots=["robot1"],
            parameters={},
            timeout=1e6,
        )
        assert task.timeout == 1e6

    def test_default_timeout(self):
        """Test default timeout is 30 seconds."""
        task = CoordinatedTask(
            task_id="task1",
            task_type=TaskType.PICK_AND_PLACE,
            robots=["robot1"],
            parameters={},
        )
        assert task.timeout == 30.0

    def test_default_priority(self):
        """Test default priority is 1."""
        task = CoordinatedTask(
            task_id="task1",
            task_type=TaskType.PICK_AND_PLACE,
            robots=["robot1"],
            parameters={},
        )
        assert task.priority == 1

    def test_custom_priority(self):
        """Test setting custom priority."""
        task = CoordinatedTask(
            task_id="task1",
            task_type=TaskType.PICK_AND_PLACE,
            robots=["robot1"],
            parameters={},
            priority=10,
        )
        assert task.priority == 10

    def test_default_status(self):
        """Test default status is 'pending'."""
        task = CoordinatedTask(
            task_id="task1",
            task_type=TaskType.PICK_AND_PLACE,
            robots=["robot1"],
            parameters={},
        )
        assert task.status == TaskStatus.PENDING

    def test_custom_status(self):
        """Test setting custom status."""
        task = CoordinatedTask(
            task_id="task1",
            task_type=TaskType.PICK_AND_PLACE,
            robots=["robot1"],
            parameters={},
            status=TaskStatus.EXECUTING,
        )
        assert task.status == TaskStatus.EXECUTING

    def test_task_types(self):
        """Test different task types."""
        for task_type in [
            TaskType.PICK_AND_PLACE,
            TaskType.ASSEMBLY,
            TaskType.HANDOVER,
            TaskType.COLLABORATIVE_TRANSPORT,
        ]:
            task = CoordinatedTask(
                task_id="task1",
                task_type=task_type,
                robots=["robot1"],
                parameters={},
            )
            assert task.task_type == task_type

    def test_complex_parameters(self):
        """Test task with complex parameters dictionary."""
        params = {
            "target_position": [1.0, 2.0, 3.0],
            "grasp_force": 10.5,
            "approach_angle": 45.0,
            "nested": {"key": "value"},
        }
        task = CoordinatedTask(
            task_id="task1",
            task_type=TaskType.PICK_AND_PLACE,
            robots=["robot1"],
            parameters=params,
        )
        assert task.parameters == params

    def test_none_start_time(self):
        """Test start_time defaults to None."""
        task = CoordinatedTask(
            task_id="task1",
            task_type=TaskType.PICK_AND_PLACE,
            robots=["robot1"],
            parameters={},
        )
        assert task.start_time is None

    def test_custom_start_time(self):
        """Test setting custom start_time."""
        import time

        start = time.time()
        task = CoordinatedTask(
            task_id="task1",
            task_type=TaskType.PICK_AND_PLACE,
            robots=["robot1"],
            parameters={},
            start_time=start,
        )
        assert task.start_time == start

    def test_frozen_dataclass(self):
        """Test that CoordinatedTask is immutable."""
        task = CoordinatedTask(
            task_id="task1",
            task_type=TaskType.PICK_AND_PLACE,
            robots=["robot1"],
            parameters={},
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            task.status = "new_status"


class TestRobotStateEdgeCases:
    """Test edge cases for RobotState."""

    def test_very_large_arrays(self):
        """Test RobotState with very large joint arrays."""
        size = 1000
        state = RobotState(
            robot_id="robot1",
            model_type="complex",
            joint_positions=np.random.random(size),
            joint_velocities=np.random.random(size),
        )
        assert len(state.joint_positions) == size
        assert len(state.joint_velocities) == size

    def test_negative_joint_values(self):
        """Test RobotState with negative joint values."""
        state = RobotState(
            robot_id="robot1",
            model_type="arm",
            joint_positions=np.array([-1.0, -2.0, -3.0]),
            joint_velocities=np.array([-0.5, -1.0, -1.5]),
        )
        assert np.all(state.joint_positions < 0)
        assert np.all(state.joint_velocities < 0)

    def test_mixed_sign_values(self):
        """Test RobotState with mixed positive/negative values."""
        state = RobotState(
            robot_id="robot1",
            model_type="arm",
            joint_positions=np.array([-1.0, 0.0, 1.0]),
            joint_velocities=np.array([1.0, 0.0, -1.0]),
        )
        assert state.joint_positions[0] < 0
        assert state.joint_positions[1] == 0
        assert state.joint_positions[2] > 0

    def test_nan_values_in_arrays(self):
        """Test RobotState with NaN values (should be allowed by dataclass but problematic)."""
        # NaN values are technically allowed by the dataclass
        # (validation only checks dimensions, not values)
        state = RobotState(
            robot_id="robot1",
            model_type="arm",
            joint_positions=np.array([np.nan, 0.0]),
            joint_velocities=np.array([0.0, np.nan]),
        )
        # State is created but contains NaN
        assert np.isnan(state.joint_positions[0])
        assert np.isnan(state.joint_velocities[1])

    def test_inf_values_in_arrays(self):
        """Test RobotState with Inf values."""
        state = RobotState(
            robot_id="robot1",
            model_type="arm",
            joint_positions=np.array([np.inf, 0.0]),
            joint_velocities=np.array([0.0, -np.inf]),
        )
        assert np.isinf(state.joint_positions[0])
        assert np.isinf(state.joint_velocities[1])


class TestCoordinatedTaskEdgeCases:
    """Test edge cases for CoordinatedTask."""

    def test_duplicate_robot_ids(self):
        """Test task with duplicate robot IDs (allowed but potentially problematic)."""
        task = CoordinatedTask(
            task_id="task1",
            task_type=TaskType.PICK_AND_PLACE,
            robots=["robot1", "robot1", "robot2"],
            parameters={},
        )
        # Duplicates are allowed by validation
        assert len(task.robots) == 3

    def test_empty_task_id(self):
        """Test task with empty task_id (allowed but not recommended)."""
        task = CoordinatedTask(
            task_id="",
            task_type=TaskType.PICK_AND_PLACE,
            robots=["robot1"],
            parameters={},
        )
        assert task.task_id == ""

    def test_empty_parameters(self):
        """Test task with empty parameters dictionary."""
        task = CoordinatedTask(
            task_id="task1",
            task_type=TaskType.PICK_AND_PLACE,
            robots=["robot1"],
            parameters={},
        )
        assert task.parameters == {}

    def test_very_high_priority(self):
        """Test task with very high priority."""
        task = CoordinatedTask(
            task_id="task1",
            task_type=TaskType.PICK_AND_PLACE,
            robots=["robot1"],
            parameters={},
            priority=1000000,
        )
        assert task.priority == 1000000

    def test_negative_priority(self):
        """Test task with negative priority (allowed)."""
        task = CoordinatedTask(
            task_id="task1",
            task_type=TaskType.PICK_AND_PLACE,
            robots=["robot1"],
            parameters={},
            priority=-5,
        )
        assert task.priority == -5

    def test_special_characters_in_robot_ids(self):
        """Test robots list with special characters."""
        task = CoordinatedTask(
            task_id="task1",
            task_type=TaskType.PICK_AND_PLACE,
            robots=["robot-1", "robot_2", "robot.3"],
            parameters={},
        )
        assert len(task.robots) == 3

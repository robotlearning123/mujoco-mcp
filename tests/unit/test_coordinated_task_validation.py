"""Additional error path tests for CoordinatedTask validation."""

import pytest

from mujoco_mcp.multi_robot_coordinator import CoordinatedTask, TaskType, TaskStatus


class TestCoordinatedTaskErrorPaths:
    """Test error paths in CoordinatedTask validation."""

    def test_empty_robots_list(self):
        """Test that empty robots list raises ValueError."""
        with pytest.raises(ValueError, match="robots list cannot be empty"):
            CoordinatedTask(
                task_id="task1",
                task_type=TaskType.PICK_AND_PLACE,
                robots=[],  # Empty list should raise error
                parameters={}
            )

    def test_negative_timeout(self):
        """Test that negative timeout raises ValueError."""
        with pytest.raises(ValueError, match="timeout must be positive"):
            CoordinatedTask(
                task_id="task1",
                task_type=TaskType.PICK_AND_PLACE,
                robots=["robot1"],
                parameters={},
                timeout=-1.0  # Negative timeout
            )

    def test_zero_timeout(self):
        """Test that zero timeout raises ValueError."""
        with pytest.raises(ValueError, match="timeout must be positive"):
            CoordinatedTask(
                task_id="task1",
                task_type=TaskType.PICK_AND_PLACE,
                robots=["robot1"],
                parameters={},
                timeout=0.0  # Zero timeout
            )

    def test_valid_task_with_single_robot(self):
        """Test that task with single robot is valid."""
        task = CoordinatedTask(
            task_id="task1",
            task_type=TaskType.PICK_AND_PLACE,
            robots=["robot1"],
            parameters={}
        )
        assert len(task.robots) == 1
        assert task.robots[0] == "robot1"

    def test_valid_task_with_multiple_robots(self):
        """Test that task with multiple robots is valid."""
        task = CoordinatedTask(
            task_id="task1",
            task_type=TaskType.COLLABORATIVE_TRANSPORT,
            robots=["robot1", "robot2", "robot3"],
            parameters={}
        )
        assert len(task.robots) == 3

    def test_valid_task_with_custom_timeout(self):
        """Test that task with valid custom timeout is accepted."""
        task = CoordinatedTask(
            task_id="task1",
            task_type=TaskType.PICK_AND_PLACE,
            robots=["robot1"],
            parameters={},
            timeout=120.0  # 2 minutes
        )
        assert task.timeout == 120.0

    def test_task_status_transitions(self):
        """Test that task status can be updated (mutable dataclass)."""
        task = CoordinatedTask(
            task_id="task1",
            task_type=TaskType.PICK_AND_PLACE,
            robots=["robot1"],
            parameters={}
        )

        # Should start as PENDING
        assert task.status == TaskStatus.PENDING

        # Should be able to update status (not frozen)
        task.status = TaskStatus.ALLOCATED
        assert task.status == TaskStatus.ALLOCATED

        task.status = TaskStatus.EXECUTING
        assert task.status == TaskStatus.EXECUTING

        task.status = TaskStatus.COMPLETED
        assert task.status == TaskStatus.COMPLETED

    def test_all_task_types(self):
        """Test that all task types are valid."""
        task_types = [
            TaskType.PICK_AND_PLACE,
            TaskType.ASSEMBLY,
            TaskType.HANDOVER,
            TaskType.COLLABORATIVE_TRANSPORT,
        ]

        for task_type in task_types:
            task = CoordinatedTask(
                task_id=f"task_{task_type.value}",
                task_type=task_type,
                robots=["robot1"],
                parameters={}
            )
            assert task.task_type == task_type

    def test_task_with_complex_parameters(self):
        """Test task with complex parameter dictionary."""
        complex_params = {
            "target_position": [1.0, 2.0, 3.0],
            "grip_force": 10.5,
            "approach_vector": [0, 0, -1],
            "constraints": {
                "max_velocity": 0.5,
                "safety_distance": 0.1
            }
        }

        task = CoordinatedTask(
            task_id="task1",
            task_type=TaskType.PICK_AND_PLACE,
            robots=["robot1"],
            parameters=complex_params
        )

        assert task.parameters == complex_params
        assert task.parameters["grip_force"] == 10.5

    def test_task_priority_values(self):
        """Test various priority values."""
        # Low priority
        task1 = CoordinatedTask(
            task_id="task1",
            task_type=TaskType.PICK_AND_PLACE,
            robots=["robot1"],
            parameters={},
            priority=1
        )
        assert task1.priority == 1

        # High priority
        task2 = CoordinatedTask(
            task_id="task2",
            task_type=TaskType.PICK_AND_PLACE,
            robots=["robot1"],
            parameters={},
            priority=100
        )
        assert task2.priority == 100

        # Default priority
        task3 = CoordinatedTask(
            task_id="task3",
            task_type=TaskType.PICK_AND_PLACE,
            robots=["robot1"],
            parameters={}
        )
        assert task3.priority == 0  # Default value


class TestRobotStateImmutability:
    """Test that RobotState numpy arrays are immutable."""

    def test_joint_positions_immutable(self):
        """Test that joint_positions array cannot be modified."""
        from mujoco_mcp.multi_robot_coordinator import RobotState
        import numpy as np

        positions = np.array([0.0, 0.5, 1.0])
        state = RobotState(
            robot_id="robot1",
            model_type="arm",
            joint_positions=positions,
            joint_velocities=np.array([0.0, 0.0, 0.0])
        )

        # Array should be marked as read-only
        assert not state.joint_positions.flags.writeable

        # Attempting to modify should raise error
        with pytest.raises(ValueError, match="read-only"):
            state.joint_positions[0] = 1.0

    def test_joint_velocities_immutable(self):
        """Test that joint_velocities array cannot be modified."""
        from mujoco_mcp.multi_robot_coordinator import RobotState
        import numpy as np

        state = RobotState(
            robot_id="robot1",
            model_type="arm",
            joint_positions=np.array([0.0, 0.5, 1.0]),
            joint_velocities=np.array([0.0, 0.0, 0.0])
        )

        assert not state.joint_velocities.flags.writeable

        with pytest.raises(ValueError, match="read-only"):
            state.joint_velocities[1] = 0.5

    def test_end_effector_pos_immutable(self):
        """Test that end_effector_pos array cannot be modified when provided."""
        from mujoco_mcp.multi_robot_coordinator import RobotState
        import numpy as np

        state = RobotState(
            robot_id="robot1",
            model_type="arm",
            joint_positions=np.array([0.0, 0.5, 1.0]),
            joint_velocities=np.array([0.0, 0.0, 0.0]),
            end_effector_pos=np.array([1.0, 0.0, 0.5])
        )

        assert state.end_effector_pos is not None
        assert not state.end_effector_pos.flags.writeable

        with pytest.raises(ValueError, match="read-only"):
            state.end_effector_pos[0] = 2.0

    def test_end_effector_vel_immutable(self):
        """Test that end_effector_vel array cannot be modified when provided."""
        from mujoco_mcp.multi_robot_coordinator import RobotState
        import numpy as np

        state = RobotState(
            robot_id="robot1",
            model_type="arm",
            joint_positions=np.array([0.0, 0.5, 1.0]),
            joint_velocities=np.array([0.0, 0.0, 0.0]),
            end_effector_vel=np.array([0.1, 0.0, 0.05])
        )

        assert state.end_effector_vel is not None
        assert not state.end_effector_vel.flags.writeable

        with pytest.raises(ValueError, match="read-only"):
            state.end_effector_vel[2] = 0.1

"""Comprehensive error path tests for RL integration configuration validation."""

import numpy as np
import pytest

from mujoco_mcp.rl_integration import RLConfig, ActionSpaceType, TaskType


class TestRLConfigValidation:
    """Test RLConfig validation and error paths."""

    def test_negative_max_episode_steps(self):
        """Test that negative max_episode_steps raises ValueError."""
        with pytest.raises(ValueError, match="max_episode_steps must be positive"):
            RLConfig(
                robot_type="test",
                task_type=TaskType.REACHING,
                max_episode_steps=-100
            )

    def test_zero_max_episode_steps(self):
        """Test that zero max_episode_steps raises ValueError."""
        with pytest.raises(ValueError, match="max_episode_steps must be positive"):
            RLConfig(
                robot_type="test",
                task_type=TaskType.REACHING,
                max_episode_steps=0
            )

    def test_negative_physics_timestep(self):
        """Test that negative physics_timestep raises ValueError."""
        with pytest.raises(ValueError, match="physics_timestep must be positive"):
            RLConfig(
                robot_type="test",
                task_type=TaskType.REACHING,
                physics_timestep=-0.001
            )

    def test_zero_physics_timestep(self):
        """Test that zero physics_timestep raises ValueError."""
        with pytest.raises(ValueError, match="physics_timestep must be positive"):
            RLConfig(
                robot_type="test",
                task_type=TaskType.REACHING,
                physics_timestep=0.0
            )

    def test_negative_control_timestep(self):
        """Test that negative control_timestep raises ValueError."""
        with pytest.raises(ValueError, match="control_timestep must be positive"):
            RLConfig(
                robot_type="test",
                task_type=TaskType.REACHING,
                control_timestep=-0.01
            )

    def test_zero_control_timestep(self):
        """Test that zero control_timestep raises ValueError."""
        with pytest.raises(ValueError, match="control_timestep must be positive"):
            RLConfig(
                robot_type="test",
                task_type=TaskType.REACHING,
                control_timestep=0.0
            )

    def test_control_timestep_less_than_physics_timestep(self):
        """Test that control_timestep < physics_timestep raises ValueError."""
        with pytest.raises(ValueError, match="control_timestep.*must be >= physics_timestep"):
            RLConfig(
                robot_type="test",
                task_type=TaskType.REACHING,
                physics_timestep=0.01,
                control_timestep=0.005  # Less than physics_timestep
            )

    def test_invalid_action_space_type(self):
        """Test that non-ActionSpaceType value raises ValueError."""
        with pytest.raises(ValueError, match="action_space_type must be an ActionSpaceType enum"):
            RLConfig(
                robot_type="test",
                task_type=TaskType.REACHING,
                action_space_type="invalid"  # type: ignore
            )

    def test_invalid_task_type(self):
        """Test that non-TaskType value raises ValueError."""
        with pytest.raises(ValueError, match="task_type must be a TaskType enum"):
            RLConfig(
                robot_type="test",
                task_type="invalid"  # type: ignore
            )

    def test_valid_config(self):
        """Test that valid configuration doesn't raise errors."""
        config = RLConfig(
            robot_type="test",
            task_type=TaskType.REACHING,
            max_episode_steps=1000,
            physics_timestep=0.002,
            control_timestep=0.02,
            action_space_type=ActionSpaceType.CONTINUOUS
        )
        assert config.max_episode_steps == 1000
        assert config.physics_timestep == 0.002
        assert config.control_timestep == 0.02

    def test_control_timestep_equal_to_physics_timestep(self):
        """Test that control_timestep == physics_timestep is valid."""
        config = RLConfig(
            robot_type="test",
            task_type=TaskType.REACHING,
            physics_timestep=0.01,
            control_timestep=0.01  # Equal to physics_timestep
        )
        assert config.physics_timestep == config.control_timestep

    def test_very_small_timesteps(self):
        """Test that very small but positive timesteps are valid."""
        config = RLConfig(
            robot_type="test",
            task_type=TaskType.REACHING,
            physics_timestep=0.0001,
            control_timestep=0.0002
        )
        assert config.physics_timestep == 0.0001
        assert config.control_timestep == 0.0002


class TestTaskRewardErrorPaths:
    """Test error paths in task reward implementations."""

    def test_reaching_reward_with_nan_observation(self):
        """Test that reaching reward handles NaN observations gracefully."""
        from mujoco_mcp.rl_integration import ReachingTaskReward

        reward = ReachingTaskReward(target_position=np.array([0.5, 0.0, 0.5]))

        # Observation with NaN
        obs = np.array([np.nan, 0.0, 0.0])
        action = np.array([0.0])
        next_obs = np.array([0.0, 0.0, 0.0])

        # Should not crash, but return a valid (possibly zero or negative) reward
        result = reward.compute_reward(obs, action, next_obs, {})
        assert isinstance(result, (int, float))

    def test_balancing_reward_with_inf_observation(self):
        """Test that balancing reward handles Inf observations."""
        from mujoco_mcp.rl_integration import BalancingTaskReward

        reward = BalancingTaskReward()

        # Observation with Inf
        obs = np.array([np.inf, 0.0])
        action = np.array([0.0])
        next_obs = np.array([0.0, 0.0])

        # Should not crash
        result = reward.compute_reward(obs, action, next_obs, {})
        assert isinstance(result, (int, float))

    def test_walking_reward_with_empty_observation(self):
        """Test that walking reward handles empty observations."""
        from mujoco_mcp.rl_integration import WalkingTaskReward

        reward = WalkingTaskReward()

        # Empty observation (edge case)
        obs = np.array([])
        action = np.array([])
        next_obs = np.array([])

        # Should handle gracefully (might return 0 or raise IndexError)
        try:
            result = reward.compute_reward(obs, action, next_obs, {})
            assert isinstance(result, (int, float))
        except IndexError:
            # Acceptable if it raises IndexError for empty arrays
            pass

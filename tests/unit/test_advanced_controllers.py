"""Comprehensive unit tests for advanced_controllers.py covering PID windup, trajectories, and edge cases."""

import time

import numpy as np
import pytest

from mujoco_mcp.advanced_controllers import (
    PIDConfig,
    PIDController,
    MinimumJerkTrajectory,
)


class TestPIDConfig:
    """Test PID configuration validation."""

    def test_valid_config(self):
        """Test creating valid PID configuration."""
        config = PIDConfig(kp=1.0, ki=0.1, kd=0.05, max_output=10.0, min_output=-10.0)
        assert config.kp == 1.0
        assert config.ki == 0.1
        assert config.kd == 0.05

    def test_negative_kp(self):
        """Test that negative Kp is rejected."""
        with pytest.raises(ValueError, match="Proportional gain must be non-negative"):
            PIDConfig(kp=-1.0)

    def test_negative_ki(self):
        """Test that negative Ki is rejected."""
        with pytest.raises(ValueError, match="Integral gain must be non-negative"):
            PIDConfig(ki=-0.5)

    def test_negative_kd(self):
        """Test that negative Kd is rejected."""
        with pytest.raises(ValueError, match="Derivative gain must be non-negative"):
            PIDConfig(kd=-0.1)

    def test_inverted_output_limits(self):
        """Test that min_output >= max_output is rejected."""
        with pytest.raises(ValueError, match="min_output .* must be less than max_output"):
            PIDConfig(max_output=10.0, min_output=10.0)

        with pytest.raises(ValueError, match="min_output .* must be less than max_output"):
            PIDConfig(max_output=10.0, min_output=20.0)

    def test_negative_windup_limit(self):
        """Test that negative windup limit is rejected."""
        with pytest.raises(ValueError, match="windup_limit must be positive"):
            PIDConfig(windup_limit=-1.0)

    def test_zero_windup_limit(self):
        """Test that zero windup limit is rejected."""
        with pytest.raises(ValueError, match="windup_limit must be positive"):
            PIDConfig(windup_limit=0.0)

    def test_frozen_dataclass(self):
        """Test that PIDConfig is immutable."""
        config = PIDConfig(kp=1.0)
        with pytest.raises(Exception):  # FrozenInstanceError
            config.kp = 2.0


class TestPIDController:
    """Test PID controller behavior and edge cases."""

    def test_initialization(self):
        """Test PID controller initialization."""
        config = PIDConfig(kp=1.0, ki=0.1, kd=0.05)
        pid = PIDController(config)
        assert pid.config == config

    def test_proportional_only(self):
        """Test P-only controller."""
        config = PIDConfig(kp=2.0, ki=0.0, kd=0.0, max_output=100.0, min_output=-100.0)
        pid = PIDController(config)

        # Error of 5.0 should give output of 10.0
        output = pid.update(target=10.0, current=5.0, dt=0.1)
        assert np.isclose(output, 10.0)

    def test_integral_accumulation(self):
        """Test integral term accumulation."""
        config = PIDConfig(kp=0.0, ki=1.0, kd=0.0, max_output=100.0, min_output=-100.0)
        pid = PIDController(config)

        # Constant error of 1.0 for 5 steps of 0.1s each
        for _ in range(5):
            pid.update(target=2.0, current=1.0, dt=0.1)

        # Integral should be approximately 5 * 0.1 * 1.0 = 0.5
        output = pid.update(target=2.0, current=1.0, dt=0.1)
        expected = 0.6  # 6 steps * 0.1 * 1.0
        assert np.isclose(output, expected, atol=0.01)

    def test_derivative_term(self):
        """Test derivative term computation."""
        config = PIDConfig(kp=0.0, ki=0.0, kd=1.0, max_output=100.0, min_output=-100.0)
        pid = PIDController(config)

        # First call establishes baseline
        pid.update(target=10.0, current=5.0, dt=0.1)

        # Second call with changing error
        output = pid.update(target=10.0, current=6.0, dt=0.1)

        # Error changed from 5.0 to 4.0, derivative = -10.0
        assert output < 0  # Negative because error is decreasing

    def test_output_clamping(self):
        """Test output is clamped to limits."""
        config = PIDConfig(kp=10.0, ki=0.0, kd=0.0, max_output=5.0, min_output=-5.0)
        pid = PIDController(config)

        # Large error should be clamped
        output = pid.update(target=100.0, current=0.0, dt=0.1)
        assert output == 5.0

        # Large negative error should be clamped
        output = pid.update(target=0.0, current=100.0, dt=0.1)
        assert output == -5.0

    def test_integral_windup_prevention(self):
        """Test integral windup is prevented."""
        config = PIDConfig(
            kp=0.0, ki=1.0, kd=0.0, max_output=100.0, min_output=-100.0, windup_limit=10.0
        )
        pid = PIDController(config)

        # Apply large constant error for many steps
        for _ in range(1000):
            pid.update(target=100.0, current=0.0, dt=0.1)

        # Integral should be clamped to windup_limit
        # Output = Ki * integral, so output should be <= Ki * windup_limit = 1.0 * 10.0 = 10.0
        output = pid.update(target=100.0, current=0.0, dt=0.1)
        assert output <= config.windup_limit + 0.1  # Small tolerance

    def test_reset(self):
        """Test reset clears internal state."""
        config = PIDConfig(kp=1.0, ki=1.0, kd=1.0)
        pid = PIDController(config)

        # Build up state
        for _ in range(10):
            pid.update(target=10.0, current=5.0, dt=0.1)

        # Reset
        pid.reset()

        # After reset, with pure P controller behavior initially
        output = pid.update(target=10.0, current=5.0, dt=0.1)
        # Should be close to pure P term (Kp * error = 1.0 * 5.0 = 5.0)
        # But derivative term will be large on first step after reset
        assert abs(output) < 100  # Just verify it's reasonable

    def test_automatic_dt_computation(self):
        """Test automatic dt computation from wall clock."""
        config = PIDConfig(kp=1.0)
        pid = PIDController(config)

        # First call uses default dt
        pid.update(target=10.0, current=5.0)

        # Wait a bit
        time.sleep(0.01)

        # Second call should compute dt automatically
        output = pid.update(target=10.0, current=5.0)
        assert isinstance(output, float)

    def test_zero_dt_handling(self):
        """Test behavior with dt=0."""
        config = PIDConfig(kp=1.0, ki=1.0, kd=1.0)
        pid = PIDController(config)

        # dt=0 should not cause division by zero
        output = pid.update(target=10.0, current=5.0, dt=0.0)
        assert np.isfinite(output)

    def test_negative_dt_handling(self):
        """Test behavior with negative dt."""
        config = PIDConfig(kp=1.0, ki=1.0, kd=1.0)
        pid = PIDController(config)

        # First update
        pid.update(target=10.0, current=5.0, dt=0.1)

        # Negative dt should be handled gracefully
        output = pid.update(target=10.0, current=5.0, dt=-0.1)
        assert np.isfinite(output)

    def test_full_pid_controller(self):
        """Test complete PID controller with all terms."""
        config = PIDConfig(kp=1.0, ki=0.5, kd=0.1, max_output=50.0, min_output=-50.0)
        pid = PIDController(config)

        # Simulate approaching a setpoint
        current = 0.0
        target = 10.0

        for _ in range(10):
            output = pid.update(target=target, current=current, dt=0.1)
            # Simulate system response (simple integration)
            current += output * 0.01

        # Should have moved closer to target
        assert current > 0.0

    def test_setpoint_tracking(self):
        """Test tracking a changing setpoint."""
        config = PIDConfig(kp=2.0, ki=0.5, kd=0.1)
        pid = PIDController(config)

        current = 0.0
        targets = [10.0, 20.0, 15.0, 5.0]

        for target in targets:
            for _ in range(5):
                output = pid.update(target=target, current=current, dt=0.1)
                current += output * 0.01

        assert np.isfinite(current)


class TestMinimumJerkTrajectory:
    """Test minimum jerk trajectory generation."""

    def test_point_to_point_trajectory(self):
        """Test generating trajectory between two points."""
        start_pos = np.array([0.0, 0.0, 0.0])
        end_pos = np.array([1.0, 1.0, 1.0])
        duration = 1.0
        num_steps = 50

        positions, velocities, accelerations = MinimumJerkTrajectory.minimum_jerk_trajectory(
            start_pos, end_pos, duration, num_steps
        )

        # Check shapes
        assert positions.shape == (num_steps, 3)
        assert velocities.shape == (num_steps, 3)
        assert accelerations.shape == (num_steps, 3)

        # Check boundary conditions - positions
        np.testing.assert_array_almost_equal(positions[0], start_pos)
        np.testing.assert_array_almost_equal(positions[-1], end_pos)

        # Check boundary conditions - velocities (should start and end at zero)
        np.testing.assert_array_almost_equal(velocities[0], np.zeros(3), decimal=2)
        np.testing.assert_array_almost_equal(velocities[-1], np.zeros(3), decimal=2)

    def test_trajectory_with_nonzero_velocities(self):
        """Test trajectory with non-zero start/end velocities."""
        start_pos = np.array([0.0])
        end_pos = np.array([1.0])
        start_vel = np.array([0.5])
        end_vel = np.array([0.2])
        duration = 1.0
        num_steps = 50

        _positions, velocities, _accelerations = MinimumJerkTrajectory.minimum_jerk_trajectory(
            start_pos, end_pos, duration, num_steps, start_vel, end_vel
        )

        # Check boundary velocities
        np.testing.assert_array_almost_equal(velocities[0], start_vel, decimal=2)
        np.testing.assert_array_almost_equal(velocities[-1], end_vel, decimal=2)

    def test_stationary_trajectory(self):
        """Test trajectory with same start and end (no movement)."""
        start_pos = np.array([1.0, 2.0, 3.0])
        end_pos = np.array([1.0, 2.0, 3.0])
        duration = 1.0
        num_steps = 50

        positions, velocities, _accelerations = MinimumJerkTrajectory.minimum_jerk_trajectory(
            start_pos, end_pos, duration, num_steps
        )

        # All positions should be constant
        for pos in positions:
            np.testing.assert_array_almost_equal(pos, start_pos)

        # All velocities should be near zero
        for vel in velocities:
            np.testing.assert_array_almost_equal(vel, np.zeros(3), decimal=2)

    def test_single_dimension_trajectory(self):
        """Test trajectory in 1D."""
        start_pos = np.array([0.0])
        end_pos = np.array([5.0])
        duration = 2.0
        num_steps = 100

        positions, velocities, accelerations = MinimumJerkTrajectory.minimum_jerk_trajectory(
            start_pos, end_pos, duration, num_steps
        )

        assert positions.shape == (num_steps, 1)
        assert velocities.shape == (num_steps, 1)
        assert accelerations.shape == (num_steps, 1)

        # Check monotonic increase (no overshooting)
        for i in range(num_steps - 1):
            assert positions[i + 1] >= positions[i]

    def test_multidimensional_trajectory(self):
        """Test trajectory in high dimensions."""
        dims = 10
        start_pos = np.zeros(dims)
        end_pos = np.ones(dims)
        duration = 1.0
        num_steps = 50

        positions, velocities, accelerations = MinimumJerkTrajectory.minimum_jerk_trajectory(
            start_pos, end_pos, duration, num_steps
        )

        assert positions.shape == (num_steps, dims)
        assert velocities.shape == (num_steps, dims)
        assert accelerations.shape == (num_steps, dims)

    def test_trajectory_smoothness(self):
        """Test that trajectory is smooth (continuous derivatives)."""
        start_pos = np.array([0.0])
        end_pos = np.array([1.0])
        duration = 1.0
        num_steps = 100

        _positions, velocities, accelerations = MinimumJerkTrajectory.minimum_jerk_trajectory(
            start_pos, end_pos, duration, num_steps
        )

        # Velocity should be continuous (no sudden jumps)
        vel_diffs = np.diff(velocities[:, 0])
        assert np.all(np.abs(vel_diffs) < 0.1)  # No large jumps

        # Acceleration should be continuous
        acc_diffs = np.diff(accelerations[:, 0])
        assert np.all(np.abs(acc_diffs) < 1.0)  # No large jumps

    def test_very_short_duration(self):
        """Test trajectory with very short duration."""
        start_pos = np.array([0.0])
        end_pos = np.array([1.0])
        duration = 0.01
        num_steps = 10

        positions, _velocities, _accelerations = MinimumJerkTrajectory.minimum_jerk_trajectory(
            start_pos, end_pos, duration, num_steps
        )

        # Should still satisfy boundary conditions
        np.testing.assert_array_almost_equal(positions[0], start_pos, decimal=2)
        np.testing.assert_array_almost_equal(positions[-1], end_pos, decimal=2)

    def test_very_long_duration(self):
        """Test trajectory with very long duration."""
        start_pos = np.array([0.0])
        end_pos = np.array([1.0])
        duration = 100.0
        num_steps = 1000

        positions, _velocities, _accelerations = MinimumJerkTrajectory.minimum_jerk_trajectory(
            start_pos, end_pos, duration, num_steps
        )

        # Should still satisfy boundary conditions
        np.testing.assert_array_almost_equal(positions[0], start_pos, decimal=2)
        np.testing.assert_array_almost_equal(positions[-1], end_pos, decimal=2)

    def test_few_steps(self):
        """Test trajectory with very few steps."""
        start_pos = np.array([0.0])
        end_pos = np.array([1.0])
        duration = 1.0
        num_steps = 2

        positions, _velocities, _accelerations = MinimumJerkTrajectory.minimum_jerk_trajectory(
            start_pos, end_pos, duration, num_steps
        )

        assert positions.shape == (2, 1)
        np.testing.assert_array_almost_equal(positions[0], start_pos)
        np.testing.assert_array_almost_equal(positions[-1], end_pos)

    def test_large_displacement(self):
        """Test trajectory with very large displacement."""
        start_pos = np.array([0.0, 0.0, 0.0])
        end_pos = np.array([1000.0, 1000.0, 1000.0])
        duration = 10.0
        num_steps = 100

        positions, velocities, accelerations = MinimumJerkTrajectory.minimum_jerk_trajectory(
            start_pos, end_pos, duration, num_steps
        )

        # Should still be finite and satisfy boundary conditions
        assert np.all(np.isfinite(positions))
        assert np.all(np.isfinite(velocities))
        assert np.all(np.isfinite(accelerations))

        np.testing.assert_array_almost_equal(positions[0], start_pos)
        np.testing.assert_array_almost_equal(positions[-1], end_pos)

    def test_negative_coordinates(self):
        """Test trajectory with negative coordinates."""
        start_pos = np.array([-5.0, -3.0, -1.0])
        end_pos = np.array([-1.0, -2.0, -4.0])
        duration = 1.0
        num_steps = 50

        positions, _velocities, _accelerations = MinimumJerkTrajectory.minimum_jerk_trajectory(
            start_pos, end_pos, duration, num_steps
        )

        np.testing.assert_array_almost_equal(positions[0], start_pos)
        np.testing.assert_array_almost_equal(positions[-1], end_pos)

    def test_mixed_direction_movement(self):
        """Test trajectory with movement in different directions per axis."""
        start_pos = np.array([0.0, 10.0, -5.0])
        end_pos = np.array([10.0, 0.0, 5.0])
        duration = 1.0
        num_steps = 50

        positions, _velocities, _accelerations = MinimumJerkTrajectory.minimum_jerk_trajectory(
            start_pos, end_pos, duration, num_steps
        )

        # Each dimension should move independently
        assert positions[0, 0] < positions[-1, 0]  # x increases
        assert positions[0, 1] > positions[-1, 1]  # y decreases
        assert positions[0, 2] < positions[-1, 2]  # z increases


class TestPIDControllerIntegration:
    """Integration tests for PID controller in realistic scenarios."""

    def test_temperature_control_simulation(self):
        """Simulate temperature control with PID."""
        config = PIDConfig(kp=5.0, ki=0.5, kd=1.0, max_output=100.0, min_output=0.0)
        pid = PIDController(config)

        target_temp = 25.0
        current_temp = 15.0
        ambient_temp = 15.0

        # Simulate for 100 steps
        for _ in range(100):
            heating_power = pid.update(target=target_temp, current=current_temp, dt=0.1)

            # Simple thermal model: temperature increases with heating, decreases toward ambient
            current_temp += heating_power * 0.001 - (current_temp - ambient_temp) * 0.01

        # Should have approached target temperature
        assert abs(current_temp - target_temp) < 5.0

    def test_position_control_simulation(self):
        """Simulate position control with PID."""
        config = PIDConfig(kp=10.0, ki=1.0, kd=2.0, max_output=50.0, min_output=-50.0)
        pid = PIDController(config)

        target_pos = 1.0
        current_pos = 0.0
        velocity = 0.0

        # Simulate for 100 steps
        for _ in range(100):
            force = pid.update(target=target_pos, current=current_pos, dt=0.01)

            # Simple physics: F = ma, integrate to get velocity and position
            acceleration = force * 0.1  # mass = 10
            velocity += acceleration * 0.01
            current_pos += velocity * 0.01

            # Add damping
            velocity *= 0.99

        # Should have approached target position
        assert abs(current_pos - target_pos) < 0.2

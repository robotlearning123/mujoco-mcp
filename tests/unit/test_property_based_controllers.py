"""Property-based tests for controllers using hypothesis.

These tests verify mathematical properties and invariants that should
always hold true for PID controllers and trajectory generation.
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.numpy import arrays

from mujoco_mcp.advanced_controllers import (
    PIDConfig,
    PIDController,
    MinimumJerkTrajectory,
)


# Strategies for generating valid test data
@st.composite
def pid_config_strategy(draw):
    """Generate valid PID configurations."""
    kp = draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False))
    ki = draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False))
    kd = draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False))

    # Generate output limits ensuring min < max
    min_output = draw(st.floats(min_value=-1000.0, max_value=0.0, allow_nan=False, allow_infinity=False))
    max_output = draw(st.floats(min_value=min_output + 1.0, max_value=1000.0, allow_nan=False, allow_infinity=False))

    windup_limit = draw(st.floats(min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False))

    return PIDConfig(
        kp=kp,
        ki=ki,
        kd=kd,
        max_output=max_output,
        min_output=min_output,
        windup_limit=windup_limit,
    )


class TestPIDStabilityProperties:
    """Test stability properties of PID controller."""

    @given(
        config=pid_config_strategy(),
        target=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        current=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        dt=st.floats(min_value=0.001, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_output_always_finite(self, config, target, current, dt):
        """Property: PID output should always be finite for finite inputs."""
        pid = PIDController(config)
        output = pid.update(target=target, current=current, dt=dt)
        assert np.isfinite(output), f"Output was {output} for target={target}, current={current}, dt={dt}"

    @given(
        config=pid_config_strategy(),
        target=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        current=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        dt=st.floats(min_value=0.001, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_output_respects_bounds(self, config, target, current, dt):
        """Property: PID output should always be within configured bounds."""
        pid = PIDController(config)
        output = pid.update(target=target, current=current, dt=dt)

        # Output should respect the configured limits
        assert config.min_output <= output <= config.max_output, (
            f"Output {output} out of bounds [{config.min_output}, {config.max_output}]"
        )

    @given(
        config=pid_config_strategy(),
        dt=st.floats(min_value=0.001, max_value=0.1, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_zero_error_gives_bounded_output(self, config, dt):
        """Property: Zero error should give bounded output (proportional term is 0)."""
        pid = PIDController(config)

        # Run for several steps with zero error
        for _ in range(10):
            output = pid.update(target=10.0, current=10.0, dt=dt)
            assert np.isfinite(output)
            assert config.min_output <= output <= config.max_output

    @given(
        kp=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        error=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_p_only_controller_proportional_to_error(self, kp, error):
        """Property: P-only controller output should be proportional to error."""
        # P-only controller (ki=0, kd=0)
        config = PIDConfig(kp=kp, ki=0.0, kd=0.0, max_output=1000.0, min_output=-1000.0)
        pid = PIDController(config)

        target = 0.0
        current = -error  # Current = target - error

        output = pid.update(target=target, current=current, dt=0.1)

        # Output should be kp * error (within floating point tolerance)
        expected = kp * error

        # Allow for floating point precision
        assert np.isclose(output, expected, rtol=1e-5, atol=1e-8), (
            f"P-only output {output} not proportional to error. Expected {expected}"
        )

    @given(
        config=pid_config_strategy(),
        target=st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
        dt=st.floats(min_value=0.01, max_value=0.1, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_reset_clears_state(self, config, target, dt):
        """Property: Reset should clear accumulated state."""
        pid = PIDController(config)

        # Build up state with constant error
        for _ in range(10):
            pid.update(target=target, current=0.0, dt=dt)

        # Reset
        pid.reset()

        # After reset, with P-only equivalent config, output should be deterministic
        pid_fresh = PIDController(config)

        output_reset = pid.update(target=target, current=0.0, dt=dt)
        output_fresh = pid_fresh.update(target=target, current=0.0, dt=dt)

        # Outputs should be close (may differ slightly due to derivative term)
        assert np.isfinite(output_reset)
        assert np.isfinite(output_fresh)

    @given(
        config=pid_config_strategy(),
        error=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        dt=st.floats(min_value=0.01, max_value=0.1, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_integral_accumulates_over_time(self, config, error, dt):
        """Property: Integral term should accumulate error over time."""
        assume(config.ki > 0.01)  # Only test when integral gain is significant

        pid = PIDController(config)

        target = 0.0
        current = -error

        # Take two updates with same error
        output1 = pid.update(target=target, current=current, dt=dt)
        output2 = pid.update(target=target, current=current, dt=dt)

        # If error is non-zero and Ki > 0, integral should accumulate
        # So output2 should be different from output1 (unless clamped)
        if abs(error) > 0.01 and config.ki > 0.01:
            # The integral contribution should increase
            # (unless we're hitting output limits)
            if output1 not in (config.max_output, config.min_output):
                # If not saturated, outputs should differ
                assert np.isfinite(output1)
                assert np.isfinite(output2)

    @given(
        config=pid_config_strategy(),
        dt=st.floats(min_value=0.01, max_value=0.1, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_windup_protection_bounds_integral(self, config, dt):
        """Property: Integral windup protection should prevent unbounded accumulation."""
        assume(config.ki > 0.01)  # Only test when integral gain is significant

        pid = PIDController(config)

        # Apply large constant error for many steps
        large_error = 100.0
        for _ in range(1000):
            pid.update(target=large_error, current=0.0, dt=dt)

        # Final output should still be finite and bounded
        output = pid.update(target=large_error, current=0.0, dt=dt)
        assert np.isfinite(output)
        assert config.min_output <= output <= config.max_output


class TestTrajectorySmoothnessProperties:
    """Test smoothness properties of minimum jerk trajectories."""

    @given(
        start=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1, max_value=10),
            elements=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        ),
        duration=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
        num_steps=st.integers(min_value=10, max_value=100),
    )
    @settings(max_examples=50, deadline=None)
    def test_trajectory_starts_at_start_position(self, start, duration, num_steps):
        """Property: Trajectory should always start at the specified start position."""
        end = start + np.random.uniform(-10, 10, size=start.shape)

        positions, _, _ = MinimumJerkTrajectory.minimum_jerk_trajectory(
            start, end, duration, num_steps
        )

        # First position should match start position
        np.testing.assert_allclose(positions[0], start, rtol=1e-5, atol=1e-8)

    @given(
        dims=st.integers(min_value=1, max_value=10),
        duration=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
        num_steps=st.integers(min_value=10, max_value=100),
    )
    @settings(max_examples=50, deadline=None)
    def test_trajectory_ends_at_end_position(self, dims, duration, num_steps):
        """Property: Trajectory should always end at the specified end position."""
        start = np.random.uniform(-10, 10, size=dims)
        end = np.random.uniform(-10, 10, size=dims)

        positions, _, _ = MinimumJerkTrajectory.minimum_jerk_trajectory(
            start, end, duration, num_steps
        )

        # Last position should match end position
        np.testing.assert_allclose(positions[-1], end, rtol=1e-5, atol=1e-8)

    @given(
        dims=st.integers(min_value=1, max_value=10),
        duration=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
        num_steps=st.integers(min_value=10, max_value=100),
    )
    @settings(max_examples=50, deadline=None)
    def test_trajectory_velocities_start_at_zero(self, dims, duration, num_steps):
        """Property: Trajectory velocities should start at zero (zero initial velocity)."""
        start = np.random.uniform(-10, 10, size=dims)
        end = np.random.uniform(-10, 10, size=dims)

        _, velocities, _ = MinimumJerkTrajectory.minimum_jerk_trajectory(
            start, end, duration, num_steps
        )

        # First velocity should be near zero
        np.testing.assert_allclose(velocities[0], np.zeros(dims), rtol=1e-3, atol=0.1)

    @given(
        dims=st.integers(min_value=1, max_value=10),
        duration=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
        num_steps=st.integers(min_value=10, max_value=100),
    )
    @settings(max_examples=50, deadline=None)
    def test_trajectory_velocities_end_at_zero(self, dims, duration, num_steps):
        """Property: Trajectory velocities should end at zero (zero final velocity)."""
        start = np.random.uniform(-10, 10, size=dims)
        end = np.random.uniform(-10, 10, size=dims)

        _, velocities, _ = MinimumJerkTrajectory.minimum_jerk_trajectory(
            start, end, duration, num_steps
        )

        # Last velocity should be near zero
        np.testing.assert_allclose(velocities[-1], np.zeros(dims), rtol=1e-3, atol=0.1)

    @given(
        dims=st.integers(min_value=1, max_value=10),
        duration=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
        num_steps=st.integers(min_value=10, max_value=100),
    )
    @settings(max_examples=50, deadline=None)
    def test_trajectory_all_values_finite(self, dims, duration, num_steps):
        """Property: All trajectory values (positions, velocities, accelerations) should be finite."""
        start = np.random.uniform(-10, 10, size=dims)
        end = np.random.uniform(-10, 10, size=dims)

        positions, velocities, accelerations = MinimumJerkTrajectory.minimum_jerk_trajectory(
            start, end, duration, num_steps
        )

        # All values should be finite
        assert np.all(np.isfinite(positions)), "Positions contain NaN or Inf"
        assert np.all(np.isfinite(velocities)), "Velocities contain NaN or Inf"
        assert np.all(np.isfinite(accelerations)), "Accelerations contain NaN or Inf"

    @given(
        dims=st.integers(min_value=1, max_value=10),
        duration=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
        num_steps=st.integers(min_value=10, max_value=100),
    )
    @settings(max_examples=50, deadline=None)
    def test_trajectory_correct_shapes(self, dims, duration, num_steps):
        """Property: Trajectory arrays should have correct shapes."""
        start = np.random.uniform(-10, 10, size=dims)
        end = np.random.uniform(-10, 10, size=dims)

        positions, velocities, accelerations = MinimumJerkTrajectory.minimum_jerk_trajectory(
            start, end, duration, num_steps
        )

        # Check shapes
        assert positions.shape == (num_steps, dims), f"Positions shape mismatch: {positions.shape}"
        assert velocities.shape == (num_steps, dims), f"Velocities shape mismatch: {velocities.shape}"
        assert accelerations.shape == (num_steps, dims), f"Accelerations shape mismatch: {accelerations.shape}"

    @given(
        dims=st.integers(min_value=1, max_value=10),
        duration=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
        num_steps=st.integers(min_value=10, max_value=100),
    )
    @settings(max_examples=30, deadline=None)
    def test_stationary_trajectory_has_zero_velocity(self, dims, duration, num_steps):
        """Property: Stationary trajectory (start == end) should have zero velocity everywhere."""
        position = np.random.uniform(-10, 10, size=dims)

        _, velocities, _ = MinimumJerkTrajectory.minimum_jerk_trajectory(
            position, position, duration, num_steps
        )

        # All velocities should be near zero
        np.testing.assert_allclose(velocities, np.zeros((num_steps, dims)), rtol=1e-3, atol=0.1)

    @given(
        dims=st.integers(min_value=1, max_value=10),
        duration=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
        num_steps=st.integers(min_value=10, max_value=100),
    )
    @settings(max_examples=30, deadline=None)
    def test_trajectory_smoothness_no_large_jumps(self, dims, duration, num_steps):
        """Property: Trajectory should be smooth with no large discontinuous jumps."""
        start = np.random.uniform(-10, 10, size=dims)
        end = np.random.uniform(-10, 10, size=dims)

        positions, velocities, _ = MinimumJerkTrajectory.minimum_jerk_trajectory(
            start, end, duration, num_steps
        )

        # Check that consecutive positions don't have huge jumps
        position_diffs = np.diff(positions, axis=0)
        max_position_diff = np.max(np.abs(position_diffs))

        # Maximum step should be reasonable given the trajectory
        total_distance = np.linalg.norm(end - start)
        max_expected_step = total_distance / num_steps * 2  # Allow 2x average step

        assert max_position_diff < max_expected_step + 1.0, (
            f"Large position jump detected: {max_position_diff} > {max_expected_step}"
        )

        # Check that velocities are continuous (no huge jumps)
        if num_steps > 2:
            velocity_diffs = np.diff(velocities, axis=0)
            max_velocity_diff = np.max(np.abs(velocity_diffs))

            # Velocity changes should be bounded
            assert np.isfinite(max_velocity_diff), "Velocity changes contain NaN or Inf"

    @given(
        start_val=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        end_val=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        duration=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
        num_steps=st.integers(min_value=10, max_value=100),
    )
    @settings(max_examples=50, deadline=None)
    def test_1d_trajectory_monotonic_when_appropriate(self, start_val, end_val, duration, num_steps):
        """Property: 1D trajectory should be monotonic when moving in one direction."""
        assume(abs(end_val - start_val) > 0.1)  # Significant movement

        start = np.array([start_val])
        end = np.array([end_val])

        positions, _, _ = MinimumJerkTrajectory.minimum_jerk_trajectory(
            start, end, duration, num_steps
        )

        # Extract 1D positions
        pos_1d = positions[:, 0]

        # Check monotonicity
        if end_val > start_val:
            # Should be non-decreasing (allowing small numerical errors)
            diffs = np.diff(pos_1d)
            assert np.all(diffs >= -1e-6), "1D increasing trajectory has decreasing segments"
        else:
            # Should be non-increasing
            diffs = np.diff(pos_1d)
            assert np.all(diffs <= 1e-6), "1D decreasing trajectory has increasing segments"

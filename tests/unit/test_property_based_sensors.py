"""Property-based tests for sensor feedback and filtering using hypothesis.

These tests verify mathematical properties and invariants for sensor
readings, filters, and sensor fusion.
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.numpy import arrays

from mujoco_mcp.sensor_feedback import (
    SensorReading,
    SensorType,
    LowPassFilter,
    KalmanFilter1D,
)


class TestSensorReadingProperties:
    """Test invariant properties of SensorReading."""

    @given(
        sensor_id=st.text(min_size=1, max_size=50),
        quality=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        timestamp=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        data_size=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=100, deadline=None)
    def test_valid_sensor_reading_always_accepted(self, sensor_id, quality, timestamp, data_size):
        """Property: Valid sensor readings should always be created successfully."""
        data = np.random.uniform(-100, 100, size=data_size)

        reading = SensorReading(
            sensor_id=sensor_id,
            sensor_type=SensorType.FORCE,
            timestamp=timestamp,
            data=data,
            quality=quality,
        )

        # Verify properties
        assert reading.sensor_id == sensor_id
        assert reading.quality == quality
        assert reading.timestamp == timestamp
        assert np.array_equal(reading.data, data)

    @given(
        quality=st.floats(allow_nan=False, allow_infinity=False).filter(lambda x: x < 0.0 or x > 1.0),
    )
    @settings(max_examples=50, deadline=None)
    def test_invalid_quality_always_rejected(self, quality):
        """Property: Quality outside [0, 1] should always be rejected."""
        data = np.array([1.0])

        with pytest.raises(ValueError, match="quality must be in"):
            SensorReading(
                sensor_id="sensor1",
                sensor_type=SensorType.FORCE,
                timestamp=1.0,
                data=data,
                quality=quality,
            )

    @given(
        timestamp=st.floats(max_value=-0.001, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_negative_timestamp_always_rejected(self, timestamp):
        """Property: Negative timestamps should always be rejected."""
        data = np.array([1.0])

        with pytest.raises(ValueError, match="timestamp must be non-negative"):
            SensorReading(
                sensor_id="sensor1",
                sensor_type=SensorType.FORCE,
                timestamp=timestamp,
                data=data,
            )


class TestLowPassFilterProperties:
    """Test mathematical properties of low-pass filter."""

    @given(
        cutoff_freq=st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False),
        sampling_rate=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        value=st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_output_always_finite(self, cutoff_freq, sampling_rate, value):
        """Property: Filter output should always be finite for finite input."""
        assume(cutoff_freq < sampling_rate / 2)  # Below Nyquist

        lpf = LowPassFilter(cutoff_freq=cutoff_freq, sampling_rate=sampling_rate)
        output = lpf.update(value)

        assert np.isfinite(output), f"Output {output} not finite for value {value}"

    @given(
        cutoff_freq=st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False),
        sampling_rate=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        constant_value=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_converges_to_constant_input(self, cutoff_freq, sampling_rate, constant_value):
        """Property: Filter should converge to constant input value."""
        assume(cutoff_freq < sampling_rate / 2)

        lpf = LowPassFilter(cutoff_freq=cutoff_freq, sampling_rate=sampling_rate)

        # Feed constant value many times
        for _ in range(1000):
            output = lpf.update(constant_value)

        # Should converge close to input value
        assert np.isclose(output, constant_value, rtol=0.05, atol=0.1), (
            f"Filter output {output} did not converge to {constant_value}"
        )

    @given(
        cutoff_freq=st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False),
        sampling_rate=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_reset_returns_to_initial_state(self, cutoff_freq, sampling_rate):
        """Property: Reset should return filter to initial state."""
        assume(cutoff_freq < sampling_rate / 2)

        lpf = LowPassFilter(cutoff_freq=cutoff_freq, sampling_rate=sampling_rate)

        # Build up state
        for _ in range(100):
            lpf.update(10.0)

        # Reset
        lpf.reset()

        # After reset, first output with zero input should be close to zero
        output_after_reset = lpf.update(0.0)

        # Fresh filter with same input
        lpf_fresh = LowPassFilter(cutoff_freq=cutoff_freq, sampling_rate=sampling_rate)
        output_fresh = lpf_fresh.update(0.0)

        assert np.isclose(output_after_reset, output_fresh, rtol=1e-5, atol=1e-8), (
            "Filter state not reset properly"
        )

    @given(
        cutoff_freq=st.floats(min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False),
        sampling_rate=st.floats(min_value=100.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30, deadline=None)
    def test_attenuates_high_frequency_noise(self, cutoff_freq, sampling_rate):
        """Property: Filter should reduce variance of high-frequency noise."""
        # Generate high-frequency noise (frequency = 2 * cutoff_freq)
        noise_freq = cutoff_freq * 3

        lpf = LowPassFilter(cutoff_freq=cutoff_freq, sampling_rate=sampling_rate)

        noisy_samples = []
        filtered_samples = []

        # Generate samples
        for i in range(100):
            t = i / sampling_rate
            # High-frequency sine wave
            noise = np.sin(2 * np.pi * noise_freq * t)
            noisy_samples.append(noise)
            filtered_samples.append(lpf.update(noise))

        # Skip first samples (transient)
        noisy_variance = np.var(noisy_samples[50:])
        filtered_variance = np.var(filtered_samples[50:])

        # Filtered signal should have lower variance
        assert filtered_variance < noisy_variance, (
            f"Filter did not reduce noise variance: {filtered_variance} >= {noisy_variance}"
        )

    @given(
        cutoff_freq=st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False),
        sampling_rate=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        value=st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_output_bounded_by_input_range(self, cutoff_freq, sampling_rate, value):
        """Property: For constant input, output should eventually be bounded by input range."""
        assume(cutoff_freq < sampling_rate / 2)

        lpf = LowPassFilter(cutoff_freq=cutoff_freq, sampling_rate=sampling_rate)

        # Feed constant value
        for _ in range(100):
            output = lpf.update(value)

        # Output should be within reasonable range of input
        # (may overshoot slightly during transient)
        assert -abs(value) * 2 <= output <= abs(value) * 2, (
            f"Output {output} far outside input range [{-abs(value)*2}, {abs(value)*2}]"
        )


class TestKalmanFilterProperties:
    """Test mathematical properties of Kalman filter."""

    @given(
        process_var=st.floats(min_value=0.001, max_value=10.0, allow_nan=False, allow_infinity=False),
        measurement_var=st.floats(min_value=0.001, max_value=10.0, allow_nan=False, allow_infinity=False),
        measurement=st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_output_always_finite(self, process_var, measurement_var, measurement):
        """Property: Kalman filter output should always be finite for finite measurement."""
        kf = KalmanFilter1D(process_variance=process_var, measurement_variance=measurement_var)
        estimate = kf.update(measurement)

        assert np.isfinite(estimate), f"Estimate {estimate} not finite for measurement {measurement}"

    @given(
        process_var=st.floats(min_value=0.001, max_value=1.0, allow_nan=False, allow_infinity=False),
        measurement_var=st.floats(min_value=0.001, max_value=1.0, allow_nan=False, allow_infinity=False),
        true_value=st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_converges_to_constant_measurements(self, process_var, measurement_var, true_value):
        """Property: Filter should converge to constant measurement value."""
        kf = KalmanFilter1D(process_variance=process_var, measurement_variance=measurement_var)

        # Feed constant measurements
        for _ in range(200):
            estimate = kf.update(true_value)

        # Should converge close to true value
        assert np.isclose(estimate, true_value, rtol=0.1, atol=0.5), (
            f"Filter estimate {estimate} did not converge to {true_value}"
        )

    @given(
        process_var=st.floats(min_value=0.001, max_value=10.0, allow_nan=False, allow_infinity=False),
        measurement_var=st.floats(min_value=0.001, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_reset_clears_state(self, process_var, measurement_var):
        """Property: Reset should clear accumulated state."""
        kf = KalmanFilter1D(process_variance=process_var, measurement_variance=measurement_var)

        # Build up state
        for _ in range(100):
            kf.update(10.0)

        # Reset
        kf.reset()

        # After reset, estimate should be fresh
        estimate_after_reset = kf.update(5.0)

        # Fresh filter
        kf_fresh = KalmanFilter1D(process_variance=process_var, measurement_variance=measurement_var)
        estimate_fresh = kf_fresh.update(5.0)

        # Should be similar (may have small differences)
        assert np.isclose(estimate_after_reset, estimate_fresh, rtol=0.1, atol=1.0), (
            "Filter state not reset properly"
        )

    @given(
        process_var=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
        measurement_var=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
        true_value=st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30, deadline=None)
    def test_reduces_measurement_noise(self, process_var, measurement_var, true_value):
        """Property: Filter should reduce noise in measurements."""
        np.random.seed(42)  # For reproducibility

        kf = KalmanFilter1D(process_variance=process_var, measurement_variance=measurement_var)

        measurements = []
        estimates = []

        # Generate noisy measurements
        for _ in range(200):
            noise = np.random.normal(0, np.sqrt(measurement_var))
            measurement = true_value + noise
            estimate = kf.update(measurement)

            measurements.append(measurement)
            estimates.append(estimate)

        # Skip transient (first 100 samples)
        measurements_steady = measurements[100:]
        estimates_steady = estimates[100:]

        # Estimates should have lower variance than measurements
        measurement_variance_actual = np.var(measurements_steady)
        estimate_variance = np.var(estimates_steady)

        assert estimate_variance < measurement_variance_actual, (
            f"Filter did not reduce variance: {estimate_variance} >= {measurement_variance_actual}"
        )

    @given(
        process_var=st.floats(min_value=0.001, max_value=10.0, allow_nan=False, allow_infinity=False),
        true_value=st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30, deadline=None)
    def test_perfect_sensor_trusts_measurements(self, process_var, true_value):
        """Property: With very low measurement variance, filter should trust measurements."""
        # Very low measurement variance (near-perfect sensor)
        measurement_var = 1e-10

        kf = KalmanFilter1D(process_variance=process_var, measurement_variance=measurement_var)

        # Single measurement
        estimate = kf.update(true_value)

        # Should be very close to measurement (high trust)
        assert np.isclose(estimate, true_value, rtol=0.01, atol=0.01), (
            f"Perfect sensor estimate {estimate} not close to measurement {true_value}"
        )

    @given(
        measurement_var=st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        true_value=st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30, deadline=None)
    def test_unreliable_sensor_changes_slowly(self, measurement_var, true_value):
        """Property: With high measurement variance, filter should change slowly."""
        # Very low process variance (static system)
        process_var = 0.001

        kf = KalmanFilter1D(process_variance=process_var, measurement_variance=measurement_var)

        # Start with zero estimate
        kf.update(0.0)

        # New measurement far from current estimate
        estimate = kf.update(true_value)

        # Should not jump immediately to measurement (low trust in noisy sensor)
        assert abs(estimate) < abs(true_value) * 0.9, (
            f"Unreliable sensor estimate {estimate} jumped too close to measurement {true_value}"
        )

    @given(
        process_var=st.floats(min_value=0.001, max_value=10.0, allow_nan=False, allow_infinity=False),
        measurement_var=st.floats(min_value=0.001, max_value=10.0, allow_nan=False, allow_infinity=False),
        true_value=st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
        outlier_value=st.floats(min_value=-200.0, max_value=200.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30, deadline=None)
    def test_recovers_from_outliers(self, process_var, measurement_var, true_value, outlier_value):
        """Property: Filter should recover from measurement outliers."""
        assume(abs(outlier_value - true_value) > 10)  # Significant outlier

        kf = KalmanFilter1D(process_variance=process_var, measurement_variance=measurement_var)

        # Build steady state
        for _ in range(100):
            kf.update(true_value)

        # Inject outlier
        kf.update(outlier_value)

        # Continue with true measurements
        for _ in range(100):
            estimate = kf.update(true_value)

        # Should recover close to true value
        assert np.isclose(estimate, true_value, rtol=0.2, atol=2.0), (
            f"Filter did not recover from outlier: {estimate} vs {true_value}"
        )


class TestFilterNumericalStability:
    """Test numerical stability properties of filters under extreme conditions."""

    @given(
        cutoff_freq=st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False),
        sampling_rate=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        value=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_lowpass_stable_with_large_values(self, cutoff_freq, sampling_rate, value):
        """Property: LowPass filter should remain stable with large input values."""
        assume(cutoff_freq < sampling_rate / 2)

        lpf = LowPassFilter(cutoff_freq=cutoff_freq, sampling_rate=sampling_rate)

        # Feed large value multiple times
        for _ in range(100):
            output = lpf.update(value)

        assert np.isfinite(output), f"Filter became unstable with value {value}"
        assert abs(output) < abs(value) * 2, "Filter output exploded"

    @given(
        process_var=st.floats(min_value=1e-6, max_value=100.0, allow_nan=False, allow_infinity=False),
        measurement_var=st.floats(min_value=1e-6, max_value=100.0, allow_nan=False, allow_infinity=False),
        measurement=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_kalman_stable_with_large_values(self, process_var, measurement_var, measurement):
        """Property: Kalman filter should remain stable with large measurements."""
        kf = KalmanFilter1D(process_variance=process_var, measurement_variance=measurement_var)

        # Feed large measurement multiple times
        for _ in range(100):
            estimate = kf.update(measurement)

        assert np.isfinite(estimate), f"Filter became unstable with measurement {measurement}"
        assert abs(estimate) < abs(measurement) * 2, "Filter estimate exploded"

    @given(
        cutoff_freq=st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False),
        sampling_rate=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_lowpass_stable_with_alternating_extremes(self, cutoff_freq, sampling_rate):
        """Property: LowPass filter should remain stable with rapidly alternating extreme values."""
        assume(cutoff_freq < sampling_rate / 2)

        lpf = LowPassFilter(cutoff_freq=cutoff_freq, sampling_rate=sampling_rate)

        # Alternate between extreme values
        for i in range(100):
            value = 1000.0 if i % 2 == 0 else -1000.0
            output = lpf.update(value)

        assert np.isfinite(output), "Filter became unstable with alternating values"
        assert abs(output) < 2000.0, "Filter output exploded with alternating values"

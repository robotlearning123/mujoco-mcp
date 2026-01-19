"""Comprehensive unit tests for sensor_feedback.py covering division by zero, filter stability, thread safety."""

import threading
import time

import numpy as np
import pytest

from mujoco_mcp.sensor_feedback import (
    SensorReading,
    SensorType,
    LowPassFilter,
    KalmanFilter1D,
)


class TestSensorReading:
    """Test SensorReading dataclass validation."""

    def test_valid_sensor_reading(self):
        """Test creating valid sensor reading."""
        data = np.array([1.0, 2.0, 3.0])
        reading = SensorReading(
            sensor_id="sensor1",
            sensor_type=SensorType.FORCE,
            timestamp=1.0,
            data=data,
            quality=0.95,
        )
        assert reading.sensor_id == "sensor1"
        assert reading.quality == 0.95

    def test_quality_bounds_lower(self):
        """Test quality must be >= 0."""
        data = np.array([1.0])
        with pytest.raises(ValueError, match="quality must be in \\[0, 1\\]"):
            SensorReading(
                sensor_id="sensor1",
                sensor_type=SensorType.FORCE,
                timestamp=1.0,
                data=data,
                quality=-0.1,
            )

    def test_quality_bounds_upper(self):
        """Test quality must be <= 1."""
        data = np.array([1.0])
        with pytest.raises(ValueError, match="quality must be in \\[0, 1\\]"):
            SensorReading(
                sensor_id="sensor1",
                sensor_type=SensorType.FORCE,
                timestamp=1.0,
                data=data,
                quality=1.1,
            )

    def test_quality_boundary_values(self):
        """Test quality exactly 0 and 1 are valid."""
        data = np.array([1.0])

        # Quality = 0 should be valid
        reading0 = SensorReading(
            sensor_id="sensor1",
            sensor_type=SensorType.FORCE,
            timestamp=1.0,
            data=data,
            quality=0.0,
        )
        assert reading0.quality == 0.0

        # Quality = 1 should be valid
        reading1 = SensorReading(
            sensor_id="sensor1",
            sensor_type=SensorType.FORCE,
            timestamp=1.0,
            data=data,
            quality=1.0,
        )
        assert reading1.quality == 1.0

    def test_negative_timestamp(self):
        """Test negative timestamp is rejected."""
        data = np.array([1.0])
        with pytest.raises(ValueError, match="timestamp must be non-negative"):
            SensorReading(
                sensor_id="sensor1",
                sensor_type=SensorType.FORCE,
                timestamp=-1.0,
                data=data,
            )

    def test_zero_timestamp(self):
        """Test zero timestamp is valid."""
        data = np.array([1.0])
        reading = SensorReading(
            sensor_id="sensor1",
            sensor_type=SensorType.FORCE,
            timestamp=0.0,
            data=data,
        )
        assert reading.timestamp == 0.0

    def test_frozen_dataclass(self):
        """Test that SensorReading is immutable."""
        data = np.array([1.0])
        reading = SensorReading(
            sensor_id="sensor1",
            sensor_type=SensorType.FORCE,
            timestamp=1.0,
            data=data,
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            reading.quality = 0.5


class TestLowPassFilter:
    """Test low-pass filter behavior and stability."""

    def test_initialization(self):
        """Test filter initialization."""
        lpf = LowPassFilter(cutoff_freq=10.0, sampling_rate=100.0)
        assert lpf.cutoff_freq == 10.0
        assert lpf.sampling_rate == 100.0

    def test_steady_state_response(self):
        """Test filter passes constant signal unchanged at steady state."""
        lpf = LowPassFilter(cutoff_freq=10.0, sampling_rate=100.0)

        # Feed constant signal
        constant_value = 5.0
        for _ in range(100):
            output = lpf.update(constant_value)

        # After many samples, should converge to input value
        assert np.isclose(output, constant_value, rtol=0.01)

    def test_step_response(self):
        """Test filter response to step input."""
        lpf = LowPassFilter(cutoff_freq=10.0, sampling_rate=100.0)

        # Start with zeros
        for _ in range(10):
            lpf.update(0.0)

        # Step to 1.0
        for _ in range(10):
            output = lpf.update(1.0)

        # Should be between 0 and 1 (not fully settled)
        assert 0.0 < output < 1.0

        # After many more samples, should approach 1.0
        for _ in range(100):
            output = lpf.update(1.0)
        assert np.isclose(output, 1.0, rtol=0.01)

    def test_high_frequency_attenuation(self):
        """Test filter attenuates high-frequency noise."""
        lpf = LowPassFilter(cutoff_freq=5.0, sampling_rate=100.0)

        # Add high-frequency noise to constant signal
        base_signal = 10.0
        noisy_outputs = []
        filtered_outputs = []

        for i in range(100):
            # Add high-frequency sine wave
            noise = 2.0 * np.sin(2 * np.pi * 20.0 * i / 100.0)
            noisy_signal = base_signal + noise

            noisy_outputs.append(noisy_signal)
            filtered_outputs.append(lpf.update(noisy_signal))

        # Filtered signal should have less variance
        assert np.std(filtered_outputs[50:]) < np.std(noisy_outputs[50:])

    def test_reset(self):
        """Test reset clears filter state."""
        lpf = LowPassFilter(cutoff_freq=10.0, sampling_rate=100.0)

        # Build up state
        for _ in range(50):
            lpf.update(5.0)

        # Reset
        lpf.reset()

        # After reset, first output should be close to zero (default state)
        output = lpf.update(0.0)
        assert np.isclose(output, 0.0, atol=0.1)

    def test_very_low_cutoff(self):
        """Test filter with very low cutoff frequency."""
        lpf = LowPassFilter(cutoff_freq=0.1, sampling_rate=100.0)

        # Should be very slow to respond
        lpf.update(0.0)
        output = lpf.update(10.0)

        # Should barely move
        assert output < 0.5

    def test_cutoff_equals_nyquist(self):
        """Test filter when cutoff equals Nyquist frequency."""
        # Nyquist frequency = sampling_rate / 2
        lpf = LowPassFilter(cutoff_freq=50.0, sampling_rate=100.0)

        # Should still be stable
        for _ in range(100):
            output = lpf.update(1.0)

        assert np.isfinite(output)

    def test_zero_division_protection(self):
        """Test filter handles edge cases that could cause division by zero."""
        # Very high cutoff frequency (near sampling rate)
        lpf = LowPassFilter(cutoff_freq=99.9, sampling_rate=100.0)

        for _ in range(10):
            output = lpf.update(1.0)

        # Should remain finite
        assert np.isfinite(output)

    def test_stability_with_large_values(self):
        """Test filter stability with large input values."""
        lpf = LowPassFilter(cutoff_freq=10.0, sampling_rate=100.0)

        # Feed very large values
        for _ in range(100):
            output = lpf.update(1e6)

        # Should converge to large value without overflow
        assert np.isfinite(output)
        assert np.isclose(output, 1e6, rtol=0.01)

    def test_stability_with_negative_values(self):
        """Test filter with negative values."""
        lpf = LowPassFilter(cutoff_freq=10.0, sampling_rate=100.0)

        for _ in range(100):
            output = lpf.update(-5.0)

        assert np.isclose(output, -5.0, rtol=0.01)

    def test_stability_with_rapid_changes(self):
        """Test filter stability with rapidly changing input."""
        lpf = LowPassFilter(cutoff_freq=10.0, sampling_rate=100.0)

        # Alternate between extreme values
        for i in range(100):
            value = 10.0 if i % 2 == 0 else -10.0
            output = lpf.update(value)

        # Should remain finite
        assert np.isfinite(output)


class TestKalmanFilter1D:
    """Test 1D Kalman filter behavior and stability."""

    def test_initialization(self):
        """Test Kalman filter initialization."""
        kf = KalmanFilter1D(process_variance=1.0, measurement_variance=1.0)
        assert kf.process_variance == 1.0
        assert kf.measurement_variance == 1.0

    def test_constant_measurement(self):
        """Test filter with constant measurements."""
        kf = KalmanFilter1D(process_variance=0.1, measurement_variance=1.0)

        constant_value = 5.0
        for _ in range(100):
            estimate = kf.update(constant_value)

        # Should converge to measured value
        assert np.isclose(estimate, constant_value, rtol=0.05)

    def test_noisy_measurements(self):
        """Test filter reduces noise in measurements."""
        kf = KalmanFilter1D(process_variance=0.01, measurement_variance=1.0)

        true_value = 10.0
        measurements = []
        estimates = []

        np.random.seed(42)
        for _ in range(100):
            # Add Gaussian noise
            measurement = true_value + np.random.normal(0, 1.0)
            estimate = kf.update(measurement)

            measurements.append(measurement)
            estimates.append(estimate)

        # Filtered estimates should have less variance than measurements
        assert np.std(estimates[50:]) < np.std(measurements[50:])

    def test_tracking_ramp(self):
        """Test filter tracking a ramp signal."""
        kf = KalmanFilter1D(process_variance=1.0, measurement_variance=0.1)

        for i in range(100):
            # Ramp from 0 to 10
            true_value = i * 0.1
            estimate = kf.update(true_value)

        # Should track the ramp
        assert np.isfinite(estimate)

    def test_reset(self):
        """Test reset clears filter state."""
        kf = KalmanFilter1D(process_variance=0.1, measurement_variance=1.0)

        # Build up state
        for _ in range(50):
            kf.update(10.0)

        # Reset
        kf.reset()

        # After reset, uncertainty should be high again
        estimate = kf.update(5.0)
        # First estimate after reset should be close to measurement
        assert np.isclose(estimate, 5.0, rtol=0.2)

    def test_zero_process_variance(self):
        """Test filter with zero process variance (static system)."""
        kf = KalmanFilter1D(process_variance=0.0, measurement_variance=1.0)

        # Should still work
        for _ in range(10):
            estimate = kf.update(5.0)

        assert np.isfinite(estimate)

    def test_zero_measurement_variance(self):
        """Test filter with zero measurement variance (perfect sensor)."""
        kf = KalmanFilter1D(process_variance=1.0, measurement_variance=1e-10)

        # Should trust measurements completely
        measurement = 7.0
        estimate = kf.update(measurement)

        # Should be very close to measurement
        assert np.isclose(estimate, measurement, rtol=0.01)

    def test_large_process_variance(self):
        """Test filter with large process variance."""
        kf = KalmanFilter1D(process_variance=100.0, measurement_variance=1.0)

        for _ in range(50):
            estimate = kf.update(5.0)

        # Should still converge
        assert np.isfinite(estimate)

    def test_large_measurement_variance(self):
        """Test filter with large measurement variance (unreliable sensor)."""
        kf = KalmanFilter1D(process_variance=0.1, measurement_variance=100.0)

        # Should change slowly due to low trust in measurements
        kf.update(0.0)
        estimate = kf.update(10.0)

        # Should not jump immediately to measurement
        assert estimate < 5.0

    def test_stability_with_outliers(self):
        """Test filter stability with measurement outliers."""
        kf = KalmanFilter1D(process_variance=0.1, measurement_variance=1.0)

        estimates = []
        for i in range(100):
            # Occasional outlier
            if i == 50:
                measurement = 100.0  # Large outlier
            else:
                measurement = 5.0

            estimate = kf.update(measurement)
            estimates.append(estimate)

        # Filter should recover from outlier
        assert estimates[-1] < 10.0  # Should return toward 5.0

    def test_division_by_zero_protection(self):
        """Test filter handles potential division by zero."""
        # Edge case: very small variances
        kf = KalmanFilter1D(process_variance=1e-10, measurement_variance=1e-10)

        for _ in range(10):
            estimate = kf.update(5.0)

        assert np.isfinite(estimate)


class TestThreadSafety:
    """Test thread safety of filter operations."""

    def test_lowpass_filter_concurrent_updates(self):
        """Test low-pass filter with concurrent updates from multiple threads."""
        lpf = LowPassFilter(cutoff_freq=10.0, sampling_rate=100.0)
        results = []
        errors = []

        def worker():
            try:
                for _ in range(100):
                    output = lpf.update(np.random.random())
                    results.append(output)
            except Exception as e:
                errors.append(e)

        # Run multiple threads
        threads = [threading.Thread(target=worker) for _ in range(5)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Should not have errors (though results may be interleaved)
        assert len(errors) == 0
        # All results should be finite
        assert all(np.isfinite(r) for r in results)

    def test_kalman_filter_concurrent_updates(self):
        """Test Kalman filter with concurrent updates from multiple threads."""
        kf = KalmanFilter1D(process_variance=0.1, measurement_variance=1.0)
        results = []
        errors = []

        def worker():
            try:
                for _ in range(100):
                    output = kf.update(np.random.random())
                    results.append(output)
            except Exception as e:
                errors.append(e)

        # Run multiple threads
        threads = [threading.Thread(target=worker) for _ in range(5)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Should not have errors
        assert len(errors) == 0
        # All results should be finite
        assert all(np.isfinite(r) for r in results)

    def test_filter_concurrent_reset_and_update(self):
        """Test filter with concurrent reset and update operations."""
        lpf = LowPassFilter(cutoff_freq=10.0, sampling_rate=100.0)
        errors = []

        def update_worker():
            try:
                for _ in range(100):
                    lpf.update(1.0)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def reset_worker():
            try:
                for _ in range(10):
                    time.sleep(0.01)
                    lpf.reset()
            except Exception as e:
                errors.append(e)

        update_thread = threading.Thread(target=update_worker)
        reset_thread = threading.Thread(target=reset_worker)

        update_thread.start()
        reset_thread.start()

        update_thread.join()
        reset_thread.join()

        # Should not crash (though results may be unpredictable)
        assert len(errors) == 0


class TestFilterNumericalStability:
    """Test numerical stability of filters under extreme conditions."""

    def test_lowpass_filter_near_zero_values(self):
        """Test low-pass filter with values near zero."""
        lpf = LowPassFilter(cutoff_freq=10.0, sampling_rate=100.0)

        for _ in range(100):
            output = lpf.update(1e-10)

        assert np.isfinite(output)
        assert output >= 0  # Should not become negative

    def test_kalman_filter_near_zero_values(self):
        """Test Kalman filter with values near zero."""
        kf = KalmanFilter1D(process_variance=0.1, measurement_variance=1.0)

        for _ in range(100):
            estimate = kf.update(1e-10)

        assert np.isfinite(estimate)

    def test_lowpass_filter_alternating_signs(self):
        """Test low-pass filter with rapidly alternating signs."""
        lpf = LowPassFilter(cutoff_freq=10.0, sampling_rate=100.0)

        for i in range(100):
            value = 10.0 if i % 2 == 0 else -10.0
            output = lpf.update(value)

        assert np.isfinite(output)
        assert abs(output) < 15.0  # Should be bounded

    def test_kalman_filter_alternating_signs(self):
        """Test Kalman filter with rapidly alternating measurements."""
        kf = KalmanFilter1D(process_variance=0.1, measurement_variance=1.0)

        for i in range(100):
            measurement = 10.0 if i % 2 == 0 else -10.0
            estimate = kf.update(measurement)

        assert np.isfinite(estimate)
        assert abs(estimate) < 15.0  # Should be bounded

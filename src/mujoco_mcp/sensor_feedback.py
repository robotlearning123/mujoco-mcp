#!/usr/bin/env python3
"""
Sensor Feedback and Closed-Loop Control System for MuJoCo MCP
Implements various sensor modalities and feedback control loops
"""

import numpy as np
import time
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum
import threading
import queue
from abc import ABC, abstractmethod
import logging


class SensorType(Enum):
    """Types of sensors supported"""
    JOINT_POSITION = "joint_position"
    JOINT_VELOCITY = "joint_velocity"
    JOINT_TORQUE = "joint_torque"
    IMU = "imu"
    FORCE_TORQUE = "force_torque"
    CAMERA = "camera"
    LIDAR = "lidar"
    CONTACT = "contact"
    PROXIMITY = "proximity"


@dataclass
class SensorReading:
    """Sensor reading data structure"""
    sensor_id: str
    sensor_type: SensorType
    timestamp: float
    data: np.ndarray
    frame_id: str = "base_link"
    quality: float = 1.0  # Sensor quality (0-1)

    def is_valid(self, max_age: float = 0.1) -> bool:
        """Check if sensor reading is valid and recent"""
        return (time.time() - self.timestamp) < max_age and self.quality > 0.5


class SensorProcessor(ABC):
    """Abstract base class for sensor processors"""

    @abstractmethod
    def process_raw_data(self, raw_data: Any) -> SensorReading:
        """Process raw sensor data into standardized format"""

    @abstractmethod
    def calibrate(self, calibration_data: Dict[str, Any]) -> bool:
        """Calibrate sensor"""


class JointSensorProcessor(SensorProcessor):
    """Processor for joint position/velocity/torque sensors"""

    def __init__(self, sensor_id: str, sensor_type: SensorType, n_joints: int):
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self.n_joints = n_joints
        self.offset = np.zeros(n_joints)
        self.scale = np.ones(n_joints)
        self.noise_filter = LowPassFilter(cutoff_freq=10.0, n_channels=n_joints)

    def process_raw_data(self, raw_data: Any) -> SensorReading:
        """Process joint sensor data"""
        if isinstance(raw_data, (list, tuple)):
            data = np.array(raw_data)
        else:
            data = raw_data

        # Apply calibration
        calibrated_data = (data - self.offset) * self.scale

        # Apply noise filtering
        filtered_data = self.noise_filter.update(calibrated_data)

        return SensorReading(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            timestamp=time.time(),
            data=filtered_data
        )

    def calibrate(self, calibration_data: Dict[str, Any]) -> bool:
        """Calibrate joint sensor"""
        if "offset" in calibration_data:
            self.offset = np.array(calibration_data["offset"])
        if "scale" in calibration_data:
            self.scale = np.array(calibration_data["scale"])
        return True


class IMUSensorProcessor(SensorProcessor):
    """Processor for IMU (Inertial Measurement Unit) sensors"""

    def __init__(self, sensor_id: str):
        self.sensor_id = sensor_id
        self.accel_offset = np.zeros(3)
        self.gyro_offset = np.zeros(3)
        self.orientation_filter = ComplementaryFilter()

    def process_raw_data(self, raw_data: Any) -> SensorReading:
        """Process IMU data"""
        # Expected format: [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
        if len(raw_data) >= 6:
            accel = np.array(raw_data[:3]) - self.accel_offset
            gyro = np.array(raw_data[3:6]) - self.gyro_offset

            # Estimate orientation using complementary filter
            orientation = self.orientation_filter.update(accel, gyro)

            # Combine all data
            processed_data = np.concatenate([accel, gyro, orientation])
        else:
            processed_data = np.array(raw_data)

        return SensorReading(
            sensor_id=self.sensor_id,
            sensor_type=SensorType.IMU,
            timestamp=time.time(),
            data=processed_data
        )

    def calibrate(self, calibration_data: Dict[str, Any]) -> bool:
        """Calibrate IMU sensor"""
        if "accel_offset" in calibration_data:
            self.accel_offset = np.array(calibration_data["accel_offset"])
        if "gyro_offset" in calibration_data:
            self.gyro_offset = np.array(calibration_data["gyro_offset"])
        return True


class ForceTorqueSensorProcessor(SensorProcessor):
    """Processor for force/torque sensors"""

    def __init__(self, sensor_id: str):
        self.sensor_id = sensor_id
        self.force_offset = np.zeros(3)
        self.torque_offset = np.zeros(3)
        self.transformation_matrix = np.eye(6)

    def process_raw_data(self, raw_data: Any) -> SensorReading:
        """Process force/torque data"""
        # Expected format: [fx, fy, fz, tx, ty, tz]
        if len(raw_data) >= 6:
            force = np.array(raw_data[:3]) - self.force_offset
            torque = np.array(raw_data[3:6]) - self.torque_offset
            ft_data = np.concatenate([force, torque])

            # Apply transformation matrix
            transformed_data = self.transformation_matrix @ ft_data
        else:
            transformed_data = np.array(raw_data)

        return SensorReading(
            sensor_id=self.sensor_id,
            sensor_type=SensorType.FORCE_TORQUE,
            timestamp=time.time(),
            data=transformed_data
        )

    def calibrate(self, calibration_data: Dict[str, Any]) -> bool:
        """Calibrate force/torque sensor"""
        if "force_offset" in calibration_data:
            self.force_offset = np.array(calibration_data["force_offset"])
        if "torque_offset" in calibration_data:
            self.torque_offset = np.array(calibration_data["torque_offset"])
        if "transformation_matrix" in calibration_data:
            self.transformation_matrix = np.array(calibration_data["transformation_matrix"])
        return True


class LowPassFilter:
    """Low-pass filter for noise reduction"""

    def __init__(self, cutoff_freq: float, n_channels: int, dt: float = 0.02):
        self.cutoff_freq = cutoff_freq
        self.dt = dt
        self.n_channels = n_channels

        # Calculate filter coefficient
        self.alpha = self.dt / (self.dt + 1.0 / (2 * np.pi * cutoff_freq))

        # Initialize filter state
        self.prev_output = np.zeros(n_channels)

    def update(self, input_data: np.ndarray) -> np.ndarray:
        """Update filter with new input"""
        output = self.alpha * input_data + (1 - self.alpha) * self.prev_output
        self.prev_output = output
        return output

    def reset(self):
        """Reset filter state"""
        self.prev_output = np.zeros(self.n_channels)


class ComplementaryFilter:
    """Complementary filter for orientation estimation"""

    def __init__(self, alpha: float = 0.98, dt: float = 0.02):
        self.alpha = alpha
        self.dt = dt
        self.orientation = np.array([0.0, 0.0, 0.0])  # Roll, pitch, yaw

    def update(self, accel: np.ndarray, gyro: np.ndarray) -> np.ndarray:
        """Update orientation estimate"""
        # Integrate gyroscope for high-frequency component
        gyro_orientation = self.orientation + gyro * self.dt

        # Calculate orientation from accelerometer (low-frequency)
        accel_roll = np.arctan2(accel[1], accel[2])
        accel_pitch = np.arctan2(-accel[0], np.sqrt(accel[1]**2 + accel[2]**2))
        accel_orientation = np.array([accel_roll, accel_pitch, self.orientation[2]])

        # Combine using complementary filter
        self.orientation = self.alpha * gyro_orientation + (1 - self.alpha) * accel_orientation

        return self.orientation.copy()


class SensorFusion:
    """Multi-sensor fusion for improved state estimation"""

    def __init__(self):
        self.sensors = {}
        self.fused_state = {}
        self.fusion_weights = {}

    def add_sensor(self, sensor_id: str, sensor_type: SensorType, weight: float = 1.0):
        """Add sensor to fusion system"""
        self.sensors[sensor_id] = sensor_type
        self.fusion_weights[sensor_id] = weight

    def fuse_sensor_data(self, sensor_readings: List[SensorReading]) -> Dict[str, np.ndarray]:
        """Fuse multiple sensor readings"""
        fused_data = {}

        # Group readings by sensor type
        readings_by_type = {}
        for reading in sensor_readings:
            if reading.is_valid():
                sensor_type = reading.sensor_type
                if sensor_type not in readings_by_type:
                    readings_by_type[sensor_type] = []
                readings_by_type[sensor_type].append(reading)

        # Fuse readings of each type
        for sensor_type, readings in readings_by_type.items():
            if len(readings) == 1:
                fused_data[sensor_type.value] = readings[0].data
            else:
                # Weighted average fusion
                total_weight = 0
                weighted_sum = None

                for reading in readings:
                    weight = self.fusion_weights.get(reading.sensor_id, 1.0) * reading.quality

                    if weighted_sum is None:
                        weighted_sum = weight * reading.data
                    else:
                        weighted_sum += weight * reading.data
                    total_weight += weight

                if total_weight > 0:
                    fused_data[sensor_type.value] = weighted_sum / total_weight

        return fused_data


class ClosedLoopController:
    """Closed-loop controller using sensor feedback"""

    def __init__(self, controller_type: str = "pid"):
        self.controller_type = controller_type
        self.sensor_fusion = SensorFusion()
        self.control_history = []
        self.error_history = []
        self.target_state = None
        self.current_state = None

        # Control parameters
        self.control_gains = {"kp": 1.0, "ki": 0.0, "kd": 0.0}
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.prev_time = None

    def set_target(self, target_state: Dict[str, np.ndarray]):
        """Set target state for control"""
        self.target_state = target_state

    def update_state(self, sensor_readings: List[SensorReading]):
        """Update current state from sensor readings"""
        self.current_state = self.sensor_fusion.fuse_sensor_data(sensor_readings)

    def compute_control(self) -> Dict[str, np.ndarray]:
        """Compute control commands based on current and target state"""
        if self.target_state is None or self.current_state is None:
            return {}

        control_commands = {}
        current_time = time.time()

        if self.prev_time is None:
            dt = 0.02
        else:
            dt = current_time - self.prev_time

        for state_key in self.target_state:
            if state_key in self.current_state:
                target = self.target_state[state_key]
                current = self.current_state[state_key]

                # Compute error
                error = target - current

                if self.controller_type == "pid":
                    control_commands[state_key] = self._pid_control(error, dt)
                elif self.controller_type == "adaptive":
                    control_commands[state_key] = self._adaptive_control(error, current)
                else:
                    control_commands[state_key] = error  # Simple proportional

        self.prev_time = current_time
        return control_commands

    def _pid_control(self, error: np.ndarray, dt: float) -> np.ndarray:
        """PID control implementation"""
        # Proportional term
        p_term = self.control_gains["kp"] * error

        # Integral term
        self.integral_error += error * dt
        i_term = self.control_gains["ki"] * self.integral_error

        # Derivative term
        if dt > 0:
            derivative_error = (error - self.prev_error) / dt
        else:
            derivative_error = np.zeros_like(error)
        d_term = self.control_gains["kd"] * derivative_error

        self.prev_error = error.copy()

        return p_term + i_term + d_term

    def _adaptive_control(self, error: np.ndarray, current_state: np.ndarray) -> np.ndarray:
        """Adaptive control implementation"""
        # Simple adaptive law based on error magnitude
        adaptive_gain = 1.0 + 0.1 * np.linalg.norm(error)
        return adaptive_gain * self.control_gains["kp"] * error


class SensorManager:
    """Manager for all sensors in the system"""

    def __init__(self):
        self.sensors: Dict[str, SensorProcessor] = {}
        self.sensor_data_queue = queue.Queue()
        self.running = False
        self.update_thread = None
        self.update_frequency = 100.0  # Hz
        self.logger = logging.getLogger(__name__)

    def add_sensor(self, sensor_id: str, processor: SensorProcessor):
        """Add sensor to manager"""
        self.sensors[sensor_id] = processor
        self.logger.info(f"Added sensor {sensor_id}")

    def remove_sensor(self, sensor_id: str):
        """Remove sensor from manager"""
        if sensor_id in self.sensors:
            del self.sensors[sensor_id]
            self.logger.info(f"Removed sensor {sensor_id}")

    def start_sensing(self):
        """Start sensor data collection"""
        if not self.running:
            self.running = True
            self.update_thread = threading.Thread(target=self._sensing_loop)
            self.update_thread.start()
            self.logger.info("Started sensor data collection")

    def stop_sensing(self):
        """Stop sensor data collection"""
        self.running = False
        if self.update_thread:
            self.update_thread.join()
        self.logger.info("Stopped sensor data collection")

    def _sensing_loop(self):
        """Main sensor data collection loop"""
        dt = 1.0 / self.update_frequency

        while self.running:
            start_time = time.time()

            # This would be replaced with actual sensor data acquisition
            # For simulation, we'll generate mock data
            self._collect_sensor_data()

            elapsed = time.time() - start_time
            sleep_time = max(0, dt - elapsed)
            time.sleep(sleep_time)

    def _collect_sensor_data(self):
        """Collect data from all sensors"""
        # This is a placeholder - in real implementation,
        # this would interface with MuJoCo to get sensor data

    def get_latest_readings(self) -> List[SensorReading]:
        """Get latest sensor readings"""
        readings = []
        try:
            while True:
                reading = self.sensor_data_queue.get_nowait()
                readings.append(reading)
        except queue.Empty:
            pass
        return readings

    def calibrate_sensor(self, sensor_id: str, calibration_data: Dict[str, Any]) -> bool:
        """Calibrate a specific sensor"""
        if sensor_id in self.sensors:
            return self.sensors[sensor_id].calibrate(calibration_data)
        return False


# Factory functions for common sensor configurations
def create_robot_sensor_suite(robot_type: str, n_joints: int) -> SensorManager:
    """Create complete sensor suite for a robot"""
    manager = SensorManager()

    # Joint position sensors
    joint_pos_processor = JointSensorProcessor(
        "joint_positions", SensorType.JOINT_POSITION, n_joints
    )
    manager.add_sensor("joint_positions", joint_pos_processor)

    # Joint velocity sensors
    joint_vel_processor = JointSensorProcessor(
        "joint_velocities", SensorType.JOINT_VELOCITY, n_joints
    )
    manager.add_sensor("joint_velocities", joint_vel_processor)

    # IMU for base orientation
    imu_processor = IMUSensorProcessor("base_imu")
    manager.add_sensor("base_imu", imu_processor)

    # Force/torque sensor for end effector (if applicable)
    if robot_type in ["franka_panda", "ur5e", "kuka_iiwa"]:
        ft_processor = ForceTorqueSensorProcessor("end_effector_ft")
        manager.add_sensor("end_effector_ft", ft_processor)

    return manager


def create_feedback_controller(robot_type: str) -> ClosedLoopController:
    """Create feedback controller optimized for robot type"""
    controller = ClosedLoopController("pid")

    # Set control gains based on robot type
    if robot_type in ["franka_panda", "ur5e", "kuka_iiwa"]:
        # Arm robots - higher precision
        controller.control_gains = {"kp": 5.0, "ki": 0.1, "kd": 0.5}
    elif robot_type in ["anymal_c", "go2", "spot"]:
        # Quadrupeds - stability focus
        controller.control_gains = {"kp": 2.0, "ki": 0.05, "kd": 0.2}
    elif robot_type in ["g1", "h1"]:
        # Humanoids - balance and coordination
        controller.control_gains = {"kp": 3.0, "ki": 0.08, "kd": 0.3}

    return controller

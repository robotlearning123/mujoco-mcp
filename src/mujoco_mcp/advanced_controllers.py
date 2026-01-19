#!/usr/bin/env python3
"""
Advanced Control Algorithms for MuJoCo MCP
Implements PID controllers, trajectory planning, and optimization-based control
"""

import numpy as np
import math
from typing import Dict, Tuple, Callable, NewType
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
import time

# Domain-specific types for type safety
Gain = NewType("Gain", float)  # PID gain values (kp, ki, kd)
OutputLimit = NewType("OutputLimit", float)  # Control output limits


@dataclass(frozen=True)
class PIDConfig:
    """PID controller configuration"""

    kp: Gain = 1.0  # Proportional gain
    ki: Gain = 0.0  # Integral gain
    kd: Gain = 0.0  # Derivative gain
    max_output: OutputLimit = 100.0  # Maximum output
    min_output: OutputLimit = -100.0  # Minimum output
    windup_limit: OutputLimit = 100.0  # Anti-windup limit

    def __post_init__(self):
        """Validate PID configuration parameters."""
        if self.kp < 0:
            raise ValueError(f"Proportional gain must be non-negative, got {self.kp}")
        if self.ki < 0:
            raise ValueError(f"Integral gain must be non-negative, got {self.ki}")
        if self.kd < 0:
            raise ValueError(f"Derivative gain must be non-negative, got {self.kd}")
        if self.min_output >= self.max_output:
            raise ValueError(
                f"min_output ({self.min_output}) must be less than "
                f"max_output ({self.max_output})"
            )
        if self.windup_limit <= 0:
            raise ValueError(f"windup_limit must be positive, got {self.windup_limit}")


class PIDController:
    """PID controller for joint position/velocity control.

    Example:
        >>> # Create PID controller for position control
        >>> config = PIDConfig(kp=10.0, ki=0.1, kd=1.0)
        >>> controller = PIDController(config)
        >>>
        >>> # Control loop
        >>> target_pos = 1.0
        >>> current_pos = 0.0
        >>> dt = 0.01
        >>>
        >>> for _ in range(100):
        ...     control_output = controller.update(target_pos, current_pos, dt)
        ...     # Apply control_output to actuator
        ...     current_pos += control_output * dt  # Simplified dynamics
    """

    def __init__(self, config: PIDConfig):
        self.config = config
        self.reset()

    def reset(self):
        """Reset PID controller state to initial conditions.

        Note:
            Clears:
            - Previous error (set to 0)
            - Integral accumulator (set to 0)
            - Previous time reference (set to None)

            Call this before starting a new control sequence or after
            discontinuities in the control loop.
        """
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = None

    def update(self, target: float, current: float, dt: float | None = None) -> float:
        """Compute PID control output for current timestep.

        Args:
            target: Desired setpoint value.
            current: Current measured value.
            dt: Time step in seconds (optional). If None, automatically computed
               from wall clock time.

        Returns:
            Control output clamped to [min_output, max_output] range.

        Note:
            Implements the PID control law:
                u(t) = Kp·e(t) + Ki·∫e(τ)dτ + Kd·de(t)/dt

            where:
            - e(t) = target - current (tracking error)
            - Kp, Ki, Kd are the proportional, integral, derivative gains
            - ∫e(τ)dτ is clamped to ±windup_limit for anti-windup
            - Output u(t) is clamped to [min_output, max_output]

            First call after reset() uses a default dt of 0.02s (50Hz).
        """
        if dt is None:
            current_time = time.time()
            if self.prev_time is None:
                dt = 0.02  # Default 50Hz
            else:
                dt = current_time - self.prev_time
            self.prev_time = current_time

        # Calculate error
        error = target - current

        # Proportional term
        p_term = self.config.kp * error

        # Integral term with anti-windup
        self.integral += error * dt
        if abs(self.integral) > self.config.windup_limit:
            self.integral = math.copysign(self.config.windup_limit, self.integral)
        i_term = self.config.ki * self.integral

        # Derivative term
        if dt > 0:
            derivative = (error - self.prev_error) / dt
        else:
            derivative = 0
        d_term = self.config.kd * derivative

        # Calculate output
        output = p_term + i_term + d_term

        # Apply output limits
        output = max(self.config.min_output, min(self.config.max_output, output))

        # Update for next iteration
        self.prev_error = error

        return output


class TrajectoryPlanner:
    """Advanced trajectory planning for smooth robot motions"""

    @staticmethod
    def minimum_jerk_trajectory(
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        duration: float,
        start_vel: np.ndarray | None = None,
        end_vel: np.ndarray | None = None,
        frequency: float = 100.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate smooth minimum jerk trajectory between two points.

        Args:
            start_pos: Initial position (ndarray of shape (n_dims,)).
            end_pos: Final position (ndarray of shape (n_dims,)).
            duration: Trajectory duration in seconds.
            start_vel: Initial velocity (default: zeros). Shape (n_dims,).
            end_vel: Final velocity (default: zeros). Shape (n_dims,).
            frequency: Sampling frequency in Hz (default: 100.0).

        Returns:
            Tuple of (positions, velocities, accelerations), each with shape
            (n_timesteps, n_dims) where n_timesteps = duration * frequency.

        Note:
            Generates trajectories that minimize the integral of jerk squared:
                J = ∫₀ᵀ ||d³x/dt³||² dt

            This produces smooth, natural-looking motions with continuous acceleration.
            The resulting trajectory is a 5th-order polynomial satisfying:
            - Position boundary conditions: x(0) = start_pos, x(T) = end_pos
            - Velocity boundary conditions: ẋ(0) = start_vel, ẋ(T) = end_vel
            - Acceleration boundary conditions: ẍ(0) = 0, ẍ(T) = 0

            Used extensively in robotics for smooth point-to-point motions.
        """

        if start_vel is None:
            start_vel = np.zeros_like(start_pos)
        if end_vel is None:
            end_vel = np.zeros_like(end_pos)

        num_points = int(duration * frequency)
        t = np.linspace(0, duration, num_points)

        # Minimum jerk polynomial coefficients
        positions = []
        velocities = []
        accelerations = []

        for i in range(len(start_pos)):
            p0, pf = start_pos[i], end_pos[i]
            v0, vf = start_vel[i], end_vel[i]
            T = duration

            # Solve for polynomial coefficients
            A = np.array(
                [
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [1, T, T**2, T**3, T**4, T**5],
                    [0, 1, 2 * T, 3 * T**2, 4 * T**3, 5 * T**4],
                    [0, 0, 2, 6 * T, 12 * T**2, 20 * T**3],
                    [0, 0, 0, 6, 24 * T, 60 * T**2],
                ]
            )

            b = np.array([p0, v0, pf, vf, 0, 0])  # Zero acceleration at endpoints
            coeffs = np.linalg.solve(A, b)

            # Generate trajectory
            pos = np.polyval(coeffs[::-1], t)
            vel = np.polyval(np.polyder(coeffs[::-1]), t)
            acc = np.polyval(np.polyder(coeffs[::-1], 2), t)

            positions.append(pos)
            velocities.append(vel)
            accelerations.append(acc)

        return np.array(positions).T, np.array(velocities).T, np.array(accelerations).T

    @staticmethod
    def spline_trajectory(
        waypoints: np.ndarray, times: np.ndarray, frequency: float = 100.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate smooth spline trajectory through waypoints"""

        t_dense = np.linspace(times[0], times[-1], int((times[-1] - times[0]) * frequency))

        positions = []
        velocities = []
        accelerations = []

        for joint_idx in range(waypoints.shape[1]):
            # Fit cubic spline
            spline = CubicSpline(times, waypoints[:, joint_idx], bc_type="natural")

            # Evaluate spline
            pos = spline(t_dense)
            vel = spline(t_dense, 1)  # First derivative
            acc = spline(t_dense, 2)  # Second derivative

            positions.append(pos)
            velocities.append(vel)
            accelerations.append(acc)

        return np.array(positions).T, np.array(velocities).T, np.array(accelerations).T

    @staticmethod
    def cartesian_to_joint_trajectory(
        cartesian_waypoints: np.ndarray,
        robot_kinematics: Callable,
        times: np.ndarray,
        frequency: float = 100.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert Cartesian trajectory to joint space"""
        joint_waypoints = np.array([
            robot_kinematics.inverse_kinematics(cart_pos)
            for cart_pos in cartesian_waypoints
        ])

        return TrajectoryPlanner.spline_trajectory(joint_waypoints, times, frequency)


class OptimizationController:
    """Model Predictive Control and optimization-based control"""

    def __init__(self, horizon: int = 10, dt: float = 0.02):
        self.horizon = horizon
        self.dt = dt

    def quadratic_programming_control(
        self,
        current_state: np.ndarray,
        target_state: np.ndarray,
        dynamics_func: Callable,
        constraints: Dict | None = None,
    ) -> np.ndarray:
        """Solve quadratic programming problem for optimal control"""

        n_states = len(current_state)
        n_controls = n_states  # Assume same dimensions for simplicity

        def objective(u_sequence):
            """Quadratic cost function"""
            u_sequence = u_sequence.reshape(self.horizon, n_controls)
            cost = 0

            state = current_state.copy()
            for i in range(self.horizon):
                # State cost
                state_error = state - target_state
                cost += np.dot(state_error, state_error)

                # Control cost
                control_cost = np.dot(u_sequence[i], u_sequence[i])
                cost += 0.1 * control_cost  # Control penalty

                # Update state
                state = dynamics_func(state, u_sequence[i], self.dt)

            return cost

        # Initial guess
        u0 = np.zeros(self.horizon * n_controls)

        # Set up constraints
        bounds = None
        if constraints and "control_bounds" in constraints:
            control_bounds = constraints["control_bounds"]
            bounds = [control_bounds] * (self.horizon * n_controls)

        # Solve optimization
        result = minimize(objective, u0, bounds=bounds, method="SLSQP")

        # Return first control action
        optimal_controls = result.x.reshape(self.horizon, n_controls)
        return optimal_controls[0]


class AdaptiveController:
    """Adaptive control with parameter estimation"""

    def __init__(self, n_params: int, learning_rate: float = 0.01):
        self.n_params = n_params
        self.learning_rate = learning_rate
        self.params = np.ones(n_params)
        self.param_history = []

    def update_parameters(self, error: np.ndarray, regressor: np.ndarray):
        """Update adaptive parameters using gradient descent"""
        # Simple gradient update
        gradient = np.dot(regressor.T, error)
        self.params -= self.learning_rate * gradient

        # Store history
        self.param_history.append(self.params.copy())

    def get_control(self, regressor: np.ndarray) -> float:
        """Get control output"""
        return np.dot(regressor, self.params)


class ForceController:
    """Force/torque control for compliant manipulation"""

    def __init__(self, force_gains: np.ndarray, position_gains: np.ndarray):
        self.force_gains = force_gains
        self.position_gains = position_gains

    def hybrid_position_force_control(
        self,
        current_pos: np.ndarray,
        target_pos: np.ndarray,
        current_force: np.ndarray,
        target_force: np.ndarray,
        selection_matrix: np.ndarray,
    ) -> np.ndarray:
        """Hybrid position/force control"""

        # Position error
        pos_error = target_pos - current_pos
        pos_command = self.position_gains * pos_error

        # Force error
        force_error = target_force - current_force
        force_command = self.force_gains * force_error

        # Combine using selection matrix
        # S = 1 for force control, S = 0 for position control
        return selection_matrix * force_command + (1 - selection_matrix) * pos_command


class RobotController:
    """High-level robot controller combining multiple control strategies"""

    def __init__(self, robot_config: Dict):
        self.config = robot_config
        self.n_joints = robot_config.get("joints", 6)

        # Initialize PID controllers for each joint
        pid_config = PIDConfig(kp=10.0, ki=0.1, kd=1.0)
        self.pid_controllers = [PIDController(pid_config) for _ in range(self.n_joints)]

        # Initialize trajectory planner
        self.trajectory_planner = TrajectoryPlanner()

        # Initialize optimization controller
        self.mpc_controller = OptimizationController()

        # Current trajectory
        self.current_trajectory = None
        self.trajectory_start_time = None
        self.trajectory_index = 0

    def set_trajectory(self, waypoints: np.ndarray, times: np.ndarray):
        """Set new trajectory to follow"""
        positions, velocities, accelerations = self.trajectory_planner.spline_trajectory(
            waypoints, times
        )

        self.current_trajectory = {
            "positions": positions,
            "velocities": velocities,
            "accelerations": accelerations,
            "times": times,
            "dt": times[1] - times[0] if len(times) > 1 else 0.02,
        }
        self.trajectory_start_time = time.time()
        self.trajectory_index = 0

    def get_trajectory_command(self, current_time: float | None = None) -> np.ndarray | None:
        """Get current trajectory command"""
        if self.current_trajectory is None:
            return None

        if current_time is None:
            current_time = time.time()

        elapsed = current_time - self.trajectory_start_time

        # Find trajectory index
        dt = self.current_trajectory["dt"]
        index = int(elapsed / dt)

        if index >= len(self.current_trajectory["positions"]):
            return None  # Trajectory complete

        return self.current_trajectory["positions"][index]

    def pid_control(
        self, target_positions: np.ndarray, current_positions: np.ndarray
    ) -> np.ndarray:
        """Apply PID control to reach target positions"""
        n_valid = min(self.n_joints, len(target_positions), len(current_positions))

        commands = [
            self.pid_controllers[i].update(target_positions[i], current_positions[i])
            for i in range(n_valid)
        ]

        # Pad with zeros if we have fewer valid joints than expected
        commands.extend([0.0] * (self.n_joints - n_valid))

        return np.array(commands)

    def impedance_control(
        self,
        current_pos: np.ndarray,
        target_pos: np.ndarray,
        current_vel: np.ndarray,
        stiffness: np.ndarray,
        damping: np.ndarray,
    ) -> np.ndarray:
        """Impedance control for compliant motion"""

        pos_error = target_pos - current_pos
        return stiffness * pos_error - damping * current_vel

    def reset_controllers(self):
        """Reset all controller states"""
        for pid in self.pid_controllers:
            pid.reset()

        self.current_trajectory = None
        self.trajectory_start_time = None
        self.trajectory_index = 0


# Robot configuration presets
_ROBOT_CONFIGS = {
    # Robotic arms
    "franka_panda": {
        "joints": 7,
        "kp": [100, 100, 100, 100, 50, 50, 25],
        "ki": [0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.01],
        "kd": [10, 10, 10, 10, 5, 5, 2.5],
    },
    "ur5e": {
        "joints": 6,
        "kp": [150, 150, 100, 100, 50, 50],
        "ki": [0.2, 0.2, 0.1, 0.1, 0.05, 0.05],
        "kd": [15, 15, 10, 10, 5, 5],
    },
    # Quadrupeds
    "anymal_c": {
        "joints": 12,
        "kp": [200] * 12,
        "ki": [0.5] * 12,
        "kd": [20] * 12,
    },
    "go2": {
        "joints": 12,
        "kp": [180] * 12,
        "ki": [0.3] * 12,
        "kd": [18] * 12,
    },
    # Humanoids
    "g1": {
        "joints": 37,
        "kp": [100] * 37,
        "ki": [0.1] * 37,
        "kd": [10] * 37,
    },
    "h1": {
        "joints": 25,
        "kp": [120] * 25,
        "ki": [0.15] * 25,
        "kd": [12] * 25,
    },
}


def _create_controller(robot_type: str, default: str) -> RobotController:
    """Create controller from preset configuration."""
    config = _ROBOT_CONFIGS.get(robot_type, _ROBOT_CONFIGS[default])
    return RobotController(config)


def create_arm_controller(robot_type: str = "franka_panda") -> RobotController:
    """Create controller optimized for robotic arms"""
    return _create_controller(robot_type, "franka_panda")


def create_quadruped_controller(robot_type: str = "anymal_c") -> RobotController:
    """Create controller optimized for quadruped robots"""
    return _create_controller(robot_type, "anymal_c")


def create_humanoid_controller(robot_type: str = "g1") -> RobotController:
    """Create controller optimized for humanoid robots"""
    return _create_controller(robot_type, "g1")

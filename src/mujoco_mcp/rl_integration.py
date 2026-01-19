#!/usr/bin/env python3
"""
Reinforcement Learning Integration for MuJoCo MCP
Provides RL environment interface and training utilities for robot control
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time
from typing import Dict, Tuple, Any, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from collections import deque
import json
from enum import Enum

from .viewer_client import MuJoCoViewerClient
from .sensor_feedback import SensorManager


class ActionSpaceType(Enum):
    """Types of action spaces for RL environments."""

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"


class TaskType(Enum):
    """Types of RL tasks."""

    REACHING = "reaching"
    BALANCING = "balancing"
    WALKING = "walking"


@dataclass(frozen=True)
class RLConfig:
    """Configuration for RL environment"""

    robot_type: str
    task_type: TaskType
    max_episode_steps: int = 1000
    reward_scale: float = 1.0
    action_space_type: ActionSpaceType = ActionSpaceType.CONTINUOUS
    observation_space_size: int = 0
    action_space_size: int = 0
    render_mode: str | None = None
    physics_timestep: float = 0.002
    control_timestep: float = 0.02

    def __post_init__(self):
        """Validate RL configuration parameters."""
        if self.max_episode_steps <= 0:
            raise ValueError(f"max_episode_steps must be positive, got {self.max_episode_steps}")
        if self.physics_timestep <= 0:
            raise ValueError(f"physics_timestep must be positive, got {self.physics_timestep}")
        if self.control_timestep <= 0:
            raise ValueError(f"control_timestep must be positive, got {self.control_timestep}")
        if self.control_timestep < self.physics_timestep:
            raise ValueError(
                f"control_timestep ({self.control_timestep}) must be >= "
                f"physics_timestep ({self.physics_timestep})"
            )
        if not isinstance(self.action_space_type, ActionSpaceType):
            raise ValueError(
                f"action_space_type must be an ActionSpaceType enum, "
                f"got {type(self.action_space_type)}"
            )
        if not isinstance(self.task_type, TaskType):
            raise ValueError(
                f"task_type must be a TaskType enum, got {type(self.task_type)}"
            )


class TaskReward(ABC):
    """Abstract base class for task-specific reward functions"""

    @abstractmethod
    def compute_reward(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        next_observation: np.ndarray,
        info: Dict[str, Any],
    ) -> float:
        """Compute reward for current step"""

    @abstractmethod
    def is_done(self, observation: np.ndarray, info: Dict[str, Any]) -> bool:
        """Check if episode is done"""


class ReachingTaskReward(TaskReward):
    """Reward function for reaching tasks"""

    def __init__(self, target_position: np.ndarray, position_tolerance: float = 0.05):
        self.target_position = target_position
        self.position_tolerance = position_tolerance
        self.prev_distance = None

    def compute_reward(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        next_observation: np.ndarray,
        info: Dict[str, Any],
    ) -> float:
        """Compute reaching reward"""
        # Extract end-effector position from observation
        end_effector_pos = next_observation[:3]  # Assume first 3 elements are position

        # Distance to target
        distance = np.linalg.norm(end_effector_pos - self.target_position)

        # Reward components
        distance_reward = -distance  # Negative distance as reward

        # Bonus for improvement
        improvement_reward = 0.0
        if self.prev_distance is not None:
            improvement = self.prev_distance - distance
            improvement_reward = improvement * 10.0  # Scale improvement

        self.prev_distance = distance

        # Success bonus
        success_reward = 100.0 if distance < self.position_tolerance else 0.0

        # Control penalty
        control_penalty = -0.01 * np.sum(np.square(action))

        return distance_reward + improvement_reward + success_reward + control_penalty

    def is_done(self, observation: np.ndarray, info: Dict[str, Any]) -> bool:
        """Episode done when target reached or max steps"""
        end_effector_pos = observation[:3]
        distance = np.linalg.norm(end_effector_pos - self.target_position)
        return distance < self.position_tolerance


class BalancingTaskReward(TaskReward):
    """Reward function for balancing tasks (e.g., cart-pole, humanoid)"""

    def __init__(self, upright_threshold: float = 0.2):
        self.upright_threshold = upright_threshold

    def compute_reward(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        next_observation: np.ndarray,
        info: Dict[str, Any],
    ) -> float:
        """Compute balancing reward"""
        # Extract relevant state (e.g., pole angle, orientation)
        if len(next_observation) >= 2:
            angle = next_observation[1]  # Assume second element is angle
            angular_velocity = next_observation[3] if len(next_observation) > 3 else 0.0
        else:
            angle = 0.0
            angular_velocity = 0.0

        # Reward for staying upright
        upright_reward = 1.0 - abs(angle) / np.pi

        # Penalty for high angular velocity
        velocity_penalty = -0.01 * abs(angular_velocity)

        # Control penalty
        control_penalty = -0.001 * np.sum(np.square(action))

        return upright_reward + velocity_penalty + control_penalty

    def is_done(self, observation: np.ndarray, info: Dict[str, Any]) -> bool:
        """Episode done when fallen over"""
        if len(observation) >= 2:
            angle = observation[1]
            return abs(angle) > self.upright_threshold
        return False


class WalkingTaskReward(TaskReward):
    """Reward function for walking/locomotion tasks"""

    def __init__(self, target_velocity: float = 1.0):
        self.target_velocity = target_velocity
        self.prev_position = None

    def compute_reward(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        next_observation: np.ndarray,
        info: Dict[str, Any],
    ) -> float:
        """Compute walking reward"""
        # Extract position and orientation
        position = next_observation[:3]  # xyz position

        # Forward velocity reward
        if self.prev_position is not None:
            velocity = position[0] - self.prev_position[0]  # Forward velocity
            velocity_reward = min(velocity / self.target_velocity, 1.0)
        else:
            velocity_reward = 0.0

        self.prev_position = position

        # Stability reward (penalize large lateral motion)
        stability_penalty = -0.1 * abs(position[1])  # Penalize y deviation

        # Energy efficiency (penalize large actions)
        energy_penalty = -0.01 * np.sum(np.square(action))

        # Height maintenance
        height_reward = max(0, 1.0 - abs(position[2] - 1.0))  # Target height ~1m

        return velocity_reward + stability_penalty + energy_penalty + height_reward

    def is_done(self, observation: np.ndarray, info: Dict[str, Any]) -> bool:
        """Episode done when fallen"""
        position = observation[:3]
        return position[2] < 0.3  # Fallen if height < 0.3m


class MuJoCoRLEnvironment(gym.Env):
    """Gymnasium-compatible RL environment for MuJoCo MCP.

    Example:
        >>> # Create configuration
        >>> config = RLConfig(
        ...     robot_type="arm",
        ...     task_type=TaskType.REACHING,
        ...     max_episode_steps=1000,
        ...     action_space_type=ActionSpaceType.CONTINUOUS,
        ...     observation_space_size=10,
        ...     action_space_size=6
        ... )
        >>>
        >>> # Create environment
        >>> env = MuJoCoRLEnvironment(config)
        >>>
        >>> # Training loop
        >>> observation, info = env.reset()
        >>> for _ in range(1000):
        ...     action = env.action_space.sample()  # Random policy
        ...     observation, reward, terminated, truncated, info = env.step(action)
        ...     if terminated or truncated:
        ...         observation, info = env.reset()
        >>> env.close()
    """

    def __init__(self, config: RLConfig):
        super().__init__()

        self.config = config
        self.viewer_client = MuJoCoViewerClient()
        self.sensor_manager = SensorManager()

        # RL state
        self.current_step = 0
        self.episode_rewards = []
        self.episode_lengths = []

        # Task-specific reward function
        self.reward_function = self._create_reward_function()

        # Define action and observation spaces
        self._setup_spaces()

        # Model and state management
        self.model_id = f"rl_env_{config.robot_type}_{config.task_type}"
        self.model_xml = self._create_model_xml()
        self.reset_state = None

        # Logging
        self.logger = logging.getLogger(__name__)

        # Performance tracking
        self.episode_start_time = None
        self.step_times = deque(maxlen=100)

    def _setup_spaces(self):
        """Setup action and observation spaces"""
        # Robot configurations (inline to avoid import issues)
        robot_configs = {
            "franka_panda": {"joints": 7},
            "ur5e": {"joints": 6},
            "anymal_c": {"joints": 12},
            "cart_pole": {"joints": 2},
            "quadruped": {"joints": 8},
        }

        if self.config.robot_type in robot_configs:
            n_joints = robot_configs[self.config.robot_type]["joints"]
        else:
            n_joints = 6  # Default

        # Action space
        if self.config.action_space_type == ActionSpaceType.CONTINUOUS:
            # Continuous joint torques/positions
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_joints,), dtype=np.float32)
        else:
            # Discrete action space
            self.action_space = spaces.Discrete(n_joints * 3)  # 3 actions per joint

        # Observation space
        obs_size = self.config.observation_space_size
        if obs_size == 0:
            # Auto-determine observation size
            obs_size = n_joints * 2 + 6  # joint pos + vel + end-effector pose

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

    def _create_reward_function(self) -> TaskReward:
        """Create task-specific reward function"""
        if self.config.task_type == TaskType.REACHING:
            target = np.array([0.5, 0.0, 0.5])  # Default target position
            return ReachingTaskReward(target)
        if self.config.task_type == TaskType.BALANCING:
            return BalancingTaskReward()
        # TaskType.WALKING
        return WalkingTaskReward()

    def _create_model_xml(self) -> str:
        """Create model XML for the RL task"""
        is_franka_reaching = (
            self.config.task_type == TaskType.REACHING
            and self.config.robot_type == "franka_panda"
        )
        if is_franka_reaching:
            return self._create_franka_reaching_xml()

        if self.config.task_type == TaskType.BALANCING:
            return self._create_cart_pole_xml()

        is_quadruped_walking = (
            self.config.task_type == TaskType.WALKING
            and "quadruped" in self.config.robot_type
        )
        if is_quadruped_walking:
            return self._create_quadruped_xml()

        return self._create_simple_arm_xml()

    def _create_franka_reaching_xml(self) -> str:
        """Create Franka Panda XML for reaching task"""
        return """
        <mujoco model="franka_reaching">
            <option timestep="0.002"/>
            <worldbody>
                <!-- Target -->
                <body name="target" pos="0.5 0 0.5">
                    <geom name="target_geom" type="sphere" size="0.05" rgba="0 1 0 0.5"/>
                </body>

                <!-- Robot base -->
                <body name="base" pos="0 0 0">
                    <geom name="base_geom" type="cylinder" size="0.1 0.1" rgba="0.5 0.5 0.5 1"/>

                    <!-- Simplified 7-DOF arm -->
                    <body name="link1" pos="0 0 0.1">
                        <joint name="joint1" type="hinge" axis="0 0 1" range="-2.8 2.8"/>
                        <geom name="link1_geom" type="capsule" size="0.05 0.15" rgba="0.8 0.2 0.2 1"/>

                        <body name="link2" pos="0 0 0.15">
                            <joint name="joint2" type="hinge" axis="0 1 0" range="-1.8 1.8"/>
                            <geom name="link2_geom" type="capsule" size="0.04 0.12" rgba="0.2 0.8 0.2 1"/>

                            <body name="link3" pos="0 0 0.12">
                                <joint name="joint3" type="hinge" axis="0 0 1" range="-2.8 2.8"/>
                                <geom name="link3_geom" type="capsule" size="0.04 0.1" rgba="0.2 0.2 0.8 1"/>

                                <body name="link4" pos="0 0 0.1">
                                    <joint name="joint4" type="hinge" axis="0 1 0" range="-3.1 0"/>
                                    <geom name="link4_geom" type="capsule" size="0.03 0.08" rgba="0.8 0.8 0.2 1"/>

                                    <body name="link5" pos="0 0 0.08">
                                        <joint name="joint5" type="hinge" axis="0 0 1" range="-2.8 2.8"/>
                                        <geom name="link5_geom" type="capsule" size="0.03 0.06" rgba="0.8 0.2 0.8 1"/>

                                        <body name="link6" pos="0 0 0.06">
                                            <joint name="joint6" type="hinge" axis="0 1 0" range="-0.1 3.8"/>
                                            <geom name="link6_geom" type="capsule" size="0.02 0.04" rgba="0.2 0.8 0.8 1"/>

                                            <body name="end_effector" pos="0 0 0.04">
                                                <joint name="joint7" type="hinge" axis="0 0 1" range="-2.8 2.8"/>
                                                <geom name="ee_geom" type="sphere" size="0.02" rgba="1 0 0 1"/>
                                            </body>
                                        </body>
                                    </body>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </worldbody>
        </mujoco>
        """

    def _create_cart_pole_xml(self) -> str:
        """Create cart-pole XML for balancing task"""
        return """
        <mujoco model="cartpole">
            <option timestep="0.002"/>
            <worldbody>
                <body name="cart" pos="0 0 0.1">
                    <joint name="slider" type="slide" axis="1 0 0" range="-2 2"/>
                    <geom name="cart_geom" type="box" size="0.1 0.1 0.1" rgba="0.8 0.2 0.2 1"/>
                    <body name="pole" pos="0 0 0.1">
                        <joint name="hinge" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
                        <geom name="pole_geom" type="capsule" size="0.02 0.5" rgba="0.2 0.8 0.2 1"/>
                    </body>
                </body>
            </worldbody>
        </mujoco>
        """

    def _create_quadruped_xml(self) -> str:
        """Create simplified quadruped XML for walking task"""
        return """
        <mujoco model="quadruped">
            <option timestep="0.002"/>
            <worldbody>
                <body name="torso" pos="0 0 0.5">
                    <joint name="free_joint" type="free"/>
                    <geom name="torso_geom" type="box" size="0.3 0.15 0.1" rgba="0.5 0.5 0.5 1"/>

                    <!-- Front legs -->
                    <body name="front_left_hip" pos="0.2 0.1 -0.05">
                        <joint name="fl_hip" type="hinge" axis="1 0 0" range="-1.57 1.57"/>
                        <geom name="fl_hip_geom" type="capsule" size="0.02 0.1" rgba="0.8 0.2 0.2 1"/>
                        <body name="front_left_knee" pos="0 0 -0.1">
                            <joint name="fl_knee" type="hinge" axis="1 0 0" range="0 2.36"/>
                            <geom name="fl_knee_geom" type="capsule" size="0.015 0.08" rgba="0.2 0.8 0.2 1"/>
                            <body name="front_left_foot" pos="0 0 -0.08">
                                <geom name="fl_foot_geom" type="sphere" size="0.03" rgba="0.2 0.2 0.8 1"/>
                            </body>
                        </body>
                    </body>

                    <body name="front_right_hip" pos="0.2 -0.1 -0.05">
                        <joint name="fr_hip" type="hinge" axis="1 0 0" range="-1.57 1.57"/>
                        <geom name="fr_hip_geom" type="capsule" size="0.02 0.1" rgba="0.8 0.2 0.2 1"/>
                        <body name="front_right_knee" pos="0 0 -0.1">
                            <joint name="fr_knee" type="hinge" axis="1 0 0" range="0 2.36"/>
                            <geom name="fr_knee_geom" type="capsule" size="0.015 0.08" rgba="0.2 0.8 0.2 1"/>
                            <body name="front_right_foot" pos="0 0 -0.08">
                                <geom name="fr_foot_geom" type="sphere" size="0.03" rgba="0.2 0.2 0.8 1"/>
                            </body>
                        </body>
                    </body>

                    <!-- Back legs -->
                    <body name="back_left_hip" pos="-0.2 0.1 -0.05">
                        <joint name="bl_hip" type="hinge" axis="1 0 0" range="-1.57 1.57"/>
                        <geom name="bl_hip_geom" type="capsule" size="0.02 0.1" rgba="0.8 0.2 0.2 1"/>
                        <body name="back_left_knee" pos="0 0 -0.1">
                            <joint name="bl_knee" type="hinge" axis="1 0 0" range="0 2.36"/>
                            <geom name="bl_knee_geom" type="capsule" size="0.015 0.08" rgba="0.2 0.8 0.2 1"/>
                            <body name="back_left_foot" pos="0 0 -0.08">
                                <geom name="bl_foot_geom" type="sphere" size="0.03" rgba="0.2 0.2 0.8 1"/>
                            </body>
                        </body>
                    </body>

                    <body name="back_right_hip" pos="-0.2 -0.1 -0.05">
                        <joint name="br_hip" type="hinge" axis="1 0 0" range="-1.57 1.57"/>
                        <geom name="br_hip_geom" type="capsule" size="0.02 0.1" rgba="0.8 0.2 0.2 1"/>
                        <body name="back_right_knee" pos="0 0 -0.1">
                            <joint name="br_knee" type="hinge" axis="1 0 0" range="0 2.36"/>
                            <geom name="br_knee_geom" type="capsule" size="0.015 0.08" rgba="0.2 0.8 0.2 1"/>
                            <body name="back_right_foot" pos="0 0 -0.08">
                                <geom name="br_foot_geom" type="sphere" size="0.03" rgba="0.2 0.2 0.8 1"/>
                            </body>
                        </body>
                    </body>
                </body>
            </worldbody>
        </mujoco>
        """

    def _create_simple_arm_xml(self) -> str:
        """Create simple arm XML for generic tasks"""
        return """
        <mujoco model="simple_arm">
            <option timestep="0.002"/>
            <worldbody>
                <body name="base" pos="0 0 0">
                    <geom name="base_geom" type="cylinder" size="0.1 0.1" rgba="0.5 0.5 0.5 1"/>
                    <body name="link1" pos="0 0 0.1">
                        <joint name="joint1" type="hinge" axis="0 0 1"/>
                        <geom name="link1_geom" type="capsule" size="0.05 0.2" rgba="0.8 0.2 0.2 1"/>
                        <body name="link2" pos="0 0 0.2">
                            <joint name="joint2" type="hinge" axis="0 1 0"/>
                            <geom name="link2_geom" type="capsule" size="0.04 0.15" rgba="0.2 0.8 0.2 1"/>
                            <body name="end_effector" pos="0 0 0.15">
                                <geom name="ee_geom" type="sphere" size="0.03" rgba="1 0 0 1"/>
                            </body>
                        </body>
                    </body>
                </body>
            </worldbody>
        </mujoco>
        """

    def reset(
        self, seed: int | None = None, options: Dict | None = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to start a new episode.

        Args:
            seed: Random seed for reproducibility (optional).
            options: Additional reset options (optional, currently unused).

        Returns:
            Tuple containing:
            - observation (np.ndarray): Initial state observation of shape determined by observation_space
            - info (Dict): Diagnostic information including episode count and current step

        Raises:
            RuntimeError: If connection to viewer server fails or model loading fails.

        Note:
            This method:
            - Resets the episode step counter to 0
            - Reloads the MuJoCo model in the viewer
            - Resets reward function internal state
            - Returns the initial observation and info dict
        """
        super().reset(seed=seed)

        # Connect to viewer if needed
        if not self.viewer_client.connected:
            success = self.viewer_client.connect()
            if not success:
                raise RuntimeError("Failed to connect to MuJoCo viewer server")

        # Load model
        response = self.viewer_client.send_command(
            {"type": "load_model", "model_id": self.model_id, "model_xml": self.model_xml}
        )

        if not response.get("success"):
            raise RuntimeError(f"Failed to load model: {response.get('error')}")

        # Reset episode state
        self.current_step = 0
        self.episode_start_time = time.time()

        # Reset reward function
        if hasattr(self.reward_function, "prev_distance"):
            self.reward_function.prev_distance = None
        if hasattr(self.reward_function, "prev_position"):
            self.reward_function.prev_position = None

        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: Union[np.ndarray, int]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step by applying an action.

        Args:
            action: Action to execute. Can be:
                   - np.ndarray: Continuous actions (for Box action space)
                   - int: Discrete action index (converted to continuous internally)

        Returns:
            Tuple containing:
            - observation (np.ndarray): State after executing the action
            - reward (float): Reward obtained from the transition
            - terminated (bool): Whether episode ended due to task completion/failure
            - truncated (bool): Whether episode ended due to time limit
            - info (Dict): Diagnostic information including step count, timing, etc.

        Note:
            The Gymnasium API separates episode termination into two flags:
            - terminated: Task-specific ending (e.g., goal reached, robot fell)
            - truncated: Episode ended due to max_episode_steps limit
        """
        step_start_time = time.time()

        # Convert action if needed
        if isinstance(action, int):
            action = self._discrete_to_continuous_action(action)

        # Ensure action is numpy array
        action = np.array(action, dtype=np.float32)

        # Apply action
        self._apply_action(action)

        # Get new observation
        prev_obs = self._get_observation()
        time.sleep(self.config.control_timestep)  # Simulate physics step
        new_obs = self._get_observation()

        # Compute reward
        info = self._get_info()
        reward = self.reward_function.compute_reward(prev_obs, action, new_obs, info)

        # Check if episode is done
        terminated = self.reward_function.is_done(new_obs, info)
        truncated = self.current_step >= self.config.max_episode_steps

        # Update step counter
        self.current_step += 1

        # Track step time
        step_time = time.time() - step_start_time
        self.step_times.append(step_time)

        # Update info
        info.update(
            {
                "step": self.current_step,
                "step_time": step_time,
                "avg_step_time": np.mean(self.step_times),
                "episode_length": self.current_step,
            }
        )

        return new_obs, reward, terminated, truncated, info

    def _discrete_to_continuous_action(self, action: int) -> np.ndarray:
        """Convert discrete action to continuous action"""
        n_joints = self.action_space.shape[0] if hasattr(self.action_space, "shape") else 2
        joint_idx = action // 3
        action_type = action % 3

        continuous_action = np.zeros(n_joints)
        if joint_idx < n_joints:
            # Map action_type: 0 -> -1.0, 1 -> 0.0, 2 -> 1.0
            action_values = {0: -1.0, 1: 0.0, 2: 1.0}
            continuous_action[joint_idx] = action_values[action_type]

        return continuous_action

    def _apply_action(self, action: np.ndarray):
        """Apply action to the robot"""
        # Scale action to appropriate range
        scaled_action = action * 10.0  # Scale to reasonable torque range

        # Send command to MuJoCo
        self.viewer_client.send_command(
            {
                "type": "set_joint_positions",
                "model_id": self.model_id,
                "positions": scaled_action.tolist(),
            }
        )

    def _get_observation(self) -> np.ndarray:
        """Get current observation from simulation.

        Returns:
            Current observation as float32 numpy array.

        Raises:
            RuntimeError: If state cannot be retrieved from simulation.
        """
        response = self.viewer_client.send_command({"type": "get_state", "model_id": self.model_id})

        if response.get("success"):
            state = response.get("state", {})
            qpos = np.array(state.get("qpos", []))
            qvel = np.array(state.get("qvel", []))

            # Combine position and velocity
            observation = np.concatenate([qpos, qvel])

            # Pad or truncate to match observation space
            obs_size = self.observation_space.shape[0]
            if len(observation) < obs_size:
                observation = np.pad(observation, (0, obs_size - len(observation)))
            elif len(observation) > obs_size:
                observation = observation[:obs_size]

            return observation.astype(np.float32)

        # State fetch failed - raise error instead of returning zeros
        error_msg = response.get("error", "Unknown error")
        logger.error(f"Failed to get observation from model {self.model_id}: {error_msg}")
        raise RuntimeError(f"Cannot get observation from simulation: {error_msg}")

    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about current state"""
        return {
            "episode_step": self.current_step,
            "model_id": self.model_id,
            "task_type": self.config.task_type,
            "robot_type": self.config.robot_type,
        }

    def render(self):
        """Render the current environment state.

        Note:
            The MuJoCo viewer server automatically renders the simulation in real-time,
            so this method is a no-op. Rendering is handled by the standalone viewer
            process that displays the simulation continuously.

            For programmatic frame capture, use the viewer_client.capture_render() method.
        """
        # The MuJoCo viewer automatically renders the simulation

    def close(self):
        """Clean up and close the environment.

        Note:
            This method:
            - Closes the model in the viewer server
            - Disconnects from the viewer server
            - Should be called when the environment is no longer needed
            - Safe to call multiple times (idempotent)
        """
        if self.viewer_client.connected:
            self.viewer_client.send_command({"type": "close_model", "model_id": self.model_id})
            self.viewer_client.disconnect()


class RLTrainer:
    """RL training utilities for MuJoCo MCP environments"""

    def __init__(self, env: MuJoCoRLEnvironment):
        self.env = env
        self.training_history = []
        self.best_reward = -np.inf
        self.logger = logging.getLogger(__name__)

    def random_policy_baseline(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate random policy performance as a baseline.

        Args:
            num_episodes: Number of episodes to run (default: 10).

        Returns:
            Dictionary containing baseline statistics:
            - mean_reward: Average total reward per episode
            - std_reward: Standard deviation of episode rewards
            - mean_length: Average episode length in steps
            - std_length: Standard deviation of episode lengths
            - min_reward: Minimum episode reward
            - max_reward: Maximum episode reward

        Note:
            This provides a performance baseline for comparing learned policies.
            Random actions are sampled uniformly from the action space.
        """
        rewards = []
        episode_lengths = []

        for episode in range(num_episodes):
            _obs, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                action = self.env.action_space.sample()
                _obs, reward, terminated, truncated, _info = self.env.step(action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated

            rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            print(
                f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}"
            )

        results = {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "mean_length": np.mean(episode_lengths),
            "std_length": np.std(episode_lengths),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
        }

        print("\nRandom Policy Baseline Results:")
        for key, value in results.items():
            print(f"  {key}: {value:.4f}")

        return results

    def evaluate_policy(self, policy_fn: Callable, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate a learned or handcrafted policy.

        Args:
            policy_fn: Callable that maps observations to actions. Should accept
                      a numpy array observation and return an action compatible
                      with the environment's action space.
            num_episodes: Number of episodes to run for evaluation (default: 10).

        Returns:
            Dictionary containing evaluation statistics:
            - mean_reward: Average total reward per episode
            - std_reward: Standard deviation of episode rewards
            - mean_length: Average episode length in steps
            - std_length: Standard deviation of episode lengths
            - min_reward: Minimum episode reward
            - max_reward: Maximum episode reward

        Note:
            The policy_fn is called at each timestep with the current observation.
            No gradient computation or training occurs during evaluation.
        """
        rewards = []
        episode_lengths = []

        for _episode in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                action = policy_fn(obs)
                obs, reward, terminated, truncated, _info = self.env.step(action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated

            rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "mean_length": np.mean(episode_lengths),
            "episodes_evaluated": num_episodes,
        }

    def save_training_data(self, filepath: str):
        """Save training history and configuration to JSON file.

        Args:
            filepath: Path where training data should be saved (with .json extension).

        Note:
            Saves:
            - training_history: List of training episodes and their statistics
            - best_reward: Best reward achieved during training
            - env_config: Environment configuration (robot_type, task_type, max_episode_steps)

            The file is written in JSON format with indentation for readability.
        """
        data = {
            "training_history": self.training_history,
            "best_reward": self.best_reward,
            "env_config": {
                "robot_type": self.env.config.robot_type,
                "task_type": self.env.config.task_type,
                "max_episode_steps": self.env.config.max_episode_steps,
            },
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)


# Factory functions for common RL setups
def create_reaching_env(robot_type: str = "franka_panda") -> MuJoCoRLEnvironment:
    """Create reaching task environment"""
    config = RLConfig(
        robot_type=robot_type,
        task_type=TaskType.REACHING,
        max_episode_steps=500,
        action_space_type=ActionSpaceType.CONTINUOUS,
    )
    return MuJoCoRLEnvironment(config)


def create_balancing_env() -> MuJoCoRLEnvironment:
    """Create balancing task environment"""
    config = RLConfig(
        robot_type="cart_pole",
        task_type=TaskType.BALANCING,
        max_episode_steps=1000,
        action_space_type=ActionSpaceType.DISCRETE,
    )
    return MuJoCoRLEnvironment(config)


def create_walking_env(robot_type: str = "quadruped") -> MuJoCoRLEnvironment:
    """Create walking task environment"""
    config = RLConfig(
        robot_type=robot_type,
        task_type=TaskType.WALKING,
        max_episode_steps=2000,
        action_space_type=ActionSpaceType.CONTINUOUS,
    )
    return MuJoCoRLEnvironment(config)


# Example training script
def example_training():
    """Example training script"""
    # Create environment
    env = create_reaching_env("franka_panda")
    trainer = RLTrainer(env)

    print("🤖 MuJoCo MCP RL Training Example")
    print("=" * 50)

    # Run random baseline
    baseline_results = trainer.random_policy_baseline(num_episodes=5)

    # Simple PID policy example
    def pid_policy(obs):
        # Simple proportional control toward target
        target_pos = np.array([0.5, 0.0, 0.5])  # Target position
        current_pos = obs[:3]  # Current end-effector position
        error = target_pos - current_pos
        action = 0.1 * error  # Simple proportional control
        return np.clip(action, -1, 1)

    # Evaluate PID policy
    pid_results = trainer.evaluate_policy(pid_policy, num_episodes=5)

    print("\nPID Policy Results:")
    for key, value in pid_results.items():
        print(f"  {key}: {value:.4f}")

    # Close environment
    env.close()


if __name__ == "__main__":
    example_training()

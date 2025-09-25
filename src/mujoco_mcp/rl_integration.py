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

from .viewer_client import MuJoCoViewerClient
from .sensor_feedback import SensorManager


@dataclass
class RLConfig:
    """Configuration for RL environment"""
    robot_type: str
    task_type: str
    max_episode_steps: int = 1000
    reward_scale: float = 1.0
    action_space_type: str = "continuous"  # "continuous" or "discrete"
    observation_space_size: int = 0
    action_space_size: int = 0
    render_mode: str | None = None
    physics_timestep: float = 0.002
    control_timestep: float = 0.02


class TaskReward(ABC):
    """Abstract base class for task-specific reward functions"""

    @abstractmethod
    def compute_reward(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        next_observation: np.ndarray,
        info: Dict[str, Any]
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
        info: Dict[str, Any]
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
        info: Dict[str, Any]
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
        info: Dict[str, Any]
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
    """Gymnasium-compatible RL environment for MuJoCo MCP"""

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
            "quadruped": {"joints": 8}
        }

        if self.config.robot_type in robot_configs:
            n_joints = robot_configs[self.config.robot_type]["joints"]
        else:
            n_joints = 6  # Default

        # Action space
        if self.config.action_space_type == "continuous":
            # Continuous joint torques/positions
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(n_joints,),
                dtype=np.float32
            )
        else:
            # Discrete action space
            self.action_space = spaces.Discrete(n_joints * 3)  # 3 actions per joint

        # Observation space
        obs_size = self.config.observation_space_size
        if obs_size == 0:
            # Auto-determine observation size
            obs_size = n_joints * 2 + 6  # joint pos + vel + end-effector pose

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )

    def _create_reward_function(self) -> TaskReward:
        """Create task-specific reward function"""
        if self.config.task_type == "reaching":
            target = np.array([0.5, 0.0, 0.5])  # Default target position
            return ReachingTaskReward(target)
        elif self.config.task_type == "balancing":
            return BalancingTaskReward()
        elif self.config.task_type == "walking":
            return WalkingTaskReward()
        else:
            # Default reward function
            return ReachingTaskReward(np.array([0.0, 0.0, 1.0]))

    def _create_model_xml(self) -> str:
        """Create model XML for the RL task"""
        if self.config.task_type == "reaching" and self.config.robot_type == "franka_panda":
            return self._create_franka_reaching_xml()
        elif self.config.task_type == "balancing":
            return self._create_cart_pole_xml()
        elif self.config.task_type == "walking" and "quadruped" in self.config.robot_type:
            return self._create_quadruped_xml()
        else:
            # Default simple arm
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

    def reset(self, seed: int | None = None, options: Dict | None = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment for new episode"""
        super().reset(seed=seed)

        # Connect to viewer if needed
        if not self.viewer_client.connected:
            success = self.viewer_client.connect()
            if not success:
                raise RuntimeError("Failed to connect to MuJoCo viewer server")

        # Load model
        response = self.viewer_client.send_command({
            "type": "load_model",
            "model_id": self.model_id,
            "model_xml": self.model_xml
        })

        if not response.get("success"):
            raise RuntimeError(f"Failed to load model: {response.get('error')}")

        # Reset episode state
        self.current_step = 0
        self.episode_start_time = time.time()

        # Reset reward function
        if hasattr(self.reward_function, 'prev_distance'):
            self.reward_function.prev_distance = None
        if hasattr(self.reward_function, 'prev_position'):
            self.reward_function.prev_position = None

        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: Union[np.ndarray, int]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
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
        info.update({
            "step": self.current_step,
            "step_time": step_time,
            "avg_step_time": np.mean(self.step_times),
            "episode_length": self.current_step
        })

        return new_obs, reward, terminated, truncated, info

    def _discrete_to_continuous_action(self, action: int) -> np.ndarray:
        """Convert discrete action to continuous action"""
        n_joints = self.action_space.shape[0] if hasattr(self.action_space, 'shape') else 2
        joint_idx = action // 3
        action_type = action % 3

        continuous_action = np.zeros(n_joints)
        if joint_idx < n_joints:
            if action_type == 0:
                continuous_action[joint_idx] = -1.0  # Negative
            elif action_type == 1:
                continuous_action[joint_idx] = 0.0   # Zero
            else:
                continuous_action[joint_idx] = 1.0   # Positive

        return continuous_action

    def _apply_action(self, action: np.ndarray):
        """Apply action to the robot"""
        # Scale action to appropriate range
        scaled_action = action * 10.0  # Scale to reasonable torque range

        # Send command to MuJoCo
        self.viewer_client.send_command({
            "type": "set_joint_positions",
            "model_id": self.model_id,
            "positions": scaled_action.tolist()
        })

    def _get_observation(self) -> np.ndarray:
        """Get current observation from simulation"""
        response = self.viewer_client.send_command({
            "type": "get_state",
            "model_id": self.model_id
        })

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

        # Return zero observation if state unavailable
        return np.zeros(self.observation_space.shape[0], dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about current state"""
        return {
            "episode_step": self.current_step,
            "model_id": self.model_id,
            "task_type": self.config.task_type,
            "robot_type": self.config.robot_type
        }

    def render(self):
        """Render environment (MuJoCo viewer handles this)"""
        # The MuJoCo viewer automatically renders the simulation

    def close(self):
        """Close environment"""
        if self.viewer_client.connected:
            self.viewer_client.send_command({
                "type": "close_model",
                "model_id": self.model_id
            })
            self.viewer_client.disconnect()


class RLTrainer:
    """RL training utilities for MuJoCo MCP environments"""

    def __init__(self, env: MuJoCoRLEnvironment):
        self.env = env
        self.training_history = []
        self.best_reward = -np.inf
        self.logger = logging.getLogger(__name__)

    def random_policy_baseline(self, num_episodes: int = 10) -> Dict[str, float]:
        """Run random policy baseline"""
        rewards = []
        episode_lengths = []

        for episode in range(num_episodes):
            obs, _ = self.env.reset()
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

            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")

        results = {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "mean_length": np.mean(episode_lengths),
            "std_length": np.std(episode_lengths),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards)
        }

        print("\nRandom Policy Baseline Results:")
        for key, value in results.items():
            print(f"  {key}: {value:.4f}")

        return results

    def evaluate_policy(self, policy_fn: Callable, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate a policy function"""
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
            "episodes_evaluated": num_episodes
        }

    def save_training_data(self, filepath: str):
        """Save training history to file"""
        data = {
            "training_history": self.training_history,
            "best_reward": self.best_reward,
            "env_config": {
                "robot_type": self.env.config.robot_type,
                "task_type": self.env.config.task_type,
                "max_episode_steps": self.env.config.max_episode_steps
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


# Factory functions for common RL setups
def create_reaching_env(robot_type: str = "franka_panda") -> MuJoCoRLEnvironment:
    """Create reaching task environment"""
    config = RLConfig(
        robot_type=robot_type,
        task_type="reaching",
        max_episode_steps=500,
        action_space_type="continuous"
    )
    return MuJoCoRLEnvironment(config)


def create_balancing_env() -> MuJoCoRLEnvironment:
    """Create balancing task environment"""
    config = RLConfig(
        robot_type="cart_pole",
        task_type="balancing",
        max_episode_steps=1000,
        action_space_type="discrete"
    )
    return MuJoCoRLEnvironment(config)


def create_walking_env(robot_type: str = "quadruped") -> MuJoCoRLEnvironment:
    """Create walking task environment"""
    config = RLConfig(
        robot_type=robot_type,
        task_type="walking",
        max_episode_steps=2000,
        action_space_type="continuous"
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

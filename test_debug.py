#!/usr/bin/env python3
"""Debug the action space issue"""

import sys
from pathlib import Path
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mujoco_mcp.rl_integration import create_balancing_env


def debug_balancing_env():
    print("Debugging balancing environment...")

    try:
        env = create_balancing_env()
        print(f"Environment created: {env}")
        print(f"Action space: {env.action_space}")
        print(f"Action space type: {type(env.action_space)}")

        if hasattr(env.action_space, "shape"):
            print(f"Action space shape: {env.action_space.shape}")

        if hasattr(env.action_space, "n"):
            print(f"Action space n: {env.action_space.n}")

        # Test observation generation
        obs = env._get_observation()
        print(f"Observation shape: {obs.shape}")
        print(f"Observation: {obs}")

        # Test action sampling
        action = env.action_space.sample()
        print(f"Sample action: {action}")

        # Test reward computation
        next_obs = np.random.randn(env.observation_space.shape[0])
        reward = env.reward_function.compute_reward(obs, np.array([0.0, 0.0]), next_obs, {})
        print(f"Reward: {reward}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    debug_balancing_env()

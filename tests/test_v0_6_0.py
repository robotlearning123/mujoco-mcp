#!/usr/bin/env python3
"""
Test suite for v0.6.0 - Learning Integration
Tests for reinforcement learning environment wrapper and training utilities
"""
import pytest
import pytest_asyncio
import asyncio
import numpy as np
from pathlib import Path
import sys

# Add project to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from mujoco_mcp.server import MuJoCoServer


class TestRLEnvironment:
    """Test reinforcement learning environment functionality"""
    
    @pytest_asyncio.fixture
    async def server(self):
        """Create server fixture"""
        server = MuJoCoServer()
        await server.initialize()
        yield server
        await server.cleanup()
    
    @pytest.mark.asyncio
    async def test_create_rl_environment_tool_exists(self, server):
        """Test that RL environment creation tool exists"""
        tool_names = list(server.mcp._tool_manager._tools.keys())
        assert "create_rl_environment" in tool_names
    
    @pytest.mark.asyncio
    async def test_create_basic_rl_environment(self, server):
        """Test creating a basic RL environment"""
        result = server._impl._handle_create_rl_environment(
            env_type="pendulum",
            observation_space="continuous",
            action_space="continuous",
            max_episode_steps=1000
        )
        
        assert result["success"] is True
        assert "env_id" in result
        assert "observation_dim" in result
        assert "action_dim" in result
        assert result["observation_space"] == "continuous"
        assert result["action_space"] == "continuous"
    
    @pytest.mark.asyncio
    async def test_reset_rl_environment(self, server):
        """Test resetting RL environment"""
        # Create environment
        env_result = server._impl._handle_create_rl_environment(
            env_type="cartpole",
            observation_space="continuous",
            action_space="discrete"
        )
        env_id = env_result["env_id"]
        
        # Reset environment
        result = server._impl._handle_reset_rl_environment(env_id=env_id)
        
        assert result["success"] is True
        assert "observation" in result
        assert "info" in result
        assert isinstance(result["observation"], list)
    
    @pytest.mark.asyncio
    async def test_step_rl_environment(self, server):
        """Test stepping RL environment with action"""
        # Create and reset environment
        env_result = server._impl._handle_create_rl_environment(
            env_type="reacher",
            observation_space="continuous",
            action_space="continuous"
        )
        env_id = env_result["env_id"]
        
        server._impl._handle_reset_rl_environment(env_id=env_id)
        
        # Take a step
        action = [0.1, -0.2]  # Example action
        result = server._impl._handle_step_rl_environment(
            env_id=env_id,
            action=action
        )
        
        assert result["success"] is True
        assert "observation" in result
        assert "reward" in result
        assert "done" in result
        assert "info" in result
        assert isinstance(result["reward"], (int, float))
        assert isinstance(result["done"], bool)
    
    @pytest.mark.asyncio
    async def test_get_rl_state(self, server):
        """Test getting current RL environment state"""
        # Create environment
        env_result = server._impl._handle_create_rl_environment(
            env_type="hopper"
        )
        env_id = env_result["env_id"]
        
        # Get state
        result = server._impl._handle_get_rl_state(env_id=env_id)
        
        assert "episode_step" in result
        assert "total_reward" in result
        assert "is_terminal" in result
        assert "observation" in result
    
    @pytest.mark.asyncio
    async def test_reward_functions(self, server):
        """Test different reward function configurations"""
        # Create environment with custom reward
        result = server._impl._handle_create_rl_environment(
            env_type="walker",
            reward_function="forward_progress",
            reward_params={
                "forward_weight": 1.0,
                "stability_weight": 0.1,
                "effort_weight": 0.01
            }
        )
        
        assert result["success"] is True
        assert result["reward_function"] == "forward_progress"
        
        # List available reward functions
        rewards = server._impl._handle_list_reward_functions()
        assert "available_functions" in rewards
        assert len(rewards["available_functions"]) > 0
    
    @pytest.mark.asyncio
    async def test_episode_management(self, server):
        """Test episode tracking and management"""
        # Create environment
        env_result = server._impl._handle_create_rl_environment(
            env_type="ant",
            max_episode_steps=500
        )
        env_id = env_result["env_id"]
        
        # Run partial episode
        server._impl._handle_reset_rl_environment(env_id=env_id)
        
        for _ in range(10):
            server._impl._handle_step_rl_environment(
                env_id=env_id,
                action=[0.1] * 2  # Default environment has 2 actuators
            )
        
        # Get episode info
        result = server._impl._handle_get_episode_info(env_id=env_id)
        
        assert result["steps_taken"] == 10
        assert result["max_steps"] == 500
        assert "cumulative_reward" in result
        assert "average_reward" in result


class TestPolicyInterface:
    """Test policy interface functionality"""
    
    @pytest_asyncio.fixture
    async def server(self):
        """Create server fixture"""
        server = MuJoCoServer()
        await server.initialize()
        yield server
        await server.cleanup()
    
    @pytest.mark.asyncio
    async def test_register_policy_tool_exists(self, server):
        """Test that policy registration tool exists"""
        tool_names = list(server.mcp._tool_manager._tools.keys())
        assert "register_policy" in tool_names
    
    @pytest.mark.asyncio
    async def test_register_random_policy(self, server):
        """Test registering a random policy"""
        result = server._impl._handle_register_policy(
            policy_type="random",
            policy_name="test_random_policy",
            action_space_dim=4
        )
        
        assert result["success"] is True
        assert "policy_id" in result
        assert result["policy_type"] == "random"
    
    @pytest.mark.asyncio
    async def test_register_neural_policy(self, server):
        """Test registering a neural network policy"""
        result = server._impl._handle_register_policy(
            policy_type="neural",
            policy_name="test_nn_policy",
            architecture={
                "input_dim": 8,
                "hidden_layers": [64, 64],
                "output_dim": 2,
                "activation": "relu"
            }
        )
        
        assert result["success"] is True
        assert "policy_id" in result
        assert result["policy_type"] == "neural"
    
    @pytest.mark.asyncio
    async def test_execute_policy(self, server):
        """Test executing a policy to get actions"""
        # Register policy
        policy_result = server._impl._handle_register_policy(
            policy_type="random",
            policy_name="test_policy",
            action_space_dim=2
        )
        policy_id = policy_result["policy_id"]
        
        # Execute policy
        observation = [0.1, 0.2, 0.3, 0.4]
        result = server._impl._handle_execute_policy(
            policy_id=policy_id,
            observation=observation
        )
        
        assert result["success"] is True
        assert "action" in result
        assert len(result["action"]) == 2
        assert all(isinstance(a, (int, float)) for a in result["action"])
    
    @pytest.mark.asyncio
    async def test_update_policy(self, server):
        """Test updating policy parameters"""
        # Register neural policy
        policy_result = server._impl._handle_register_policy(
            policy_type="neural",
            policy_name="test_update_policy",
            architecture={
                "input_dim": 4,
                "hidden_layers": [32],
                "output_dim": 2
            }
        )
        policy_id = policy_result["policy_id"]
        
        # Update policy
        result = server._impl._handle_update_policy(
            policy_id=policy_id,
            parameters={
                "learning_rate": 0.001,
                "weights": "random_init"  # Simplified
            }
        )
        
        assert result["success"] is True
        assert result["parameters_updated"] is True


class TestTrainingUtilities:
    """Test training utilities functionality"""
    
    @pytest_asyncio.fixture
    async def server(self):
        """Create server fixture"""
        server = MuJoCoServer()
        await server.initialize()
        yield server
        await server.cleanup()
    
    @pytest.mark.asyncio
    async def test_create_training_session_tool_exists(self, server):
        """Test that training session creation tool exists"""
        tool_names = list(server.mcp._tool_manager._tools.keys())
        assert "create_training_session" in tool_names
    
    @pytest.mark.asyncio
    async def test_create_training_session(self, server):
        """Test creating a training session"""
        # Create environment and policy first
        env_result = server._impl._handle_create_rl_environment(
            env_type="pendulum"
        )
        env_id = env_result["env_id"]
        
        policy_result = server._impl._handle_register_policy(
            policy_type="neural",
            policy_name="pendulum_policy"
        )
        policy_id = policy_result["policy_id"]
        
        # Create training session
        result = server._impl._handle_create_training_session(
            env_id=env_id,
            policy_id=policy_id,
            algorithm="ppo",
            hyperparameters={
                "learning_rate": 3e-4,
                "batch_size": 64,
                "n_epochs": 10
            }
        )
        
        assert result["success"] is True
        assert "session_id" in result
        assert result["algorithm"] == "ppo"
    
    @pytest.mark.asyncio
    async def test_run_training_steps(self, server):
        """Test running training steps"""
        # Setup
        env_result = server._impl._handle_create_rl_environment(
            env_type="cartpole"
        )
        policy_result = server._impl._handle_register_policy(
            policy_type="neural",
            policy_name="test_policy"
        )
        
        session_result = server._impl._handle_create_training_session(
            env_id=env_result["env_id"],
            policy_id=policy_result["policy_id"],
            algorithm="dqn"
        )
        session_id = session_result["session_id"]
        
        # Run training steps
        result = server._impl._handle_run_training_steps(
            session_id=session_id,
            num_steps=100
        )
        
        assert result["success"] is True
        assert result["steps_completed"] == 100
        assert "metrics" in result
        assert "average_reward" in result["metrics"]
        assert "loss" in result["metrics"]
    
    @pytest.mark.asyncio
    async def test_evaluate_policy(self, server):
        """Test policy evaluation"""
        # Setup
        env_result = server._impl._handle_create_rl_environment(
            env_type="reacher"
        )
        policy_result = server._impl._handle_register_policy(
            policy_type="random",
            policy_name="test_policy"
        )
        
        # Evaluate policy
        result = server._impl._handle_evaluate_policy(
            env_id=env_result["env_id"],
            policy_id=policy_result["policy_id"],
            num_episodes=5,
            render=False
        )
        
        assert result["success"] is True
        assert result["episodes_evaluated"] == 5
        assert "average_reward" in result
        assert "std_reward" in result
        assert "success_rate" in result
    
    @pytest.mark.asyncio
    async def test_save_and_load_checkpoint(self, server):
        """Test saving and loading training checkpoints"""
        # Setup training session
        env_result = server._impl._handle_create_rl_environment(
            env_type="walker"
        )
        policy_result = server._impl._handle_register_policy(
            policy_type="neural",
            policy_name="test_policy"
        )
        session_result = server._impl._handle_create_training_session(
            env_id=env_result["env_id"],
            policy_id=policy_result["policy_id"]
        )
        session_id = session_result["session_id"]
        
        # Save checkpoint
        save_result = server._impl._handle_save_checkpoint(
            session_id=session_id,
            checkpoint_name="test_checkpoint"
        )
        
        assert save_result["success"] is True
        assert "checkpoint_id" in save_result
        assert "path" in save_result
        
        # Load checkpoint
        load_result = server._impl._handle_load_checkpoint(
            checkpoint_id=save_result["checkpoint_id"]
        )
        
        assert load_result["success"] is True
        assert load_result["session_id"] == session_id
    
    @pytest.mark.asyncio
    async def test_training_metrics(self, server):
        """Test retrieving training metrics"""
        # Setup and run some training
        env_result = server._impl._handle_create_rl_environment(
            env_type="humanoid"
        )
        policy_result = server._impl._handle_register_policy(
            policy_type="neural",
            policy_name="test_policy"
        )
        session_result = server._impl._handle_create_training_session(
            env_id=env_result["env_id"],
            policy_id=policy_result["policy_id"]
        )
        
        # Run some training
        server._impl._handle_run_training_steps(
            session_id=session_result["session_id"],
            num_steps=50
        )
        
        # Get metrics
        result = server._impl._handle_get_training_metrics(
            session_id=session_result["session_id"],
            metric_types=["reward", "loss", "entropy"]
        )
        
        assert "metrics" in result
        assert "reward" in result["metrics"]
        assert "loss" in result["metrics"]
        assert "timestamps" in result


class TestServerVersion060:
    """Test server version update"""
    
    @pytest.mark.asyncio
    async def test_version_updated(self):
        """Test that server version is 0.6.0"""
        server = MuJoCoServer()
        await server.initialize()
        
        try:
            assert server.version == "0.6.0"
            
            info = server.get_server_info()
            assert info["version"] == "0.6.0"
        finally:
            await server.cleanup()
    
    @pytest.mark.asyncio
    async def test_learning_capability(self):
        """Test that learning integration capability is advertised"""
        server = MuJoCoServer()
        await server.initialize()
        
        try:
            info = server.get_server_info()
            
            assert "capabilities" in info
            assert "reinforcement_learning" in info["capabilities"]
            assert info["capabilities"]["reinforcement_learning"] is True
        finally:
            await server.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
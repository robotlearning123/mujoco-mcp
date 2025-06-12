#!/usr/bin/env python3
"""
Reinforcement Learning Demo for MuJoCo MCP v0.6.0

This demo showcases the reinforcement learning integration features including:
- RL environment creation
- Policy management
- Training sessions
- Performance evaluation
"""

import asyncio
import time
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from mujoco_mcp.server import MuJoCoServer


def print_section(title):
    """Print a section header"""
    print(f"\n{'='*70}")
    print(f" {title}")
    print('='*70)


async def main():
    """Run reinforcement learning demo"""
    # Create and initialize server
    server = MuJoCoServer()
    await server.initialize()
    
    print(f"MuJoCo MCP Server v{server.version} - Reinforcement Learning Demo")
    
    # 1. Create RL Environments
    print_section("1. Creating RL Environments")
    
    environments = {}
    env_types = ["pendulum", "cartpole", "reacher"]
    
    for env_type in env_types:
        result = server._impl._handle_create_rl_environment(
            env_type=env_type,
            observation_space="continuous",
            action_space="continuous" if env_type != "cartpole" else "discrete",
            max_episode_steps=1000
        )
        
        environments[env_type] = result["env_id"]
        print(f"✓ Created {env_type} environment")
        print(f"  - Environment ID: {result['env_id'][:8]}...")
        print(f"  - Observation dim: {result['observation_dim']}")
        print(f"  - Action dim: {result['action_dim']}")
    
    # 2. Test Environment Reset and Step
    print_section("2. Testing Environment Interaction")
    
    # Use pendulum environment for testing
    pendulum_env = environments["pendulum"]
    
    # Reset environment
    reset_result = server._impl._handle_reset_rl_environment(env_id=pendulum_env)
    print(f"✓ Reset pendulum environment")
    print(f"  - Initial observation: {reset_result['observation']}")
    
    # Take a few steps
    print("\nTaking 5 steps with random actions...")
    for i in range(5):
        action = [0.5 - i * 0.2]  # Varying torque
        step_result = server._impl._handle_step_rl_environment(
            env_id=pendulum_env,
            action=action
        )
        print(f"  Step {i+1}: reward={step_result['reward']:.3f}, done={step_result['done']}")
    
    # 3. Reward Functions
    print_section("3. Available Reward Functions")
    
    reward_functions = server._impl._handle_list_reward_functions()
    print("Available reward functions:")
    for func in reward_functions["available_functions"]:
        print(f"  - {func['name']}: {func['description']}")
        if func['parameters']:
            print(f"    Parameters: {', '.join(func['parameters'])}")
    
    # Create environment with custom reward
    print("\nCreating walker with forward progress reward...")
    walker_result = server._impl._handle_create_rl_environment(
        env_type="walker",
        reward_function="forward_progress",
        reward_params={
            "forward_weight": 1.0,
            "stability_weight": 0.1,
            "effort_weight": 0.01
        }
    )
    print(f"✓ Created walker with custom reward function")
    
    # 4. Policy Management
    print_section("4. Policy Management")
    
    # Register different types of policies
    policies = {}
    
    # Random policy
    random_policy = server._impl._handle_register_policy(
        policy_type="random",
        policy_name="random_explorer",
        action_space_dim=1
    )
    policies["random"] = random_policy["policy_id"]
    print(f"✓ Registered random policy: {random_policy['policy_name']}")
    
    # Neural network policy
    nn_policy = server._impl._handle_register_policy(
        policy_type="neural",
        policy_name="pendulum_controller",
        architecture={
            "input_dim": 3,
            "hidden_layers": [64, 64],
            "output_dim": 1,
            "activation": "relu"
        }
    )
    policies["neural"] = nn_policy["policy_id"]
    print(f"✓ Registered neural policy: {nn_policy['policy_name']}")
    print(f"  - Architecture: {nn_policy['policy_type']}")
    
    # Execute policies
    print("\nExecuting policies...")
    observation = [0.5, -0.2, 0.1]  # Example observation
    
    for policy_type, policy_id in policies.items():
        action_result = server._impl._handle_execute_policy(
            policy_id=policy_id,
            observation=observation
        )
        print(f"  {policy_type} policy action: {action_result['action']}")
    
    # 5. Training Sessions
    print_section("5. Training Sessions")
    
    # Create training session for pendulum
    session_result = server._impl._handle_create_training_session(
        env_id=pendulum_env,
        policy_id=policies["neural"],
        algorithm="ppo",
        hyperparameters={
            "learning_rate": 3e-4,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99
        }
    )
    session_id = session_result["session_id"]
    print(f"✓ Created training session")
    print(f"  - Session ID: {session_id[:8]}...")
    print(f"  - Algorithm: {session_result['algorithm']}")
    print(f"  - Learning rate: {session_result['hyperparameters']['learning_rate']}")
    
    # Run training steps
    print("\nRunning training...")
    for epoch in range(3):
        train_result = server._impl._handle_run_training_steps(
            session_id=session_id,
            num_steps=50
        )
        print(f"  Epoch {epoch+1}: avg_reward={train_result['metrics']['average_reward']:.3f}, "
              f"loss={train_result['metrics']['loss']:.3f}")
    
    # 6. Policy Evaluation
    print_section("6. Policy Evaluation")
    
    # Evaluate trained policy
    eval_result = server._impl._handle_evaluate_policy(
        env_id=pendulum_env,
        policy_id=policies["neural"],
        num_episodes=5,
        render=False
    )
    
    print(f"✓ Policy evaluation complete")
    print(f"  - Episodes: {eval_result['episodes_evaluated']}")
    print(f"  - Average reward: {eval_result['average_reward']:.3f} ± {eval_result['std_reward']:.3f}")
    print(f"  - Average length: {eval_result['average_length']:.1f}")
    print(f"  - Success rate: {eval_result['success_rate']:.2%}")
    
    # Compare with random policy
    print("\nComparing with random policy...")
    random_eval = server._impl._handle_evaluate_policy(
        env_id=pendulum_env,
        policy_id=policies["random"],
        num_episodes=5,
        render=False
    )
    print(f"  Random policy avg reward: {random_eval['average_reward']:.3f}")
    
    # 7. Checkpointing
    print_section("7. Checkpointing")
    
    # Save checkpoint
    save_result = server._impl._handle_save_checkpoint(
        session_id=session_id,
        checkpoint_name="pendulum_ppo_checkpoint"
    )
    checkpoint_id = save_result["checkpoint_id"]
    print(f"✓ Saved checkpoint: {save_result['checkpoint_name']}")
    print(f"  - Checkpoint ID: {checkpoint_id[:8]}...")
    print(f"  - Path: {save_result['path']}")
    
    # Load checkpoint
    load_result = server._impl._handle_load_checkpoint(
        checkpoint_id=checkpoint_id
    )
    print(f"✓ Loaded checkpoint")
    print(f"  - Restored session: {load_result['session_id'][:8]}...")
    
    # 8. Training Metrics
    print_section("8. Training Metrics")
    
    # Get training metrics
    metrics_result = server._impl._handle_get_training_metrics(
        session_id=session_id,
        metric_types=["reward", "loss"]
    )
    
    print("Training metrics (last 10 steps):")
    if "reward" in metrics_result["metrics"]:
        rewards = metrics_result["metrics"]["reward"][-10:]
        print(f"  Rewards: min={min(rewards):.3f}, max={max(rewards):.3f}, "
              f"avg={sum(rewards)/len(rewards):.3f}")
    
    if "loss" in metrics_result["metrics"]:
        losses = metrics_result["metrics"]["loss"][-10:]
        print(f"  Losses: min={min(losses):.3f}, max={max(losses):.3f}, "
              f"avg={sum(losses)/len(losses):.3f}")
    
    # 9. Episode Information
    print_section("9. Episode Information")
    
    # Get episode info
    episode_info = server._impl._handle_get_episode_info(env_id=pendulum_env)
    print(f"Current episode status:")
    print(f"  - Steps: {episode_info['steps_taken']}/{episode_info['max_steps']}")
    print(f"  - Cumulative reward: {episode_info['cumulative_reward']:.3f}")
    print(f"  - Average reward: {episode_info['average_reward']:.3f}")
    print(f"  - Episode done: {episode_info['is_done']}")
    
    # 10. Multi-Environment Training
    print_section("10. Multi-Environment Training")
    
    # Create sessions for multiple environments
    print("Creating training sessions for multiple environments...")
    sessions = {}
    
    for env_type, env_id in environments.items():
        if env_type != "walker":  # Skip walker (created separately)
            # Create appropriate policy
            obs_dim = 3 if env_type == "pendulum" else 4 if env_type == "cartpole" else 6
            act_dim = 1 if env_type in ["pendulum", "cartpole"] else 2
            
            policy = server._impl._handle_register_policy(
                policy_type="neural",
                policy_name=f"{env_type}_policy",
                architecture={
                    "input_dim": obs_dim,
                    "hidden_layers": [32, 32],
                    "output_dim": act_dim
                }
            )
            
            session = server._impl._handle_create_training_session(
                env_id=env_id,
                policy_id=policy["policy_id"],
                algorithm="sac" if env_type == "reacher" else "dqn"
            )
            
            sessions[env_type] = session["session_id"]
            print(f"  ✓ {env_type}: {session['algorithm']} algorithm")
    
    # Cleanup
    await server.cleanup()
    
    print_section("Demo Complete!")
    print("\nReinforcement learning features demonstrated:")
    print("- RL environment creation (pendulum, cartpole, reacher, walker)")
    print("- Environment reset and step operations")
    print("- Custom reward functions")
    print("- Policy registration (random, neural)")
    print("- Policy execution")
    print("- Training session management")
    print("- Training execution with metrics")
    print("- Policy evaluation and comparison")
    print("- Checkpoint save/load")
    print("- Multi-environment training setup")


if __name__ == "__main__":
    asyncio.run(main())
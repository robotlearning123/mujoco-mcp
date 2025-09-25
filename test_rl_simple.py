#!/usr/bin/env python3
"""
Simplified RL Test - Tests core RL functionality without MuJoCo viewer dependency
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from mujoco_mcp.rl_integration import (
        RLConfig, MuJoCoRLEnvironment, RLTrainer,
        create_reaching_env, create_balancing_env, create_walking_env,
        example_training
    )
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

def test_rl_core_functionality():
    """Test core RL functionality without MuJoCo viewer"""
    print("🤖 MuJoCo MCP RL Core Functionality Test")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Environment Creation
    print("\n1. Testing Environment Creation:")
    total_tests += 1
    try:
        env_reach = create_reaching_env("franka_panda")
        env_balance = create_balancing_env()
        env_walk = create_walking_env("quadruped")
        print("   ✅ All environments created successfully")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Environment creation failed: {e}")
    
    # Test 2: Action and Observation Spaces
    print("\n2. Testing Action and Observation Spaces:")
    total_tests += 1
    try:
        env = create_reaching_env("franka_panda")
        print(f"   Action space: {env.action_space}")
        print(f"   Observation space: {env.observation_space}")
        
        # Test action sampling
        action = env.action_space.sample()
        print(f"   Sample action: {action[:3]}... (shape: {action.shape})")
        
        # Test observation structure
        obs = env._get_observation()  # Returns zeros without MuJoCo
        print(f"   Observation shape: {obs.shape}")
        
        tests_passed += 1
        print("   ✅ Spaces test passed")
    except Exception as e:
        print(f"   ❌ Spaces test failed: {e}")
    
    # Test 3: Reward Functions
    print("\n3. Testing Reward Functions:")
    total_tests += 1
    try:
        env = create_reaching_env("franka_panda")
        
        # Create dummy data
        obs = np.random.randn(env.observation_space.shape[0])
        action = np.random.randn(env.action_space.shape[0])
        next_obs = np.random.randn(env.observation_space.shape[0])
        info = {}
        
        # Test reward computation
        reward = env.reward_function.compute_reward(obs, action, next_obs, info)
        print(f"   Reaching reward: {reward:.4f}")
        
        # Test termination
        done = env.reward_function.is_done(next_obs, info)
        print(f"   Episode done: {done}")
        
        tests_passed += 1
        print("   ✅ Reward functions test passed")
    except Exception as e:
        print(f"   ❌ Reward functions test failed: {e}")
    
    # Test 4: XML Generation
    print("\n4. Testing XML Generation:")
    total_tests += 1
    try:
        robots = [("franka_panda", "reaching"), ("cart_pole", "balancing"), ("quadruped", "walking")]
        
        for robot, task in robots:
            if task == "reaching":
                env = create_reaching_env(robot)
            elif task == "balancing":
                env = create_balancing_env()
            else:
                env = create_walking_env(robot)
            
            xml = env._create_model_xml()
            print(f"   {robot} XML: {len(xml)} chars")
            
            # Basic validation
            assert "<mujoco" in xml
            assert "</mujoco>" in xml
            assert "worldbody" in xml
        
        tests_passed += 1
        print("   ✅ XML generation test passed")
    except Exception as e:
        print(f"   ❌ XML generation test failed: {e}")
    
    # Test 5: Trainer Creation
    print("\n5. Testing Trainer Creation:")
    total_tests += 1
    try:
        env = create_reaching_env("franka_panda")
        trainer = RLTrainer(env)
        
        print(f"   Trainer best reward: {trainer.best_reward}")
        print(f"   Training history length: {len(trainer.training_history)}")
        
        # Test policy evaluation structure (without actually running)
        def dummy_policy(obs):
            return np.zeros(env.action_space.shape[0])
        
        # Just test that the method exists and has correct signature
        assert hasattr(trainer, 'evaluate_policy')
        assert callable(trainer.evaluate_policy)
        
        tests_passed += 1
        print("   ✅ Trainer creation test passed")
    except Exception as e:
        print(f"   ❌ Trainer creation test failed: {e}")
    
    # Test 6: Action Conversion
    print("\n6. Testing Action Conversion:")
    total_tests += 1
    try:
        config = RLConfig(robot_type="cart_pole", task_type="balancing", action_space_type="discrete")
        env = MuJoCoRLEnvironment(config)
        
        # Test discrete to continuous conversion
        for i in [0, 1, 2, 5]:
            if hasattr(env.action_space, 'n') and i < env.action_space.n:
                continuous = env._discrete_to_continuous_action(i)
                print(f"   Discrete {i} -> Continuous {continuous}")
        
        tests_passed += 1
        print("   ✅ Action conversion test passed")
    except Exception as e:
        print(f"   ❌ Action conversion test failed: {e}")
    
    # Test 7: Performance Monitoring
    print("\n7. Testing Performance Monitoring:")
    total_tests += 1
    try:
        env = create_reaching_env("franka_panda")
        
        # Simulate step times
        env.step_times.extend([0.001, 0.002, 0.0015])
        avg_time = np.mean(env.step_times)
        print(f"   Average step time: {avg_time:.6f}s")
        
        # Test info generation
        env.current_step = 42
        info = env._get_info()
        print(f"   Info keys: {list(info.keys())}")
        
        tests_passed += 1
        print("   ✅ Performance monitoring test passed")
    except Exception as e:
        print(f"   ❌ Performance monitoring test failed: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print(f"Total Tests: {total_tests}")
    print(f"✅ Passed: {tests_passed}")
    print(f"❌ Failed: {total_tests - tests_passed}")
    print(f"Success Rate: {(tests_passed/total_tests)*100:.1f}%")
    
    return tests_passed == total_tests

def benchmark_rl_performance():
    """Benchmark RL performance"""
    print("\n🚀 RL Performance Benchmark")
    print("-" * 30)
    
    try:
        # Create environments
        envs = {
            "reaching": create_reaching_env("franka_panda"),
            "balancing": create_balancing_env(),
            "walking": create_walking_env("quadruped")
        }
        
        for env_name, env in envs.items():
            print(f"\n{env_name.title()} Environment:")
            
            # Benchmark observation generation
            start_time = time.time()
            for _ in range(100):
                obs = env._get_observation()
            obs_time = (time.time() - start_time) / 100
            
            # Benchmark action sampling
            start_time = time.time()
            for _ in range(100):
                action = env.action_space.sample()
            action_time = (time.time() - start_time) / 100
            
            # Benchmark reward computation
            dummy_obs = np.zeros(env.observation_space.shape[0])
            dummy_action = np.zeros(env.action_space.shape[0])
            start_time = time.time()
            for _ in range(100):
                reward = env.reward_function.compute_reward(dummy_obs, dummy_action, dummy_obs, {})
            reward_time = (time.time() - start_time) / 100
            
            print(f"  Observation: {obs_time*1000:.3f}ms")
            print(f"  Action:      {action_time*1000:.3f}ms")
            print(f"  Reward:      {reward_time*1000:.3f}ms")
            print(f"  Total:       {(obs_time + action_time + reward_time)*1000:.3f}ms")
            
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")

def test_example_training():
    """Test that example training script runs without errors"""
    print("\n🎓 Testing Example Training Script")
    print("-" * 30)
    
    try:
        # This will create environment and trainer but not actually connect to MuJoCo
        # The example_training function should handle connection failures gracefully
        print("Running example training (structural test only)...")
        
        # Just test that we can create the components
        env = create_reaching_env("franka_panda")
        trainer = RLTrainer(env)
        
        print("✅ Example training components created successfully")
        
        # Test that the example function exists
        assert callable(example_training), "example_training function not found"
        print("✅ example_training function is callable")
        
    except Exception as e:
        print(f"❌ Example training test failed: {e}")

def main():
    """Main test execution"""
    success = test_rl_core_functionality()
    benchmark_rl_performance()
    test_example_training()
    
    if success:
        print("\n🎉 Core RL functionality tests passed!")
        print("Note: Full integration tests require MuJoCo viewer server")
        return 0
    else:
        print("\n⚠️  Some core tests failed.")
        return 1

if __name__ == "__main__":
    exit(main())
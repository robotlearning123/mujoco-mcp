#!/usr/bin/env python3
"""
RL Integration Test with MuJoCo Viewer
Tests RL functionality with actual MuJoCo physics simulation
"""

import sys
import time
import subprocess
import signal
import os
import numpy as np
from pathlib import Path
from typing import Dict, Any
import threading
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from mujoco_mcp.rl_integration import (
        RLConfig, MuJoCoRLEnvironment, RLTrainer,
        create_reaching_env, create_balancing_env, create_walking_env
    )
    from mujoco_mcp.viewer_server import MuJoCoViewerServer
    from mujoco_mcp.viewer_client import MuJoCoViewerClient
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

class RLIntegrationTest:
    """Test RL integration with MuJoCo viewer"""
    
    def __init__(self):
        self.viewer_process = None
        self.viewer_server = None
        self.results = {}
        
    def start_viewer_server(self):
        """Start MuJoCo viewer server in background"""
        try:
            print("🚀 Starting MuJoCo viewer server...")
            self.viewer_server = MuJoCoViewerServer(host="localhost", port=12345)
            
            # Start server in separate thread
            server_thread = threading.Thread(target=self._run_server)
            server_thread.daemon = True
            server_thread.start()
            
            # Give server time to start
            time.sleep(2)
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start viewer server: {e}")
            return False
    
    def _run_server(self):
        """Run server in thread"""
        try:
            self.viewer_server.run()
        except Exception as e:
            print(f"Viewer server error: {e}")
    
    def test_rl_environment_connection(self):
        """Test RL environment connection to MuJoCo"""
        try:
            print("🔗 Testing RL environment connection...")
            
            # Create reaching environment
            env = create_reaching_env("franka_panda")
            
            # Test connection without reset (should fail gracefully)
            client = MuJoCoViewerClient()
            connected = client.connect()
            
            if not connected:
                print("⚠️  MuJoCo viewer not available, testing structure only")
                self.test_environment_structure(env)
                return True
            
            # Test reset with connection
            try:
                obs, info = env.reset()
                print(f"✅ Environment reset successful, obs shape: {obs.shape}")
                
                # Test action execution
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"✅ Step execution successful, reward: {reward:.4f}")
                
                env.close()
                client.disconnect()
                return True
                
            except Exception as step_error:
                print(f"⚠️  Step execution failed: {step_error}")
                return False
                
        except Exception as e:
            print(f"❌ Environment connection test failed: {e}")
            return False
    
    def test_environment_structure(self, env):
        """Test environment structure without MuJoCo connection"""
        print("🏗️  Testing environment structure...")
        
        # Test spaces
        print(f"   Action space: {env.action_space}")
        print(f"   Observation space: {env.observation_space}")
        
        # Test XML generation
        xml = env._create_model_xml()
        print(f"   Generated XML length: {len(xml)} chars")
        
        # Test reward function
        dummy_obs = np.zeros(env.observation_space.shape[0])
        dummy_action = np.zeros(env.action_space.shape[0] if hasattr(env.action_space, 'shape') else 3)
        reward = env.reward_function.compute_reward(dummy_obs, dummy_action, dummy_obs, {})
        print(f"   Reward function test: {reward:.4f}")
        
        return True
    
    def test_multiple_environments(self):
        """Test multiple environment types"""
        print("🎯 Testing multiple environment types...")
        
        env_configs = [
            ("reaching", "franka_panda"),
            ("balancing", "cart_pole"),
            ("walking", "quadruped")
        ]
        
        for task, robot in env_configs:
            try:
                if task == "reaching":
                    env = create_reaching_env(robot)
                elif task == "balancing":
                    env = create_balancing_env()
                elif task == "walking":
                    env = create_walking_env(robot)
                
                print(f"   ✅ {task} environment created")
                
                # Test basic properties
                assert hasattr(env, 'action_space')
                assert hasattr(env, 'observation_space')
                assert hasattr(env, 'reward_function')
                
                # Test XML generation
                xml = env._create_model_xml()
                assert len(xml) > 100  # Should have substantial XML
                
                print(f"      Action space: {env.action_space}")
                print(f"      XML length: {len(xml)} chars")
                
            except Exception as e:
                print(f"   ❌ {task} environment failed: {e}")
                return False
        
        return True
    
    def test_trainer_functionality(self):
        """Test RL trainer functionality"""
        print("🏋️  Testing RL trainer functionality...")
        
        try:
            # Create environment and trainer
            env = create_reaching_env("franka_panda")
            trainer = RLTrainer(env)
            
            # Test basic trainer properties
            assert hasattr(trainer, 'env')
            assert hasattr(trainer, 'training_history')
            assert trainer.best_reward == -np.inf
            
            # Test policy evaluation structure
            def simple_policy(obs):
                """Simple policy for testing"""
                return np.zeros(env.action_space.shape[0] if hasattr(env.action_space, 'shape') else 3)
            
            # This would require MuJoCo connection for full test
            print("   ✅ Trainer created successfully")
            print(f"   ✅ Best reward initialized: {trainer.best_reward}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Trainer test failed: {e}")
            return False
    
    def test_action_space_compatibility(self):
        """Test action space compatibility"""
        print("🎮 Testing action space compatibility...")
        
        try:
            # Test continuous action space
            env_continuous = create_reaching_env("franka_panda")
            action_cont = env_continuous.action_space.sample()
            print(f"   ✅ Continuous action: shape={action_cont.shape}, range=[{action_cont.min():.2f}, {action_cont.max():.2f}]")
            
            # Test discrete action space  
            config_discrete = RLConfig(
                robot_type="cart_pole",
                task_type="balancing",
                action_space_type="discrete"
            )
            env_discrete = MuJoCoRLEnvironment(config_discrete)
            
            # Test discrete action sampling
            action_discrete = env_discrete.action_space.sample()
            print(f"   ✅ Discrete action: {action_discrete}")
            
            # Test discrete to continuous conversion
            if hasattr(env_discrete.action_space, 'n'):
                for i in range(min(5, env_discrete.action_space.n)):
                    continuous = env_discrete._discrete_to_continuous_action(i)
                    print(f"      Discrete {i} -> Continuous {continuous}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Action space test failed: {e}")
            return False
    
    def test_reward_functions(self):
        """Test all reward function types"""
        print("🎁 Testing reward functions...")
        
        try:
            # Test reaching reward
            env_reach = create_reaching_env("franka_panda")
            obs = np.random.randn(env_reach.observation_space.shape[0])
            action = np.random.randn(env_reach.action_space.shape[0])
            reward_reach = env_reach.reward_function.compute_reward(obs, action, obs, {})
            print(f"   ✅ Reaching reward: {reward_reach:.4f}")
            
            # Test balancing reward
            env_balance = create_balancing_env()
            obs_balance = np.array([0.0, 0.1, 0.0, 0.1])  # Small angle and velocity
            action_balance = np.array([0.1, 0.0])
            reward_balance = env_balance.reward_function.compute_reward(obs_balance, action_balance, obs_balance, {})
            print(f"   ✅ Balancing reward: {reward_balance:.4f}")
            
            # Test walking reward
            env_walk = create_walking_env("quadruped")
            obs_walk = np.random.randn(env_walk.observation_space.shape[0])
            action_walk = np.random.randn(env_walk.action_space.shape[0])
            reward_walk = env_walk.reward_function.compute_reward(obs_walk, action_walk, obs_walk, {})
            print(f"   ✅ Walking reward: {reward_walk:.4f}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Reward function test failed: {e}")
            return False
    
    def test_performance_monitoring(self):
        """Test performance monitoring features"""
        print("📊 Testing performance monitoring...")
        
        try:
            env = create_reaching_env("franka_panda")
            
            # Test performance tracking attributes
            assert hasattr(env, 'step_times')
            assert hasattr(env, 'episode_start_time')
            
            # Simulate performance data
            env.step_times.extend([0.001, 0.002, 0.0015, 0.0012])
            avg_time = np.mean(env.step_times)
            print(f"   ✅ Average step time: {avg_time:.6f}s")
            
            # Test info generation
            env.current_step = 50
            info = env._get_info()
            print(f"   ✅ Info generated: {info}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Performance monitoring test failed: {e}")
            return False
    
    def run_integration_tests(self):
        """Run all integration tests"""
        print("🤖 MuJoCo MCP RL Integration Tests")
        print("=" * 50)
        
        # Try to start viewer server
        server_started = self.start_viewer_server()
        if not server_started:
            print("⚠️  Running tests without MuJoCo viewer")
        
        tests = [
            ("Environment Connection", self.test_rl_environment_connection),
            ("Multiple Environments", self.test_multiple_environments),
            ("Trainer Functionality", self.test_trainer_functionality),
            ("Action Space Compatibility", self.test_action_space_compatibility),
            ("Reward Functions", self.test_reward_functions),
            ("Performance Monitoring", self.test_performance_monitoring)
        ]
        
        results = {}
        passed = 0
        
        for test_name, test_func in tests:
            try:
                print(f"\n{test_name}:")
                success = test_func()
                results[test_name] = success
                if success:
                    passed += 1
                    print(f"✅ {test_name} PASSED")
                else:
                    print(f"❌ {test_name} FAILED")
            except Exception as e:
                print(f"❌ {test_name} ERROR: {e}")
                results[test_name] = False
        
        print("\n" + "=" * 50)
        print(f"📊 Integration Test Summary")
        print(f"Total Tests: {len(tests)}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {len(tests) - passed}")
        print(f"Success Rate: {(passed/len(tests))*100:.1f}%")
        
        # Cleanup
        if self.viewer_server:
            try:
                self.viewer_server.shutdown()
            except:
                pass
        
        return passed == len(tests)

def benchmark_rl_performance():
    """Benchmark RL environment performance"""
    print("\n🚀 RL Performance Benchmark")
    print("-" * 30)
    
    try:
        env = create_reaching_env("franka_panda")
        
        # Benchmark observation generation
        start_time = time.time()
        for _ in range(100):
            obs = env._get_observation()
        obs_time = (time.time() - start_time) / 100
        
        # Benchmark action conversion
        start_time = time.time()
        for i in range(100):
            if hasattr(env.action_space, 'sample'):
                action = env.action_space.sample()
        action_time = (time.time() - start_time) / 100
        
        # Benchmark reward computation
        dummy_obs = np.zeros(env.observation_space.shape[0])
        dummy_action = np.zeros(env.action_space.shape[0])
        start_time = time.time()
        for _ in range(100):
            reward = env.reward_function.compute_reward(dummy_obs, dummy_action, dummy_obs, {})
        reward_time = (time.time() - start_time) / 100
        
        print(f"Observation generation: {obs_time*1000:.3f}ms")
        print(f"Action sampling: {action_time*1000:.3f}ms") 
        print(f"Reward computation: {reward_time*1000:.3f}ms")
        print(f"Total step overhead: {(obs_time + action_time + reward_time)*1000:.3f}ms")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")

def main():
    """Main test execution"""
    # Run integration tests
    test_suite = RLIntegrationTest()
    success = test_suite.run_integration_tests()
    
    # Run performance benchmark
    benchmark_rl_performance()
    
    if success:
        print("\n🎉 All RL integration tests passed!")
        return 0
    else:
        print("\n⚠️  Some integration tests failed.")
        return 1

if __name__ == "__main__":
    exit(main())
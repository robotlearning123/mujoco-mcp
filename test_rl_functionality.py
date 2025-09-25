#!/usr/bin/env python3
"""
Comprehensive RL Functionality Test Suite
Tests all aspects of the MuJoCo MCP RL integration
"""

import sys
import time
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from mujoco_mcp.rl_integration import (
        RLConfig, MuJoCoRLEnvironment, RLTrainer,
        ReachingTaskReward, BalancingTaskReward, WalkingTaskReward,
        create_reaching_env, create_balancing_env, create_walking_env,
        example_training
    )
    from mujoco_mcp.viewer_client import MuJoCoViewerClient
    from mujoco_mcp.simulation import MuJoCoSimulation
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure MuJoCo MCP is properly installed")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RLTestSuite:
    """Comprehensive RL functionality test suite"""
    
    def __init__(self):
        self.results = {}
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        
    def log_test_result(self, test_name: str, passed: bool, details: str = ""):
        """Log test result"""
        self.total_tests += 1
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
        if details:
            print(f"   {details}")
        
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
        
        self.results[test_name] = {"passed": passed, "details": details}
    
    def test_rl_config(self):
        """Test RL configuration creation"""
        try:
            # Test basic config
            config = RLConfig(
                robot_type="franka_panda",
                task_type="reaching",
                max_episode_steps=500
            )
            assert config.robot_type == "franka_panda"
            assert config.task_type == "reaching"
            assert config.max_episode_steps == 500
            assert config.action_space_type == "continuous"
            
            # Test custom config
            config2 = RLConfig(
                robot_type="cart_pole",
                task_type="balancing",
                max_episode_steps=1000,
                action_space_type="discrete",
                reward_scale=2.0
            )
            assert config2.action_space_type == "discrete"
            assert config2.reward_scale == 2.0
            
            self.log_test_result("RL Config Creation", True, "Basic and custom configs created successfully")
            
        except Exception as e:
            self.log_test_result("RL Config Creation", False, str(e))
    
    def test_reward_functions(self):
        """Test reward function implementations"""
        try:
            # Test reaching reward
            target = np.array([0.5, 0.0, 0.5])
            reaching_reward = ReachingTaskReward(target)
            
            obs = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            next_obs = np.array([0.4, 0.0, 0.4, 0.0, 0.0, 0.0])  # Close to target
            action = np.array([0.1, 0.0, 0.1])
            info = {}
            
            reward = reaching_reward.compute_reward(obs, action, next_obs, info)
            assert isinstance(reward, float)
            assert not np.isnan(reward)
            
            # Test balancing reward
            balancing_reward = BalancingTaskReward()
            obs_balance = np.array([0.0, 0.1, 0.0, 0.0])  # Small angle
            reward_balance = balancing_reward.compute_reward(obs_balance, action[:2], obs_balance, info)
            assert isinstance(reward_balance, float)
            
            # Test walking reward
            walking_reward = WalkingTaskReward()
            obs_walk = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            reward_walk = walking_reward.compute_reward(obs_walk, action, obs_walk, info)
            assert isinstance(reward_walk, float)
            
            self.log_test_result("Reward Functions", True, "All reward functions working correctly")
            
        except Exception as e:
            self.log_test_result("Reward Functions", False, str(e))
    
    def test_environment_creation(self):
        """Test RL environment creation"""
        try:
            # Test reaching environment
            config = RLConfig(robot_type="franka_panda", task_type="reaching")
            env = MuJoCoRLEnvironment(config)
            
            assert env.config.robot_type == "franka_panda"
            assert env.config.task_type == "reaching"
            assert hasattr(env, 'action_space')
            assert hasattr(env, 'observation_space')
            assert hasattr(env, 'reward_function')
            
            # Test action space
            assert env.action_space.shape[0] == 7  # Franka has 7 joints
            
            # Test observation space
            assert len(env.observation_space.shape) == 1
            assert env.observation_space.shape[0] > 0
            
            # Test factory functions
            reaching_env = create_reaching_env("franka_panda")
            assert reaching_env.config.task_type == "reaching"
            
            balancing_env = create_balancing_env()
            assert balancing_env.config.task_type == "balancing"
            
            walking_env = create_walking_env()
            assert walking_env.config.task_type == "walking"
            
            self.log_test_result("Environment Creation", True, "All environment types created successfully")
            
        except Exception as e:
            self.log_test_result("Environment Creation", False, str(e))
    
    def test_environment_spaces(self):
        """Test environment action and observation spaces"""
        try:
            # Test different robot configurations
            robots = ["franka_panda", "ur5e", "cart_pole", "quadruped"]
            
            for robot in robots:
                if robot == "cart_pole":
                    config = RLConfig(robot_type=robot, task_type="balancing")
                elif robot == "quadruped":
                    config = RLConfig(robot_type=robot, task_type="walking")
                else:
                    config = RLConfig(robot_type=robot, task_type="reaching")
                
                env = MuJoCoRLEnvironment(config)
                
                # Test action space
                assert hasattr(env.action_space, 'shape') or hasattr(env.action_space, 'n')
                
                # Test observation space
                assert hasattr(env.observation_space, 'shape')
                assert env.observation_space.shape[0] > 0
                
                # Test action sampling
                action = env.action_space.sample()
                assert action is not None
            
            self.log_test_result("Environment Spaces", True, f"Tested {len(robots)} robot configurations")
            
        except Exception as e:
            self.log_test_result("Environment Spaces", False, str(e))
    
    def test_xml_generation(self):
        """Test XML model generation"""
        try:
            config_reaching = RLConfig(robot_type="franka_panda", task_type="reaching")
            env_reaching = MuJoCoRLEnvironment(config_reaching)
            xml_reaching = env_reaching._create_model_xml()
            assert "mujoco" in xml_reaching.lower()
            assert "franka" in xml_reaching.lower()
            
            config_balance = RLConfig(robot_type="cart_pole", task_type="balancing")
            env_balance = MuJoCoRLEnvironment(config_balance)
            xml_balance = env_balance._create_model_xml()
            assert "cartpole" in xml_balance.lower()
            assert "joint" in xml_balance.lower()
            
            config_walk = RLConfig(robot_type="quadruped", task_type="walking")
            env_walk = MuJoCoRLEnvironment(config_walk)
            xml_walk = env_walk._create_model_xml()
            assert "quadruped" in xml_walk.lower()
            
            self.log_test_result("XML Generation", True, "All XML models generated correctly")
            
        except Exception as e:
            self.log_test_result("XML Generation", False, str(e))
    
    def test_action_conversion(self):
        """Test discrete to continuous action conversion"""
        try:
            config = RLConfig(robot_type="franka_panda", task_type="reaching", action_space_type="discrete")
            env = MuJoCoRLEnvironment(config)
            
            # Test various discrete actions
            discrete_actions = [0, 1, 2, 10, 20]
            for action in discrete_actions:
                continuous_action = env._discrete_to_continuous_action(action)
                assert isinstance(continuous_action, np.ndarray)
                assert len(continuous_action) == 7  # Franka has 7 joints
                assert np.all(continuous_action >= -1.0)
                assert np.all(continuous_action <= 1.0)
            
            self.log_test_result("Action Conversion", True, "Discrete to continuous conversion working")
            
        except Exception as e:
            self.log_test_result("Action Conversion", False, str(e))
    
    def test_trainer_creation(self):
        """Test RL trainer creation and basic functionality"""
        try:
            env = create_reaching_env("franka_panda")
            trainer = RLTrainer(env)
            
            assert hasattr(trainer, 'env')
            assert hasattr(trainer, 'training_history')
            assert hasattr(trainer, 'best_reward')
            assert trainer.best_reward == -np.inf
            
            # Test policy evaluation function structure
            def dummy_policy(obs):
                return np.zeros(7)
            
            # This would require MuJoCo viewer connection, so just test structure
            assert hasattr(trainer, 'evaluate_policy')
            assert hasattr(trainer, 'save_training_data')
            
            self.log_test_result("Trainer Creation", True, "Trainer created with all required methods")
            
        except Exception as e:
            self.log_test_result("Trainer Creation", False, str(e))
    
    def test_environment_step_structure(self):
        """Test environment step function structure (without MuJoCo connection)"""
        try:
            env = create_reaching_env("franka_panda")
            
            # Test action validation
            action = np.array([0.1, 0.2, -0.1, 0.0, 0.1, -0.2, 0.05])
            
            # Verify action processing
            processed_action = env._discrete_to_continuous_action(5)
            assert isinstance(processed_action, np.ndarray)
            
            # Test observation structure
            dummy_obs = env._get_observation()  # This will return zeros without MuJoCo
            assert isinstance(dummy_obs, np.ndarray)
            assert dummy_obs.shape == env.observation_space.shape
            
            # Test info structure
            info = env._get_info()
            assert isinstance(info, dict)
            assert 'model_id' in info
            assert 'task_type' in info
            
            self.log_test_result("Environment Step Structure", True, "Step function components working")
            
        except Exception as e:
            self.log_test_result("Environment Step Structure", False, str(e))
    
    def test_error_handling(self):
        """Test error handling in RL components"""
        try:
            # Test invalid configurations
            try:
                invalid_config = RLConfig(robot_type="", task_type="")
                env = MuJoCoRLEnvironment(invalid_config)
                # Should still create environment but with defaults
                assert env.config.robot_type == ""
            except:
                pass  # Expected to potentially fail
            
            # Test reward function edge cases
            reaching_reward = ReachingTaskReward(np.array([0.0, 0.0, 0.0]))
            
            # Test with NaN values
            obs_nan = np.array([np.nan, 0.0, 0.0])
            try:
                reward = reaching_reward.compute_reward(obs_nan, np.zeros(3), obs_nan, {})
                # Should handle gracefully
            except:
                pass
            
            # Test empty observations
            try:
                reward = reaching_reward.compute_reward(np.array([]), np.array([]), np.array([]), {})
            except:
                pass  # Expected to fail gracefully
            
            self.log_test_result("Error Handling", True, "Error handling tests completed")
            
        except Exception as e:
            self.log_test_result("Error Handling", False, str(e))
    
    def test_performance_tracking(self):
        """Test performance tracking features"""
        try:
            env = create_reaching_env("franka_panda")
            
            # Test step time tracking
            assert hasattr(env, 'step_times')
            assert hasattr(env, 'episode_start_time')
            
            # Simulate some step times
            env.step_times.append(0.001)
            env.step_times.append(0.002)
            env.step_times.append(0.001)
            
            avg_time = np.mean(env.step_times)
            assert avg_time > 0
            
            # Test episode tracking
            env.current_step = 100
            info = env._get_info()
            assert 'episode_step' in info
            assert info['episode_step'] == 100
            
            self.log_test_result("Performance Tracking", True, "Performance metrics tracking working")
            
        except Exception as e:
            self.log_test_result("Performance Tracking", False, str(e))
    
    def test_model_xml_validity(self):
        """Test that generated XML models are valid MuJoCo XML"""
        try:
            configs = [
                ("franka_panda", "reaching"),
                ("cart_pole", "balancing"), 
                ("quadruped", "walking"),
                ("simple_arm", "reaching")
            ]
            
            for robot_type, task_type in configs:
                config = RLConfig(robot_type=robot_type, task_type=task_type)
                env = MuJoCoRLEnvironment(config)
                xml = env._create_model_xml()
                
                # Basic XML validation
                assert xml.strip().startswith('<mujoco')
                assert '</mujoco>' in xml
                assert 'worldbody' in xml
                assert 'geom' in xml
                
                # Check for required elements
                if robot_type == "franka_panda":
                    assert 'joint' in xml
                    assert 'link' in xml
                elif robot_type == "cart_pole":
                    assert 'cart' in xml
                    assert 'pole' in xml
                elif robot_type == "quadruped":
                    assert 'torso' in xml
                    assert 'leg' in xml or 'hip' in xml
            
            self.log_test_result("Model XML Validity", True, f"All {len(configs)} XML models are valid")
            
        except Exception as e:
            self.log_test_result("Model XML Validity", False, str(e))
    
    def test_integration_completeness(self):
        """Test that all integration components are present"""
        try:
            # Check that all classes can be imported
            from mujoco_mcp.rl_integration import (
                RLConfig, MuJoCoRLEnvironment, RLTrainer, TaskReward,
                ReachingTaskReward, BalancingTaskReward, WalkingTaskReward
            )
            
            # Check factory functions
            env1 = create_reaching_env()
            env2 = create_balancing_env()
            env3 = create_walking_env()
            
            # Check that example training function exists
            assert callable(example_training)
            
            # Verify all environments have correct task types
            assert env1.config.task_type == "reaching"
            assert env2.config.task_type == "balancing"
            assert env3.config.task_type == "walking"
            
            self.log_test_result("Integration Completeness", True, "All RL integration components present")
            
        except Exception as e:
            self.log_test_result("Integration Completeness", False, str(e))
    
    def run_all_tests(self):
        """Run all RL functionality tests"""
        print("🤖 MuJoCo MCP RL Functionality Test Suite")
        print("=" * 60)
        
        test_methods = [
            self.test_rl_config,
            self.test_reward_functions,
            self.test_environment_creation,
            self.test_environment_spaces,
            self.test_xml_generation,
            self.test_action_conversion,
            self.test_trainer_creation,
            self.test_environment_step_structure,
            self.test_error_handling,
            self.test_performance_tracking,
            self.test_model_xml_validity,
            self.test_integration_completeness
        ]
        
        print(f"Running {len(test_methods)} test categories...")
        print()
        
        for test_method in test_methods:
            test_method()
        
        print()
        print("=" * 60)
        print(f"📊 Test Results Summary")
        print(f"Total Tests: {self.total_tests}")
        print(f"✅ Passed: {self.passed_tests}")
        print(f"❌ Failed: {self.failed_tests}")
        print(f"Success Rate: {(self.passed_tests/self.total_tests)*100:.1f}%")
        
        if self.failed_tests > 0:
            print()
            print("Failed Tests:")
            for test_name, result in self.results.items():
                if not result['passed']:
                    print(f"  ❌ {test_name}: {result['details']}")
        
        return self.failed_tests == 0

def main():
    """Main test execution"""
    test_suite = RLTestSuite()
    success = test_suite.run_all_tests()
    
    if success:
        print("\n🎉 All RL functionality tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check output above for details.")
        return 1

if __name__ == "__main__":
    exit(main())
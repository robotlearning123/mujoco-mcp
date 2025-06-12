#!/usr/bin/env python3
"""
Comprehensive Test Suite for MuJoCo MCP v0.6.2
Tests all MCP tools and generates detailed report
"""

import json
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple
import sys
import os

# Add project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mujoco_mcp.remote_server import MuJoCoRemoteServer

class TestResult:
    def __init__(self, name: str, success: bool, result: Any = None, error: str = None, duration: float = 0):
        self.name = name
        self.success = success
        self.result = result
        self.error = error
        self.duration = duration

class ComprehensiveTest:
    def __init__(self):
        self.server = MuJoCoRemoteServer()
        self.results: List[TestResult] = []
        self.start_time = time.time()
        
    def run_test(self, name: str, test_func) -> TestResult:
        """Run a single test and capture results"""
        print(f"\n🧪 Testing: {name}")
        start = time.time()
        try:
            result = test_func()
            duration = time.time() - start
            print(f"✅ {name}: Success ({duration:.2f}s)")
            return TestResult(name, True, result, duration=duration)
        except Exception as e:
            duration = time.time() - start
            error = str(e)
            print(f"❌ {name}: Failed - {error} ({duration:.2f}s)")
            return TestResult(name, False, error=error, duration=duration)
    
    def test_get_server_info(self):
        """Test getting server information"""
        info = self.server._handle_get_server_info()
        assert info["version"] == "0.6.2", f"Expected version 0.6.2, got {info['version']}"
        assert info["mode"] == "remote_viewer", f"Expected mode remote_viewer, got {info['mode']}"
        return info
    
    def test_create_pendulum(self):
        """Test creating pendulum scene"""
        result = self.server._handle_create_scene("pendulum")
        assert "model_id" in result, "No model_id in result"
        assert result["status"] == "created", f"Expected status 'created', got {result['status']}"
        return result
    
    def test_create_double_pendulum(self):
        """Test creating double pendulum scene"""
        result = self.server._handle_create_scene("double_pendulum")
        assert "model_id" in result, "No model_id in result"
        assert result["status"] == "created", f"Expected status 'created', got {result['status']}"
        return result
    
    def test_create_cart_pole(self):
        """Test creating cart pole scene"""
        result = self.server._handle_create_scene("cart_pole")
        assert "model_id" in result, "No model_id in result"
        assert result["status"] == "created", f"Expected status 'created', got {result['status']}"
        return result
    
    def test_create_robotic_arm(self):
        """Test creating robotic arm scene"""
        result = self.server._handle_create_scene("robotic_arm")
        assert "model_id" in result, "No model_id in result"
        assert result["status"] == "created", f"Expected status 'created', got {result['status']}"
        return result
    
    def test_get_state(self, model_id: str = None):
        """Test getting simulation state"""
        state = self.server._handle_get_state(model_id)
        assert "time" in state, "No time in state"
        assert "positions" in state, "No positions in state"
        return state
    
    def test_set_joint_positions(self, model_id: str = None):
        """Test setting joint positions"""
        positions = [1.57]  # 90 degrees
        result = self.server._handle_set_joint_positions(positions, model_id)
        assert result["success"], "Failed to set joint positions"
        return result
    
    def test_step_simulation(self, model_id: str = None):
        """Test stepping simulation"""
        steps = 50
        result = self.server._handle_step_simulation(steps, model_id)
        assert "time" in result, "No time in result after stepping"
        return result
    
    def test_reset_simulation(self, model_id: str = None):
        """Test resetting simulation"""
        result = self.server._handle_reset_simulation(model_id)
        assert result["success"], "Failed to reset simulation"
        return result
    
    def test_get_loaded_models(self):
        """Test getting loaded models"""
        models = self.server._handle_get_loaded_models()
        assert isinstance(models, dict), "Models should be a dictionary"
        return models
    
    def test_execute_command(self, command: str, model_id: str = None):
        """Test executing natural language command"""
        context = {"model_id": model_id} if model_id else None
        result = self.server._handle_execute_command(command, context)
        return result
    
    def test_scene_lifecycle(self, scene_type: str):
        """Test complete lifecycle of a scene"""
        print(f"\n📋 Testing {scene_type} lifecycle...")
        
        # Create scene
        create_result = self.run_test(
            f"Create {scene_type}", 
            lambda: self.server._handle_create_scene(scene_type)
        )
        
        if not create_result.success:
            return create_result
        
        model_id = create_result.result.get("model_id")
        
        # Test operations on the scene
        time.sleep(0.5)  # Give time for scene to initialize
        
        # Get initial state
        self.run_test(
            f"Get {scene_type} state",
            lambda: self.test_get_state(model_id)
        )
        
        # Set joint positions
        self.run_test(
            f"Set {scene_type} joint positions",
            lambda: self.test_set_joint_positions(model_id)
        )
        
        # Step simulation
        self.run_test(
            f"Step {scene_type} simulation",
            lambda: self.test_step_simulation(model_id)
        )
        
        # Reset simulation
        self.run_test(
            f"Reset {scene_type} simulation",
            lambda: self.test_reset_simulation(model_id)
        )
        
        return create_result
    
    def test_natural_language_commands(self):
        """Test various natural language commands"""
        commands = [
            "show current state",
            "create cart pole simulation",
            "swing the pendulum",
            "set position to 45 degrees",
            "reset the simulation",
            "step forward 100 times",
            "show energy",
            "list all models"
        ]
        
        print("\n🗣️ Testing Natural Language Commands...")
        for cmd in commands:
            self.run_test(
                f"Command: '{cmd}'",
                lambda c=cmd: self.test_execute_command(c)
            )
    
    def test_connection_recovery(self):
        """Test connection recovery scenarios"""
        print("\n🔌 Testing Connection Recovery...")
        
        # Test creating multiple scenes rapidly
        for i in range(3):
            self.run_test(
                f"Rapid scene creation {i+1}",
                lambda: self.server._handle_create_scene("pendulum")
            )
            time.sleep(0.1)
    
    def run_all_tests(self):
        """Run all tests and generate report"""
        print("🚀 Starting MuJoCo MCP v0.6.2 Comprehensive Test Suite")
        print("=" * 60)
        
        # 1. Basic server info
        info_result = self.run_test("Get Server Info", self.test_get_server_info)
        self.results.append(info_result)
        
        # 2. Test each scene type individually
        scene_types = ["pendulum", "double_pendulum", "cart_pole", "robotic_arm"]
        for scene_type in scene_types:
            result = self.test_scene_lifecycle(scene_type)
            # Results are already added in test_scene_lifecycle
            time.sleep(1)  # Give time between scenes
        
        # 3. Test loaded models
        models_result = self.run_test("Get Loaded Models", self.test_get_loaded_models)
        self.results.append(models_result)
        
        # 4. Test natural language commands
        self.test_natural_language_commands()
        
        # 5. Test connection recovery
        self.test_connection_recovery()
        
        # Generate report
        self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive test report"""
        total_time = time.time() - self.start_time
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        
        report = f"""
# MuJoCo MCP v0.6.2 Comprehensive Test Report

**Test Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Duration**: {total_time:.2f} seconds
**Tests Run**: {len(self.results)}
**Successful**: {len(successful)} ({len(successful)/len(self.results)*100:.1f}%)
**Failed**: {len(failed)} ({len(failed)/len(self.results)*100:.1f}%)

## Test Results Summary

### ✅ Successful Tests ({len(successful)})
"""
        
        for test in successful:
            report += f"- **{test.name}** ({test.duration:.2f}s)\n"
        
        report += f"\n### ❌ Failed Tests ({len(failed)})\n"
        for test in failed:
            report += f"- **{test.name}**: {test.error} ({test.duration:.2f}s)\n"
        
        # Categorize failures
        connection_failures = [t for t in failed if "connect" in t.error.lower()]
        timeout_failures = [t for t in failed if "timeout" in t.error.lower()]
        command_failures = [t for t in failed if "command" in t.name.lower()]
        
        report += f"""
## Failure Analysis

### Connection Issues: {len(connection_failures)}
### Timeout Issues: {len(timeout_failures)}
### Command Recognition Issues: {len(command_failures)}
### Other Issues: {len(failed) - len(connection_failures) - len(timeout_failures) - len(command_failures)}

## Key Findings

1. **Server Status**: {"✅ Running" if any(t.name == "Get Server Info" and t.success for t in self.results) else "❌ Not responding"}
2. **Scene Creation Success Rate**: {self._calculate_scene_success_rate():.1f}%
3. **Command Recognition Rate**: {self._calculate_command_success_rate():.1f}%
4. **Average Response Time**: {self._calculate_avg_response_time():.2f}s

## Recommendations

"""
        
        # Add recommendations based on failures
        if len(connection_failures) > 2:
            report += "- **Critical**: Multiple connection failures detected. Check viewer server status.\n"
        if len(timeout_failures) > 0:
            report += "- **Important**: Timeout issues present. Consider increasing timeout values.\n"
        if len(command_failures) > 3:
            report += "- **Enhancement**: Natural language command recognition needs improvement.\n"
        
        # Save report
        report_file = f"test_report_v062_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\n📄 Report saved to: {report_file}")
        print("\n" + "=" * 60)
        print(f"Test Summary: {len(successful)}/{len(self.results)} passed ({len(successful)/len(self.results)*100:.1f}%)")
    
    def _calculate_scene_success_rate(self):
        scene_tests = [t for t in self.results if "Create" in t.name and any(s in t.name for s in ["pendulum", "cart", "arm"])]
        if not scene_tests:
            return 0
        successful_scenes = [t for t in scene_tests if t.success]
        return len(successful_scenes) / len(scene_tests) * 100
    
    def _calculate_command_success_rate(self):
        command_tests = [t for t in self.results if "Command:" in t.name]
        if not command_tests:
            return 0
        successful_commands = [t for t in command_tests if t.success]
        return len(successful_commands) / len(command_tests) * 100
    
    def _calculate_avg_response_time(self):
        if not self.results:
            return 0
        return sum(t.duration for t in self.results) / len(self.results)


def main():
    """Main test execution"""
    tester = ComprehensiveTest()
    
    try:
        tester.run_all_tests()
    except KeyboardInterrupt:
        print("\n\n⚠️ Tests interrupted by user")
        tester.generate_report()
    except Exception as e:
        print(f"\n\n❌ Test suite failed: {e}")
        tester.generate_report()


if __name__ == "__main__":
    main()
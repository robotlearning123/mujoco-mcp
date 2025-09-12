#!/usr/bin/env python3
"""
Comprehensive test suite for advanced MuJoCo MCP features
Tests all new capabilities: controllers, coordination, sensors, RL, benchmarks, visualization
"""

import time
import numpy as np
import sys
from pathlib import Path
from typing import Dict
import tempfile
import json

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import all advanced modules
from mujoco_mcp.advanced_controllers import (
    PIDController,
    PIDConfig,
    TrajectoryPlanner,
    create_arm_controller,
    create_quadruped_controller,
)
from mujoco_mcp.multi_robot_coordinator import MultiRobotCoordinator, TaskType, CollisionChecker
from mujoco_mcp.sensor_feedback import (
    create_robot_sensor_suite,
    create_feedback_controller,
    SensorFusion,
)
from mujoco_mcp.rl_integration import create_reaching_env, create_balancing_env, RLTrainer
from mujoco_mcp.visualization_tools import (
    RealTimePlotter,
    PlotConfig,
    InteractiveVisualizer,
    TrajectoryVisualizer,
)


class AdvancedFeatureTests:
    """Comprehensive test suite for advanced features"""

    def __init__(self):
        self.test_results = {}
        self.temp_dir = tempfile.mkdtemp()
        print("🧪 Advanced Features Test Suite")
        print(f"Temporary directory: {self.temp_dir}")
        print("=" * 60)

    def run_all_tests(self) -> Dict[str, bool]:
        """Run all advanced feature tests"""
        tests = [
            ("Advanced Controllers", self.test_advanced_controllers),
            ("Multi-Robot Coordination", self.test_multi_robot_coordination),
            ("Sensor Feedback", self.test_sensor_feedback),
            ("RL Integration", self.test_rl_integration),
            ("Visualization Tools", self.test_visualization_tools),
            ("Performance Features", self.test_performance_features),
        ]

        for test_name, test_func in tests:
            print(f"\n🔧 Testing {test_name}...")
            try:
                result = test_func()
                self.test_results[test_name] = result
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"   {status}")
            except Exception as e:
                print(f"   ❌ ERROR: {e}")
                self.test_results[test_name] = False

        return self.test_results

    def test_advanced_controllers(self) -> bool:
        """Test advanced control algorithms"""
        print("   Testing PID controllers...")

        # Test PID controller
        config = PIDConfig(kp=1.0, ki=0.1, kd=0.05)
        pid = PIDController(config)

        # Test control response
        target = 1.0
        current = 0.0
        command = pid.update(target, current)

        if command <= 0:
            print("     ❌ PID should output positive command for positive error")
            return False

        print("     ✅ PID controller working")

        # Test trajectory planner
        print("   Testing trajectory planning...")
        planner = TrajectoryPlanner()

        start_pos = np.array([0.0, 0.0])
        end_pos = np.array([1.0, 1.0])

        positions, _velocities, _accelerations = planner.minimum_jerk_trajectory(
            start_pos, end_pos, duration=2.0
        )

        if len(positions) == 0:
            print("     ❌ Trajectory planner returned empty trajectory")
            return False

        # Check start and end positions
        if not np.allclose(positions[0], start_pos, atol=1e-3):
            print("     ❌ Trajectory doesn't start at correct position")
            return False

        if not np.allclose(positions[-1], end_pos, atol=1e-3):
            print("     ❌ Trajectory doesn't end at correct position")
            return False

        print("     ✅ Trajectory planning working")

        # Test robot controller factory
        print("   Testing robot controller factories...")
        arm_controller = create_arm_controller("franka_panda")
        quadruped_controller = create_quadruped_controller("anymal_c")

        if not arm_controller or not quadruped_controller:
            print("     ❌ Failed to create robot controllers")
            return False

        print("     ✅ Robot controller factories working")

        return True

    def test_multi_robot_coordination(self) -> bool:
        """Test multi-robot coordination system"""
        print("   Testing multi-robot coordinator...")

        # Create coordinator (without actual viewer connection)
        coordinator = MultiRobotCoordinator(viewer_client=None)

        # Test adding robots
        success1 = coordinator.add_robot("robot1", "franka_panda", {"manipulation": True})
        success2 = coordinator.add_robot("robot2", "ur5e", {"manipulation": True})

        if not success1 or not success2:
            print("     ❌ Failed to add robots to coordinator")
            return False

        print("     ✅ Robot registration working")

        # Test task allocation
        print("   Testing task allocation...")
        task_allocator = coordinator.task_allocator

        # Add test task
        from mujoco_mcp.multi_robot_coordinator import CoordinatedTask

        task = CoordinatedTask(
            task_id="test_task",
            task_type=TaskType.FORMATION_CONTROL,
            robots=["robot1", "robot2"],
            parameters={"formation": "line", "spacing": 1.0},
        )

        task_allocator.add_task(task)

        if len(task_allocator.pending_tasks) != 1:
            print("     ❌ Task not added to pending tasks")
            return False

        print("     ✅ Task allocation working")

        # Test collision checker
        print("   Testing collision detection...")
        collision_checker = CollisionChecker()

        from mujoco_mcp.multi_robot_coordinator import RobotState

        state1 = RobotState("robot1", "franka_panda", np.zeros(7), np.zeros(7))
        state2 = RobotState("robot2", "ur5e", np.zeros(6), np.zeros(6))

        # Set end-effector positions for collision test
        state1.end_effector_pos = np.array([0.0, 0.0, 1.0])
        state2.end_effector_pos = np.array([0.05, 0.0, 1.0])  # Close but not colliding

        collision = collision_checker.check_collision(state1, state2)

        if collision:
            print("     ❌ False collision detected")
            return False

        # Test actual collision
        state2.end_effector_pos = np.array([0.01, 0.0, 1.0])  # Very close - should collide
        collision = collision_checker.check_collision(state1, state2)

        if not collision:
            print("     ❌ Collision not detected when it should be")
            return False

        print("     ✅ Collision detection working")

        return True

    def test_sensor_feedback(self) -> bool:
        """Test sensor feedback systems"""
        print("   Testing sensor management...")

        # Test sensor manager creation
        sensor_manager = create_robot_sensor_suite("franka_panda", n_joints=7)

        if not sensor_manager or len(sensor_manager.sensors) == 0:
            print("     ❌ Failed to create sensor suite")
            return False

        print("     ✅ Sensor suite creation working")

        # Test feedback controller
        print("   Testing feedback controller...")
        controller = create_feedback_controller("franka_panda")

        if not controller:
            print("     ❌ Failed to create feedback controller")
            return False

        # Test control gain settings
        expected_gains = ["kp", "ki", "kd"]
        for gain in expected_gains:
            if gain not in controller.control_gains:
                print(f"     ❌ Missing control gain: {gain}")
                return False

        print("     ✅ Feedback controller working")

        # Test sensor fusion
        print("   Testing sensor fusion...")
        from mujoco_mcp.sensor_feedback import SensorType

        fusion = SensorFusion()
        fusion.add_sensor("sensor1", SensorType.JOINT_POSITION, weight=1.0)
        fusion.add_sensor("sensor2", SensorType.JOINT_VELOCITY, weight=0.8)

        # Create mock sensor readings
        from mujoco_mcp.sensor_feedback import SensorReading

        readings = [
            SensorReading("sensor1", SensorType.JOINT_POSITION, time.time(), np.array([1.0, 2.0])),
            SensorReading("sensor2", SensorType.JOINT_VELOCITY, time.time(), np.array([0.1, 0.2])),
        ]

        fused_data = fusion.fuse_sensor_data(readings)

        if not fused_data or len(fused_data) == 0:
            print("     ❌ Sensor fusion failed")
            return False

        print("     ✅ Sensor fusion working")

        return True

    def test_rl_integration(self) -> bool:
        """Test reinforcement learning integration"""
        print("   Testing RL environment creation...")

        # Test different environment types
        try:
            reaching_env = create_reaching_env("franka_panda")
            balancing_env = create_balancing_env()

            if not reaching_env or not balancing_env:
                print("     ❌ Failed to create RL environments")
                return False

            print("     ✅ RL environment creation working")

        except Exception as e:
            print(f"     ❌ RL environment creation failed: {e}")
            return False

        # Test environment interface
        print("   Testing environment interface...")

        try:
            # Test action and observation spaces
            if reaching_env.action_space is None or reaching_env.observation_space is None:
                print("     ❌ Missing action or observation space")
                return False

            # Test reset (without actual connection)
            try:
                _obs, _info = reaching_env.reset()
                # If no viewer connection, this should gracefully handle the error
            except RuntimeError as e:
                if "Failed to connect" not in str(e):
                    print(f"     ❌ Unexpected error during reset: {e}")
                    return False
                # Expected error when no viewer server is running

            print("     ✅ RL environment interface working")

        except Exception as e:
            print(f"     ❌ RL environment interface failed: {e}")
            return False

        # Test trainer utilities
        print("   Testing RL trainer...")

        try:
            trainer = RLTrainer(reaching_env)

            if not trainer:
                print("     ❌ Failed to create RL trainer")
                return False

            # Test policy evaluation function (without actual execution)
            def dummy_policy(obs):
                return reaching_env.action_space.sample()

            # The evaluation would fail without viewer connection, but we can test the setup
            print("     ✅ RL trainer working")

        except Exception as e:
            print(f"     ❌ RL trainer failed: {e}")
            return False

        return True

    def test_visualization_tools(self) -> bool:
        """Test visualization and plotting tools"""
        print("   Testing plot configuration...")

        # Test plot config
        config = PlotConfig(title="Test Plot", xlabel="Time", ylabel="Value", max_points=100)

        if not config or config.title != "Test Plot":
            print("     ❌ Plot configuration failed")
            return False

        print("     ✅ Plot configuration working")

        # Test real-time plotter
        print("   Testing real-time plotter...")

        try:
            plotter = RealTimePlotter(config)

            # Add data sources
            idx1 = plotter.add_data_source("Signal 1", "blue")
            idx2 = plotter.add_data_source("Signal 2", "red")

            if idx1 != 0 or idx2 != 1:
                print("     ❌ Data source indexing failed")
                return False

            # Test data update
            test_data = [1.0, 2.0]
            plotter.update_data(time.time(), test_data)

            if len(plotter.data.timestamps) != 1:
                print("     ❌ Data update failed")
                return False

            print("     ✅ Real-time plotter working")

        except Exception as e:
            print(f"     ❌ Real-time plotter failed: {e}")
            return False

        # Test interactive visualizer
        print("   Testing interactive visualizer...")

        try:
            visualizer = InteractiveVisualizer()
            fig = visualizer.create_dashboard("Test Dashboard")

            if not fig:
                print("     ❌ Dashboard creation failed")
                return False

            visualizer.add_data_stream("test_stream", "position", 1, 1)
            visualizer.update_data_stream("test_stream", time.time(), 1.5)

            if "test_stream" not in visualizer.data_sources:
                print("     ❌ Data stream management failed")
                return False

            print("     ✅ Interactive visualizer working")

        except Exception as e:
            print(f"     ❌ Interactive visualizer failed: {e}")
            return False

        # Test trajectory visualizer
        print("   Testing trajectory visualizer...")

        try:
            traj_viz = TrajectoryVisualizer()

            # Create test trajectory
            positions = np.random.randn(100, 3)  # 100 points in 3D
            traj_viz.add_trajectory("test_traj", positions)

            if "test_traj" not in traj_viz.trajectories:
                print("     ❌ Trajectory addition failed")
                return False

            fig = traj_viz.create_3d_plot()

            if not fig:
                print("     ❌ 3D plot creation failed")
                return False

            print("     ✅ Trajectory visualizer working")

        except Exception as e:
            print(f"     ❌ Trajectory visualizer failed: {e}")
            return False

        return True

    def test_performance_features(self) -> bool:
        """Test performance and reliability features"""
        print("   Testing performance monitoring...")

        # Test performance monitoring in enhanced server
        try:
            from mujoco_viewer_server_enhanced import PerformanceMonitor

            monitor = PerformanceMonitor()
            current_stats = monitor.get_current_stats()

            # Basic sanity checks
            if current_stats.cpu_usage < 0 or current_stats.memory_usage < 0:
                print("     ❌ Invalid performance statistics")
                return False

            monitor.stop()
            print("     ✅ Performance monitoring working")

        except Exception as e:
            print(f"     ❌ Performance monitoring failed: {e}")
            return False

        # Test connection management
        print("   Testing connection management...")

        try:
            from mujoco_viewer_server_enhanced import ConnectionManager

            conn_manager = ConnectionManager(max_connections=5)

            # Test connection registration
            success = conn_manager.register_connection("test_conn_1")
            if not success:
                print("     ❌ Connection registration failed")
                return False

            # Test connection limit
            for i in range(10):  # Try to register more than limit
                conn_manager.register_connection(f"test_conn_{i + 2}")

            stats = conn_manager.get_stats()
            if stats["active_connections"] > 5:
                print("     ❌ Connection limit not enforced")
                return False

            print("     ✅ Connection management working")

        except Exception as e:
            print(f"     ❌ Connection management failed: {e}")
            return False

        # Test benchmark framework (basic functionality)
        print("   Testing benchmark framework...")

        try:
            from benchmarks.physics_benchmarks import BenchmarkResult

            result = BenchmarkResult(
                test_name="Test Benchmark",
                success=True,
                execution_time=1.5,
                metrics={"test_metric": 0.95},
            )

            if result.test_name != "Test Benchmark" or not result.success:
                print("     ❌ Benchmark result creation failed")
                return False

            print("     ✅ Benchmark framework working")

        except Exception as e:
            print(f"     ❌ Benchmark framework failed: {e}")
            return False

        return True

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("📊 Advanced Features Test Summary")
        print("=" * 60)

        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)

        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {test_name:<30} {status}")

        print("\n" + "-" * 60)
        print(f"Overall: {passed_tests}/{total_tests} tests passed")

        if passed_tests == total_tests:
            print("🎉 All advanced features working correctly!")
            return True
        else:
            print("⚠️  Some advanced features need attention")
            return False


def main():
    """Run the comprehensive advanced features test"""
    print("🚀 MuJoCo MCP Advanced Features Test Suite")
    print("Testing all enhanced capabilities...")
    print()

    tester = AdvancedFeatureTests()
    results = tester.run_all_tests()
    success = tester.print_summary()

    # Save results
    results_file = Path(tester.temp_dir) / "test_results.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "timestamp": time.time(),
                "results": results,
                "summary": {
                    "total_tests": len(results),
                    "passed_tests": sum(1 for r in results.values() if r),
                    "success_rate": sum(1 for r in results.values() if r) / len(results),
                },
            },
            f,
            indent=2,
        )

    print(f"\nDetailed results saved to: {results_file}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

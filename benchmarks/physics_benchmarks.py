#!/usr/bin/env python3
"""
Physics Simulation Benchmarks for MuJoCo MCP
Comprehensive benchmarking suite for performance, accuracy, and stability testing
"""

import time
import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mujoco_mcp.viewer_client import MuJoCoViewerClient


@dataclass
class BenchmarkResult:
    """Result of a benchmark test"""
    test_name: str
    success: bool
    execution_time: float
    metrics: Dict[str, float] = field(default_factory=dict)
    error_message: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics for benchmarking"""
    fps: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    physics_time: float = 0.0
    rendering_time: float = 0.0
    communication_latency: float = 0.0


class PhysicsBenchmark:
    """Base class for physics benchmarks"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.viewer_client = MuJoCoViewerClient()
        self.results = []

    def setup(self) -> bool:
        """Setup benchmark environment"""
        return self.viewer_client.connect()

    def teardown(self):
        """Cleanup benchmark environment"""
        if self.viewer_client.connected:
            self.viewer_client.disconnect()

    def run(self) -> BenchmarkResult:
        """Run the benchmark"""
        start_time = time.time()

        try:
            if not self.setup():
                return BenchmarkResult(
                    test_name=self.name,
                    success=False,
                    execution_time=0.0,
                    error_message="Setup failed"
                )

            result = self._execute_benchmark()
            result.execution_time = time.time() - start_time

        except Exception as e:
            result = BenchmarkResult(
                test_name=self.name,
                success=False,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
        finally:
            self.teardown()

        self.results.append(result)
        return result

    def _execute_benchmark(self) -> BenchmarkResult:
        """Execute the specific benchmark - to be overridden"""
        raise NotImplementedError


class SimulationStabilityBenchmark(PhysicsBenchmark):
    """Test simulation stability under various conditions"""

    def __init__(self):
        super().__init__(
            "Simulation Stability",
            "Tests simulation stability with different models and physics parameters"
        )

    def _execute_benchmark(self) -> BenchmarkResult:
        """Execute stability benchmark"""

        # Test different models
        models_to_test = [
            ("pendulum", self._create_pendulum_xml()),
            ("double_pendulum", self._create_double_pendulum_xml()),
            ("cart_pole", self._create_cart_pole_xml())
        ]

        stability_scores = []

        for model_name, model_xml in models_to_test:
            # Load model
            response = self.viewer_client.send_command({
                "type": "load_model",
                "model_id": model_name,
                "model_xml": model_xml
            })

            if not response.get("success"):
                continue

            # Run simulation for extended period
            start_energy = None
            energy_values = []

            for _step in range(1000):  # 20 seconds at 50Hz
                state_response = self.viewer_client.send_command({
                    "type": "get_state",
                    "model_id": model_name
                })

                if state_response.get("success"):
                    state = state_response.get("state", {})
                    positions = np.array(state.get("qpos", []))
                    velocities = np.array(state.get("qvel", []))

                    # Calculate total energy (simplified)
                    kinetic_energy = 0.5 * np.sum(velocities**2)
                    potential_energy = 9.81 * np.sum(positions)  # Simplified
                    total_energy = kinetic_energy + potential_energy

                    energy_values.append(total_energy)

                    if start_energy is None:
                        start_energy = total_energy

                time.sleep(0.02)  # 50Hz

            # Calculate stability score based on energy conservation
            if energy_values and start_energy is not None:
                energy_drift = abs(energy_values[-1] - start_energy) / abs(start_energy + 1e-6)
                stability_score = max(0, 1.0 - energy_drift)
                stability_scores.append(stability_score)

        # Overall stability score
        avg_stability = np.mean(stability_scores) if stability_scores else 0.0

        return BenchmarkResult(
            test_name=self.name,
            success=len(stability_scores) > 0,
            execution_time=0.0,
            metrics={
                "stability_score": avg_stability,
                "models_tested": len(stability_scores),
                "energy_conservation": min(stability_scores) if stability_scores else 0.0
            }
        )

    def _create_pendulum_xml(self) -> str:
        """Create pendulum model XML"""
        return """
        <mujoco>
            <worldbody>
                <body name="pole" pos="0 0 1">
                    <joint name="hinge" type="hinge" axis="1 0 0"/>
                    <geom name="pole" type="capsule" size="0.02 0.6" rgba="0.8 0.2 0.2 1"/>
                    <body name="mass" pos="0 0 -0.6">
                        <geom name="mass" type="sphere" size="0.05" rgba="0.2 0.8 0.2 1"/>
                    </body>
                </body>
            </worldbody>
        </mujoco>
        """

    def _create_double_pendulum_xml(self) -> str:
        """Create double pendulum model XML"""
        return """
        <mujoco>
            <worldbody>
                <body name="pole1" pos="0 0 1">
                    <joint name="hinge1" type="hinge" axis="1 0 0"/>
                    <geom name="pole1" type="capsule" size="0.02 0.4" rgba="0.8 0.2 0.2 1"/>
                    <body name="pole2" pos="0 0 -0.4">
                        <joint name="hinge2" type="hinge" axis="1 0 0"/>
                        <geom name="pole2" type="capsule" size="0.02 0.4" rgba="0.2 0.8 0.2 1"/>
                        <body name="mass" pos="0 0 -0.4">
                            <geom name="mass" type="sphere" size="0.05" rgba="0.2 0.2 0.8 1"/>
                        </body>
                    </body>
                </body>
            </worldbody>
        </mujoco>
        """

    def _create_cart_pole_xml(self) -> str:
        """Create cart-pole model XML"""
        return """
        <mujoco>
            <worldbody>
                <body name="cart" pos="0 0 0.1">
                    <joint name="slider" type="slide" axis="1 0 0"/>
                    <geom name="cart" type="box" size="0.1 0.1 0.1" rgba="0.8 0.2 0.2 1"/>
                    <body name="pole" pos="0 0 0.1">
                        <joint name="hinge" type="hinge" axis="0 1 0"/>
                        <geom name="pole" type="capsule" size="0.02 0.5" rgba="0.2 0.8 0.2 1"/>
                    </body>
                </body>
            </worldbody>
        </mujoco>
        """


class PerformanceBenchmark(PhysicsBenchmark):
    """Test simulation performance and throughput"""

    def __init__(self):
        super().__init__(
            "Performance",
            "Tests simulation performance, FPS, and resource usage"
        )
        self.performance_monitor = PerformanceMonitor()

    def _execute_benchmark(self) -> BenchmarkResult:
        """Execute performance benchmark"""

        # Load a complex model for performance testing
        complex_model_xml = self._create_complex_model_xml()

        response = self.viewer_client.send_command({
            "type": "load_model",
            "model_id": "performance_test",
            "model_xml": complex_model_xml
        })

        if not response.get("success"):
            return BenchmarkResult(
                test_name=self.name,
                success=False,
                execution_time=0.0,
                error_message="Failed to load test model"
            )

        # Start performance monitoring
        self.performance_monitor.start()

        # Run simulation for performance measurement
        num_steps = 1000
        step_times = []

        for step in range(num_steps):
            step_start = time.time()

            # Simulate one step
            state_response = self.viewer_client.send_command({
                "type": "get_state",
                "model_id": "performance_test"
            })

            step_end = time.time()
            step_times.append(step_end - step_start)

            if step % 100 == 0:  # Sample performance every 100 steps
                self.performance_monitor.sample()

        # Stop monitoring
        metrics = self.performance_monitor.stop()

        # Calculate performance metrics
        avg_step_time = np.mean(step_times)
        max_step_time = np.max(step_times)
        fps = 1.0 / avg_step_time if avg_step_time > 0 else 0.0

        return BenchmarkResult(
            test_name=self.name,
            success=True,
            execution_time=0.0,
            metrics={
                "fps": fps,
                "avg_step_time": avg_step_time * 1000,  # ms
                "max_step_time": max_step_time * 1000,  # ms
                "cpu_usage": metrics.cpu_usage,
                "memory_usage": metrics.memory_usage,
                "steps_completed": num_steps
            }
        )

    def _create_complex_model_xml(self) -> str:
        """Create complex model for performance testing"""
        return """
        <mujoco>
            <worldbody>
                <body name="base" pos="0 0 0.5">
                    <geom name="base" type="box" size="0.2 0.2 0.1" rgba="0.5 0.5 0.5 1"/>
                    <body name="arm1" pos="0 0 0.1">
                        <joint name="joint1" type="hinge" axis="0 0 1"/>
                        <geom name="link1" type="capsule" size="0.05 0.3" rgba="0.8 0.2 0.2 1"/>
                        <body name="arm2" pos="0 0 0.3">
                            <joint name="joint2" type="hinge" axis="1 0 0"/>
                            <geom name="link2" type="capsule" size="0.04 0.25"
                                  rgba="0.2 0.8 0.2 1"/>
                            <body name="arm3" pos="0 0 0.25">
                                <joint name="joint3" type="hinge" axis="0 1 0"/>
                                <geom name="link3" type="capsule" size="0.03 0.2"
                                      rgba="0.2 0.2 0.8 1"/>
                                <body name="end_effector" pos="0 0 0.2">
                                    <joint name="joint4" type="hinge" axis="0 0 1"/>
                                    <geom name="gripper" type="box" size="0.05 0.05 0.05"
                                          rgba="0.8 0.8 0.2 1"/>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </worldbody>
        </mujoco>
        """


class AccuracyBenchmark(PhysicsBenchmark):
    """Test simulation accuracy against known analytical solutions"""

    def __init__(self):
        super().__init__(
            "Accuracy",
            "Tests simulation accuracy against analytical solutions"
        )

    def _execute_benchmark(self) -> BenchmarkResult:
        """Execute accuracy benchmark"""

        # Test simple pendulum against analytical solution
        pendulum_xml = self._create_simple_pendulum_xml()

        response = self.viewer_client.send_command({
            "type": "load_model",
            "model_id": "accuracy_test",
            "model_xml": pendulum_xml
        })

        if not response.get("success"):
            return BenchmarkResult(
                test_name=self.name,
                success=False,
                execution_time=0.0,
                error_message="Failed to load test model"
            )

        # Set initial conditions
        initial_angle = 0.1  # Small angle for linear approximation
        self.viewer_client.send_command({
            "type": "set_joint_positions",
            "model_id": "accuracy_test",
            "positions": [initial_angle]
        })

        # Simulate and compare with analytical solution
        dt = 0.02  # 50Hz
        duration = 2.0  # 2 seconds
        num_steps = int(duration / dt)

        # Analytical solution parameters
        g = 9.81  # gravity
        L = 1.0   # pendulum length
        omega = np.sqrt(g / L)  # natural frequency

        simulation_angles = []
        analytical_angles = []
        times = []

        for step in range(num_steps):
            t = step * dt
            times.append(t)

            # Get simulation state
            state_response = self.viewer_client.send_command({
                "type": "get_state",
                "model_id": "accuracy_test"
            })

            if state_response.get("success"):
                state = state_response.get("state", {})
                positions = state.get("qpos", [])
                if positions:
                    simulation_angles.append(positions[0])
                else:
                    simulation_angles.append(0.0)
            else:
                simulation_angles.append(0.0)

            # Analytical solution (small angle approximation)
            analytical_angle = initial_angle * np.cos(omega * t)
            analytical_angles.append(analytical_angle)

        # Calculate accuracy metrics
        if simulation_angles and analytical_angles:
            simulation_angles = np.array(simulation_angles)
            analytical_angles = np.array(analytical_angles)

            # Mean absolute error
            mae = np.mean(np.abs(simulation_angles - analytical_angles))

            # Root mean square error
            rmse = np.sqrt(np.mean((simulation_angles - analytical_angles)**2))

            # Relative error
            relative_error = mae / (np.max(np.abs(analytical_angles)) + 1e-6)

            # Accuracy score (higher is better)
            accuracy_score = max(0, 1.0 - relative_error)

            return BenchmarkResult(
                test_name=self.name,
                success=True,
                execution_time=0.0,
                metrics={
                    "mae": mae,
                    "rmse": rmse,
                    "relative_error": relative_error,
                    "accuracy_score": accuracy_score,
                    "simulation_steps": len(simulation_angles)
                }
            )

        return BenchmarkResult(
            test_name=self.name,
            success=False,
            execution_time=0.0,
            error_message="Failed to collect simulation data"
        )

    def _create_simple_pendulum_xml(self) -> str:
        """Create simple pendulum for accuracy testing"""
        return """
        <mujoco>
            <option gravity="0 0 -9.81" timestep="0.002"/>
            <worldbody>
                <body name="pendulum" pos="0 0 0">
                    <joint name="hinge" type="hinge" axis="1 0 0" damping="0"/>
                    <geom name="rod" type="capsule" size="0.01 1.0" pos="0 0 -1"
                          rgba="0.8 0.2 0.2 1"/>
                    <geom name="mass" type="sphere" size="0.1" pos="0 0 -2" mass="1.0"
                          rgba="0.2 0.8 0.2 1"/>
                </body>
            </worldbody>
        </mujoco>
        """


class ScalabilityBenchmark(PhysicsBenchmark):
    """Test simulation scalability with increasing complexity"""

    def __init__(self):
        super().__init__(
            "Scalability",
            "Tests simulation scalability with increasing number of objects"
        )

    def _execute_benchmark(self) -> BenchmarkResult:
        """Execute scalability benchmark"""

        object_counts = [1, 5, 10, 20, 50]
        fps_results = []

        for count in object_counts:
            # Create model with specified number of objects
            model_xml = self._create_multi_object_xml(count)

            response = self.viewer_client.send_command({
                "type": "load_model",
                "model_id": f"scalability_test_{count}",
                "model_xml": model_xml
            })

            if not response.get("success"):
                continue

            # Measure performance with this configuration
            num_steps = 100
            start_time = time.time()

            for _step in range(num_steps):
                self.viewer_client.send_command({
                    "type": "get_state",
                    "model_id": f"scalability_test_{count}"
                })

            end_time = time.time()
            total_time = end_time - start_time
            fps = num_steps / total_time if total_time > 0 else 0.0
            fps_results.append((count, fps))

        # Calculate scalability metrics
        if len(fps_results) >= 2:
            # Linear regression to find performance degradation
            counts = [r[0] for r in fps_results]
            fps_values = [r[1] for r in fps_results]

            # Simple linear fit
            A = np.vstack([counts, np.ones(len(counts))]).T
            slope, _intercept = np.linalg.lstsq(A, fps_values, rcond=None)[0]

            # Scalability score (less negative slope is better)
            scalability_score = max(0, 1.0 + slope / max(fps_values))

            return BenchmarkResult(
                test_name=self.name,
                success=True,
                execution_time=0.0,
                metrics={
                    "max_objects_tested": max(counts),
                    "min_fps": min(fps_values),
                    "max_fps": max(fps_values),
                    "performance_slope": slope,
                    "scalability_score": scalability_score,
                    "fps_results": fps_results
                }
            )

        return BenchmarkResult(
            test_name=self.name,
            success=False,
            execution_time=0.0,
            error_message="Insufficient data for scalability analysis"
        )

    def _create_multi_object_xml(self, num_objects: int) -> str:
        """Create model with multiple objects"""
        xml_header = """
        <mujoco>
            <worldbody>
        """

        xml_objects = ""
        for i in range(num_objects):
            x = (i % 10) * 0.5 - 2.5  # Arrange in grid
            y = (i // 10) * 0.5 - 2.5
            z = 1.0

            xml_objects += f"""
                <body name="object_{i}" pos="{x} {y} {z}">
                    <joint name="free_{i}" type="free"/>
                    <geom name="box_{i}" type="box" size="0.1 0.1 0.1" rgba="0.8 0.2 0.2 1"/>
                </body>
            """

        xml_footer = """
            </worldbody>
        </mujoco>
        """

        return xml_header + xml_objects + xml_footer


class PerformanceMonitor:
    """Monitor system performance during benchmarks"""

    def __init__(self):
        self.monitoring = False
        self.samples = []
        self.start_time = None

    def start(self):
        """Start performance monitoring"""
        self.monitoring = True
        self.samples = []
        self.start_time = time.time()

    def sample(self):
        """Take a performance sample"""
        if self.monitoring:
            cpu_percent = psutil.cpu_percent()
            memory_info = psutil.virtual_memory()

            sample = {
                "timestamp": time.time() - self.start_time,
                "cpu_usage": cpu_percent,
                "memory_usage": memory_info.percent,
                "memory_available": memory_info.available / (1024**3)  # GB
            }
            self.samples.append(sample)

    def stop(self) -> PerformanceMetrics:
        """Stop monitoring and return metrics"""
        self.monitoring = False

        if not self.samples:
            return PerformanceMetrics()

        cpu_values = [s["cpu_usage"] for s in self.samples]
        memory_values = [s["memory_usage"] for s in self.samples]

        return PerformanceMetrics(
            cpu_usage=np.mean(cpu_values),
            memory_usage=np.mean(memory_values)
        )


class BenchmarkSuite:
    """Complete benchmark suite for MuJoCo MCP"""

    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.benchmarks = [
            SimulationStabilityBenchmark(),
            PerformanceBenchmark(),
            AccuracyBenchmark(),
            ScalabilityBenchmark()
        ]

        self.results = []

    def run_all_benchmarks(self, parallel: bool = False) -> List[BenchmarkResult]:
        """Run all benchmarks"""
        print("🚀 Starting MuJoCo MCP Benchmark Suite")
        print("=" * 60)

        if parallel:
            self.results = self._run_parallel()
        else:
            self.results = self._run_sequential()

        # Generate report
        self._generate_report()

        return self.results

    def _run_sequential(self) -> List[BenchmarkResult]:
        """Run benchmarks sequentially"""
        results = []

        for i, benchmark in enumerate(self.benchmarks):
            print(f"\n[{i+1}/{len(self.benchmarks)}] Running {benchmark.name}...")
            print(f"Description: {benchmark.description}")

            result = benchmark.run()
            results.append(result)

            if result.success:
                print(f"✅ {benchmark.name} completed successfully")
                for metric, value in result.metrics.items():
                    print(f"   {metric}: {value:.4f}")
            else:
                print(f"❌ {benchmark.name} failed: {result.error_message}")

        return results

    def _run_parallel(self) -> List[BenchmarkResult]:
        """Run benchmarks in parallel"""
        results = []

        with ThreadPoolExecutor(max_workers=len(self.benchmarks)) as executor:
            future_to_benchmark = {
                executor.submit(benchmark.run): benchmark
                for benchmark in self.benchmarks
            }

            for future in as_completed(future_to_benchmark):
                benchmark = future_to_benchmark[future]
                try:
                    result = future.result()
                    results.append(result)

                    if result.success:
                        print(f"✅ {benchmark.name} completed")
                    else:
                        print(f"❌ {benchmark.name} failed")

                except Exception as e:
                    print(f"❌ {benchmark.name} crashed: {e}")

        return results

    def _generate_report(self):
        """Generate comprehensive benchmark report"""
        report_data = {
            "timestamp": time.time(),
            "summary": self._generate_summary(),
            "detailed_results": [
                {
                    "test_name": result.test_name,
                    "success": result.success,
                    "execution_time": result.execution_time,
                    "metrics": result.metrics,
                    "error_message": result.error_message
                }
                for result in self.results
            ]
        }

        # Save JSON report
        json_file = self.output_dir / "benchmark_report.json"
        with json_file.open('w') as f:
            json.dump(report_data, f, indent=2)

        # Generate text summary
        self._generate_text_report()

        # Generate plots if matplotlib is available
        try:
            self._generate_plots()
        except ImportError:
            print("Matplotlib not available - skipping plot generation")

        print(f"\n📊 Benchmark report saved to: {self.output_dir}")

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate benchmark summary"""
        successful_tests = [r for r in self.results if r.success]
        failed_tests = [r for r in self.results if not r.success]

        return {
            "total_tests": len(self.results),
            "successful_tests": len(successful_tests),
            "failed_tests": len(failed_tests),
            "success_rate": len(successful_tests) / len(self.results) if self.results else 0.0,
            "total_execution_time": sum(r.execution_time for r in self.results)
        }

    def _generate_text_report(self):
        """Generate text summary report"""
        report_file = self.output_dir / "benchmark_summary.txt"

        with report_file.open('w') as f:
            f.write("MuJoCo MCP Benchmark Suite Results\n")
            f.write("=" * 50 + "\n\n")

            summary = self._generate_summary()
            f.write(f"Total Tests: {summary['total_tests']}\n")
            f.write(f"Successful: {summary['successful_tests']}\n")
            f.write(f"Failed: {summary['failed_tests']}\n")
            f.write(f"Success Rate: {summary['success_rate']:.1%}\n")
            f.write(f"Total Time: {summary['total_execution_time']:.2f}s\n\n")

            for result in self.results:
                f.write(f"Test: {result.test_name}\n")
                f.write(f"Status: {'PASS' if result.success else 'FAIL'}\n")
                f.write(f"Time: {result.execution_time:.2f}s\n")

                if result.success and result.metrics:
                    f.write("Metrics:\n")
                    for metric, value in result.metrics.items():
                        f.write(f"  {metric}: {value:.4f}\n")

                if not result.success and result.error_message:
                    f.write(f"Error: {result.error_message}\n")

                f.write("\n" + "-" * 30 + "\n\n")

    def _generate_plots(self):
        """Generate performance plots"""
        # Performance metrics plot
        performance_results = [
            r for r in self.results if r.test_name == "Performance" and r.success
        ]

        if performance_results:
            result = performance_results[0]
            metrics = result.metrics

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('MuJoCo MCP Performance Metrics')

            # FPS
            axes[0, 0].bar(['FPS'], [metrics.get('fps', 0)])
            axes[0, 0].set_title('Frames Per Second')
            axes[0, 0].set_ylabel('FPS')

            # CPU and Memory usage
            axes[0, 1].bar(
                ['CPU', 'Memory'],
                [metrics.get('cpu_usage', 0), metrics.get('memory_usage', 0)]
            )
            axes[0, 1].set_title('Resource Usage (%)')
            axes[0, 1].set_ylabel('Percentage')

            # Step times
            axes[1, 0].bar(['Avg Step Time', 'Max Step Time'],
                          [metrics.get('avg_step_time', 0), metrics.get('max_step_time', 0)])
            axes[1, 0].set_title('Step Times (ms)')
            axes[1, 0].set_ylabel('Milliseconds')

            # Overall scores
            stability_score = next((r.metrics.get('stability_score', 0) for r in self.results
                                  if r.test_name == "Simulation Stability" and r.success), 0)
            accuracy_score = next((r.metrics.get('accuracy_score', 0) for r in self.results
                                 if r.test_name == "Accuracy" and r.success), 0)
            scalability_score = next((r.metrics.get('scalability_score', 0) for r in self.results
                                    if r.test_name == "Scalability" and r.success), 0)

            axes[1, 1].bar(['Stability', 'Accuracy', 'Scalability'],
                          [stability_score, accuracy_score, scalability_score])
            axes[1, 1].set_title('Quality Scores')
            axes[1, 1].set_ylabel('Score (0-1)')
            axes[1, 1].set_ylim(0, 1)

            plt.tight_layout()
            plt.savefig(self.output_dir / 'performance_metrics.png', dpi=300, bbox_inches='tight')
            plt.close()


def main():
    """Run the complete benchmark suite"""
    import argparse

    parser = argparse.ArgumentParser(description='MuJoCo MCP Benchmark Suite')
    parser.add_argument('--parallel', action='store_true', help='Run benchmarks in parallel')
    parser.add_argument('--output', default='benchmark_results', help='Output directory')

    args = parser.parse_args()

    # Create and run benchmark suite
    suite = BenchmarkSuite(args.output)
    results = suite.run_all_benchmarks(parallel=args.parallel)

    # Print final summary
    successful = sum(1 for r in results if r.success)
    total = len(results)

    print(f"\n🎯 Benchmark Summary: {successful}/{total} tests passed")

    if successful == total:
        print("🎉 All benchmarks passed!")
        return 0
    else:
        print("⚠️  Some benchmarks failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

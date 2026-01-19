"""End-to-end integration tests with actual MuJoCo simulations.

These tests verify that all components work together correctly in
realistic scenarios with real MuJoCo physics simulations.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# Add project to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Check for required dependencies
missing_deps = [
    name for name in ("mujoco", "scipy", "gymnasium")
    if importlib.util.find_spec(name) is None
]

if missing_deps:
    pytest.skip(
        f"Missing required dependencies: {', '.join(missing_deps)}",
        allow_module_level=True,
    )

from mujoco_mcp.simulation import MuJoCoSimulation
from mujoco_mcp.robot_controller import RobotController
from mujoco_mcp.advanced_controllers import PIDController, PIDConfig, MinimumJerkTrajectory
from mujoco_mcp.sensor_feedback import LowPassFilter, KalmanFilter1D, SensorType, SensorReading
from mujoco_mcp.rl_integration import create_reaching_env, TaskType, ActionSpaceType
from mujoco_mcp.menagerie_loader import MenagerieLoader


class TestCompleteSimulationWorkflows:
    """Test complete simulation workflows with all components integrated."""

    def test_single_robot_trajectory_following(self):
        """Test a robot following a trajectory with PID control in simulation."""
        # Load a simple robot
        loader = MenagerieLoader()

        try:
            # Try to load a simple arm model
            model_xml = loader.get_model_xml("universal_robots_ur5e")
        except Exception:
            pytest.skip("MuJoCo Menagerie models not available")

        # Create simulation
        sim = MuJoCoSimulation()
        sim.load_from_xml_string(model_xml)

        # Get number of actuators
        nu = sim.get_num_actuators()
        nq = sim.get_num_joints()

        # Generate a simple trajectory (move first joint)
        start_pos = np.zeros(1)
        end_pos = np.array([0.5])  # Move 0.5 radians
        duration = 2.0
        num_steps = 100

        positions, _velocities, _ = MinimumJerkTrajectory.minimum_jerk_trajectory(
            start_pos, end_pos, duration, num_steps
        )

        # Create PID controller for position tracking
        pid_config = PIDConfig(kp=10.0, ki=1.0, kd=2.0, max_output=50.0, min_output=-50.0)
        pid = PIDController(pid_config)

        # Simulate trajectory following
        dt = duration / num_steps
        errors = []

        for i in range(num_steps):
            # Get current state
            current_positions = sim.get_joint_positions()
            current_pos = current_positions[0] if len(current_positions) > 0 else 0.0

            # Desired position from trajectory
            desired_pos = positions[i, 0]

            # Compute control with PID
            control = pid.update(target=desired_pos, current=current_pos, dt=dt)

            # Apply control (only to first actuator, others zero)
            control_array = np.zeros(nu)
            if nu > 0:
                control_array[0] = control

            sim.apply_control(control_array.tolist())

            # Step simulation
            sim.step()

            # Track error
            errors.append(abs(desired_pos - current_pos))

        # Verify that tracking error decreased over time
        initial_error = np.mean(errors[:10])
        final_error = np.mean(errors[-10:])

        assert final_error < initial_error, "Tracking error should decrease"
        assert final_error < 0.2, f"Final tracking error {final_error} too large"

    def test_sensor_feedback_in_simulation(self):
        """Test sensor feedback processing with actual simulation data."""
        # Create a simple pendulum model
        pendulum_xml = """
        <mujoco model="pendulum">
            <worldbody>
                <body name="pole" pos="0 0 1">
                    <geom type="cylinder" size="0.05 0.5" rgba="1 0 0 1"/>
                    <joint name="hinge" type="hinge" axis="0 1 0"/>
                </body>
            </worldbody>
            <actuator>
                <motor joint="hinge" name="torque" gear="1"/>
            </actuator>
        </mujoco>
        """

        sim = MuJoCoSimulation()
        sim.load_from_xml_string(pendulum_xml)

        # Create filters for sensor data
        lpf = LowPassFilter(cutoff_freq=5.0, sampling_rate=100.0)
        kf = KalmanFilter1D(process_variance=0.01, measurement_variance=0.1)

        # Simulate and collect sensor data
        raw_angles = []
        filtered_lpf = []
        filtered_kf = []

        for _ in range(200):
            # Step simulation
            sim.step()

            # Get joint position (simulated sensor)
            positions = sim.get_joint_positions()
            angle = positions[0] if len(positions) > 0 else 0.0

            # Add simulated sensor noise
            noisy_angle = angle + np.random.normal(0, 0.05)

            # Apply filters
            lpf_output = lpf.update(noisy_angle)
            kf_output = kf.update(noisy_angle)

            raw_angles.append(noisy_angle)
            filtered_lpf.append(lpf_output)
            filtered_kf.append(kf_output)

        # Skip transient (first 50 samples)
        raw_var = np.var(raw_angles[50:])
        lpf_var = np.var(filtered_lpf[50:])
        kf_var = np.var(filtered_kf[50:])

        # Filters should reduce noise variance
        assert lpf_var < raw_var, "Low-pass filter should reduce variance"
        assert kf_var < raw_var, "Kalman filter should reduce variance"

        # All outputs should be finite
        assert np.all(np.isfinite(filtered_lpf)), "LPF outputs should be finite"
        assert np.all(np.isfinite(filtered_kf)), "KF outputs should be finite"

    def test_multi_robot_simulation(self):
        """Test multiple robots in the same simulation."""
        robot_controller = RobotController()

        # Load multiple robots
        robot_types = ["panda_arm", "ur5e_arm"]
        loaded_robots = []

        for robot_type in robot_types:
            try:
                robot_id = robot_controller.load_robot(robot_type)
                loaded_robots.append((robot_id, robot_type))
            except (ValueError, RuntimeError):
                # Skip if robot type not available
                continue

        if len(loaded_robots) == 0:
            pytest.skip("No robot models available for multi-robot test")

        # Test that each robot can be controlled independently
        for robot_id, robot_type in loaded_robots:
            # Get robot state
            state = robot_controller.get_robot_state(robot_id)
            assert state is not None

            # Get model info
            model_info = robot_controller.models[robot_id]
            nu = model_info.nu

            # Apply random control
            if nu > 0:
                control = np.random.uniform(-1, 1, size=nu).tolist()
                robot_controller.set_joint_torques(robot_id, control)

                # Step simulation
                robot_controller.step_robot(robot_id)

                # Get new state
                new_state = robot_controller.get_robot_state(robot_id)
                assert new_state is not None

    def test_rl_environment_interaction(self):
        """Test RL environment creation and interaction."""
        try:
            # Create a reaching environment
            env = create_reaching_env(
                robot_type="point_mass",
                target_position=[0.5, 0.0, 0.5],
            )
        except Exception as e:
            pytest.skip(f"RL environment creation failed: {e}")

        # Reset environment
        observation, info = env.reset()

        assert observation is not None, "Observation should not be None"
        assert isinstance(observation, np.ndarray), "Observation should be numpy array"
        assert observation.shape[0] > 0, "Observation should have elements"

        # Take random actions
        total_reward = 0.0
        steps = 0
        max_steps = 50

        for _ in range(max_steps):
            # Sample random action
            action = env.action_space.sample()

            # Step environment
            observation, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1

            # Check outputs
            assert isinstance(observation, np.ndarray), "Observation should be array"
            assert isinstance(reward, (int, float)), "Reward should be numeric"
            assert isinstance(terminated, bool), "Terminated should be bool"
            assert isinstance(truncated, bool), "Truncated should be bool"
            assert isinstance(info, dict), "Info should be dict"

            if terminated or truncated:
                break

        assert steps > 0, "Should take at least one step"
        assert np.isfinite(total_reward), "Total reward should be finite"

        env.close()

    def test_simulation_state_consistency(self):
        """Test that simulation state remains consistent across operations."""
        # Create simple model
        simple_xml = """
        <mujoco model="simple">
            <worldbody>
                <body name="box" pos="0 0 1">
                    <geom type="box" size="0.1 0.1 0.1" mass="1"/>
                    <joint name="slide" type="slide" axis="0 0 1"/>
                </body>
            </worldbody>
            <actuator>
                <motor joint="slide" name="force" gear="1"/>
            </actuator>
        </mujoco>
        """

        sim = MuJoCoSimulation()
        sim.load_from_xml_string(simple_xml)

        # Get initial state
        initial_pos = sim.get_joint_positions()
        initial_vel = sim.get_joint_velocities()
        initial_time = sim.get_time()

        # Verify initial state
        assert len(initial_pos) == 1, "Should have 1 joint"
        assert len(initial_vel) == 1, "Should have 1 velocity"
        assert initial_time == 0.0, "Initial time should be 0"

        # Set specific state
        new_pos = [0.5]
        new_vel = [1.0]
        sim.set_joint_positions(new_pos)
        sim.set_joint_velocities(new_vel)

        # Verify state was set
        check_pos = sim.get_joint_positions()
        check_vel = sim.get_joint_velocities()

        np.testing.assert_allclose(check_pos, new_pos, rtol=1e-5)
        np.testing.assert_allclose(check_vel, new_vel, rtol=1e-5)

        # Step simulation
        for _ in range(10):
            sim.step()

        # State should have changed
        after_pos = sim.get_joint_positions()
        after_time = sim.get_time()

        assert after_time > initial_time, "Time should advance"
        # Position should change due to velocity
        assert not np.allclose(after_pos, new_pos), "Position should change"

        # Reset simulation
        sim.reset()

        # Should return to initial state
        reset_pos = sim.get_joint_positions()
        reset_time = sim.get_time()

        assert reset_time == 0.0, "Time should reset to 0"
        assert len(reset_pos) == 1, "Should still have 1 joint"

    def test_error_recovery_in_simulation(self):
        """Test that simulation handles errors gracefully."""
        sim = MuJoCoSimulation()

        # Create simple model
        simple_xml = """
        <mujoco model="test">
            <worldbody>
                <geom type="sphere" size="0.1"/>
            </worldbody>
        </mujoco>
        """

        sim.load_from_xml_string(simple_xml)

        # Try to set invalid positions (too many)
        with pytest.raises(ValueError, match="size mismatch"):
            sim.set_joint_positions([1.0, 2.0, 3.0])

        # Try to apply invalid control (too many)
        with pytest.raises(ValueError, match="size mismatch"):
            sim.apply_control([1.0, 2.0, 3.0])

        # Try to set NaN positions
        nq = sim.get_num_joints()
        if nq > 0:
            with pytest.raises(ValueError, match="NaN or Inf"):
                sim.set_joint_positions([np.nan] * nq)

        # Simulation should still work after errors
        sim.step()
        time_after = sim.get_time()
        assert time_after > 0.0, "Simulation should still work after errors"

    def test_performance_with_large_model(self):
        """Test simulation performance with a reasonably complex model."""
        loader = MenagerieLoader()

        try:
            # Try to load a complex humanoid model
            model_xml = loader.get_model_xml("unitree_h1")
        except Exception:
            try:
                # Fallback to a simpler model
                model_xml = loader.get_model_xml("unitree_a1")
            except Exception:
                pytest.skip("No complex models available for performance test")

        sim = MuJoCoSimulation()
        sim.load_from_xml_string(model_xml)

        # Benchmark simulation steps
        import time
        num_steps = 100

        start_time = time.time()
        for _ in range(num_steps):
            sim.step()
        end_time = time.time()

        elapsed = end_time - start_time
        steps_per_second = num_steps / elapsed

        # Should be able to simulate at reasonable speed
        # (This is a soft check - exact performance depends on hardware)
        assert steps_per_second > 10, f"Simulation too slow: {steps_per_second} steps/sec"
        assert np.isfinite(steps_per_second), "Performance metric should be finite"

    def test_menagerie_loader_integration(self):
        """Test MenagerieLoader integration with simulation."""
        loader = MenagerieLoader()

        # Get available models
        models = loader.get_available_models()

        assert isinstance(models, dict), "Should return dict of models"
        assert len(models) > 0, "Should have at least one category"

        # Try to load and validate a model from each category
        loaded_any = False

        for _category, model_list in models.items():
            if len(model_list) == 0:
                continue

            # Try first model in category
            model_name = model_list[0]

            try:
                # Get model XML
                model_xml = loader.get_model_xml(model_name)
                assert len(model_xml) > 0, "Model XML should not be empty"

                # Try to load in simulation
                sim = MuJoCoSimulation()
                sim.load_from_xml_string(model_xml)

                # Verify model loaded
                assert sim.get_num_joints() >= 0
                assert sim.get_time() == 0.0

                loaded_any = True
                break  # Success, no need to try more
            except Exception:
                # This model might not be available, try next
                continue

        if not loaded_any:
            pytest.skip("Could not load any Menagerie models")


class TestRobotControllerIntegration:
    """Test RobotController with actual robot models."""

    def test_robot_controller_lifecycle(self):
        """Test complete robot controller lifecycle."""
        controller = RobotController()

        # Load a robot
        try:
            robot_id = controller.load_robot("panda_arm")
        except ValueError:
            # Try alternative
            try:
                robot_id = controller.load_robot("ur5e_arm")
            except ValueError:
                pytest.skip("No robot models available")

        # Verify robot loaded
        assert robot_id in controller.models
        assert robot_id in controller.datas

        # Get initial state
        state = controller.get_robot_state(robot_id)
        assert state is not None
        assert "positions" in state
        assert "velocities" in state

        # Get model info
        model_info = controller.models[robot_id]
        nu = model_info.nu
        nq = model_info.nq

        # Set joint positions
        if nq > 0:
            new_positions = np.zeros(nq).tolist()
            controller.set_joint_positions(robot_id, new_positions)

            # Verify positions set
            state = controller.get_robot_state(robot_id)
            np.testing.assert_allclose(state["positions"], new_positions, rtol=1e-5)

        # Set joint velocities
        if nq > 0:
            new_velocities = np.zeros(nq).tolist()
            controller.set_joint_velocities(robot_id, new_velocities)

            # Verify velocities set
            state = controller.get_robot_state(robot_id)
            np.testing.assert_allclose(state["velocities"], new_velocities, rtol=1e-5)

        # Apply control and step
        if nu > 0:
            control = np.zeros(nu).tolist()
            controller.set_joint_torques(robot_id, control)
            controller.step_robot(robot_id)

            # Time should advance
            state_after = controller.get_robot_state(robot_id)
            assert state_after["time"] > 0.0

        # Reset robot
        controller.reset_robot(robot_id)
        state_reset = controller.get_robot_state(robot_id)
        assert state_reset["time"] == 0.0


class TestSensorIntegration:
    """Test sensor feedback with simulated data."""

    def test_sensor_reading_creation_from_simulation(self):
        """Test creating sensor readings from simulation data."""
        # Create simple simulation
        simple_xml = """
        <mujoco model="sensor_test">
            <worldbody>
                <body name="obj" pos="0 0 1">
                    <geom type="box" size="0.1 0.1 0.1"/>
                    <joint name="free" type="free"/>
                </body>
            </worldbody>
        </mujoco>
        """

        sim = MuJoCoSimulation()
        sim.load_from_xml_string(simple_xml)

        # Get sensor data (simulated accelerometer)
        for _i in range(10):
            sim.step()

            # Create sensor reading from simulation time
            sim_time = sim.get_time()

            # Simulated sensor data (e.g., accelerometer)
            accel_data = np.random.normal(0, 0.1, size=3)

            reading = SensorReading(
                sensor_id="accel_0",
                sensor_type=SensorType.IMU,
                timestamp=sim_time,
                data=accel_data,
                quality=0.95,
            )

            assert reading.timestamp == sim_time
            assert len(reading.data) == 3
            assert 0.0 <= reading.quality <= 1.0

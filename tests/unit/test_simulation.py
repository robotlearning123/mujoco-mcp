"""Comprehensive unit tests for simulation.py covering edge cases and error handling."""

import numpy as np
import pytest

from mujoco_mcp.simulation import MuJoCoSimulation


# Sample valid MuJoCo XML for testing
VALID_PENDULUM_XML = """
<mujoco model="pendulum">
    <worldbody>
        <body name="pole" pos="0 0 0">
            <joint name="hinge" type="hinge" axis="0 1 0"/>
            <geom name="cpole" type="capsule" fromto="0 0 0 0 0 0.6" size="0.045"/>
        </body>
    </worldbody>
    <actuator>
        <motor name="torque" joint="hinge" gear="1"/>
    </actuator>
</mujoco>
"""

MINIMAL_XML = """
<mujoco>
    <worldbody>
        <geom type="box" size="1 1 1"/>
    </worldbody>
</mujoco>
"""


class TestSimulationInitialization:
    """Test simulation initialization and model loading."""

    def test_init_without_model(self):
        """Test initializing simulation without a model."""
        sim = MuJoCoSimulation()
        assert not sim.is_initialized()
        assert sim.model is None
        assert sim.data is None

    def test_init_with_xml_string(self):
        """Test initializing with XML string."""
        sim = MuJoCoSimulation(model_xml=VALID_PENDULUM_XML)
        assert sim.is_initialized()
        assert sim.model is not None
        assert sim.data is not None

    def test_load_empty_model(self):
        """Test loading empty model raises ValueError."""
        sim = MuJoCoSimulation()
        with pytest.raises(ValueError, match="Empty MuJoCo model is not valid"):
            sim.load_from_xml_string("<mujoco></mujoco>")

        # Also test with whitespace/newlines
        with pytest.raises(ValueError, match="Empty MuJoCo model is not valid"):
            sim.load_from_xml_string("<mujoco>\n  \n</mujoco>")

    def test_load_from_xml_string(self):
        """Test loading model from XML string."""
        sim = MuJoCoSimulation()
        sim.load_from_xml_string(VALID_PENDULUM_XML)
        assert sim.is_initialized()
        assert sim.get_model_name() == "pendulum"

    def test_load_model_from_string_alias(self):
        """Test backward compatibility alias."""
        sim = MuJoCoSimulation()
        sim.load_model_from_string(VALID_PENDULUM_XML)
        assert sim.is_initialized()

    def test_load_from_nonexistent_file(self):
        """Test loading from non-existent file raises error."""
        sim = MuJoCoSimulation()
        with pytest.raises(Exception):  # MuJoCo raises various exceptions
            sim.load_from_file("/nonexistent/path/to/model.xml")


class TestUninitializedAccess:
    """Test that operations on uninitialized simulation raise RuntimeError."""

    def test_step_uninitialized(self):
        """Test stepping uninitialized simulation raises RuntimeError."""
        sim = MuJoCoSimulation()
        with pytest.raises(RuntimeError, match="Simulation not initialized"):
            sim.step()

    def test_reset_uninitialized(self):
        """Test resetting uninitialized simulation raises RuntimeError."""
        sim = MuJoCoSimulation()
        with pytest.raises(RuntimeError, match="Simulation not initialized"):
            sim.reset()

    def test_get_joint_positions_uninitialized(self):
        """Test getting joint positions from uninitialized simulation."""
        sim = MuJoCoSimulation()
        with pytest.raises(RuntimeError, match="Simulation not initialized"):
            sim.get_joint_positions()

    def test_get_joint_velocities_uninitialized(self):
        """Test getting joint velocities from uninitialized simulation."""
        sim = MuJoCoSimulation()
        with pytest.raises(RuntimeError, match="Simulation not initialized"):
            sim.get_joint_velocities()

    def test_set_joint_positions_uninitialized(self):
        """Test setting joint positions on uninitialized simulation."""
        sim = MuJoCoSimulation()
        with pytest.raises(RuntimeError, match="Simulation not initialized"):
            sim.set_joint_positions([0.0])

    def test_set_joint_velocities_uninitialized(self):
        """Test setting joint velocities on uninitialized simulation."""
        sim = MuJoCoSimulation()
        with pytest.raises(RuntimeError, match="Simulation not initialized"):
            sim.set_joint_velocities([0.0])

    def test_apply_control_uninitialized(self):
        """Test applying control to uninitialized simulation."""
        sim = MuJoCoSimulation()
        with pytest.raises(RuntimeError, match="Simulation not initialized"):
            sim.apply_control([0.0])

    def test_get_sensor_data_uninitialized(self):
        """Test getting sensor data from uninitialized simulation."""
        sim = MuJoCoSimulation()
        with pytest.raises(RuntimeError, match="Simulation not initialized"):
            sim.get_sensor_data()

    def test_get_rigid_body_states_uninitialized(self):
        """Test getting rigid body states from uninitialized simulation."""
        sim = MuJoCoSimulation()
        with pytest.raises(RuntimeError, match="Simulation not initialized"):
            sim.get_rigid_body_states()

    def test_get_time_uninitialized(self):
        """Test getting time from uninitialized simulation."""
        sim = MuJoCoSimulation()
        with pytest.raises(RuntimeError, match="Simulation not initialized"):
            sim.get_time()

    def test_get_timestep_uninitialized(self):
        """Test getting timestep from uninitialized simulation."""
        sim = MuJoCoSimulation()
        with pytest.raises(RuntimeError, match="Simulation not initialized"):
            sim.get_timestep()

    def test_get_num_joints_uninitialized(self):
        """Test getting number of joints from uninitialized simulation."""
        sim = MuJoCoSimulation()
        with pytest.raises(RuntimeError, match="Simulation not initialized"):
            sim.get_num_joints()

    def test_get_num_actuators_uninitialized(self):
        """Test getting number of actuators from uninitialized simulation."""
        sim = MuJoCoSimulation()
        with pytest.raises(RuntimeError, match="Simulation not initialized"):
            sim.get_num_actuators()

    def test_get_joint_names_uninitialized(self):
        """Test getting joint names from uninitialized simulation."""
        sim = MuJoCoSimulation()
        with pytest.raises(RuntimeError, match="Simulation not initialized"):
            sim.get_joint_names()

    def test_get_model_name_uninitialized(self):
        """Test getting model name from uninitialized simulation."""
        sim = MuJoCoSimulation()
        with pytest.raises(RuntimeError, match="Simulation not initialized"):
            sim.get_model_name()

    def test_get_model_info_uninitialized(self):
        """Test getting model info from uninitialized simulation."""
        sim = MuJoCoSimulation()
        with pytest.raises(RuntimeError, match="Simulation not initialized"):
            sim.get_model_info()

    def test_render_frame_uninitialized(self):
        """Test rendering frame from uninitialized simulation."""
        sim = MuJoCoSimulation()
        with pytest.raises(RuntimeError, match="Simulation not initialized"):
            sim.render_frame()

    def test_render_ascii_uninitialized(self):
        """Test ASCII rendering from uninitialized simulation."""
        sim = MuJoCoSimulation()
        with pytest.raises(RuntimeError, match="Simulation not initialized"):
            sim.render_ascii()


class TestArrayMismatches:
    """Test that array size mismatches are detected."""

    def test_set_joint_positions_wrong_size(self):
        """Test setting joint positions with wrong array size."""
        sim = MuJoCoSimulation(model_xml=VALID_PENDULUM_XML)
        nq = sim.get_num_joints()

        # Too few positions
        with pytest.raises(ValueError, match="Position array size mismatch"):
            sim.set_joint_positions([0.0] * (nq - 1) if nq > 1 else [0.0, 0.0])

        # Too many positions
        with pytest.raises(ValueError, match="Position array size mismatch"):
            sim.set_joint_positions([0.0] * (nq + 1))

    def test_set_joint_velocities_wrong_size(self):
        """Test setting joint velocities with wrong array size."""
        sim = MuJoCoSimulation(model_xml=VALID_PENDULUM_XML)
        model_info = sim.get_model_info()
        nv = model_info["nv"]

        # Too few velocities
        with pytest.raises(ValueError, match="Velocity array size mismatch"):
            sim.set_joint_velocities([0.0] * (nv - 1) if nv > 1 else [0.0, 0.0])

        # Too many velocities
        with pytest.raises(ValueError, match="Velocity array size mismatch"):
            sim.set_joint_velocities([0.0] * (nv + 1))

    def test_apply_control_wrong_size(self):
        """Test applying control with wrong array size."""
        sim = MuJoCoSimulation(model_xml=VALID_PENDULUM_XML)
        nu = sim.get_num_actuators()

        # Too few controls
        with pytest.raises(ValueError, match="Control array size mismatch"):
            sim.apply_control([0.0] * (nu - 1) if nu > 1 else [0.0, 0.0])

        # Too many controls
        with pytest.raises(ValueError, match="Control array size mismatch"):
            sim.apply_control([0.0] * (nu + 1))


class TestNaNInfValidation:
    """Test that NaN and Inf values are rejected."""

    def test_set_joint_positions_with_nan(self):
        """Test setting joint positions with NaN values."""
        sim = MuJoCoSimulation(model_xml=VALID_PENDULUM_XML)
        nq = sim.get_num_joints()

        with pytest.raises(ValueError, match="Position array contains NaN or Inf"):
            sim.set_joint_positions([np.nan] * nq)

    def test_set_joint_positions_with_inf(self):
        """Test setting joint positions with Inf values."""
        sim = MuJoCoSimulation(model_xml=VALID_PENDULUM_XML)
        nq = sim.get_num_joints()

        with pytest.raises(ValueError, match="Position array contains NaN or Inf"):
            sim.set_joint_positions([np.inf] * nq)

    def test_set_joint_velocities_with_nan(self):
        """Test setting joint velocities with NaN values."""
        sim = MuJoCoSimulation(model_xml=VALID_PENDULUM_XML)
        model_info = sim.get_model_info()
        nv = model_info["nv"]

        with pytest.raises(ValueError, match="Velocity array contains NaN or Inf"):
            sim.set_joint_velocities([np.nan] * nv)

    def test_set_joint_velocities_with_inf(self):
        """Test setting joint velocities with Inf values."""
        sim = MuJoCoSimulation(model_xml=VALID_PENDULUM_XML)
        model_info = sim.get_model_info()
        nv = model_info["nv"]

        with pytest.raises(ValueError, match="Velocity array contains NaN or Inf"):
            sim.set_joint_velocities([np.inf] * nv)

    def test_apply_control_with_nan(self):
        """Test applying control with NaN values."""
        sim = MuJoCoSimulation(model_xml=VALID_PENDULUM_XML)
        nu = sim.get_num_actuators()

        with pytest.raises(ValueError, match="Control array contains NaN or Inf"):
            sim.apply_control([np.nan] * nu)

    def test_apply_control_with_inf(self):
        """Test applying control with Inf values."""
        sim = MuJoCoSimulation(model_xml=VALID_PENDULUM_XML)
        nu = sim.get_num_actuators()

        with pytest.raises(ValueError, match="Control array contains NaN or Inf"):
            sim.apply_control([np.inf] * nu)


class TestSimulationOperations:
    """Test normal simulation operations."""

    def test_step_single(self):
        """Test stepping simulation once."""
        sim = MuJoCoSimulation(model_xml=VALID_PENDULUM_XML)
        initial_time = sim.get_time()
        sim.step()
        assert sim.get_time() > initial_time

    def test_step_multiple(self):
        """Test stepping simulation multiple times."""
        sim = MuJoCoSimulation(model_xml=VALID_PENDULUM_XML)
        initial_time = sim.get_time()
        sim.step(10)
        assert sim.get_time() > initial_time

    def test_reset(self):
        """Test resetting simulation."""
        sim = MuJoCoSimulation(model_xml=VALID_PENDULUM_XML)

        # Step simulation and change state
        sim.step(10)
        sim.set_joint_positions([1.0])

        # Reset should restore initial state
        sim.reset()
        assert sim.get_time() == 0.0
        # Position might not be exactly zero depending on model, just verify reset works
        positions = sim.get_joint_positions()
        assert len(positions) == 1

    def test_get_set_joint_positions(self):
        """Test getting and setting joint positions."""
        sim = MuJoCoSimulation(model_xml=VALID_PENDULUM_XML)

        # Set positions
        test_pos = [0.5]
        sim.set_joint_positions(test_pos)

        # Get positions
        positions = sim.get_joint_positions()
        assert len(positions) == 1
        assert np.isclose(positions[0], test_pos[0])

    def test_get_set_joint_velocities(self):
        """Test getting and setting joint velocities."""
        sim = MuJoCoSimulation(model_xml=VALID_PENDULUM_XML)

        # Set velocities
        test_vel = [1.5]
        sim.set_joint_velocities(test_vel)

        # Get velocities
        velocities = sim.get_joint_velocities()
        assert len(velocities) == 1
        assert np.isclose(velocities[0], test_vel[0])

    def test_apply_control(self):
        """Test applying control inputs."""
        sim = MuJoCoSimulation(model_xml=VALID_PENDULUM_XML)

        # Apply control
        sim.apply_control([0.5])

        # Step to apply the control
        sim.step()

        # Verify simulation advanced
        assert sim.get_time() > 0

    def test_get_sensor_data(self):
        """Test getting sensor data."""
        sim = MuJoCoSimulation(model_xml=VALID_PENDULUM_XML)
        sensor_data = sim.get_sensor_data()
        assert isinstance(sensor_data, dict)

    def test_get_rigid_body_states(self):
        """Test getting rigid body states."""
        sim = MuJoCoSimulation(model_xml=VALID_PENDULUM_XML)
        body_states = sim.get_rigid_body_states()
        assert isinstance(body_states, dict)
        assert "pole" in body_states
        assert "position" in body_states["pole"]
        assert "orientation" in body_states["pole"]
        assert len(body_states["pole"]["position"]) == 3
        assert len(body_states["pole"]["orientation"]) == 4

    def test_get_time(self):
        """Test getting simulation time."""
        sim = MuJoCoSimulation(model_xml=VALID_PENDULUM_XML)
        time = sim.get_time()
        assert time == 0.0

        sim.step()
        assert sim.get_time() > 0.0

    def test_get_timestep(self):
        """Test getting simulation timestep."""
        sim = MuJoCoSimulation(model_xml=VALID_PENDULUM_XML)
        timestep = sim.get_timestep()
        assert timestep > 0.0

    def test_get_num_joints(self):
        """Test getting number of joints."""
        sim = MuJoCoSimulation(model_xml=VALID_PENDULUM_XML)
        nq = sim.get_num_joints()
        assert nq > 0

    def test_get_num_actuators(self):
        """Test getting number of actuators."""
        sim = MuJoCoSimulation(model_xml=VALID_PENDULUM_XML)
        nu = sim.get_num_actuators()
        assert nu > 0

    def test_get_joint_names(self):
        """Test getting joint names."""
        sim = MuJoCoSimulation(model_xml=VALID_PENDULUM_XML)
        names = sim.get_joint_names()
        assert isinstance(names, list)
        assert "hinge" in names

    def test_get_model_name(self):
        """Test getting model name."""
        sim = MuJoCoSimulation(model_xml=VALID_PENDULUM_XML)
        name = sim.get_model_name()
        assert name == "pendulum"

    def test_get_model_info(self):
        """Test getting model info."""
        sim = MuJoCoSimulation(model_xml=VALID_PENDULUM_XML)
        info = sim.get_model_info()
        assert "nq" in info
        assert "nv" in info
        assert "nbody" in info
        assert "njoint" in info
        assert "ngeom" in info
        assert "nsensor" in info
        assert "nu" in info
        assert "timestep" in info

    def test_render_ascii(self):
        """Test ASCII rendering."""
        sim = MuJoCoSimulation(model_xml=VALID_PENDULUM_XML)
        ascii_art = sim.render_ascii()
        assert isinstance(ascii_art, str)
        assert "Angle:" in ascii_art
        assert "Time:" in ascii_art

    def test_positions_velocities_are_copies(self):
        """Test that getters return copies, not references."""
        sim = MuJoCoSimulation(model_xml=VALID_PENDULUM_XML)

        # Get positions
        pos1 = sim.get_joint_positions()
        pos2 = sim.get_joint_positions()

        # Modify one copy
        pos1[0] = 999.0

        # Other copy should be unchanged
        assert pos2[0] != 999.0

        # Same for velocities
        vel1 = sim.get_joint_velocities()
        vel2 = sim.get_joint_velocities()
        vel1[0] = 999.0
        assert vel2[0] != 999.0


class TestMinimalModel:
    """Test simulation with minimal model (no joints/actuators)."""

    def test_minimal_model_initialization(self):
        """Test loading minimal model without joints."""
        sim = MuJoCoSimulation(model_xml=MINIMAL_XML)
        assert sim.is_initialized()

    def test_minimal_model_no_joints(self):
        """Test minimal model has no joints."""
        sim = MuJoCoSimulation(model_xml=MINIMAL_XML)
        assert sim.get_num_joints() == 0
        assert sim.get_num_actuators() == 0

    def test_minimal_model_step(self):
        """Test stepping minimal model."""
        sim = MuJoCoSimulation(model_xml=MINIMAL_XML)
        initial_time = sim.get_time()
        sim.step()
        assert sim.get_time() > initial_time


class TestRenderingEdgeCases:
    """Test rendering edge cases and fallbacks."""

    def test_render_frame_default_params(self):
        """Test rendering with default parameters."""
        sim = MuJoCoSimulation(model_xml=VALID_PENDULUM_XML)
        frame = sim.render_frame()
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (480, 640, 3)

    def test_render_frame_custom_size(self):
        """Test rendering with custom size."""
        sim = MuJoCoSimulation(model_xml=VALID_PENDULUM_XML)
        frame = sim.render_frame(width=320, height=240)
        assert isinstance(frame, np.ndarray)
        # Might fall back to software rendering with different size
        assert frame.ndim == 3
        assert frame.shape[2] == 3

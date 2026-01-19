"""MuJoCo simulation wrapper providing physics simulation functionality."""

import logging
import uuid
from typing import Dict, Any, List, Tuple

import mujoco
import numpy as np

logger = logging.getLogger("mujoco_mcp.simulation")


class MuJoCoSimulation:
    """Basic MuJoCo simulation class providing core functionality.

    Example:
        >>> # Create simulation from XML string
        >>> model_xml = '''
        ... <mujoco>
        ...   <worldbody>
        ...     <body name="box" pos="0 0 1">
        ...       <joint type="free"/>
        ...       <geom type="box" size="0.1 0.1 0.1"/>
        ...     </body>
        ...   </worldbody>
        ... </mujoco>
        ... '''
        >>> sim = MuJoCoSimulation(model_xml=model_xml)
        >>>
        >>> # Step the simulation
        >>> sim.step(num_steps=100)
        >>>
        >>> # Get joint positions
        >>> positions = sim.get_joint_positions()
        >>>
        >>> # Reset simulation
        >>> sim.reset()
    """

    def __init__(self, model_xml: str | None = None, model_path: str | None = None):
        """Initialize MuJoCo simulation."""
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.sim_id = str(uuid.uuid4())
        self._initialized = False

        if model_xml:
            self.load_from_xml_string(model_xml)
        elif model_path:
            self.load_from_file(model_path)

    def load_from_xml_string(self, model_xml: str):
        """Load MuJoCo model from XML string.

        Args:
            model_xml: Complete MuJoCo model definition in XML format.

        Raises:
            ValueError: If model_xml is empty or invalid.
            RuntimeError: If MuJoCo fails to parse the XML.

        Note:
            This method initializes both the model and simulation data.
            Empty models (containing only <mujoco></mujoco>) are rejected.
        """
        # Check for empty model
        if "<mujoco></mujoco>" in model_xml.replace(" ", "").replace("\n", ""):
            raise ValueError("Empty MuJoCo model is not valid")

        self.model = mujoco.MjModel.from_xml_string(model_xml)
        self.data = mujoco.MjData(self.model)
        self._initialized = True
        logger.info(f"Loaded model from XML string, sim_id: {self.sim_id}")

    def load_model_from_string(self, xml_string: str):
        """Alias for load_from_xml_string for backward compatibility."""
        return self.load_from_xml_string(xml_string)

    def load_from_file(self, model_path: str):
        """Load MuJoCo model from XML file.

        Args:
            model_path: Path to MuJoCo XML model file (absolute or relative).

        Raises:
            FileNotFoundError: If model_path does not exist.
            RuntimeError: If MuJoCo fails to parse the XML file.
            PermissionError: If model_path is not readable.

        Note:
            This method initializes both the model and simulation data.
            The path can be absolute or relative to the current working directory.
        """
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self._initialized = True
        logger.info(f"Loaded model from file: {model_path}, sim_id: {self.sim_id}")

    def is_initialized(self) -> bool:
        """Check if simulation is properly initialized with a model.

        Returns:
            True if a model has been loaded and simulation data exists, False otherwise.

        Note:
            A simulation is considered initialized after successfully calling either
            load_from_xml_string() or load_from_file().
        """
        return self._initialized

    def _require_sim(self) -> Tuple[mujoco.MjModel, mujoco.MjData]:
        if not self._initialized or self.model is None or self.data is None:
            raise RuntimeError("Simulation not initialized")
        return self.model, self.data

    def step(self, num_steps: int = 1):
        """Step the physics simulation forward in time.

        Args:
            num_steps: Number of simulation timesteps to advance (default: 1).
                      Each step advances time by model.opt.timestep seconds.

        Raises:
            RuntimeError: If simulation not initialized.
            ValueError: If num_steps is not positive.

        Note:
            This integrates the equations of motion using the configured integrator
            (Euler, RK4, etc.). Control inputs set via apply_control() are applied
            during each step.
        """
        model, data = self._require_sim()

        for _ in range(num_steps):
            mujoco.mj_step(model, data)

    def reset(self):
        """Reset simulation to initial state defined in the model.

        Raises:
            RuntimeError: If simulation not initialized.

        Note:
            This resets:
            - All joint positions (qpos) to keyframe 0 or default values
            - All joint velocities (qvel) to zero
            - All actuator activations to zero
            - Simulation time to zero
            - All cached physics quantities are recomputed
        """
        model, data = self._require_sim()
        mujoco.mj_resetData(model, data)

    def get_joint_positions(self) -> np.ndarray:
        """Get current generalized coordinates (joint positions).

        Returns:
            Numpy array of shape (nq,) containing current joint positions.
            For rotational joints, values are in radians.
            For prismatic joints, values are in meters.
            A copy is returned to prevent accidental modification.

        Raises:
            RuntimeError: If simulation not initialized.

        Note:
            This returns qpos which includes both joint coordinates and free body
            positions/orientations for floating bodies. The ordering matches the
            model's joint definition order.
        """
        _, data = self._require_sim()
        return data.qpos.copy()

    def get_joint_velocities(self) -> np.ndarray:
        """Get current generalized velocities (joint velocities).

        Returns:
            Numpy array of shape (nv,) containing current joint velocities.
            For rotational joints, values are in radians/second.
            For prismatic joints, values are in meters/second.
            A copy is returned to prevent accidental modification.

        Raises:
            RuntimeError: If simulation not initialized.

        Note:
            This returns qvel which represents the time derivative of qpos.
            For most joints, nv equals nq, but for quaternion-based free joints,
            nv may differ from nq (3 rotational velocities vs 4 quaternion components).
        """
        _, data = self._require_sim()
        return data.qvel.copy()

    def set_joint_positions(self, positions: List[float]):
        """Set joint positions.

        Args:
            positions: Joint position values. Length must match model.nq.

        Raises:
            RuntimeError: If simulation not initialized.
            ValueError: If positions length doesn't match model.nq or contains NaN/Inf.
        """
        model, data = self._require_sim()

        # Validate array size
        if len(positions) != model.nq:
            raise ValueError(
                f"Position array size mismatch: got {len(positions)}, expected {model.nq}"
            )

        # Convert to numpy array for validation
        pos_array = np.array(positions)

        # Validate for NaN/Inf
        if not np.isfinite(pos_array).all():
            raise ValueError(f"Position array contains NaN or Inf values: {positions}")

        data.qpos[:] = pos_array
        mujoco.mj_forward(model, data)

    def set_joint_velocities(self, velocities: List[float]):
        """Set joint velocities.

        Args:
            velocities: Joint velocity values. Length must match model.nv.

        Raises:
            RuntimeError: If simulation not initialized.
            ValueError: If velocities length doesn't match model.nv or contains NaN/Inf.
        """
        model, data = self._require_sim()

        # Validate array size
        if len(velocities) != model.nv:
            raise ValueError(
                f"Velocity array size mismatch: got {len(velocities)}, expected {model.nv}"
            )

        # Convert to numpy array for validation
        vel_array = np.array(velocities)

        # Validate for NaN/Inf
        if not np.isfinite(vel_array).all():
            raise ValueError(f"Velocity array contains NaN or Inf values: {velocities}")

        data.qvel[:] = vel_array

    def apply_control(self, control: List[float]):
        """Apply control inputs.

        Args:
            control: Control input values. Length must match model.nu.

        Raises:
            RuntimeError: If simulation not initialized.
            ValueError: If control length doesn't match model.nu or contains NaN/Inf.
        """
        model, data = self._require_sim()

        # Validate array size
        if len(control) != model.nu:
            raise ValueError(
                f"Control array size mismatch: got {len(control)}, expected {model.nu}"
            )

        # Convert to numpy array for validation
        ctrl_array = np.array(control)

        # Validate for NaN/Inf
        if not np.isfinite(ctrl_array).all():
            raise ValueError(f"Control array contains NaN or Inf values: {control}")

        data.ctrl[:] = ctrl_array

    def get_sensor_data(self) -> Dict[str, List[float]]:
        """Get readings from all sensors defined in the model.

        Returns:
            Dictionary mapping sensor names to their current readings.
            Each sensor value is a list of floats (sensors can be multi-dimensional).

        Raises:
            RuntimeError: If simulation not initialized.

        Note:
            Sensor types include: touch, accelerometer, velocimeter, gyro, force,
            torque, magnetometer, rangefinder, jointpos, jointvel, tendonpos,
            tendonvel, actuatorpos, actuatorvel, actuatorfrc, ballquat, ballangvel,
            jointlimitpos, jointlimitvel, jointlimitfrc, tendonlimitpos,
            tendonlimitvel, tendonlimitfrc, framepos, framequat, framexaxis,
            frameyaxis, framezaxis, framelinvel, frameangvel, framelinacc, frameangacc,
            subtreecom, subtreelinvel, subtreeangmom, and user-defined sensors.
        """
        model, data = self._require_sim()

        sensor_data: Dict[str, List[float]] = {}
        for i in range(model.nsensor):
            name = model.sensor(i).name
            sensor_data[name] = data.sensordata[i : i + 1].tolist()
        return sensor_data

    def get_rigid_body_states(self) -> Dict[str, Dict[str, List[float]]]:
        """Get Cartesian positions and orientations of all rigid bodies.

        Returns:
            Dictionary mapping body names to their states. Each state contains:
            - 'position': [x, y, z] in world coordinates (meters)
            - 'orientation': [w, x, y, z] quaternion (scalar-first convention)

            Unnamed bodies are excluded from the result.

        Raises:
            RuntimeError: If simulation not initialized.

        Note:
            Positions (xpos) are the body's center of mass in world coordinates.
            Orientations (xquat) use the scalar-first quaternion convention: [w, x, y, z]
            where w is the scalar part and (x, y, z) is the vector part.
            The world body (index 0) is typically named "world" or left unnamed.
        """
        model, data = self._require_sim()

        body_states: Dict[str, Dict[str, List[float]]] = {}
        for i in range(model.nbody):
            name = model.body(i).name
            if name:  # Skip unnamed bodies
                pos = data.xpos[i].tolist()
                quat = data.xquat[i].tolist()
                body_states[name] = {"position": pos, "orientation": quat}
        return body_states

    def get_time(self) -> float:
        """Get simulation time.

        Returns:
            Current simulation time in seconds.

        Raises:
            RuntimeError: If simulation not initialized.
        """
        _, data = self._require_sim()
        return data.time

    def get_timestep(self) -> float:
        """Get simulation timestep.

        Returns:
            Simulation timestep in seconds.

        Raises:
            RuntimeError: If simulation not initialized.
        """
        model, _ = self._require_sim()
        return model.opt.timestep

    def get_num_joints(self) -> int:
        """Get number of joints.

        Returns:
            Number of generalized coordinates (joints).

        Raises:
            RuntimeError: If simulation not initialized.
        """
        model, _ = self._require_sim()
        return model.nq

    def get_num_actuators(self) -> int:
        """Get number of actuators.

        Returns:
            Number of actuators in the model.

        Raises:
            RuntimeError: If simulation not initialized.
        """
        model, _ = self._require_sim()
        return model.nu

    def get_joint_names(self) -> List[str]:
        """Get joint names.

        Returns:
            List of joint names in the model.

        Raises:
            RuntimeError: If simulation not initialized.
        """
        model, _ = self._require_sim()
        return [model.joint(i).name for i in range(model.njnt)]

    def get_model_name(self) -> str:
        """Get model name.

        Returns:
            Name of the MuJoCo model, or "unnamed" if not set.

        Raises:
            RuntimeError: If simulation not initialized.
        """
        model, _ = self._require_sim()
        return model.meta.model_name or "unnamed"

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the loaded model.

        Returns:
            Dictionary containing model dimensions and configuration:
            - nq: Number of generalized coordinates (position dimensions)
            - nv: Number of degrees of freedom (velocity dimensions)
            - nbody: Number of rigid bodies
            - njoint: Number of joints
            - ngeom: Number of geometric collision/visual elements
            - nsensor: Number of sensors
            - nu: Number of actuators (controls)
            - timestep: Simulation timestep in seconds

        Raises:
            RuntimeError: If simulation not initialized.

        Note:
            For most models, nq equals nv. However, models with quaternion-based
            free joints will have nq > nv (7 vs 6 per free joint).
        """
        model, _ = self._require_sim()

        return {
            "nq": model.nq,  # number of generalized coordinates
            "nv": model.nv,  # number of degrees of freedom
            "nbody": model.nbody,  # number of bodies
            "njoint": model.njnt,  # number of joints
            "ngeom": model.ngeom,  # number of geoms
            "nsensor": model.nsensor,  # number of sensors
            "nu": model.nu,  # number of actuators
            "timestep": model.opt.timestep,
        }

    def render_frame(
        self, width: int = 640, height: int = 480, camera_id: int = -1, scene_option=None
    ) -> np.ndarray:
        """Render a frame from the simulation.

        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.
            camera_id: Camera ID to render from (-1 for default).
            scene_option: Optional scene rendering options.

        Returns:
            RGB image as numpy array of shape (height, width, 3).

        Raises:
            RuntimeError: If simulation not initialized.

        Note:
            Falls back to software rendering if hardware rendering fails.
        """
        model, data = self._require_sim()

        try:
            renderer = mujoco.Renderer(model, height=height, width=width)
            renderer.update_scene(data, camera=camera_id, scene_option=scene_option)

            # Render and return RGB array
            return renderer.render()

        except (RuntimeError, OSError, ValueError) as e:
            # Hardware rendering failed - fall back to software
            logger.warning(
                f"Hardware rendering failed (camera={camera_id}, "
                f"{width}x{height}): {e}. Falling back to software rendering."
            )
            return self._render_software_fallback(width, height)
        except Exception as e:
            # Unexpected error during rendering
            logger.exception(f"Unexpected rendering error: {e}")
            logger.info("Attempting software fallback rendering")
            return self._render_software_fallback(width, height)

    def _render_software_fallback(self, width: int, height: int) -> np.ndarray:
        """Fallback software rendering when hardware rendering fails."""
        # Create a simple visualization using simulation state
        # This is a placeholder that creates a visual representation of the pendulum

        import math

        # Get joint positions for visualization
        model, data = self._require_sim()

        if model.nq > 0:
            joint_pos = data.qpos[0] if len(data.qpos) > 0 else 0.0
        else:
            joint_pos = 0.0

        # Create image array
        image = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light gray background

        # Draw a simple pendulum representation
        center_x, center_y = width // 2, height // 4
        length = min(width, height) // 3

        # Calculate pendulum end position
        end_x = int(center_x + length * math.sin(joint_pos))
        end_y = int(center_y + length * math.cos(joint_pos))

        # Draw pendulum rod (simple line)
        self._draw_line(image, (center_x, center_y), (end_x, end_y), (50, 50, 50))

        # Draw pivot point
        self._draw_circle(image, (center_x, center_y), 5, (100, 100, 100))

        # Draw pendulum mass
        self._draw_circle(image, (end_x, end_y), 10, (200, 100, 100))

        # Add angle text
        angle_deg = math.degrees(joint_pos)
        self._draw_text(image, f"Angle: {angle_deg:.1f}°", (10, height - 30))

        return image

    def _draw_line(self, image, start, end, color):
        """Draw a simple line on the image."""
        x1, y1 = start
        x2, y2 = end

        # Simple line drawing using Bresenham's algorithm (simplified)
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1

        if dx > dy:
            err = dx / 2.0
            while x != x2:
                if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                    image[y, x] = color
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y2:
                if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                    image[y, x] = color
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy

    def _draw_circle(self, image, center, radius, color):
        """Draw a simple filled circle on the image."""
        cx, cy = center
        for y in range(max(0, cy - radius), min(image.shape[0], cy + radius + 1)):
            for x in range(max(0, cx - radius), min(image.shape[1], cx + radius + 1)):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius**2:
                    image[y, x] = color

    def _draw_text(self, image, text, position):
        """Draw simple text on the image."""
        # Simple text rendering - just place colored pixels
        x, y = position
        for i, char in enumerate(text[:20]):  # Limit text length
            char_x = x + i * 8
            if char_x + 8 < image.shape[1] and y + 10 < image.shape[0]:
                # Draw a simple character representation
                if char.isdigit() or char.isalpha() or char in ".:-°":
                    image[y : y + 8, char_x : char_x + 6] = [50, 50, 50]

    def render_ascii(self, width: int = 60, height: int = 20) -> str:
        """Render ASCII art visualization of the simulation state.

        Args:
            width: Character width of the ASCII canvas (default: 60).
            height: Character height of the ASCII canvas (default: 20).

        Returns:
            Multi-line string containing ASCII art visualization with:
            - Pendulum visualization (for single-joint models)
            - Current angle in degrees
            - Current simulation time

        Raises:
            RuntimeError: If simulation not initialized.

        Note:
            This is a simplified visualization primarily designed for pendulum-like
            systems. For complex multi-body systems, use render_frame() instead.
            The rendering shows: '+' for pivot, 'O' for mass, '|' for rod.
        """
        model, data = self._require_sim()

        # Get first joint position for ASCII art
        joint_pos = data.qpos[0] if model.nq > 0 and len(data.qpos) > 0 else 0.0

        import math

        # Create ASCII grid
        grid = [[" " for _ in range(width)] for _ in range(height)]

        # Draw pendulum in ASCII
        center_x, center_y = width // 2, height // 4
        length = min(width // 2, height // 2)

        # Calculate pendulum end position
        end_x = int(center_x + length * math.sin(joint_pos))
        end_y = int(center_y + length * math.cos(joint_pos))

        # Ensure positions are within bounds
        end_x = max(0, min(width - 1, end_x))
        end_y = max(0, min(height - 1, end_y))

        # Draw pendulum rod
        self._draw_ascii_line(grid, (center_x, center_y), (end_x, end_y), "|")

        # Draw pivot and mass
        if 0 <= center_y < height and 0 <= center_x < width:
            grid[center_y][center_x] = "+"
        if 0 <= end_y < height and 0 <= end_x < width:
            grid[end_y][end_x] = "O"

        # Convert grid to string
        result = "\n".join("".join(row) for row in grid)
        result += f"\nAngle: {math.degrees(joint_pos):.1f}°"
        result += f"\nTime: {self.data.time:.2f}s"

        return result

    def _draw_ascii_line(self, grid, start, end, char):
        """Draw a line in ASCII grid."""
        x1, y1 = start
        x2, y2 = end

        # Simple line drawing
        steps = max(abs(x2 - x1), abs(y2 - y1))
        if steps == 0:
            return

        for i in range(steps + 1):
            t = i / steps
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))

            if 0 <= y < len(grid) and 0 <= x < len(grid[0]):
                grid[y][x] = char


__all__ = ["MuJoCoSimulation"]

"""MuJoCo simulation wrapper providing physics simulation functionality."""

import logging
import uuid
import numpy as np
from typing import Dict, Any, List
import mujoco

logger = logging.getLogger("mujoco_mcp.simulation")


class MuJoCoSimulation:
    """Basic MuJoCo simulation class providing core functionality."""
    
    def __init__(self, model_xml: str | None = None, model_path: str | None = None):
        """Initialize MuJoCo simulation."""
        self.model = None
        self.data = None
        self.sim_id = str(uuid.uuid4())
        self._initialized = False
        
        if model_xml:
            self.load_from_xml_string(model_xml)
        elif model_path:
            self.load_from_file(model_path)
    
    def load_from_xml_string(self, model_xml: str):
        """Load model from XML string."""
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
        """Load model from file."""
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self._initialized = True
        logger.info(f"Loaded model from file: {model_path}, sim_id: {self.sim_id}")
    
    def is_initialized(self) -> bool:
        """Check if simulation is initialized."""
        return self._initialized
    
    def step(self, num_steps: int = 1):
        """Step simulation forward."""
        if not self._initialized:
            raise RuntimeError("Simulation not initialized")
        
        for _ in range(num_steps):
            mujoco.mj_step(self.model, self.data)
    
    def reset(self):
        """Reset simulation to initial state."""
        if not self._initialized:
            raise RuntimeError("Simulation not initialized")
        
        mujoco.mj_resetData(self.model, self.data)
    
    def get_joint_positions(self) -> np.ndarray:
        """Get current joint positions."""
        if not self._initialized:
            raise RuntimeError("Simulation not initialized")
        return self.data.qpos.copy()
    
    def get_joint_velocities(self) -> np.ndarray:
        """Get current joint velocities."""
        if not self._initialized:
            raise RuntimeError("Simulation not initialized")
        return self.data.qvel.copy()
    
    def set_joint_positions(self, positions: List[float]):
        """Set joint positions."""
        if not self._initialized:
            raise RuntimeError("Simulation not initialized")
        self.data.qpos[:] = positions
        mujoco.mj_forward(self.model, self.data)
    
    def set_joint_velocities(self, velocities: List[float]):
        """Set joint velocities."""
        if not self._initialized:
            raise RuntimeError("Simulation not initialized")
        self.data.qvel[:] = velocities
    
    def apply_control(self, control: List[float]):
        """Apply control inputs."""
        if not self._initialized:
            raise RuntimeError("Simulation not initialized")
        self.data.ctrl[:] = control
    
    def get_sensor_data(self) -> Dict[str, List[float]]:
        """Get sensor readings."""
        if not self._initialized:
            raise RuntimeError("Simulation not initialized")
        
        sensor_data = {}
        for i in range(self.model.nsensor):
            name = self.model.sensor(i).name
            sensor_data[name] = self.data.sensordata[i:i+1].tolist()
        return sensor_data
    
    def get_rigid_body_states(self) -> Dict[str, Dict[str, List[float]]]:
        """Get rigid body states."""
        if not self._initialized:
            raise RuntimeError("Simulation not initialized")
        
        body_states = {}
        for i in range(self.model.nbody):
            name = self.model.body(i).name
            if name:  # Skip unnamed bodies
                pos = self.data.xpos[i].tolist()
                quat = self.data.xquat[i].tolist()
                body_states[name] = {
                    "position": pos,
                    "orientation": quat
                }
        return body_states
    
    def get_time(self) -> float:
        """Get simulation time."""
        if not self._initialized:
            return 0.0
        return self.data.time
    
    def get_timestep(self) -> float:
        """Get simulation timestep."""
        if not self._initialized:
            return 0.0
        return self.model.opt.timestep
    
    def get_num_joints(self) -> int:
        """Get number of joints."""
        if not self._initialized:
            return 0
        return self.model.nq
    
    def get_num_actuators(self) -> int:
        """Get number of actuators."""
        if not self._initialized:
            return 0
        return self.model.nu
    
    def get_joint_names(self) -> List[str]:
        """Get joint names."""
        if not self._initialized:
            return []
        
        names = []
        for i in range(self.model.njnt):
            names.append(self.model.joint(i).name)
        return names
    
    def get_model_name(self) -> str:
        """Get model name."""
        if not self._initialized:
            return ""
        return self.model.meta.model_name or "unnamed"

    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if not self._initialized:
            raise RuntimeError("Simulation not initialized")
        
        return {
            "nq": self.model.nq,      # number of generalized coordinates
            "nv": self.model.nv,      # number of degrees of freedom
            "nbody": self.model.nbody,  # number of bodies
            "njoint": self.model.njnt,  # number of joints
            "ngeom": self.model.ngeom,  # number of geoms
            "nsensor": self.model.nsensor,  # number of sensors
            "nu": self.model.nu,      # number of actuators
            "timestep": self.model.opt.timestep
        }
    
    def render_frame(self, width: int = 640, height: int = 480, 
                    camera_id: int = -1, scene_option=None) -> np.ndarray:
        """Render a frame from the simulation."""
        if not self._initialized:
            raise RuntimeError("Simulation not initialized")
        
        try:
            # Create a renderer
            renderer = mujoco.Renderer(self.model, height=height, width=width)
            
            # Update scene
            renderer.update_scene(self.data, camera=camera_id, scene_option=scene_option)
            
            # Render and return RGB array
            image = renderer.render()
            
            return image
            
        except Exception as e:
            logger.warning(f"Hardware rendering failed: {e}, falling back to software rendering")
            # Fallback to software rendering
            return self._render_software_fallback(width, height)
    
    def _render_software_fallback(self, width: int, height: int) -> np.ndarray:
        """Fallback software rendering when hardware rendering fails."""
        # Create a simple visualization using simulation state
        # This is a placeholder that creates a visual representation of the pendulum
        
        import math
        
        # Get joint positions for visualization
        if self.model.nq > 0:
            joint_pos = self.data.qpos[0] if len(self.data.qpos) > 0 else 0.0
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
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
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
                    image[y:y+8, char_x:char_x+6] = [50, 50, 50]
    
    def render_ascii(self, width: int = 60, height: int = 20) -> str:
        """Render ASCII art representation of the simulation."""
        if not self._initialized:
            raise RuntimeError("Simulation not initialized")
        
        # Get joint position for ASCII art
        if self.model.nq > 0:
            joint_pos = self.data.qpos[0] if len(self.data.qpos) > 0 else 0.0
        else:
            joint_pos = 0.0
        
        import math
        
        # Create ASCII grid
        grid = [[' ' for _ in range(width)] for _ in range(height)]
        
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
        self._draw_ascii_line(grid, (center_x, center_y), (end_x, end_y), '|')
        
        # Draw pivot and mass
        if 0 <= center_y < height and 0 <= center_x < width:
            grid[center_y][center_x] = '+'
        if 0 <= end_y < height and 0 <= end_x < width:
            grid[end_y][end_x] = 'O'
        
        # Convert grid to string
        result = '\n'.join(''.join(row) for row in grid)
        result += f'\nAngle: {math.degrees(joint_pos):.1f}°'
        result += f'\nTime: {self.data.time:.2f}s'
        
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

__all__ = ['MuJoCoSimulation']
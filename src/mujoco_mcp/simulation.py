"""MuJoCo simulation wrapper providing physics simulation functionality."""

import logging
import uuid
import numpy as np
from typing import Dict, Any, List, Optional
import mujoco

logger = logging.getLogger("mujoco_mcp.simulation")


class MuJoCoSimulation:
    """Basic MuJoCo simulation class providing core functionality."""
    
    def __init__(self, model_xml: Optional[str] = None, model_path: Optional[str] = None):
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

__all__ = ['MuJoCoSimulation']
#!/usr/bin/env python3
"""
Robot Controller for MuJoCo MCP
Provides full robot control capabilities via MCP protocol
"""

import numpy as np
from typing import Dict, Any, List
import mujoco
import time

class RobotController:
    """Advanced robot control interface for MuJoCo"""

    def __init__(self):
        self.models = {}
        self.data = {}
        self.controllers = {}

    def load_robot(self, robot_type: str, robot_id: str = None) -> Dict[str, Any]:
        """Load a robot model into the simulation"""
        if robot_id is None:
            robot_id = f"{robot_type}_{int(time.time())}"

        # Robot XML definitions
        robot_xmls = {
            "arm": self._get_arm_robot_xml(),
            "gripper": self._get_gripper_robot_xml(),
            "mobile": self._get_mobile_robot_xml(),
            "humanoid": self._get_humanoid_robot_xml()
        }

        if robot_type not in robot_xmls:
            return {"error": f"Unknown robot type: {robot_type}"}

        xml = robot_xmls[robot_type]

        try:
            model = mujoco.MjModel.from_xml_string(xml)
            data = mujoco.MjData(model)

            self.models[robot_id] = model
            self.data[robot_id] = data
            self.controllers[robot_id] = {
                "type": robot_type,
                "control_mode": "position",
                "target_positions": np.zeros(model.nu),
                "target_velocities": np.zeros(model.nu),
                "target_torques": np.zeros(model.nu)
            }

            return {
                "robot_id": robot_id,
                "robot_type": robot_type,
                "num_joints": model.nu,
                "num_sensors": model.nsensor,
                "joint_names": [model.joint(i).name for i in range(model.njnt)],
                "actuator_names": [model.actuator(i).name for i in range(model.nu)],
                "status": "loaded"
            }

        except Exception as e:
            return {"error": str(e)}

    def set_joint_positions(self, robot_id: str, positions: List[float]) -> Dict[str, Any]:
        """Set target joint positions for the robot"""
        if robot_id not in self.models:
            return {"error": f"Robot {robot_id} not found"}

        model = self.models[robot_id]
        data = self.data[robot_id]
        controller = self.controllers[robot_id]

        if len(positions) != model.nu:
            return {"error": f"Expected {model.nu} positions, got {len(positions)}"}

        # Set target positions
        controller["target_positions"] = np.array(positions)
        controller["control_mode"] = "position"

        # Apply position control
        data.ctrl[:] = positions

        return {
            "robot_id": robot_id,
            "positions_set": positions,
            "control_mode": "position",
            "status": "success"
        }

    def set_joint_velocities(self, robot_id: str, velocities: List[float]) -> Dict[str, Any]:
        """Set target joint velocities for the robot"""
        if robot_id not in self.models:
            return {"error": f"Robot {robot_id} not found"}

        model = self.models[robot_id]
        data = self.data[robot_id]
        controller = self.controllers[robot_id]

        if len(velocities) != model.nu:
            return {"error": f"Expected {model.nu} velocities, got {len(velocities)}"}

        # Set target velocities
        controller["target_velocities"] = np.array(velocities)
        controller["control_mode"] = "velocity"

        # Apply velocity control (simplified PD controller)
        kp = 100.0  # Position gain
        kv = 10.0   # Velocity gain

        for i in range(model.nu):
            error_vel = velocities[i] - data.qvel[i]
            data.ctrl[i] = kv * error_vel

        return {
            "robot_id": robot_id,
            "velocities_set": velocities,
            "control_mode": "velocity",
            "status": "success"
        }

    def set_joint_torques(self, robot_id: str, torques: List[float]) -> Dict[str, Any]:
        """Set joint torques for direct force control"""
        if robot_id not in self.models:
            return {"error": f"Robot {robot_id} not found"}

        model = self.models[robot_id]
        data = self.data[robot_id]
        controller = self.controllers[robot_id]

        if len(torques) != model.nu:
            return {"error": f"Expected {model.nu} torques, got {len(torques)}"}

        # Set torques directly
        controller["target_torques"] = np.array(torques)
        controller["control_mode"] = "torque"

        data.ctrl[:] = torques

        return {
            "robot_id": robot_id,
            "torques_set": torques,
            "control_mode": "torque",
            "status": "success"
        }

    def get_robot_state(self, robot_id: str) -> Dict[str, Any]:
        """Get complete robot state including positions, velocities, and sensors"""
        if robot_id not in self.models:
            return {"error": f"Robot {robot_id} not found"}

        model = self.models[robot_id]
        data = self.data[robot_id]
        controller = self.controllers[robot_id]

        # Get joint positions and velocities
        joint_positions = data.qpos[:model.nq].tolist()
        joint_velocities = data.qvel[:model.nv].tolist()

        # Get actuator forces
        actuator_forces = data.ctrl[:model.nu].tolist()

        # Get sensor data if available
        sensor_data = {}
        if model.nsensor > 0:
            sensor_data = {
                "sensor_values": data.sensordata.tolist(),
                "sensor_names": [model.sensor(i).name for i in range(model.nsensor)]
            }

        # Get end-effector position (if applicable)
        ee_pos = None
        ee_orient = None
        if robot_id in ["arm", "humanoid"]:
            # Get end-effector body id (last body)
            ee_body_id = model.nbody - 1
            ee_pos = data.xpos[ee_body_id].tolist()
            ee_orient = data.xquat[ee_body_id].tolist()

        return {
            "robot_id": robot_id,
            "robot_type": controller["type"],
            "control_mode": controller["control_mode"],
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "actuator_forces": actuator_forces,
            "target_positions": controller["target_positions"].tolist(),
            "target_velocities": controller["target_velocities"].tolist(),
            "target_torques": controller["target_torques"].tolist(),
            "end_effector": {
                "position": ee_pos,
                "orientation": ee_orient
            } if ee_pos else None,
            "sensors": sensor_data,
            "simulation_time": data.time
        }

    def step_robot(self, robot_id: str, steps: int = 1) -> Dict[str, Any]:
        """Step the robot simulation forward"""
        if robot_id not in self.models:
            return {"error": f"Robot {robot_id} not found"}

        model = self.models[robot_id]
        data = self.data[robot_id]

        try:
            for _ in range(steps):
                mujoco.mj_step(model, data)

            return {
                "robot_id": robot_id,
                "steps_completed": steps,
                "simulation_time": data.time,
                "status": "success"
            }
        except Exception as e:
            return {"error": str(e)}

    def execute_trajectory(self, robot_id: str, trajectory: List[List[float]],
                          time_steps: int = 10) -> Dict[str, Any]:
        """Execute a trajectory of joint positions"""
        if robot_id not in self.models:
            return {"error": f"Robot {robot_id} not found"}

        results = []
        for waypoint in trajectory:
            # Set positions
            self.set_joint_positions(robot_id, waypoint)

            # Step simulation
            self.step_robot(robot_id, time_steps)

            # Get state
            state = self.get_robot_state(robot_id)
            results.append({
                "waypoint": waypoint,
                "achieved_positions": state["joint_positions"],
                "time": state["simulation_time"]
            })

        return {
            "robot_id": robot_id,
            "trajectory_executed": True,
            "num_waypoints": len(trajectory),
            "results": results,
            "status": "success"
        }

    def reset_robot(self, robot_id: str) -> Dict[str, Any]:
        """Reset robot to initial configuration"""
        if robot_id not in self.models:
            return {"error": f"Robot {robot_id} not found"}

        model = self.models[robot_id]
        data = self.data[robot_id]

        # Reset simulation
        mujoco.mj_resetData(model, data)

        # Reset controller
        self.controllers[robot_id]["target_positions"] = np.zeros(model.nu)
        self.controllers[robot_id]["target_velocities"] = np.zeros(model.nu)
        self.controllers[robot_id]["target_torques"] = np.zeros(model.nu)

        return {
            "robot_id": robot_id,
            "status": "reset",
            "simulation_time": 0.0
        }

    def _get_arm_robot_xml(self) -> str:
        """Get XML for a simple robot arm"""
        return """
        <mujoco model="robot_arm">
            <option gravity="0 0 -9.81" timestep="0.002"/>

            <worldbody>
                <body name="base">
                    <geom type="box" size="0.1 0.1 0.05" rgba="0.5 0.5 0.5 1"/>
                    <body name="link1" pos="0 0 0.1">
                        <joint name="joint1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
                        <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.2" rgba="0.8 0.2 0.2 1"/>
                        <body name="link2" pos="0 0 0.2">
                            <joint name="joint2" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
                            <geom type="capsule" size="0.025" fromto="0 0 0 0.15 0 0" rgba="0.2 0.8 0.2 1"/>
                            <body name="link3" pos="0.15 0 0">
                                <joint name="joint3" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
                                <geom type="capsule" size="0.02" fromto="0 0 0 0.1 0 0" rgba="0.2 0.2 0.8 1"/>
                                <body name="end_effector" pos="0.1 0 0">
                                    <geom type="sphere" size="0.02" rgba="1 0.5 0 1"/>
                                    <site name="ee_site" pos="0 0 0" size="0.01"/>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </worldbody>

            <actuator>
                <motor name="motor1" joint="joint1" gear="50"/>
                <motor name="motor2" joint="joint2" gear="50"/>
                <motor name="motor3" joint="joint3" gear="30"/>
            </actuator>

            <sensor>
                <jointpos name="joint1_pos" joint="joint1"/>
                <jointpos name="joint2_pos" joint="joint2"/>
                <jointpos name="joint3_pos" joint="joint3"/>
                <jointvel name="joint1_vel" joint="joint1"/>
                <jointvel name="joint2_vel" joint="joint2"/>
                <jointvel name="joint3_vel" joint="joint3"/>
            </sensor>
        </mujoco>
        """

    def _get_gripper_robot_xml(self) -> str:
        """Get XML for a simple gripper"""
        return """
        <mujoco model="gripper">
            <option gravity="0 0 -9.81" timestep="0.002"/>

            <worldbody>
                <body name="palm">
                    <geom type="box" size="0.04 0.06 0.01" rgba="0.5 0.5 0.5 1"/>
                    <body name="finger1" pos="0.03 0 0">
                        <joint name="finger1_joint" type="slide" axis="1 0 0" range="0 0.04"/>
                        <geom type="box" size="0.01 0.03 0.005" rgba="0.8 0.2 0.2 1"/>
                    </body>
                    <body name="finger2" pos="-0.03 0 0">
                        <joint name="finger2_joint" type="slide" axis="-1 0 0" range="0 0.04"/>
                        <geom type="box" size="0.01 0.03 0.005" rgba="0.2 0.8 0.2 1"/>
                    </body>
                </body>
            </worldbody>

            <actuator>
                <motor name="finger1_motor" joint="finger1_joint" gear="10"/>
                <motor name="finger2_motor" joint="finger2_joint" gear="10"/>
            </actuator>
        </mujoco>
        """

    def _get_mobile_robot_xml(self) -> str:
        """Get XML for a simple mobile robot"""
        return """
        <mujoco model="mobile_robot">
            <option gravity="0 0 -9.81" timestep="0.002"/>

            <worldbody>
                <body name="chassis" pos="0 0 0.1">
                    <joint name="x_joint" type="slide" axis="1 0 0" range="-10 10"/>
                    <joint name="y_joint" type="slide" axis="0 1 0" range="-10 10"/>
                    <joint name="theta_joint" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
                    <geom type="box" size="0.2 0.15 0.05" rgba="0.3 0.3 0.8 1"/>
                    <geom name="wheel1" pos="0.15 0.1 -0.05" type="cylinder" size="0.04 0.02" euler="1.57 0 0" rgba="0.2 0.2 0.2 1"/>
                    <geom name="wheel2" pos="0.15 -0.1 -0.05" type="cylinder" size="0.04 0.02" euler="1.57 0 0" rgba="0.2 0.2 0.2 1"/>
                    <geom name="wheel3" pos="-0.15 0.1 -0.05" type="cylinder" size="0.04 0.02" euler="1.57 0 0" rgba="0.2 0.2 0.2 1"/>
                    <geom name="wheel4" pos="-0.15 -0.1 -0.05" type="cylinder" size="0.04 0.02" euler="1.57 0 0" rgba="0.2 0.2 0.2 1"/>
                </body>
            </worldbody>

            <actuator>
                <motor name="x_motor" joint="x_joint" gear="100"/>
                <motor name="y_motor" joint="y_joint" gear="100"/>
                <motor name="theta_motor" joint="theta_joint" gear="50"/>
            </actuator>
        </mujoco>
        """

    def _get_humanoid_robot_xml(self) -> str:
        """Get XML for a simple humanoid robot"""
        return """
        <mujoco model="simple_humanoid">
            <option gravity="0 0 -9.81" timestep="0.002"/>

            <worldbody>
                <body name="torso" pos="0 0 1.0">
                    <geom type="box" size="0.1 0.05 0.2" rgba="0.5 0.5 0.5 1"/>

                    <!-- Right Arm -->
                    <body name="right_shoulder" pos="0.15 0 0.1">
                        <joint name="r_shoulder_pitch" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
                        <geom type="capsule" size="0.03" fromto="0 0 0 0.2 0 0" rgba="0.8 0.2 0.2 1"/>
                        <body name="right_elbow" pos="0.2 0 0">
                            <joint name="r_elbow" type="hinge" axis="0 1 0" range="0 2.0"/>
                            <geom type="capsule" size="0.025" fromto="0 0 0 0.15 0 0" rgba="0.8 0.4 0.4 1"/>
                        </body>
                    </body>

                    <!-- Left Arm -->
                    <body name="left_shoulder" pos="-0.15 0 0.1">
                        <joint name="l_shoulder_pitch" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
                        <geom type="capsule" size="0.03" fromto="0 0 0 -0.2 0 0" rgba="0.2 0.2 0.8 1"/>
                        <body name="left_elbow" pos="-0.2 0 0">
                            <joint name="l_elbow" type="hinge" axis="0 1 0" range="0 2.0"/>
                            <geom type="capsule" size="0.025" fromto="0 0 0 -0.15 0 0" rgba="0.4 0.4 0.8 1"/>
                        </body>
                    </body>
                </body>
            </worldbody>

            <actuator>
                <motor name="r_shoulder_motor" joint="r_shoulder_pitch" gear="50"/>
                <motor name="r_elbow_motor" joint="r_elbow" gear="30"/>
                <motor name="l_shoulder_motor" joint="l_shoulder_pitch" gear="50"/>
                <motor name="l_elbow_motor" joint="l_elbow" gear="30"/>
            </actuator>
        </mujoco>
        """

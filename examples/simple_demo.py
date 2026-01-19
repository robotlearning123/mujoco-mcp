#!/usr/bin/env python3
"""
MuJoCo Simplified Demo Script
Demonstrates basic MuJoCo functionality

Usage:
python simple_demo.py
"""

import numpy as np
import mujoco
import argparse
import logging
from typing import List

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("mujoco_simple_demo")

# Example MuJoCo model XML (simple manipulator and some objects)
EXAMPLE_MODEL_XML = """
<mujoco>
    <worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
        <geom name="floor" type="plane" size="5 5 0.1" rgba=".9 .9 .9 1"/>

        <!-- Simple robot -->
        <body name="robot1" pos="0 0 0.5">
            <joint name="robot1_base_rot" type="ball"/>
            <geom name="robot1_base" type="cylinder" size="0.1 0.1" rgba="0.7 0.7 0.7 1"/>
            <body name="robot1_arm" pos="0 0 0.1">
                <joint name="robot1_shoulder" axis="0 1 0" range="-180 180"/>
                <geom name="robot1_arm_geom" type="capsule" size="0.05"
                      fromto="0 0 0 0.5 0 0" rgba="0.7 0.7 0.7 1"/>
                <body name="robot1_forearm" pos="0.5 0 0">
                    <joint name="robot1_elbow" axis="0 1 0" range="-90 90"/>
                    <geom name="robot1_forearm_geom" type="capsule" size="0.04"
                          fromto="0 0 0 0.5 0 0" rgba="0.7 0.7 0.7 1"/>
                    <body name="robot1_wrist" pos="0.5 0 0">
                        <joint name="robot1_wrist_rot" axis="1 0 0" range="-180 180"/>
                        <geom name="robot1_wrist_geom" type="sphere" size="0.05"
                              rgba="0.5 0.5 0.5 1"/>
                        <site name="robot1_grasp_site" pos="0 0 0.1" size="0.02"/>
                    </body>
                </body>
            </body>
        </body>

        <!-- Objects in scene -->
        <body name="red_cube" pos="1 0 0.1">
            <joint type="free"/>
            <geom type="box" size="0.1 0.1 0.1" rgba="1 0 0 1"/>
        </body>

        <body name="blue_cube" pos="0 1 0.1">
            <joint type="free"/>
            <geom type="box" size="0.1 0.1 0.1" rgba="0 0 1 1"/>
        </body>

        <body name="green_sphere" pos="-1 0 0.1">
            <joint type="free"/>
            <geom type="sphere" size="0.1" rgba="0 1 0 1"/>
        </body>
    </worldbody>

    <actuator>
        <motor name="robot1_base_rot_motor" joint="robot1_base_rot"/>
        <motor name="robot1_shoulder_motor" joint="robot1_shoulder"/>
        <motor name="robot1_elbow_motor" joint="robot1_elbow"/>
        <motor name="robot1_wrist_motor" joint="robot1_wrist_rot"/>
    </actuator>
</mujoco>
"""


class MuJoCoSimulation:
    """MuJoCo simulation class"""

    def __init__(self, model_xml: str):
        """Initialize MuJoCo simulation"""
        logger.info("Initializing MuJoCo simulation...")
        self.model = mujoco.MjModel.from_xml_string(model_xml)
        self.data = mujoco.MjData(self.model)

        # Find objects and robot parts
        self._find_bodies()

        # Create rendering context
        self.create_renderer()

        logger.info("Simulation initialization complete")

    def _find_bodies(self):
        """Find objects and robot parts in the scene"""
        self.body_names = {}
        self.robot_parts = {}

        # Get all body indices and names
        for i in range(self.model.nbody):
            name = self.model.body(i).name
            if name:
                self.body_names[name] = i

                # Identify robot parts
                if name.startswith("robot1_"):
                    self.robot_parts[name] = i

        logger.info(f"Found objects: {list(self.body_names.keys())}")

    def create_renderer(self):
        """Create rendering context"""
        try:
            # Create an offscreen rendering context
            self.renderer = mujoco.Renderer(self.model, 640, 480)
            logger.info("Rendering context created successfully")
        except Exception as e:
            logger.exception(f"Failed to create rendering context: {str(e)}")
            self.renderer = None

    def step(self, num_steps: int = 1):
        """Step simulation"""
        for _ in range(num_steps):
            mujoco.mj_step(self.model, self.data)

    def reset(self):
        """Reset simulation"""
        mujoco.mj_resetData(self.model, self.data)
        logger.info("Simulation reset")

    def render(self):
        """Render current scene"""
        if self.renderer:
            self.renderer.update_scene(self.data)
            return self.renderer.render()
        return None

    def set_joint_positions(self, positions: List[float]):
        """Set joint positions"""
        for i, pos in enumerate(positions):
            if i < self.model.nq:
                self.data.qpos[i] = pos
        mujoco.mj_forward(self.model, self.data)

    def get_joint_positions(self) -> List[float]:
        """Get joint positions"""
        return self.data.qpos.copy()

    def apply_control(self, control: List[float]):
        """Apply control signal"""
        for i, ctrl in enumerate(control):
            if i < self.model.nu:
                self.data.ctrl[i] = ctrl

    def get_body_position(self, body_name: str) -> List[float] | None:
        """Get body position"""
        if body_name in self.body_names:
            body_id = self.body_names[body_name]
            return self.data.body(body_id).xpos.copy()
        return None

    def apply_force(self, body_name: str, force: List[float]):
        """Apply force to body"""
        if body_name in self.body_names:
            body_id = self.body_names[body_name]
            self.data.xfrc_applied[body_id, :3] = force
            logger.info(f"Applying force {force} to {body_name}")

    def move_robot_arm(self, shoulder_angle: float, elbow_angle: float, wrist_angle: float):
        """Move robot arm"""
        # Find indices of relevant joints
        shoulder_idx = -1
        elbow_idx = -1
        wrist_idx = -1

        for i in range(self.model.njnt):
            name = self.model.joint(i).name
            if name == "robot1_shoulder":
                shoulder_idx = self.model.joint(i).qposadr[0]
            elif name == "robot1_elbow":
                elbow_idx = self.model.joint(i).qposadr[0]
            elif name == "robot1_wrist_rot":
                wrist_idx = self.model.joint(i).qposadr[0]

        # Set joint angles
        if shoulder_idx >= 0:
            self.data.qpos[shoulder_idx] = np.deg2rad(shoulder_angle)
        if elbow_idx >= 0:
            self.data.qpos[elbow_idx] = np.deg2rad(elbow_angle)
        if wrist_idx >= 0:
            self.data.qpos[wrist_idx] = np.deg2rad(wrist_angle)

        # Update simulation
        mujoco.mj_forward(self.model, self.data)
        logger.info(
            f"Moving robot arm to shoulder={shoulder_angle}°, elbow={elbow_angle}°, wrist={wrist_angle}°"
        )

    def move_to_target(self, target_pos: List[float], steps: int = 100):
        """Move robot arm to target position"""
        # This is a very simplified motion planner, real scenarios require more complex inverse kinematics

        # Get end effector position
        wrist_pos = self.get_body_position("robot1_wrist")
        if wrist_pos is None:
            logger.error("Cannot find robot wrist")
            return

        # Calculate direction vector to target
        direction = np.array(target_pos) - wrist_pos
        distance = np.linalg.norm(direction)

        logger.info(f"Starting move to target position {target_pos}, distance {distance:.2f}")

        # Move gradually
        for i in range(steps):
            # Get current joint angles
            qpos = self.get_joint_positions()

            # Simple gradient-based control
            # Note: This is not true inverse kinematics, just a simplified demonstration
            shoulder_idx = -1
            elbow_idx = -1

            for j in range(self.model.njnt):
                name = self.model.joint(j).name
                if name == "robot1_shoulder":
                    shoulder_idx = self.model.joint(j).qposadr[0]
                elif name == "robot1_elbow":
                    elbow_idx = self.model.joint(j).qposadr[0]

            # Use simple heuristics to adjust joint angles
            wrist_pos = self.get_body_position("robot1_wrist")
            direction = np.array(target_pos) - wrist_pos

            # Adjust shoulder angle
            if shoulder_idx >= 0:
                qpos[shoulder_idx] += np.sign(direction[0]) * 0.01

            # Adjust elbow angle
            if elbow_idx >= 0:
                qpos[elbow_idx] += np.sign(direction[2]) * 0.01

            # Update position
            self.set_joint_positions(qpos)

            # Step simulation
            self.step(5)

            # Check if close to target
            wrist_pos = self.get_body_position("robot1_wrist")
            distance = np.linalg.norm(np.array(target_pos) - wrist_pos)

            if distance < 0.2:
                logger.info(f"Close to target position, remaining distance {distance:.2f}")
                break

            if i % 10 == 0:
                logger.info(f"Moving... step {i}, remaining distance {distance:.2f}")

    def grasp_object(self, object_name: str):
        """Grasp object (simplified version)"""
        # Get object position
        object_pos = self.get_body_position(object_name)
        if object_pos is None:
            logger.error(f"Cannot find object {object_name}")
            return

        logger.info(f"Attempting to grasp {object_name} at position {object_pos}")

        # Move to above object position
        target_pos = object_pos.copy()
        target_pos[2] += 0.2  # Slightly above object

        self.move_to_target(target_pos)

        # Simulate grasp operation (in real MuJoCo, this typically involves creating constraints)
        logger.info(f"Grasped {object_name}")


def run_demo():
    """Run demo"""
    logger.info("Starting MuJoCo simplified demo")

    # Create simulation
    sim = MuJoCoSimulation(EXAMPLE_MODEL_XML)

    # Reset simulation
    sim.reset()

    # Display objects in scene
    for name in sim.body_names:
        pos = sim.get_body_position(name)
        if pos is not None:
            logger.info(f"Object {name}: position {pos}")

    # Step 1: Move robot arm
    logger.info("Step 1: Move robot arm")
    sim.move_robot_arm(30, 45, 0)
    sim.step(100)  # Step simulation to apply motion

    # Step 2: Approach red cube
    logger.info("Step 2: Move to red cube")
    red_cube_pos = sim.get_body_position("red_cube")
    if red_cube_pos is not None:
        # Move to above cube
        target_pos = red_cube_pos.copy()
        target_pos[2] += 0.3  # 0.3 units above cube
        sim.move_to_target(target_pos)

    # Step 3: Simulate grasp
    logger.info("Step 3: Grasp red cube")
    sim.grasp_object("red_cube")

    # Step 4: Apply some control signals
    logger.info("Step 4: Apply control signals")
    for _i in range(5):
        # Apply random control signal
        control = np.random.uniform(-1, 1, sim.model.nu)
        sim.apply_control(control.tolist())
        sim.step(20)
        logger.info(f"Applying control {control}")

    # Step 5: Apply force to green sphere
    logger.info("Step 5: Apply force to green sphere")
    sim.apply_force("green_sphere", [10.0, 0.0, 5.0])
    for _ in range(5):
        sim.step(20)
        pos = sim.get_body_position("green_sphere")
        logger.info(f"Green sphere position: {pos}")

    logger.info("Demo complete")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="MuJoCo simplified demo")
    parser.add_argument("--verbose", "-v", action="store_true", help="Output detailed information")
    args = parser.parse_args()

    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run demo
    run_demo()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDemo terminated")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        print("Demo ended")

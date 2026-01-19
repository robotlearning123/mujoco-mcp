#!/usr/bin/env python3
"""
Basic example: Demonstrates how to start and interact with MuJoCo MCP server
"""

import sys
import time
import threading
import logging
from model_context_protocol import MCPClient

# Import mujoco_mcp module
try:
    import mujoco_mcp
except ImportError:
    print("mujoco_mcp module not found. Please ensure the module is installed.")
    print("Try running: pip install -e .")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_example")

# MuJoCo model XML - simple pendulum
PENDULUM_XML = """
<mujoco model="pendulum">
  <option timestep="0.01" />
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="1 1 0.1" rgba=".9 .9 .9 1"/>
    <body name="pendulum" pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.1"/>
      <geom name="pole" type="capsule" size="0.045 0.5" rgba="0 0.7 0.7 1" fromto="0 0 0 0 0 -1"/>
      <site name="tip" pos="0 0 -1" size="0.01"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="torque" joint="hinge" gear="1" ctrllimited="true" ctrlrange="-2.0 2.0"/>
  </actuator>
  <sensor>
    <jointpos name="joint_pos" joint="hinge"/>
    <jointvel name="joint_vel" joint="hinge"/>
  </sensor>
</mujoco>
"""


def start_server():
    """Start MCP server in a separate thread"""
    logger.info("Starting MuJoCo MCP server...")
    server_thread = threading.Thread(
        target=mujoco_mcp.start,
        kwargs={"host": "localhost", "port": 8000, "blocking": True},
        daemon=True,
    )
    server_thread.start()
    time.sleep(1)  # Give server some startup time
    return server_thread


def run_client():
    """Run MCP client example"""
    logger.info("Connecting to MuJoCo MCP server...")
    client = MCPClient("http://localhost:8000")

    # Start simulation
    logger.info("Starting new simulation...")
    result = client.call_tool("start_simulation", {"model_xml": PENDULUM_XML})
    sim_id = result["simulation_id"]
    logger.info(f"Simulation ID: {sim_id}")

    # Get simulation info
    sim_info = client.get_resource("simulation_info", {"simulation_id": sim_id})
    logger.info(f"Simulation info: {sim_info}")

    # Reset simulation
    client.call_tool("reset_simulation", {"simulation_id": sim_id})

    # Run control loop
    logger.info("Starting control loop, running 50 steps...")
    for i in range(50):
        # Get joint positions and velocities
        positions = client.get_resource("joint_positions", {"simulation_id": sim_id})
        velocities = client.get_resource("joint_velocities", {"simulation_id": sim_id})

        # Apply simple control - pull pendulum down
        control = [1.0 * (i % 10)]  # Change direction every 10 steps
        client.call_tool("apply_control", {"simulation_id": sim_id, "control": control})

        # Get sensor data
        sensors = client.get_resource("sensor_data", {"simulation_id": sim_id})

        # Print status
        if i % 5 == 0:  # Print every 5 steps
            logger.info(f"Step {i}: position={positions}, velocity={velocities}, control={control}")
            logger.info(f"Sensors: {sensors}")

        # Step simulation forward
        client.call_tool("step_simulation", {"simulation_id": sim_id, "num_steps": 1})
        time.sleep(0.01)  # Slow down loop slightly for observation

    # Cleanup
    logger.info("Deleting simulation...")
    client.call_tool("delete_simulation", {"simulation_id": sim_id})
    logger.info("Example complete!")


def main():
    """Main program"""
    logger.info("Starting basic MuJoCo MCP example")

    server_thread = start_server()

    try:
        run_client()
    except KeyboardInterrupt:
        logger.info("User interrupt, shutting down...")
    except Exception as e:
        logger.exception(f"Error occurred: {str(e)}")
    finally:
        # Stop server
        logger.info("Stopping server...")
        mujoco_mcp.stop()
        # Wait for server thread to end
        server_thread.join(timeout=2)
        logger.info("Program exiting")


if __name__ == "__main__":
    main()

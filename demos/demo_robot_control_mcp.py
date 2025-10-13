#!/usr/bin/env python3
"""
Full Robot Control Demo via MCP Protocol
Shows complete robot manipulation using Claude Code MCP interface
"""

import asyncio
import json
import sys
import math
from pathlib import Path
from typing import Dict, Any


class RobotControlDemo:
    """Demonstrates full robot control via MCP"""

    def __init__(self):
        self.process = None
        self.request_id = 0

    async def start_server(self):
        """Start the enhanced MCP server with robot control"""
        # Start the robot control MCP server
        self.process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "mujoco_mcp.mcp_server_robot",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=Path(__file__).parent,
        )

        # Initialize MCP connection
        await self.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "robot-control-demo", "version": "1.0.0"},
            },
        )

        # Send initialized notification
        await self.send_notification("notifications/initialized", {})
        print("✅ MCP Robot Control Server initialized")

    async def send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send JSON-RPC request to MCP server"""
        self.request_id += 1
        request = {"jsonrpc": "2.0", "id": self.request_id, "method": method, "params": params}

        request_line = json.dumps(request) + "\n"
        self.process.stdin.write(request_line.encode())
        await self.process.stdin.drain()

        response_line = await asyncio.wait_for(self.process.stdout.readline(), timeout=10.0)
        response = json.loads(response_line.decode())

        if "error" in response:
            raise RuntimeError(f"MCP Error: {response['error']}")

        return response.get("result", {})

    async def send_notification(self, method: str, params: Dict[str, Any]):
        """Send JSON-RPC notification"""
        notification = {"jsonrpc": "2.0", "method": method, "params": params}

        notification_line = json.dumps(notification) + "\n"
        self.process.stdin.write(notification_line.encode())
        await self.process.stdin.drain()

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call an MCP tool and return the result"""
        result = await self.send_request("tools/call", {"name": name, "arguments": arguments})

        # Parse the result
        if result.get("content"):
            text = result["content"][0].get("text", "{}")
            return json.loads(text)
        return result

    async def stop_server(self):
        """Stop the MCP server"""
        if self.process:
            self.process.terminate()
            await asyncio.wait_for(self.process.wait(), timeout=3.0)


async def demonstrate_robot_control():
    """Main demonstration of robot control via MCP"""
    print("🤖 FULL ROBOT CONTROL VIA MCP DEMONSTRATION")
    print("=" * 60)

    demo = RobotControlDemo()

    try:
        # Start server
        print("\n📡 Step 1: Starting MCP Robot Control Server")
        print("-" * 40)
        await demo.start_server()

        # List available tools
        print("\n🔧 Step 2: Discovering Robot Control Tools")
        print("-" * 40)
        tools_result = await demo.send_request("tools/list", {})
        tools = tools_result.get("tools", [])
        print(f"✅ Found {len(tools)} robot control tools:")
        for tool in tools:
            print(f"   📋 {tool['name']}")

        # Get server info
        print("\n📊 Step 3: Server Information")
        print("-" * 40)
        server_info = await demo.call_tool("get_server_info", {})
        print(f"✅ Server: {server_info['name']}")
        print(f"   Version: {server_info['version']}")
        print(f"   Robots: {', '.join(server_info['supported_robots'])}")

        # Load a robot arm
        print("\n🦾 Step 4: Loading Robot Arm")
        print("-" * 40)
        robot_info = await demo.call_tool(
            "load_robot", {"robot_type": "arm", "robot_id": "demo_arm"}
        )
        print(f"✅ Robot loaded: {robot_info['robot_id']}")
        print(f"   Type: {robot_info['robot_type']}")
        print(f"   Joints: {robot_info['num_joints']}")
        print(f"   Actuators: {robot_info['actuator_names']}")

        # Test position control
        print("\n🎯 Step 5: Position Control Test")
        print("-" * 40)
        print("Moving to position [0.5, 1.0, 0.3] radians...")

        position_result = await demo.call_tool(
            "set_joint_positions", {"robot_id": "demo_arm", "positions": [0.5, 1.0, 0.3]}
        )
        print(f"✅ Positions set: {position_result['positions_set']}")

        # Step simulation
        await demo.call_tool("step_robot", {"robot_id": "demo_arm", "steps": 50})

        # Get robot state
        state = await demo.call_tool("get_robot_state", {"robot_id": "demo_arm"})
        print(f"📊 Current joint positions: {state['joint_positions']}")
        print(f"   Simulation time: {state['simulation_time']:.3f}s")

        # Test velocity control
        print("\n⚡ Step 6: Velocity Control Test")
        print("-" * 40)
        print("Setting velocities [0.2, -0.1, 0.15] rad/s...")

        velocity_result = await demo.call_tool(
            "set_joint_velocities", {"robot_id": "demo_arm", "velocities": [0.2, -0.1, 0.15]}
        )
        print(f"✅ Velocities set: {velocity_result['velocities_set']}")

        # Step and check
        await demo.call_tool("step_robot", {"robot_id": "demo_arm", "steps": 30})

        state = await demo.call_tool("get_robot_state", {"robot_id": "demo_arm"})
        print(f"📊 Joint velocities: {state['joint_velocities']}")

        # Execute a trajectory
        print("\n🎪 Step 7: Trajectory Execution")
        print("-" * 40)
        print("Executing circular trajectory...")

        # Create circular trajectory
        trajectory = []
        for i in range(8):
            angle = i * math.pi / 4
            trajectory.append(
                [
                    math.sin(angle) * 0.5,  # Joint 1
                    math.cos(angle) * 0.7 + 0.5,  # Joint 2
                    math.sin(angle * 2) * 0.3,  # Joint 3
                ]
            )

        traj_result = await demo.call_tool(
            "execute_trajectory",
            {"robot_id": "demo_arm", "trajectory": trajectory, "time_steps": 20},
        )

        print(f"✅ Trajectory executed: {traj_result['num_waypoints']} waypoints")
        for i, result in enumerate(traj_result["results"][:3]):  # Show first 3
            print(f"   Waypoint {i + 1}: {result['achieved_positions']}")

        # Test torque control
        print("\n💪 Step 8: Torque Control Test")
        print("-" * 40)
        print("Applying torques [0.5, -0.3, 0.2] Nm...")

        torque_result = await demo.call_tool(
            "set_joint_torques", {"robot_id": "demo_arm", "torques": [0.5, -0.3, 0.2]}
        )
        print(f"✅ Torques applied: {torque_result['torques_set']}")

        # Step and observe
        await demo.call_tool("step_robot", {"robot_id": "demo_arm", "steps": 50})

        state = await demo.call_tool("get_robot_state", {"robot_id": "demo_arm"})
        print(f"📊 Actuator forces: {state['actuator_forces']}")

        # Load and control a gripper
        print("\n🤏 Step 9: Gripper Control")
        print("-" * 40)

        gripper_info = await demo.call_tool(
            "load_robot", {"robot_type": "gripper", "robot_id": "demo_gripper"}
        )
        print(f"✅ Gripper loaded: {gripper_info['robot_id']}")

        # Open gripper
        print("Opening gripper...")
        await demo.call_tool(
            "set_joint_positions",
            {
                "robot_id": "demo_gripper",
                "positions": [0.04, 0.04],  # Max open
            },
        )
        await demo.call_tool("step_robot", {"robot_id": "demo_gripper", "steps": 30})

        # Close gripper
        print("Closing gripper...")
        await demo.call_tool(
            "set_joint_positions",
            {
                "robot_id": "demo_gripper",
                "positions": [0.0, 0.0],  # Closed
            },
        )
        await demo.call_tool("step_robot", {"robot_id": "demo_gripper", "steps": 30})

        gripper_state = await demo.call_tool("get_robot_state", {"robot_id": "demo_gripper"})
        print(f"✅ Gripper positions: {gripper_state['joint_positions']}")

        # Load mobile robot
        print("\n🚗 Step 10: Mobile Robot Control")
        print("-" * 40)

        mobile_info = await demo.call_tool(
            "load_robot", {"robot_type": "mobile", "robot_id": "demo_mobile"}
        )
        print(f"✅ Mobile robot loaded: {mobile_info['robot_id']}")

        # Move in a square pattern
        print("Moving in square pattern...")
        square_trajectory = [
            [1.0, 0.0, 0.0],  # Move forward
            [1.0, 1.0, 1.57],  # Turn left
            [0.0, 1.0, 1.57],  # Move left
            [0.0, 0.0, 3.14],  # Turn around
        ]

        await demo.call_tool(
            "execute_trajectory",
            {"robot_id": "demo_mobile", "trajectory": square_trajectory, "time_steps": 50},
        )
        print("✅ Square pattern completed")

        # Reset robots
        print("\n🔄 Step 11: Reset All Robots")
        print("-" * 40)

        for robot_id in ["demo_arm", "demo_gripper", "demo_mobile"]:
            await demo.call_tool("reset_robot", {"robot_id": robot_id})
            print(f"✅ Reset {robot_id}")

        # Final summary
        print(f"\n{'=' * 60}")
        print("🎉 ROBOT CONTROL DEMONSTRATION COMPLETE")
        print(f"{'=' * 60}")
        print("\n✅ DEMONSTRATED CAPABILITIES:")
        print("   🦾 Robot arm control (3 DOF)")
        print("   🤏 Gripper control (open/close)")
        print("   🚗 Mobile robot navigation")
        print("   🎯 Position control mode")
        print("   ⚡ Velocity control mode")
        print("   💪 Torque control mode")
        print("   🎪 Trajectory execution")
        print("   📊 State feedback")
        print("   🔄 Reset functionality")

        print("\n🔌 HOW TO USE IN CLAUDE CODE:")
        print("   1. Configure MCP server in Claude Desktop")
        print("   2. Ask: 'Load a robot arm'")
        print("   3. Ask: 'Move joint 1 to 0.5 radians'")
        print("   4. Ask: 'Execute a circular trajectory'")
        print("   5. Ask: 'Get current robot state'")
        print("   6. Ask: 'Control gripper to pick up object'")

    except Exception as e:
        print(f"\n💥 Demo failed: {e}")
        import traceback

        traceback.print_exc()

    finally:
        print("\n🧹 Cleaning up...")
        await demo.stop_server()
        print("✅ Server stopped")


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     MUJOCO ROBOT CONTROL VIA MCP - FULL DEMONSTRATION   ║
    ╠══════════════════════════════════════════════════════════╣
    ║  This demo shows complete robot control including:       ║
    ║  • Loading different robot types (arm, gripper, mobile)  ║
    ║  • Position, velocity, and torque control modes         ║
    ║  • Trajectory execution and path following              ║
    ║  • Sensor feedback and state queries                    ║
    ║  • Multi-robot coordination                             ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    try:
        asyncio.run(demonstrate_robot_control())
        print("\n🚀 Full robot control demo completed successfully!")
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Demo crashed: {e}")
        sys.exit(1)

#!/usr/bin/env python3
"""
MCP Motion Control Demo
Demonstrates motion control through MCP interface with Menagerie models

This demo shows how to control various robots through natural language
commands via the MCP interface.
"""

import asyncio
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mujoco_mcp.mcp_server import handle_call_tool, handle_list_tools


async def demo_sequence():
    """Run demo sequences using MCP tools"""

    print("🤖 MuJoCo MCP Motion Control Demo")
    print("=" * 50)

    # List available tools
    print("\n📋 Available MCP tools:")
    tools = await handle_list_tools()
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")

    print("\n" + "=" * 50)
    print("Starting demo sequences...\n")

    # Demo 1: Pendulum
    print("1️⃣ Demo: Simple Pendulum")
    result = await handle_call_tool(
        "create_scene", {"scene_type": "pendulum", "parameters": {"length": 0.6, "mass": 0.5}}
    )
    print(f"   {result[0].text}")
    await asyncio.sleep(2)

    # Get initial state
    result = await handle_call_tool("get_state", {})
    print(f"   Initial state: {result[0].text[:100]}...")

    # Set position
    result = await handle_call_tool(
        "set_joint_positions",
        {
            "positions": [1.57]  # 90 degrees
        },
    )
    print(f"   {result[0].text}")
    await asyncio.sleep(2)

    # Step simulation
    print("   Stepping simulation...")
    for _ in range(5):
        result = await handle_call_tool("step_simulation", {"steps": 100})
        await asyncio.sleep(0.5)

    # Demo 2: Double Pendulum
    print("\n2️⃣ Demo: Double Pendulum")
    result = await handle_call_tool(
        "create_scene",
        {"scene_type": "double_pendulum", "parameters": {"length1": 0.4, "length2": 0.4}},
    )
    print(f"   {result[0].text}")
    await asyncio.sleep(2)

    # Set initial positions
    result = await handle_call_tool(
        "set_joint_positions",
        {
            "positions": [0.785, -0.785]  # 45 and -45 degrees
        },
    )
    print(f"   {result[0].text}")
    await asyncio.sleep(1)

    # Run simulation
    print("   Running chaotic motion...")
    for _ in range(10):
        result = await handle_call_tool("step_simulation", {"steps": 50})
        await asyncio.sleep(0.2)

    # Demo 3: Cart-Pole
    print("\n3️⃣ Demo: Cart-Pole Balance")
    result = await handle_call_tool("create_scene", {"scene_type": "cart_pole"})
    print(f"   {result[0].text}")
    await asyncio.sleep(2)

    # Simple balance control
    print("   Attempting to balance...")
    for _i in range(20):
        # Get current state
        state_result = await handle_call_tool("get_state", {})
        # In real implementation, parse state and apply control
        # For demo, just step
        await handle_call_tool("step_simulation", {"steps": 10})
        await asyncio.sleep(0.1)

    # Demo 4: Robotic Arm (if available)
    print("\n4️⃣ Demo: Robotic Arm")
    result = await handle_call_tool("create_scene", {"scene_type": "robotic_arm"})
    print(f"   {result[0].text}")

    if "Created robotic_arm" in result[0].text:
        await asyncio.sleep(2)

        # Move joints
        print("   Moving arm joints...")
        positions_sequence = [
            [0.0, -0.785, 0.0],  # Position 1
            [0.785, -0.785, 0.785],  # Position 2
            [0.0, -1.57, 0.0],  # Position 3
            [0.0, 0.0, 0.0],  # Home
        ]

        for pos in positions_sequence:
            result = await handle_call_tool("set_joint_positions", {"positions": pos})
            print(f"   Moving to: {pos}")
            await asyncio.sleep(1.5)

    print("\n✅ Demo sequence completed!")


async def interactive_mode():
    """Interactive control through MCP"""
    print("\n🎮 Interactive MCP Control Mode")
    print("=" * 50)
    print("Examples of natural language commands:")
    print("  - Create a pendulum simulation")
    print("  - Set the pendulum angle to 45 degrees")
    print("  - Step the simulation 100 times")
    print("  - Show me the current state")
    print("  - Reset the simulation")
    print("  - Close the viewer")
    print("\nType 'quit' to exit")

    while True:
        try:
            command = input("\n> ").strip()
            if command.lower() == "quit":
                break

            # Use execute_command tool for natural language
            result = await handle_call_tool("execute_command", {"command": command})
            print(result[0].text)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")

    # Close viewer before exiting
    print("\nClosing viewer...")
    await handle_call_tool("close_viewer", {})


async def test_menagerie_models():
    """Test loading Menagerie models through MCP"""
    print("\n🦾 Testing Menagerie Models")
    print("=" * 50)

    # Test commands for loading Menagerie models
    test_commands = [
        "Load Franka Panda robot",
        "Create a Unitree Go2 scene",
        "Show me the Anymal robot",
        "Load Shadow Hand model",
    ]

    for cmd in test_commands:
        print(f"\n📌 Command: {cmd}")
        try:
            result = await handle_call_tool("execute_command", {"command": cmd})
            print(f"   Result: {result[0].text}")

            if "Created" in result[0].text or "Loaded" in result[0].text:
                # Let it run for a bit
                await asyncio.sleep(2)

                # Try to move it
                print("   Testing motion...")
                await handle_call_tool("step_simulation", {"steps": 100})
                await asyncio.sleep(1)
        except Exception as e:
            print(f"   Error: {e}")


async def main():
    """Main entry point"""
    print("🚀 MuJoCo MCP Motion Control Demo")
    print("Choose demo mode:")
    print("1. Run demo sequences")
    print("2. Interactive control")
    print("3. Test Menagerie models")
    print("4. Run all demos")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == "1":
        await demo_sequence()
    elif choice == "2":
        await interactive_mode()
    elif choice == "3":
        await test_menagerie_models()
    elif choice == "4":
        await demo_sequence()
        await test_menagerie_models()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo terminated")

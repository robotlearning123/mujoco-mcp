#!/usr/bin/env python3
"""
Test basic MCP scene creation and control
Tests the built-in scenes without requiring Menagerie
"""

import asyncio
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mujoco_mcp.mcp_server import handle_list_tools, handle_call_tool


async def test_basic_scenes():
    """Test creating and controlling basic scenes"""
    print("🔧 Testing basic scene creation...")

    scenes = ["pendulum", "double_pendulum", "cart_pole"]

    for scene_type in scenes:
        print(f"\n   Testing {scene_type}...")

        # Create scene
        result = await handle_call_tool("create_scene", {"scene_type": scene_type})
        print(f"   Create: {result[0].text}")

        if "successfully" in result[0].text or "Created" in result[0].text:
            await asyncio.sleep(1)

            # Step simulation
            result = await handle_call_tool(
                "step_simulation", {"model_id": scene_type, "steps": 100}
            )
            print(f"   Step: {result[0].text}")

            # Get state
            result = await handle_call_tool("get_state", {"model_id": scene_type})
            print(f"   State: {result[0].text[:100]}...")

            # Reset
            result = await handle_call_tool("reset_simulation", {"model_id": scene_type})
            print(f"   Reset: {result[0].text}")

            await asyncio.sleep(0.5)

    print("\n✅ Basic scene tests completed")


async def test_complete_workflow():
    """Test a complete workflow"""
    print("\n🔧 Testing complete workflow...")

    # Create pendulum
    result = await handle_call_tool("create_scene", {"scene_type": "pendulum"})
    print(f"   {result[0].text}")

    if "successfully" in result[0].text or "Created" in result[0].text:
        # Run simulation for a few steps
        for _i in range(5):
            result = await handle_call_tool(
                "step_simulation", {"model_id": "pendulum", "steps": 20}
            )
            await asyncio.sleep(0.2)

        # Get final state
        result = await handle_call_tool("get_state", {"model_id": "pendulum"})
        print(f"   Final state obtained: {len(result[0].text)} characters")

        # Close viewer
        result = await handle_call_tool("close_viewer", {"model_id": "pendulum"})
        print(f"   {result[0].text}")

        print("   ✅ Complete workflow test passed")
        return True

    print("   ❌ Complete workflow test failed")
    return False


async def main():
    """Run all basic tests"""
    print("🚀 Testing Basic MCP Scenes")
    print("=" * 50)

    # List tools
    tools = await handle_list_tools()
    print(f"Available tools: {[t.name for t in tools]}")

    # Test server info
    result = await handle_call_tool("get_server_info", {})
    print(f"Server info: {result[0].text[:100]}...")

    # Test scenes
    await test_basic_scenes()

    # Test complete workflow
    success = await test_complete_workflow()

    if success:
        print("\n🎉 All basic tests passed!")
    else:
        print("\n⚠️  Some tests had issues")


if __name__ == "__main__":
    asyncio.run(main())

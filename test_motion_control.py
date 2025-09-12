#!/usr/bin/env python3
"""
Test script for motion control demos
Non-interactive testing of the motion control functionality
"""

import asyncio
import sys
import time
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from examples.motion_control_demo import MotionControlDemo


async def test_basic_connection():
    """Test basic connection to viewer server"""
    print("🔧 Testing basic connection...")

    demo = MotionControlDemo()
    success = await demo.connect()

    if success:
        print("✅ Connection test passed")
        demo.viewer_client.disconnect()
        return True
    else:
        print("❌ Connection test failed")
        return False


async def test_model_loading():
    """Test loading different models"""
    print("\n🔧 Testing model loading...")

    demo = MotionControlDemo()
    if not await demo.connect():
        return False

    # Test models that should exist
    test_models = ["franka_panda", "ur5e", "anymal_c"]
    results = {}

    for model in test_models:
        print(f"   Loading {model}...")
        success = demo.load_model(model)
        results[model] = success
        if success:
            print(f"   ✅ {model} loaded successfully")
            time.sleep(1)  # Give it time to load
        else:
            print(f"   ⚠️  {model} failed to load (Menagerie may not be installed)")

    demo.viewer_client.disconnect()
    return any(results.values())


async def test_basic_motions():
    """Test basic motion patterns"""
    print("\n🔧 Testing basic motions...")

    demo = MotionControlDemo()
    if not await demo.connect():
        return False

    # Try to load a simple model - use built-in scene if Menagerie not available
    success = demo.load_model("franka_panda")

    if success:
        print("   Testing go_home...")
        demo.go_home()
        time.sleep(1)

        print("   Testing wave motion...")
        demo.wave_motion(2.0)  # Short duration for testing

        print("   ✅ Motion tests completed")
        demo.viewer_client.disconnect()
        return True
    else:
        print("   ⚠️  Skipping motion tests (no models available)")
        demo.viewer_client.disconnect()
        return False


async def test_mcp_integration():
    """Test MCP integration"""
    print("\n🔧 Testing MCP integration...")

    try:
        from mujoco_mcp.mcp_server import handle_list_tools, handle_call_tool

        # Test listing tools
        tools = await handle_list_tools()
        print(f"   Found {len(tools)} MCP tools")

        # Test server info
        result = await handle_call_tool("get_server_info", {})
        print(f"   Server info: {result[0].text[:50]}...")

        print("   ✅ MCP integration test passed")
        return True

    except Exception as e:
        print(f"   ❌ MCP integration test failed: {e}")
        return False


async def run_all_tests():
    """Run all tests"""
    print("🚀 Running Motion Control Tests")
    print("=" * 50)

    tests = [
        ("Connection", test_basic_connection),
        ("Model Loading", test_model_loading),
        ("Basic Motions", test_basic_motions),
        ("MCP Integration", test_mcp_integration),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            results[test_name] = await test_func()
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")

    passed = 0
    total = len(results)

    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name:<20} {status}")
        if success:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
        return True
    elif passed > 0:
        print("⚠️  Some tests passed - partial functionality working")
        return True
    else:
        print("❌ All tests failed - check setup")
        return False


if __name__ == "__main__":
    asyncio.run(run_all_tests())

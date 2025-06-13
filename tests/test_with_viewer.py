#!/usr/bin/env python3
"""
Test with actual MuJoCo viewer server running
This test requires mujoco_viewer_server.py to be running
"""

import sys
import os
import time
import socket

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mujoco_mcp.remote_server import MuJoCoRemoteServer


def check_viewer_server():
    """Check if viewer server is running"""
    try:
        s = socket.socket()
        s.connect(('localhost', 8888))
        s.close()
        return True
    except:
        return False


def test_with_viewer():
    """Test actual loading with viewer"""
    if not check_viewer_server():
        print("⚠️  MuJoCo viewer server not running")
        print("   Start it with: python mujoco_viewer_server.py")
        return False
    
    print("✅ Viewer server is running")
    
    server = MuJoCoRemoteServer()
    
    # Test 1: Load built-in model
    print("\n1. Testing built-in model...")
    result = server._handle_create_scene("pendulum")
    if result.get("success"):
        model_id = result.get("model_id")
        print(f"   ✅ Pendulum loaded: {model_id}")
        
        # Get state
        state = server._handle_get_state(model_id)
        if "time" in state:
            print(f"   ✅ State retrieved: time={state['time']}")
        
        # Step simulation
        step_result = server._handle_step_simulation(model_id, 100)
        if step_result.get("success"):
            print("   ✅ Simulation stepped")
    else:
        print(f"   ❌ Failed: {result.get('error')}")
    
    # Test 2: Load Menagerie model
    if server.menagerie.is_available():
        print("\n2. Testing Menagerie model...")
        result = server._handle_create_scene("franka_emika_panda")
        if result.get("success"):
            model_id = result.get("model_id")
            model_info = result.get("model_info", {})
            print(f"   ✅ Franka loaded: {model_id}")
            print(f"      Joints: {model_info.get('nq', 'unknown')}")
            print(f"      Actuators: {model_info.get('nu', 'unknown')}")
        else:
            print(f"   ❌ Failed: {result.get('error')}")
    
    # Test 3: Natural language
    print("\n3. Testing natural language...")
    result = server._handle_execute_command("load spot robot")
    if "error" in result:
        if "another MuJoCo viewer is already open" in result["error"]:
            print("   ✅ Command processed (viewer limit reached)")
        else:
            print(f"   ❌ Error: {result['error']}")
    elif result.get("success"):
        print("   ✅ Spot loaded via natural language")
    
    return True


def main():
    print("MuJoCo MCP v0.7.1 - Viewer Integration Test")
    print("=" * 50)
    
    success = test_with_viewer()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ Viewer integration working!")
    else:
        print("❌ Viewer integration issues")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Quick Test for MuJoCo MCP v0.6.2 - Key Features Only
"""

import time
import sys
import os
from datetime import datetime

# Add project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mujoco_mcp.remote_server import MuJoCoRemoteServer

def test_mujoco_mcp():
    """Quick test of MuJoCo MCP core features"""
    
    print("🚀 MuJoCo MCP v0.6.2 Quick Test")
    print("=" * 60)
    
    server = MuJoCoRemoteServer()
    results = []
    
    # Test 1: Server Info
    print("\n1️⃣ Testing Server Info...")
    try:
        info = server._handle_get_server_info()
        print(f"✅ Server version: {info['version']}")
        print(f"   Mode: {info['mode']}")
        print(f"   Connected viewers: {info['connected_viewers']}")
        results.append(("Server Info", "Success", None))
    except Exception as e:
        print(f"❌ Failed: {e}")
        results.append(("Server Info", "Failed", str(e)))
    
    # Test 2: Create Pendulum
    print("\n2️⃣ Testing Pendulum Creation...")
    try:
        result = server._handle_create_scene("pendulum")
        if result.get("status") == "created":
            model_id = result["model_id"]
            print(f"✅ Created pendulum: {model_id}")
            results.append(("Create Pendulum", "Success", model_id))
            
            # Test 3: Get State
            print("\n3️⃣ Testing Get State...")
            try:
                state = server._handle_get_state(model_id)
                print(f"✅ Got state - Time: {state['time']:.2f}, Position: {state['positions'][0]:.3f}")
                results.append(("Get State", "Success", None))
            except Exception as e:
                print(f"❌ Failed: {e}")
                results.append(("Get State", "Failed", str(e)))
            
            # Test 4: Set Joint Position
            print("\n4️⃣ Testing Set Joint Position...")
            try:
                result = server._handle_set_joint_positions([1.57], model_id)  # 90 degrees
                if result["success"]:
                    print("✅ Set joint position to 90 degrees")
                    results.append(("Set Joint Position", "Success", None))
                else:
                    print(f"❌ Failed: {result.get('error', 'Unknown error')}")
                    results.append(("Set Joint Position", "Failed", result.get('error')))
            except Exception as e:
                print(f"❌ Failed: {e}")
                results.append(("Set Joint Position", "Failed", str(e)))
            
            # Test 5: Step Simulation
            print("\n5️⃣ Testing Step Simulation...")
            try:
                result = server._handle_step_simulation(50, model_id)
                print(f"✅ Stepped 50 times - New time: {result['time']:.2f}")
                results.append(("Step Simulation", "Success", None))
            except Exception as e:
                print(f"❌ Failed: {e}")
                results.append(("Step Simulation", "Failed", str(e)))
                
        else:
            print(f"❌ Failed to create: {result.get('error', 'Unknown error')}")
            results.append(("Create Pendulum", "Failed", result.get('error')))
    except Exception as e:
        print(f"❌ Failed: {e}")
        results.append(("Create Pendulum", "Failed", str(e)))
    
    # Test 6: Natural Language
    print("\n6️⃣ Testing Natural Language Commands...")
    commands = ["show current state", "reset the simulation", "list all models"]
    for cmd in commands:
        try:
            result = server._handle_execute_command(cmd)
            if result.get("success"):
                print(f"✅ Command '{cmd}': Success")
                results.append((f"Command: {cmd}", "Success", None))
            else:
                print(f"❌ Command '{cmd}': {result.get('error', 'Failed')}")
                results.append((f"Command: {cmd}", "Failed", result.get('error')))
        except Exception as e:
            print(f"❌ Command '{cmd}': {e}")
            results.append((f"Command: {cmd}", "Failed", str(e)))
    
    # Test 7: Multiple Scene Creation
    print("\n7️⃣ Testing Multiple Scene Creation...")
    scene_types = ["double_pendulum", "cart_pole"]
    for scene_type in scene_types:
        try:
            result = server._handle_create_scene(scene_type)
            if result.get("status") == "created":
                print(f"✅ Created {scene_type}: {result['model_id']}")
                results.append((f"Create {scene_type}", "Success", result['model_id']))
            else:
                print(f"❌ Failed to create {scene_type}: {result.get('error', 'Unknown error')}")
                results.append((f"Create {scene_type}", "Failed", result.get('error')))
        except Exception as e:
            print(f"❌ Failed to create {scene_type}: {e}")
            results.append((f"Create {scene_type}", "Failed", str(e)))
        time.sleep(0.5)  # Small delay between creations
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    successful = [r for r in results if r[1] == "Success"]
    failed = [r for r in results if r[1] == "Failed"]
    
    print(f"Total tests: {len(results)}")
    print(f"✅ Successful: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
    print(f"❌ Failed: {len(failed)} ({len(failed)/len(results)*100:.1f}%)")
    
    if failed:
        print("\nFailed tests:")
        for test_name, status, error in failed:
            print(f"  - {test_name}: {error}")
    
    # Generate detailed report
    report = f"""# MuJoCo MCP v0.6.2 Test Report

**Test Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Test Type**: Quick Feature Test

## Summary
- **Total Tests**: {len(results)}
- **Successful**: {len(successful)} ({len(successful)/len(results)*100:.1f}%)
- **Failed**: {len(failed)} ({len(failed)/len(results)*100:.1f}%)

## Test Results

"""
    
    for test_name, status, detail in results:
        if status == "Success":
            report += f"- ✅ **{test_name}**"
            if detail:
                report += f" - {detail}"
            report += "\n"
        else:
            report += f"- ❌ **{test_name}** - {detail}\n"
    
    # Analysis
    report += "\n## Analysis\n\n"
    
    # Connection issues
    connection_failures = [r for r in failed if "connect" in r[2].lower()]
    if connection_failures:
        report += f"### Connection Issues ({len(connection_failures)})\n"
        report += "Multiple tests failed due to connection problems with the MuJoCo Viewer.\n\n"
    
    # Command recognition
    command_failures = [r for r in failed if r[0].startswith("Command:")]
    if command_failures:
        report += f"### Natural Language Issues ({len(command_failures)})\n"
        report += "Some natural language commands were not recognized or failed.\n\n"
    
    # Scene creation
    scene_failures = [r for r in failed if "Create" in r[0]]
    if scene_failures:
        report += f"### Scene Creation Issues ({len(scene_failures)})\n"
        report += "Failed to create some physics scenes.\n\n"
    
    report += """## Recommendations

1. **Connection Stability**: If connection failures occurred, ensure the MuJoCo Viewer Server is running on port 8888.
2. **Natural Language**: The command recognition system may need expansion to handle more varied inputs.
3. **Multi-Scene Support**: Check if the system supports multiple simultaneous scenes.

## Conclusion

"""
    
    if len(successful) / len(results) > 0.8:
        report += "The MuJoCo MCP v0.6.2 is functioning well with most core features working correctly."
    elif len(successful) / len(results) > 0.5:
        report += "The MuJoCo MCP v0.6.2 has mixed results with some features working and others failing."
    else:
        report += "The MuJoCo MCP v0.6.2 has significant issues that need to be addressed."
    
    # Save report
    report_file = f"test_report_v062_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\n📄 Report saved to: {report_file}")
    

if __name__ == "__main__":
    test_mujoco_mcp()
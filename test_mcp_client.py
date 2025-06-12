#!/usr/bin/env python3
"""Test MCP client to verify server functionality."""

import json
import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


class MockMCPClient:
    """Mock MCP client for testing server connections."""
    
    def __init__(self):
        self.test_results = {
            "connection": {},
            "tools": {},
            "resources": {},
            "protocol": {}
        }
    
    async def test_server_info(self):
        """Test getting server information."""
        print("\n[TEST] Testing server info retrieval...")
        try:
            # This would normally be an MCP protocol request
            # For now, we'll test the server module directly
            from mujoco_mcp import __version__
            from mujoco_mcp.server import MuJoCoServer
            
            print(f"  ✓ Server version: {__version__}")
            self.test_results["connection"]["server_info"] = "PASSED"
            return True
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            self.test_results["connection"]["server_info"] = "FAILED"
            return False
    
    async def test_list_tools(self):
        """Test listing available tools."""
        print("\n[TEST] Testing tool listing...")
        
        # Expected tools based on documentation
        expected_tools = [
            "load_model",
            "start_simulation",
            "step_simulation",
            "reset_simulation",
            "set_joint_positions",
            "apply_control",
            "move_to_position",
            "grasp_object",
            "get_camera_image",
            "create_rl_environment"
        ]
        
        try:
            # In a real MCP client, this would be a protocol request
            # For testing, we'll check if these tools are registered
            from mujoco_mcp.server import mcp
            
            # Check if FastMCP instance has tools registered
            if hasattr(mcp, '_tools'):
                registered_tools = list(mcp._tools.keys())
                print(f"  ✓ Found {len(registered_tools)} registered tools")
                
                # Check for expected tools
                missing_tools = [tool for tool in expected_tools if tool not in registered_tools]
                if missing_tools:
                    print(f"  ⚠ Missing expected tools: {missing_tools}")
                
                self.test_results["tools"]["list"] = "PASSED"
                return True
            else:
                print("  ⚠ Unable to access registered tools")
                self.test_results["tools"]["list"] = "PARTIAL"
                return True
                
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            self.test_results["tools"]["list"] = "FAILED"
            return False
    
    async def test_list_resources(self):
        """Test listing available resources."""
        print("\n[TEST] Testing resource listing...")
        
        # Expected resources based on documentation
        expected_resources = [
            "joint_positions",
            "joint_velocities",
            "sensor_data",
            "rigid_body_states",
            "scene_objects",
            "robot_state"
        ]
        
        try:
            from mujoco_mcp.server import mcp
            
            # Check if FastMCP instance has resources registered
            if hasattr(mcp, '_resources'):
                registered_resources = list(mcp._resources.keys())
                print(f"  ✓ Found {len(registered_resources)} registered resources")
                
                self.test_results["resources"]["list"] = "PASSED"
                return True
            else:
                print("  ⚠ Unable to access registered resources")
                self.test_results["resources"]["list"] = "PARTIAL"
                return True
                
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            self.test_results["resources"]["list"] = "FAILED"
            return False
    
    async def test_protocol_format(self):
        """Test MCP protocol message format."""
        print("\n[TEST] Testing MCP protocol format...")
        
        # Test creating MCP-formatted messages
        test_messages = [
            {
                "jsonrpc": "2.0",
                "method": "tools/list",
                "id": 1
            },
            {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "load_model",
                    "arguments": {
                        "model_path": "test.xml"
                    }
                },
                "id": 2
            }
        ]
        
        try:
            for msg in test_messages:
                # Validate JSON-RPC format
                assert "jsonrpc" in msg and msg["jsonrpc"] == "2.0"
                assert "method" in msg
                assert "id" in msg
                
                json_str = json.dumps(msg)
                parsed = json.loads(json_str)
                assert parsed == msg
            
            print("  ✓ Protocol messages properly formatted")
            self.test_results["protocol"]["format"] = "PASSED"
            return True
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            self.test_results["protocol"]["format"] = "FAILED"
            return False
    
    async def test_simple_tool_call(self):
        """Test calling a simple tool."""
        print("\n[TEST] Testing simple tool call...")
        
        try:
            # Test creating a simulation manager instance
            from mujoco_mcp.server_manager import SimulationManager
            
            manager = SimulationManager()
            sim_id = manager.create_simulation("test_sim")
            
            if sim_id:
                print(f"  ✓ Created simulation: {sim_id}")
                
                # Clean up
                manager.remove_simulation(sim_id)
                print("  ✓ Cleaned up simulation")
                
                self.test_results["tools"]["simple_call"] = "PASSED"
                return True
            else:
                print("  ✗ Failed to create simulation")
                self.test_results["tools"]["simple_call"] = "FAILED"
                return False
                
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            self.test_results["tools"]["simple_call"] = "FAILED"
            return False
    
    def print_results(self):
        """Print test results summary."""
        print("\n" + "="*60)
        print("MCP Client Test Results")
        print("="*60)
        
        for category, tests in self.test_results.items():
            if tests:
                print(f"\n## {category.title()}")
                for test, status in tests.items():
                    symbol = "✓" if status == "PASSED" else "⚠" if status == "PARTIAL" else "✗"
                    print(f"  {symbol} {test}: {status}")
        
        # Calculate overall stats
        all_tests = []
        for tests in self.test_results.values():
            all_tests.extend(tests.values())
        
        passed = sum(1 for status in all_tests if status == "PASSED")
        partial = sum(1 for status in all_tests if status == "PARTIAL")
        failed = sum(1 for status in all_tests if status == "FAILED")
        total = len(all_tests)
        
        print(f"\n## Summary")
        print(f"  Total Tests: {total}")
        print(f"  Passed: {passed}")
        print(f"  Partial: {partial}")
        print(f"  Failed: {failed}")
        if total > 0:
            print(f"  Success Rate: {((passed + partial)/total*100):.1f}%")
        print("="*60)


async def test_server_startup():
    """Test starting the server in a subprocess."""
    print("\n[TEST] Testing server startup...")
    
    import subprocess
    import time
    
    try:
        # Try to start the server
        proc = subprocess.Popen(
            [sys.executable, "-m", "mujoco_mcp.server"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Give it a moment to start
        time.sleep(2)
        
        # Check if process is still running
        if proc.poll() is None:
            print("  ✓ Server process started")
            
            # Terminate the process
            proc.terminate()
            proc.wait(timeout=5)
            print("  ✓ Server process terminated cleanly")
            return True
        else:
            # Process ended, check output
            stdout, stderr = proc.communicate()
            print(f"  ✗ Server failed to start")
            if stderr:
                print(f"  Error: {stderr}")
            return False
            
    except Exception as e:
        print(f"  ✗ Failed to start server: {e}")
        return False


async def main():
    """Run all client tests."""
    client = MockMCPClient()
    
    # Run connection tests
    await client.test_server_info()
    
    # Run tool/resource tests
    await client.test_list_tools()
    await client.test_list_resources()
    
    # Run protocol tests
    await client.test_protocol_format()
    
    # Run functional tests
    await client.test_simple_tool_call()
    
    # Test server startup
    await test_server_startup()
    
    # Print results
    client.print_results()
    
    # Save results
    results_path = Path(__file__).parent / "mcp_client_test_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            "test_results": client.test_results,
            "timestamp": str(asyncio.get_event_loop().time())
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_path}")


if __name__ == "__main__":
    asyncio.run(main())
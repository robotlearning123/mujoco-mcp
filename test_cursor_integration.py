#!/usr/bin/env python3
"""
Test Cursor MCP Integration
Validates that MuJoCo MCP works correctly with Cursor configuration
"""
import json
import subprocess
import sys
import time
from pathlib import Path


def test_cursor_config():
    """Test Cursor MCP configuration"""
    print("🔍 Testing Cursor MCP Configuration")
    print("=" * 50)
    
    # Check if config file exists and is valid
    cursor_config = Path.home() / ".cursor" / "mcp.json"
    
    if not cursor_config.exists():
        print("❌ Cursor MCP config file not found")
        return False
    
    try:
        with open(cursor_config, 'r') as f:
            config = json.load(f)
        print("✅ Cursor MCP config file is valid JSON")
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in Cursor config: {e}")
        return False
    
    # Check if mujoco-mcp is configured
    if "mujoco-mcp" not in config:
        print("❌ mujoco-mcp not found in Cursor config")
        return False
    
    mujoco_config = config["mujoco-mcp"]
    print("✅ mujoco-mcp found in Cursor config")
    
    # Validate configuration structure
    required_fields = ["command", "args", "cwd", "env"]
    for field in required_fields:
        if field not in mujoco_config:
            print(f"❌ Missing required field: {field}")
            return False
    
    print("✅ All required configuration fields present")
    
    # Check working directory exists
    cwd = Path(mujoco_config["cwd"])
    if not cwd.exists():
        print(f"❌ Working directory not found: {cwd}")
        return False
    
    print(f"✅ Working directory exists: {cwd}")
    return True


def test_server_startup():
    """Test that the MuJoCo MCP server can start"""
    print("\n🚀 Testing Server Startup")
    print("=" * 50)
    
    try:
        # Test configuration check
        result = subprocess.run(
            ["python", "-m", "mujoco_mcp", "--check"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd="/Users/robert/workspace/04-code/03-mujoco/mujoco-mcp"
        )
        
        if result.returncode == 0:
            print("✅ Server configuration check passed")
        else:
            print(f"❌ Server configuration check failed: {result.stderr}")
            return False
        
        # Test version command
        result = subprocess.run(
            ["python", "-m", "mujoco_mcp", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd="/Users/robert/workspace/04-code/03-mujoco/mujoco-mcp"
        )
        
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✅ Server version: {version}")
        else:
            print(f"❌ Version check failed: {result.stderr}")
            return False
        
        return True
        
    except subprocess.TimeoutExpired:
        print("❌ Server startup timed out")
        return False
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        return False


def test_mcp_tools_registration():
    """Test that MCP tools are properly registered"""
    print("\n🔧 Testing MCP Tools Registration")
    print("=" * 50)
    
    try:
        # Import and test server initialization
        sys.path.insert(0, "/Users/robert/workspace/04-code/03-mujoco/mujoco-mcp/src")
        
        from mujoco_mcp.server import MuJoCoServer
        import asyncio
        
        async def test_server():
            server = MuJoCoServer()
            await server.initialize()
            
            # Check tool registration
            tools = list(server.mcp._tool_manager._tools.values())
            print(f"✅ {len(tools)} MCP tools registered")
            
            # Check resource registration
            resources = list(server.mcp._resource_manager._resources.values())
            print(f"✅ {len(resources)} MCP resources registered")
            
            # Test basic tool execution
            result = await server.mcp.call_tool("get_server_info", {})
            if result:
                print("✅ Basic tool execution successful")
            else:
                print("❌ Basic tool execution failed")
                return False
            
            await server.cleanup()
            return True
        
        success = asyncio.run(test_server())
        return success
        
    except Exception as e:
        print(f"❌ Tool registration test failed: {e}")
        return False


def print_usage_instructions():
    """Print instructions for using MuJoCo MCP with Cursor"""
    print("\n📖 Usage Instructions")
    print("=" * 50)
    
    print("1. Restart Cursor completely")
    print("2. Open a new chat session")
    print("3. Try these test commands:")
    print("")
    print("   🔹 'What MCP servers are available?'")
    print("   🔹 'Create a pendulum simulation using MuJoCo'")
    print("   🔹 'Show me the current simulation state'")
    print("   🔹 'Move the robot to position [1, 0, 1]'")
    print("")
    print("4. Expected results:")
    print("   ✅ MuJoCo MCP should be listed as available")
    print("   ✅ Physics simulations should work")
    print("   ✅ Natural language commands should be understood")
    print("")
    print("🎯 MuJoCo MCP Features Available:")
    print("   • 98 simulation and control tools")
    print("   • 3 data resources (state, sensors, config)")
    print("   • Natural language interface")
    print("   • Reinforcement learning integration")
    print("   • Real-time visualization")


def main():
    """Main test function"""
    print("🧪 MuJoCo MCP + Cursor Integration Test")
    print("=" * 60)
    
    tests = [
        ("Cursor Configuration", test_cursor_config),
        ("Server Startup", test_server_startup),
        ("MCP Tools Registration", test_mcp_tools_registration),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            if success:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! MuJoCo MCP is ready for Cursor!")
        print_usage_instructions()
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        print("\nTroubleshooting:")
        print("• Run: pip install -e .")
        print("• Check: python -m mujoco_mcp --check")
        print("• Review: CURSOR_SETUP.md")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Test Claude Desktop MCP Integration
Validates that MuJoCo MCP works correctly with Claude Desktop configuration
"""
import json
import subprocess
import sys
from pathlib import Path


def test_claude_desktop_config():
    """Test Claude Desktop MCP configuration"""
    print("🔍 Testing Claude Desktop MCP Configuration")
    print("=" * 50)
    
    # Check if config file exists and is valid
    claude_config = Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    
    if not claude_config.exists():
        print("❌ Claude Desktop config file not found")
        print(f"Expected location: {claude_config}")
        return False
    
    try:
        with open(claude_config, 'r') as f:
            config = json.load(f)
        print("✅ Claude Desktop config file is valid JSON")
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in Claude Desktop config: {e}")
        return False
    
    # Check if mcpServers section exists
    if "mcpServers" not in config:
        print("❌ mcpServers section not found in Claude Desktop config")
        return False
    
    # Check if mujoco-mcp is configured
    if "mujoco-mcp" not in config["mcpServers"]:
        print("❌ mujoco-mcp not found in Claude Desktop config")
        return False
    
    mujoco_config = config["mcpServers"]["mujoco-mcp"]
    print("✅ mujoco-mcp found in Claude Desktop config")
    
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
    
    # Check for proper command structure
    if mujoco_config["command"] != "python":
        print("❌ Command should be 'python'")
        return False
    
    if mujoco_config["args"] != ["-m", "mujoco_mcp"]:
        print("❌ Args should be ['-m', 'mujoco_mcp']")
        return False
    
    print("✅ Command and args are correct")
    return True


def test_config_validation():
    """Test that the configuration can be validated"""
    print("\n🔧 Testing Configuration Validation")
    print("=" * 50)
    
    try:
        # Test that our server can validate itself
        result = subprocess.run(
            ["python", "-m", "mujoco_mcp", "--check"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd="/Users/robert/workspace/04-code/03-mujoco/mujoco-mcp"
        )
        
        if result.returncode == 0:
            print("✅ MuJoCo MCP configuration validation passed")
            print(f"   Output: {result.stdout.strip()}")
        else:
            print(f"❌ Configuration validation failed: {result.stderr}")
            return False
        
        return True
        
    except subprocess.TimeoutExpired:
        print("❌ Configuration validation timed out")
        return False
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False


def check_other_servers():
    """Check other MCP servers in the configuration"""
    print("\n📋 Checking Other MCP Servers")
    print("=" * 50)
    
    claude_config = Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    
    try:
        with open(claude_config, 'r') as f:
            config = json.load(f)
        
        servers = config.get("mcpServers", {})
        
        print(f"Found {len(servers)} MCP servers configured:")
        for server_name, server_config in servers.items():
            if server_name == "mujoco-mcp":
                print(f"  ✅ {server_name} (MuJoCo Physics Simulation)")
            else:
                print(f"  📦 {server_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to check other servers: {e}")
        return False


def print_claude_desktop_instructions():
    """Print instructions for using MuJoCo MCP with Claude Desktop"""
    print("\n📖 Claude Desktop Usage Instructions")
    print("=" * 50)
    
    print("1. **Restart Claude Desktop completely**")
    print("   - Quit Claude Desktop (Cmd+Q)")
    print("   - Reopen Claude Desktop")
    print("")
    print("2. **Start a new conversation**")
    print("   - Click on 'New Chat'")
    print("   - Wait for MCP servers to initialize")
    print("")
    print("3. **Test MCP connection**")
    print("   Try: 'What MCP servers are connected?'")
    print("   Expected: Should list mujoco-mcp among others")
    print("")
    print("4. **Test MuJoCo functionality**")
    print("   🔹 'Create a pendulum simulation'")
    print("   🔹 'Show me the current simulation state'")
    print("   🔹 'Move the pendulum to 45 degrees'")
    print("   🔹 'Generate a robot arm and plan a trajectory'")
    print("")
    print("5. **Advanced features to try**")
    print("   🔹 'Set up a reinforcement learning environment'")
    print("   🔹 'Train a policy to balance a cartpole'")
    print("   🔹 'Optimize controller parameters for energy efficiency'")
    print("   🔹 'Visualize contact forces in the simulation'")
    print("")
    print("🎯 **Available Capabilities:**")
    print("   • 98 physics simulation tools")
    print("   • Natural language robot control")
    print("   • Real-time visualization")
    print("   • Reinforcement learning integration")
    print("   • Parameter optimization")
    print("   • Multi-agent coordination")


def main():
    """Main test function"""
    print("🧪 Claude Desktop + MuJoCo MCP Integration Test")
    print("=" * 60)
    
    tests = [
        ("Claude Desktop Configuration", test_claude_desktop_config),
        ("Configuration Validation", test_config_validation),
        ("Other MCP Servers", check_other_servers),
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
        print("🎉 All tests passed! MuJoCo MCP is ready for Claude Desktop!")
        print_claude_desktop_instructions()
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        print("\nTroubleshooting:")
        print("• Ensure Claude Desktop is completely closed")
        print("• Check config file syntax")
        print("• Run: python -m mujoco_mcp --check")
        print("• Review: CURSOR_SETUP.md for debugging")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
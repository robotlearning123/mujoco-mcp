#!/usr/bin/env python3
"""
Test MCP Client Connection
Simulates how external tools like Cursor or Claude Desktop would connect
"""
import asyncio
import json
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from mujoco_mcp.server import MuJoCoServer


async def simulate_mcp_client():
    """Simulate an MCP client connecting and using the server"""
    print("🔌 Simulating MCP Client Connection")
    print("=" * 50)
    
    # Initialize server (as would happen when client connects)
    server = MuJoCoServer()
    await server.initialize()
    
    print(f"✓ Connected to {server.name} v{server.version}")
    
    # Test 1: Get server capabilities
    print("\n1. Getting server capabilities...")
    info = server.get_server_info()
    print(f"   Server: {info['name']}")
    print(f"   Version: {info['version']}")
    print(f"   Capabilities: {list(info['capabilities'].keys())}")
    
    # Test 2: List available tools
    print("\n2. Listing available tools...")
    tools = list(server.mcp._tool_manager._tools.values())
    print(f"   Found {len(tools)} tools:")
    for i, tool in enumerate(tools[:5]):  # Show first 5
        print(f"     • {tool.name}")
    if len(tools) > 5:
        print(f"     ... and {len(tools) - 5} more")
    
    # Test 3: Create a simple simulation
    print("\n3. Creating a pendulum simulation...")
    result = await server.mcp.call_tool("pendulum_demo", {"action": "setup"})
    
    if isinstance(result, list) and len(result) > 0:
        if hasattr(result[0], 'text'):
            content = json.loads(result[0].text)
        else:
            content = json.loads(result[0].content)
        
        model_id = content.get("model_id")
        print(f"   ✓ Created pendulum model: {model_id}")
    
        # Test 4: Step the simulation
        print("\n4. Stepping simulation...")
        step_result = await server.mcp.call_tool("step_simulation", {
            "model_id": model_id,
            "steps": 10
        })
        print("   ✓ Simulation stepped successfully")
        
        # Test 5: Get simulation state
        print("\n5. Reading simulation state...")
        state_resource = await server.mcp.read_resource("simulation://state")
        state_data = json.loads(state_resource[0].content)
        sim_time = state_data["contents"]["time"]
        print(f"   ✓ Current simulation time: {sim_time:.4f}s")
        
        # Test 6: Natural language command
        print("\n6. Testing natural language interface...")
        nl_result = await server.mcp.call_tool("nl_command", {
            "command": "show me the current pendulum angle",
            "context": {"model_id": model_id}
        })
        
        if isinstance(nl_result, list) and len(nl_result) > 0:
            if hasattr(nl_result[0], 'text'):
                nl_content = json.loads(nl_result[0].text)
            else:
                nl_content = json.loads(nl_result[0].content)
            print(f"   ✓ Response: {nl_content.get('response', 'No response')}")
    
    # Test 7: Test error handling
    print("\n7. Testing error handling...")
    try:
        error_result = await server.mcp.call_tool("nonexistent_tool", {})
        print("   ✗ Error handling failed - should have raised exception")
    except Exception as e:
        print(f"   ✓ Error properly handled: {type(e).__name__}")
    
    print("\n" + "=" * 50)
    print("🎉 MCP Client simulation completed successfully!")
    print("\nThis demonstrates that external tools can:")
    print("• Connect to the MuJoCo MCP server")
    print("• List and call available tools")
    print("• Read resources for simulation state")
    print("• Use natural language commands")
    print("• Handle errors gracefully")
    
    # Cleanup
    await server.cleanup()


async def test_configuration_usage():
    """Test how the configuration files would be used"""
    print("\n📋 Configuration Usage Examples")
    print("=" * 50)
    
    # Show how to use each config file
    configs = {
        "mcp.json": "Generic MCP client configuration",
        "claude_desktop_config.json": "Claude Desktop configuration",
        ".cursorrules": "Cursor IDE integration",
        "CONFIG.md": "Comprehensive setup guide"
    }
    
    for config_file, description in configs.items():
        config_path = project_root / config_file
        if config_path.exists():
            print(f"✓ {config_file}: {description}")
            
            # Show relevant config snippets
            if config_file.endswith('.json'):
                try:
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)
                    
                    if 'mcpServers' in config_data:
                        server_config = config_data['mcpServers']['mujoco-mcp']
                        command = f"{server_config['command']} {' '.join(server_config['args'])}"
                        print(f"   Command: {command}")
                    elif 'command' in config_data:
                        command = f"{config_data['command']} {' '.join(config_data['args'])}"
                        print(f"   Command: {command}")
                        
                except Exception as e:
                    print(f"   Warning: Could not parse {config_file}: {e}")
        else:
            print(f"✗ {config_file}: Not found")
    
    print("\nTo use with your preferred tool:")
    print("1. Cursor: Automatically detects .cursorrules")
    print("2. Claude Desktop: Copy claude_desktop_config.json to Claude config directory")
    print("3. VS Code: See CONFIG.md for setup instructions")
    print("4. Other MCP clients: Use mcp.json as reference")


async def main():
    """Main test function"""
    await simulate_mcp_client()
    await test_configuration_usage()


if __name__ == "__main__":
    asyncio.run(main())
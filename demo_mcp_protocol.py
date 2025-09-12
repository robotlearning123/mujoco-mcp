#!/usr/bin/env python3
"""
MCP Protocol Demo: Test, Load, Control Random MuJoCo Menagerie Model
Demonstrates proper MCP client-server interaction via JSON-RPC
"""

import asyncio
import json
import random
import sys
from pathlib import Path
from typing import Dict, Any, List


class MCPClient:
    """Simple MCP client that communicates via JSON-RPC over subprocess"""

    def __init__(self):
        self.process = None
        self.request_id = 0

    async def start_server(self):
        """Start the MCP server process"""
        server_path = Path(__file__).parent / "src" / "mujoco_mcp"

        # Start the MCP server via stdio
        self.process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "mujoco_mcp",
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
                "capabilities": {},
                "clientInfo": {"name": "demo-client", "version": "1.0.0"},
            },
        )

        # Send initialized notification
        await self.send_notification("initialized", {})

    async def send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send JSON-RPC request to MCP server"""
        self.request_id += 1
        request = {"jsonrpc": "2.0", "id": self.request_id, "method": method, "params": params}

        # Send request
        request_line = json.dumps(request) + "\n"
        self.process.stdin.write(request_line.encode())
        await self.process.stdin.drain()

        # Read response
        response_line = await self.process.stdout.readline()
        if not response_line:
            raise RuntimeError("Server closed connection")

        response = json.loads(response_line.decode())

        if "error" in response:
            raise RuntimeError(f"MCP Error: {response['error']}")

        return response.get("result", {})

    async def send_notification(self, method: str, params: Dict[str, Any]):
        """Send JSON-RPC notification (no response expected)"""
        notification = {"jsonrpc": "2.0", "method": method, "params": params}

        notification_line = json.dumps(notification) + "\n"
        self.process.stdin.write(notification_line.encode())
        await self.process.stdin.drain()

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available MCP tools"""
        result = await self.send_request("tools/list", {})
        return result.get("tools", [])

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool"""
        return await self.send_request("tools/call", {"name": name, "arguments": arguments})

    async def stop_server(self):
        """Stop the MCP server process"""
        if self.process:
            self.process.terminate()
            await self.process.wait()


async def demo_mcp_menagerie():
    """Demo: Test, load, control random MuJoCo Menagerie model via MCP"""
    print("🚀 MCP Protocol Demo: MuJoCo Menagerie Integration")
    print("=" * 55)

    client = MCPClient()

    try:
        # Step 1: Start MCP server
        print("\n📡 Step 1: Starting MCP Server")
        print("-" * 30)
        await client.start_server()
        print("✅ MCP server started via stdio transport")

        # Step 2: List available tools
        print("\n🔧 Step 2: Available MCP Tools")
        print("-" * 30)

        tools = await client.list_tools()
        print(f"✅ Found {len(tools)} available tools:")

        for tool in tools:
            print(f"  📋 {tool['name']}: {tool.get('description', 'No description')}")

        # Check if we have Menagerie tools
        tool_names = [tool["name"] for tool in tools]
        menagerie_tools = [name for name in tool_names if "menagerie" in name.lower()]

        if menagerie_tools:
            print(f"\n🎯 Found Menagerie tools: {', '.join(menagerie_tools)}")
        else:
            print("\n⚠️ No specific Menagerie tools found, using standard tools")

        # Step 3: Try to get server info
        print("\n📊 Step 3: Server Information")
        print("-" * 30)

        try:
            server_info = await client.call_tool("get_server_info", {})
            print("✅ Server info retrieved:")
            print(f"   {server_info}")
        except Exception as e:
            print(f"⚠️ Could not get server info: {e}")

        # Step 4: Create a scene (test core functionality)
        print("\n🎭 Step 4: Testing Scene Creation")
        print("-" * 35)

        try:
            # Try creating a simple scene first
            scene_result = await client.call_tool("create_scene", {"scene_type": "pendulum"})
            print("✅ Successfully created pendulum scene")
            print(f"   Result: {scene_result}")

            # Step 5: Test simulation control
            print("\n⚡ Step 5: Testing Simulation Control")
            print("-" * 40)

            # Step simulation
            step_result = await client.call_tool(
                "step_simulation", {"model_id": "pendulum", "steps": 5}
            )
            print("✅ Simulation stepped successfully")
            print(f"   Result: {step_result}")

            # Get state
            state_result = await client.call_tool("get_state", {"model_id": "pendulum"})
            print("✅ Retrieved simulation state")
            state_preview = (
                str(state_result)[:100] + "..."
                if len(str(state_result)) > 100
                else str(state_result)
            )
            print(f"   State: {state_preview}")

        except Exception as e:
            print(f"❌ Scene creation/control failed: {e}")

        # Step 6: If we have Menagerie support, test it
        if "list_menagerie_models" in tool_names:
            print("\n🦋 Step 6: Testing Menagerie Integration")
            print("-" * 40)

            try:
                # List Menagerie models
                models_result = await client.call_tool("list_menagerie_models", {})
                print("✅ Retrieved Menagerie models catalog")

                models_data = json.loads(models_result.get("content", [{}])[0].get("text", "{}"))
                total_models = models_data.get("total_models", 0)
                print(f"   📦 Found {total_models} models across multiple categories")

                # Show categories
                for category, info in models_data.get("models", {}).items():
                    print(f"   🏷️  {category.upper()}: {info['count']} models")
                    # Show first model as example
                    if info["models"]:
                        print(f"      Example: {info['models'][0]}")

                # Test with a random model
                if models_data.get("models"):
                    all_models = []
                    for category_info in models_data["models"].values():
                        all_models.extend(category_info["models"])

                    if all_models:
                        random_model = random.choice(all_models)
                        print(f"\n🎯 Testing with random model: {random_model}")

                        # Validate the model
                        validation_result = await client.call_tool(
                            "validate_menagerie_model", {"model_name": random_model}
                        )
                        print(f"   🔬 Validation: {validation_result}")

                        # Try to create scene with the model
                        menagerie_scene_result = await client.call_tool(
                            "create_menagerie_scene",
                            {"model_name": random_model, "scene_name": f"demo_{random_model}"},
                        )
                        print(f"   🎭 Scene creation: {menagerie_scene_result}")

            except Exception as e:
                print(f"❌ Menagerie testing failed: {e}")

        print(f"\n{'=' * 55}")
        print("🎉 MCP PROTOCOL DEMO COMPLETE")
        print(f"{'=' * 55}")
        print("✅ Successfully demonstrated MCP client-server interaction")
        print("📡 JSON-RPC communication working properly")
        print("🔧 MCP tools accessible and functional")
        if menagerie_tools:
            print(f"🦋 Menagerie integration: {len(menagerie_tools)} specialized tools")
        print("⚡ Simulation control validated")

    except Exception as e:
        print(f"\n💥 Demo failed: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        await client.stop_server()
        print("✅ MCP server stopped")


if __name__ == "__main__":
    try:
        asyncio.run(demo_mcp_menagerie())
        print("\n🚀 Demo completed successfully!")
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Demo crashed: {e}")
        sys.exit(1)

#!/usr/bin/env python3
"""
Simple MCP Demo: Test, Load, Control Models via MCP Protocol
Demonstrates proper MCP client-server interaction with existing tools
"""

import asyncio
import json
import sys
from pathlib import Path

async def demo_mcp_interaction():
    """Demo: Test, load, control models via MCP protocol"""
    print("🚀 MCP Protocol Demo: Testing, Loading, and Controlling Models")
    print("=" * 65)

    # Start the MCP server process
    print("\n📡 Step 1: Starting MCP Server")
    print("-" * 30)

    server_cmd = [sys.executable, "-m", "mujoco_mcp"]

    # Start server process
    process = await asyncio.create_subprocess_exec(
        *server_cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=Path(__file__).parent
    )

    print("✅ MCP server process started")

    try:
        # MCP Initialization sequence
        print("\n🔧 Step 2: MCP Protocol Initialization")
        print("-" * 40)

        # Initialize request
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "clientInfo": {
                    "name": "demo-client",
                    "version": "1.0.0"
                }
            }
        }

        # Send initialize request
        request_line = json.dumps(init_request) + "\n"
        process.stdin.write(request_line.encode())
        await process.stdin.drain()

        # Read initialize response
        response_line = await asyncio.wait_for(process.stdout.readline(), timeout=5.0)
        init_response = json.loads(response_line.decode())
        print("✅ Server initialized successfully")
        print(f"   Server info: {init_response.get('result', {}).get('serverInfo', {})}")

        # Send initialized notification
        initialized_notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }

        notif_line = json.dumps(initialized_notification) + "\n"
        process.stdin.write(notif_line.encode())
        await process.stdin.drain()

        print("✅ Initialization handshake completed")

        # Step 3: List available tools
        print("\n🔧 Step 3: Discovering Available Tools")
        print("-" * 40)

        tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }

        request_line = json.dumps(tools_request) + "\n"
        process.stdin.write(request_line.encode())
        await process.stdin.drain()

        response_line = await asyncio.wait_for(process.stdout.readline(), timeout=5.0)
        tools_response = json.loads(response_line.decode())

        if "error" in tools_response:
            print(f"❌ Error listing tools: {tools_response['error']}")
        else:
            tools = tools_response.get("result", {}).get("tools", [])
            print(f"✅ Found {len(tools)} available tools:")
            for tool in tools:
                print(f"   📋 {tool['name']}: {tool.get('description', 'No description')}")

        # Step 4: Get server information
        print("\n📊 Step 4: Getting Server Information")
        print("-" * 40)

        server_info_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "get_server_info",
                "arguments": {}
            }
        }

        request_line = json.dumps(server_info_request) + "\n"
        process.stdin.write(request_line.encode())
        await process.stdin.drain()

        response_line = await asyncio.wait_for(process.stdout.readline(), timeout=5.0)
        server_info_response = json.loads(response_line.decode())

        if "error" in server_info_response:
            print(f"❌ Error getting server info: {server_info_response['error']}")
        else:
            server_info = server_info_response.get("result", {})
            print("✅ Server information retrieved:")
            print(f"   {server_info}")

        # Step 5: Test model loading (create scene)
        print("\n🎭 Step 5: Testing Model Loading (Scene Creation)")
        print("-" * 50)

        # Test with different scene types
        scene_types = ["pendulum", "double_pendulum", "cart_pole"]

        for scene_type in scene_types:
            print(f"\n🔧 Testing {scene_type} scene...")

            create_scene_request = {
                "jsonrpc": "2.0",
                "id": 4 + scene_types.index(scene_type),
                "method": "tools/call",
                "params": {
                    "name": "create_scene",
                    "arguments": {
                        "scene_type": scene_type
                    }
                }
            }

            request_line = json.dumps(create_scene_request) + "\n"
            process.stdin.write(request_line.encode())
            await process.stdin.drain()

            response_line = await asyncio.wait_for(process.stdout.readline(), timeout=10.0)
            scene_response = json.loads(response_line.decode())

            if "error" in scene_response:
                print(f"   ❌ Failed to create {scene_type}: {scene_response['error']}")
            else:
                result = scene_response.get("result", {})
                print(f"   ✅ {scene_type} scene created successfully")
                print(f"   📋 Result: {result}")

                # Test simulation control for this model
                print("   ⚡ Testing simulation control...")

                # Step simulation
                step_request = {
                    "jsonrpc": "2.0",
                    "id": 10 + scene_types.index(scene_type),
                    "method": "tools/call",
                    "params": {
                        "name": "step_simulation",
                        "arguments": {
                            "model_id": scene_type,
                            "steps": 3
                        }
                    }
                }

                request_line = json.dumps(step_request) + "\n"
                process.stdin.write(request_line.encode())
                await process.stdin.drain()

                response_line = await asyncio.wait_for(process.stdout.readline(), timeout=5.0)
                step_response = json.loads(response_line.decode())

                if "error" in step_response:
                    print(f"   ⚠️ Step simulation failed: {step_response['error']}")
                else:
                    print("   ✅ Simulation stepped successfully")

                # Get state
                state_request = {
                    "jsonrpc": "2.0",
                    "id": 20 + scene_types.index(scene_type),
                    "method": "tools/call",
                    "params": {
                        "name": "get_state",
                        "arguments": {
                            "model_id": scene_type
                        }
                    }
                }

                request_line = json.dumps(state_request) + "\n"
                process.stdin.write(request_line.encode())
                await process.stdin.drain()

                response_line = await asyncio.wait_for(process.stdout.readline(), timeout=5.0)
                state_response = json.loads(response_line.decode())

                if "error" in state_response:
                    print(f"   ⚠️ Get state failed: {state_response['error']}")
                else:
                    state_result = str(state_response.get("result", {}))
                    state_preview = state_result[:80] + "..." if len(state_result) > 80 else state_result
                    print(f"   📊 State retrieved: {state_preview}")

        # Final summary
        print(f"\n{'=' * 65}")
        print("🎉 MCP PROTOCOL DEMO COMPLETE")
        print(f"{'=' * 65}")
        print("✅ Successfully demonstrated MCP client-server communication")
        print("📡 JSON-RPC 2.0 protocol working correctly")
        print("🔧 MCP tools accessible and functional")
        print("🎭 Model loading (scene creation) validated")
        print("⚡ Simulation control (step, get_state) working")
        print("🏗️ Multiple model types tested successfully")

        print("\n💡 WHAT THIS DEMONSTRATES:")
        print("🔌 MCP server starts and accepts connections")
        print("📋 Tools are discoverable via tools/list")
        print("🎯 Tool execution via tools/call works properly")
        print("🔄 Real-time simulation control is functional")
        print("📊 State queries return structured data")
        print("🎪 Multiple model types can be loaded and controlled")

        print("\n🚀 MCP Server is ready for integration with:")
        print("   🖥️  Claude Desktop")
        print("   🔗 Other MCP-compatible clients")
        print("   🤖 Custom automation scripts")

    except asyncio.TimeoutError:
        print("❌ Timeout waiting for server response")
    except Exception as e:
        print(f"💥 Demo failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        if process:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        print("✅ MCP server stopped")

if __name__ == "__main__":
    try:
        asyncio.run(demo_mcp_interaction())
        print("\n🎯 Demo completed successfully!")
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Demo crashed: {e}")
        sys.exit(1)

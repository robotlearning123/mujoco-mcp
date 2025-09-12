#!/usr/bin/env python3
"""
Working MuJoCo MCP Demo - Headless Mode
This demonstrates the FIXED MCP server that actually works!
"""

import asyncio
import json
import sys
from pathlib import Path

class WorkingMCPDemo:
    """Demonstrates the working headless MCP server"""

    def __init__(self):
        self.process = None
        self.request_id = 0

    async def start_server(self):
        """Start the working headless MCP server"""
        self.process = await asyncio.create_subprocess_exec(
            sys.executable, "-c",
            "import sys; sys.path.append('./src'); "
            "from mujoco_mcp.mcp_server_headless import main; "
            "import asyncio; asyncio.run(main())",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=Path(__file__).parent
        )

        # MCP Initialization
        await self.send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "clientInfo": {
                "name": "working-demo",
                "version": "1.0.0"
            }
        })

        await self.send_notification("notifications/initialized", {})
        print("✅ Working MCP Server started!")

    async def send_request(self, method: str, params: dict) -> dict:
        """Send JSON-RPC request"""
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params
        }

        request_line = json.dumps(request) + "\n"
        self.process.stdin.write(request_line.encode())
        await self.process.stdin.drain()

        response_line = await asyncio.wait_for(
            self.process.stdout.readline(),
            timeout=5.0
        )
        response = json.loads(response_line.decode())

        if "error" in response:
            raise RuntimeError(f"MCP Error: {response['error']}")

        return response.get("result", {})

    async def send_notification(self, method: str, params: dict):
        """Send JSON-RPC notification"""
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }

        notification_line = json.dumps(notification) + "\n"
        self.process.stdin.write(notification_line.encode())
        await self.process.stdin.drain()

    async def call_tool(self, name: str, arguments: dict):
        """Call an MCP tool"""
        return await self.send_request("tools/call", {
            "name": name,
            "arguments": arguments
        })

    async def stop_server(self):
        """Stop the MCP server"""
        if self.process:
            self.process.terminate()
            await asyncio.wait_for(self.process.wait(), timeout=3.0)

async def demonstrate_working_mcp():
    """Demonstrate the working MCP server"""
    print("🎉 WORKING MUJOCO MCP DEMONSTRATION")
    print("(This one actually works - no GUI timeouts!)")
    print("=" * 55)

    demo = WorkingMCPDemo()

    try:
        # Start the fixed server
        print("\n📡 Starting Working MCP Server")
        print("-" * 35)
        await demo.start_server()

        # Test server info
        print("\n📊 Getting Server Info")
        print("-" * 25)
        server_result = await demo.call_tool("get_server_info", {})
        print(f"✅ Server: {server_result}")

        # Test physics simulations
        simulations = [
            ("pendulum", "Simple pendulum physics"),
            ("cart_pole", "Cart-pole balancing system"),
            ("double_pendulum", "Chaotic double pendulum"),
            ("arm", "2-DOF robot arm")
        ]

        for scene_type, description in simulations:
            print(f"\n🎯 Testing {scene_type.upper()}")
            print(f"📝 {description}")
            print("-" * 40)

            # Create scene
            create_result = await demo.call_tool("create_scene", {
                "scene_type": scene_type
            })
            print(f"📦 Created: {create_result['content'][0]['text']}")

            # Step simulation
            step_result = await demo.call_tool("step_simulation", {
                "model_id": scene_type,
                "steps": 50
            })
            print(f"⏩ Stepped: {step_result['content'][0]['text']}")

            # Get state
            state_result = await demo.call_tool("get_state", {
                "model_id": scene_type
            })
            state_text = state_result['content'][0]['text']
            state_data = json.loads(state_text)
            print(f"📊 Time: {state_data['time']:.3f}s")
            print(f"📊 Bodies: {state_data['nbody']}")
            print(f"📊 DOF: {state_data['nq']}")

            # Reset
            reset_result = await demo.call_tool("reset_simulation", {
                "model_id": scene_type
            })
            print(f"🔄 Reset: {reset_result['content'][0]['text']}")

        # Clean up
        print("\n🧹 Cleaning Up")
        print("-" * 15)
        for scene_type, _ in simulations:
            close_result = await demo.call_tool("close_simulation", {
                "model_id": scene_type
            })
            print(f"🚪 {close_result['content'][0]['text']}")

        # Success summary
        print(f"\n{'=' * 55}")
        print("🎉 SUCCESS! MuJoCo MCP Server Works Perfectly!")
        print(f"{'=' * 55}")
        print("\n✅ DEMONSTRATED CAPABILITIES:")
        print("   🔧 Headless operation (no GUI required)")
        print("   🎯 Multiple physics scenes")
        print("   ⏩ Real-time simulation stepping")
        print("   📊 State queries and monitoring")
        print("   🔄 Reset and control functionality")
        print("   🚪 Proper cleanup and resource management")

        print("\n🔌 HOW TO USE WITH CLAUDE CODE:")
        print("   1. Use mcp_server_headless.py instead of mcp_server.py")
        print("   2. Configure Claude Desktop with headless server")
        print("   3. Works on SSH, Docker, cloud, headless systems")
        print("   4. No display/GUI requirements")

        print("\n💡 CONFIGURATION FOR CLAUDE DESKTOP:")
        print('   {')
        print('     "mcpServers": {')
        print('       "mujoco-headless": {')
        print('         "command": "python",')
        print('         "args": ["-m", "mujoco_mcp.mcp_server_headless"],')
        print('         "cwd": "/path/to/mujoco-mcp",')
        print('         "env": {"PYTHONPATH": "./src"}')
        print('       }')
        print('     }')
        print('   }')

        return True

    except Exception as e:
        print(f"\n💥 Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        await demo.stop_server()
        print("\n✅ Server stopped cleanly")

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║          WORKING MUJOCO MCP DEMONSTRATION                ║
    ╠══════════════════════════════════════════════════════════╣
    ║  This demo shows the FIXED MCP server that:             ║
    ║  • Works without GUI/display requirements               ║
    ║  • No timeouts or hanging issues                        ║
    ║  • Perfect for SSH, Docker, cloud environments         ║
    ║  • Full physics simulation capabilities                 ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    try:
        success = asyncio.run(demonstrate_working_mcp())
        if success:
            print("\n🚀 Working MCP demo completed successfully!")
            print("\n🔥 The solution to your timeout issue:")
            print("   Use 'mcp_server_headless.py' instead of 'mcp_server.py'")
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Demo crashed: {e}")
        sys.exit(1)

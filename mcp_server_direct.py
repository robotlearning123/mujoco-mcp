#!/usr/bin/env python3
"""
Direct MuJoCo MCP Server for Claude Desktop
Uses FastMCP's native execution pattern
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the server class
from mujoco_mcp.server import MuJoCoServer

# Direct execution using FastMCP's preferred pattern
if __name__ == "__main__":
    # Create server
    server = MuJoCoServer()
    
    # Use FastMCP's direct run method (no asyncio.run wrapper)
    import asyncio
    
    # Get or create event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Initialize and run server
    async def setup_and_run():
        await server.initialize()
        # Call the mcp.run() directly which handles stdio properly
        await server.mcp.run()
    
    # Run the server
    loop.run_until_complete(setup_and_run())
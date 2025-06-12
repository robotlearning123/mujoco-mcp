#!/usr/bin/env python3
"""
Simple MuJoCo MCP Server for Claude Desktop
Uses the correct FastMCP pattern without asyncio conflicts
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the server class
from mujoco_mcp.server import MuJoCoServer

# Simple direct execution - let FastMCP handle everything
if __name__ == "__main__":
    import asyncio
    
    async def main():
        server = MuJoCoServer()
        await server.initialize()
        await server.run()
    
    asyncio.run(main())
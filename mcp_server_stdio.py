#!/usr/bin/env python3
"""
MuJoCo MCP Server for Claude Desktop (STDIO mode)
Direct stdio interface for Claude Desktop MCP integration
"""
import sys
import os
import asyncio
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mujoco_mcp.server import MuJoCoServer


def main():
    """Main entry point for Claude Desktop"""
    # Setup logging to stderr (Claude Desktop captures this)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stderr
    )
    
    logger = logging.getLogger("mujoco_mcp.stdio")
    logger.info("Starting MuJoCo MCP Server for Claude Desktop")
    
    async def run_server():
        try:
            # Create and initialize server
            server = MuJoCoServer()
            await server.initialize()
            
            logger.info(f"MuJoCo MCP Server v{server.version} initialized successfully")
            logger.info(f"Tools registered: {len(list(server.mcp._tool_manager._tools.values()))}")
            logger.info(f"Resources registered: {len(list(server.mcp._resource_manager._resources.values()))}")
            
            # Run the server (this will handle stdio communication)
            await server.mcp.run()
            
        except Exception as e:
            logger.error(f"Failed to start MuJoCo MCP Server: {e}")
            raise
    
    # Check if we're already in an event loop
    try:
        # Try to get the current event loop
        loop = asyncio.get_running_loop()
        # If we get here, we're in an event loop, run as task
        task = asyncio.create_task(run_server())
        return task
    except RuntimeError:
        # No event loop running, safe to create one
        asyncio.run(run_server())


if __name__ == "__main__":
    main()
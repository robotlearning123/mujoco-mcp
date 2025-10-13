#!/usr/bin/env python3
"""
Debug script to check MCP protocol version handling
"""

import asyncio
import json
import sys
from typing import Dict, Any, List

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

# Create server instance
server = Server("test-server")

@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """Return empty tool list for testing"""
    return []

@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle tool calls"""
    return [types.TextContent(type="text", text="Test response")]

async def test_initialization():
    """Test MCP initialization to see protocol version"""
    print("Testing MCP server initialization...", file=sys.stderr)
    
    # Initialize server capabilities
    server_options = InitializationOptions(
        server_name="test-mcp",
        server_version="0.1.0",
        capabilities=server.get_capabilities(
            notification_options=NotificationOptions(),
            experimental_capabilities={}
        )
    )
    
    print(f"Server options: {server_options}", file=sys.stderr)
    print(f"Capabilities: {server_options.capabilities}", file=sys.stderr)
    
    # Try to inspect the server object
    print(f"Server attributes: {[attr for attr in dir(server) if not attr.startswith('_')]}", file=sys.stderr)

if __name__ == "__main__":
    asyncio.run(test_initialization())
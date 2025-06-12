#!/usr/bin/env python3
"""
MuJoCo MCP Server for Claude Desktop - Final STDIO Version
Uses sync approach to avoid asyncio conflicts
"""
import sys
import os
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Setup logging to stderr for Claude Desktop
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)

logger = logging.getLogger("mujoco_mcp.stdio_final")

def main():
    """Main entry point that works with Claude Desktop's stdio expectations"""
    logger.info("Starting MuJoCo MCP Server for Claude Desktop")
    
    try:
        # Import here to ensure path is set
        from mujoco_mcp.simple_server import MuJoCoMCPServer
        
        # Create the simple server (not the FastMCP one)
        server = MuJoCoMCPServer()
        logger.info(f"MuJoCo MCP Server v{server.version} initialized")
        logger.info(f"Available tools: {len(server._tools)}")
        
        # For Claude Desktop, we need to implement a basic JSON-RPC stdio handler
        import json
        
        while True:
            try:
                # Read from stdin
                line = sys.stdin.readline()
                if not line:
                    break
                
                # Parse JSON-RPC request
                request = json.loads(line.strip())
                logger.info(f"Received request: {request.get('method', 'unknown')}")
                
                # Handle different MCP methods
                if request.get('method') == 'initialize':
                    response = {
                        "jsonrpc": "2.0",
                        "id": request.get('id'),
                        "result": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {
                                "tools": {},
                                "resources": {}
                            },
                            "serverInfo": {
                                "name": "mujoco-mcp",
                                "version": "0.6.0"
                            }
                        }
                    }
                    
                elif request.get('method') == 'notifications/initialized':
                    # This is a notification, no response needed
                    continue
                    
                elif request.get('method') == 'tools/list':
                    tools = server.get_tools()
                    response = {
                        "jsonrpc": "2.0", 
                        "id": request.get('id'),
                        "result": {
                            "tools": tools
                        }
                    }
                    
                elif request.get('method') == 'tools/call':
                    tool_name = request.get('params', {}).get('name')
                    arguments = request.get('params', {}).get('arguments', {})
                    
                    try:
                        result = server.call_tool(tool_name, arguments)
                        response = {
                            "jsonrpc": "2.0",
                            "id": request.get('id'),
                            "result": {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": json.dumps(result)
                                    }
                                ]
                            }
                        }
                    except Exception as e:
                        response = {
                            "jsonrpc": "2.0",
                            "id": request.get('id'),
                            "error": {
                                "code": -32000,
                                "message": str(e)
                            }
                        }
                
                else:
                    # Unknown method
                    response = {
                        "jsonrpc": "2.0",
                        "id": request.get('id'),
                        "error": {
                            "code": -32601,
                            "message": f"Method not found: {request.get('method')}"
                        }
                    }
                
                # Send response
                print(json.dumps(response), flush=True)
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                continue
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
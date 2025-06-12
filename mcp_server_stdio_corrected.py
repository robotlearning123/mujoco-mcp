#!/usr/bin/env python3
"""
MuJoCo MCP Server for Claude Desktop - Corrected STDIO Version
Implements proper MCP protocol compliance
"""
import sys
import os
import logging
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Setup logging to stderr for Claude Desktop
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)

logger = logging.getLogger("mujoco_mcp.stdio_corrected")

def main():
    """Main entry point with proper MCP protocol implementation"""
    logger.info("Starting MuJoCo MCP Server v0.6.0 for Claude Desktop")
    
    try:
        # Import here to ensure path is set
        from mujoco_mcp.simple_server import MuJoCoMCPServer
        
        # Create the server
        server = MuJoCoMCPServer()
        logger.info(f"MuJoCo MCP Server initialized with {len(server._tools)} tools")
        
        # MCP protocol handler
        while True:
            try:
                # Read from stdin
                line = sys.stdin.readline()
                if not line:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                # Parse JSON-RPC request
                try:
                    request = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}")
                    continue
                
                method = request.get('method')
                request_id = request.get('id')
                
                logger.info(f"Received request: {method}")
                
                # Handle MCP methods
                response = None
                
                if method == 'initialize':
                    # Initialize response
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {
                                "tools": {
                                    "listChanged": True
                                },
                                "resources": {
                                    "listChanged": True
                                },
                                "prompts": {
                                    "listChanged": True
                                }
                            },
                            "serverInfo": {
                                "name": "mujoco-mcp",
                                "version": "0.6.0"
                            }
                        }
                    }
                    
                elif method == 'notifications/initialized':
                    # This is a notification, no response needed
                    logger.info("Client initialized notification received")
                    continue
                    
                elif method == 'tools/list':
                    # List available tools
                    tools = server.get_tools()
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "tools": tools
                        }
                    }
                    
                elif method == 'tools/call':
                    # Call a tool
                    params = request.get('params', {})
                    tool_name = params.get('name')
                    arguments = params.get('arguments', {})
                    
                    try:
                        result = server.call_tool(tool_name, arguments)
                        response = {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "result": {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": json.dumps(result, indent=2)
                                    }
                                ]
                            }
                        }
                    except Exception as e:
                        logger.error(f"Tool call error: {e}")
                        response = {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {
                                "code": -32000,
                                "message": str(e)
                            }
                        }
                
                elif method == 'resources/list':
                    # List available resources
                    resources = server.get_resources()
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "resources": resources
                        }
                    }
                
                elif method == 'resources/read':
                    # Read a resource
                    params = request.get('params', {})
                    uri = params.get('uri', '')
                    
                    try:
                        # Simple resource handling - can be expanded
                        content = f"Resource {uri} content"
                        response = {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "result": {
                                "contents": [
                                    {
                                        "uri": uri,
                                        "mimeType": "text/plain",
                                        "text": content
                                    }
                                ]
                            }
                        }
                    except Exception as e:
                        logger.error(f"Resource read error: {e}")
                        response = {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {
                                "code": -32000,
                                "message": str(e)
                            }
                        }
                
                elif method == 'prompts/list':
                    # List available prompts
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "prompts": []
                        }
                    }
                
                elif method == 'prompts/get':
                    # Get a specific prompt
                    params = request.get('params', {})
                    prompt_name = params.get('name', '')
                    
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32602,
                            "message": f"Prompt not found: {prompt_name}"
                        }
                    }
                
                else:
                    # Unknown method
                    logger.warning(f"Unknown method: {method}")
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32601,
                            "message": f"Method not found: {method}"
                        }
                    }
                
                # Send response (only if not a notification)
                if response is not None:
                    print(json.dumps(response), flush=True)
                    
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                # Send error response if we can
                if 'request_id' in locals() and request_id is not None:
                    error_response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32603,
                            "message": f"Internal error: {str(e)}"
                        }
                    }
                    print(json.dumps(error_response), flush=True)
                continue
                
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
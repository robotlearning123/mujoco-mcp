#!/usr/bin/env python3
"""
MCP Bridge for OpenAI ChatGPT
Provides a bridge between ChatGPT and MCP servers via OpenAI API
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import subprocess
import time

import openai
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-bridge")

@dataclass
class MCPServer:
    """MCP Server configuration"""
    name: str
    command: str
    args: List[str]
    env: Dict[str, str]
    process: Optional[subprocess.Popen] = None

class MCPBridge:
    """Bridge between OpenAI ChatGPT and MCP servers"""
    
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
        self.servers: Dict[str, MCPServer] = {}
        self.conversations: Dict[str, List[Dict]] = {}
        
    def add_server(self, server: MCPServer):
        """Add an MCP server to the bridge"""
        self.servers[server.name] = server
        logger.info(f"Added MCP server: {server.name}")
        
    async def start_server(self, server_name: str) -> bool:
        """Start an MCP server"""
        if server_name not in self.servers:
            logger.error(f"Unknown server: {server_name}")
            return False
            
        server = self.servers[server_name]
        if server.process and server.process.poll() is None:
            logger.info(f"Server {server_name} already running")
            return True
            
        try:
            # Start the MCP server process
            env = os.environ.copy()
            env.update(server.env)
            
            server.process = subprocess.Popen(
                [server.command] + server.args,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Give it a moment to start
            await asyncio.sleep(1)
            
            if server.process.poll() is None:
                logger.info(f"Started MCP server: {server_name}")
                return True
            else:
                logger.error(f"Failed to start server {server_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error starting server {server_name}: {e}")
            return False
            
    async def stop_server(self, server_name: str):
        """Stop an MCP server"""
        if server_name not in self.servers:
            return
            
        server = self.servers[server_name]
        if server.process:
            server.process.terminate()
            await asyncio.sleep(0.5)
            if server.process.poll() is None:
                server.process.kill()
            server.process = None
            logger.info(f"Stopped MCP server: {server_name}")
            
    async def send_mcp_request(self, server_name: str, request: Dict) -> Dict:
        """Send a request to an MCP server"""
        if server_name not in self.servers:
            return {"error": f"Unknown server: {server_name}"}
            
        server = self.servers[server_name]
        if not server.process or server.process.poll() is not None:
            if not await self.start_server(server_name):
                return {"error": f"Could not start server: {server_name}"}
                
        try:
            # Send JSON-RPC request
            json_request = json.dumps(request) + "\\n"
            server.process.stdin.write(json_request)
            server.process.stdin.flush()
            
            # Read response
            response_line = server.process.stdout.readline()
            if response_line:
                return json.loads(response_line.strip())
            else:
                return {"error": "No response from server"}
                
        except Exception as e:
            logger.error(f"Error communicating with {server_name}: {e}")
            return {"error": str(e)}
            
    def create_function_schema(self, tool: Dict) -> Dict:
        """Convert MCP tool schema to OpenAI function schema"""
        return {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool.get("inputSchema", {
                "type": "object",
                "properties": {},
                "required": []
            })
        }
        
    async def get_available_functions(self, server_name: str) -> List[Dict]:
        """Get available functions from an MCP server"""
        response = await self.send_mcp_request(server_name, {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list"
        })
        
        if "error" in response:
            logger.error(f"Error getting tools from {server_name}: {response['error']}")
            return []
            
        tools = response.get("result", {}).get("tools", [])
        return [self.create_function_schema(tool) for tool in tools]
        
    async def execute_function(self, server_name: str, function_name: str, arguments: Dict) -> str:
        """Execute a function on an MCP server"""
        response = await self.send_mcp_request(server_name, {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": function_name,
                "arguments": arguments
            }
        })
        
        if "error" in response:
            return f"Error: {response['error']}"
            
        result = response.get("result", {})
        content = result.get("content", [])
        
        if content and isinstance(content, list):
            return "\\n".join([item.get("text", "") for item in content if item.get("type") == "text"])
        
        return str(result)
        
    async def chat_with_functions(self, message: str, conversation_id: str = "default") -> str:
        """Chat with OpenAI using MCP functions"""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
            
        conversation = self.conversations[conversation_id]
        
        # Add user message
        conversation.append({"role": "user", "content": message})
        
        # Get available functions from all servers
        all_functions = []
        for server_name in self.servers:
            functions = await self.get_available_functions(server_name)
            # Add server name to function metadata
            for func in functions:
                func["_server"] = server_name
            all_functions.extend(functions)
            
        if not all_functions:
            logger.warning("No MCP functions available")
            
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=conversation,
                functions=all_functions if all_functions else None,
                function_call="auto" if all_functions else None,
                temperature=0.7
            )
            
            message = response.choices[0].message
            
            # Check if function call was requested
            if message.function_call:
                function_name = message.function_call.name
                function_args = json.loads(message.function_call.arguments)
                
                # Find which server this function belongs to
                server_name = None
                for func in all_functions:
                    if func["name"] == function_name:
                        server_name = func.get("_server")
                        break
                        
                if server_name:
                    # Execute function on MCP server
                    function_result = await self.execute_function(
                        server_name, function_name, function_args
                    )
                    
                    # Add function call and result to conversation
                    conversation.append({
                        "role": "assistant", 
                        "content": None,
                        "function_call": {
                            "name": function_name,
                            "arguments": message.function_call.arguments
                        }
                    })
                    
                    conversation.append({
                        "role": "function",
                        "name": function_name,
                        "content": function_result
                    })
                    
                    # Get final response from OpenAI
                    final_response = self.client.chat.completions.create(
                        model="gpt-4",
                        messages=conversation,
                        temperature=0.7
                    )
                    
                    final_message = final_response.choices[0].message.content
                    conversation.append({"role": "assistant", "content": final_message})
                    
                    return final_message
                else:
                    error_msg = f"Unknown function: {function_name}"
                    conversation.append({"role": "assistant", "content": error_msg})
                    return error_msg
            else:
                # Regular response
                response_content = message.content
                conversation.append({"role": "assistant", "content": response_content})
                return response_content
                
        except Exception as e:
            error_msg = f"Error calling OpenAI API: {e}"
            logger.error(error_msg)
            return error_msg
            
    async def shutdown(self):
        """Shutdown all MCP servers"""
        for server_name in self.servers:
            await self.stop_server(server_name)

async def main():
    """Main entry point"""
    # Check for required environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable required")
        sys.exit(1)
        
    # Create bridge
    bridge = MCPBridge(openai_api_key)
    
    # Add MuJoCo MCP server
    mujoco_server = MCPServer(
        name="mujoco-mcp",
        command="python",
        args=["-m", "mujoco_mcp"],
        env={
            "PYTHONUNBUFFERED": "1",
            "MUJOCO_MCP_LOG_LEVEL": "INFO",
            "MUJOCO_GL": "osmesa"
        }
    )
    bridge.add_server(mujoco_server)
    
    try:
        print("MCP Bridge for OpenAI ChatGPT")
        print("Commands:")
        print("  /quit - Exit")
        print("  /servers - List servers")
        print("  /functions - List available functions")
        print("")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input == "/quit":
                    break
                elif user_input == "/servers":
                    print("Servers:", list(bridge.servers.keys()))
                    continue
                elif user_input == "/functions":
                    for server_name in bridge.servers:
                        functions = await bridge.get_available_functions(server_name)
                        print(f"{server_name}: {[f['name'] for f in functions]}")
                    continue
                elif not user_input:
                    continue
                    
                # Get response from ChatGPT with MCP functions
                response = await bridge.chat_with_functions(user_input)
                print(f"Assistant: {response}")
                print()
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                
    finally:
        print("Shutting down...")
        await bridge.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
Comprehensive MCP functionality tests
Tests MuJoCo MCP server functionality across different scenarios
"""

import asyncio
import json
import subprocess
import sys
import time
from typing import Dict, List, Optional
import pytest
import tempfile
import os

class MCPTester:
    """Test MuJoCo MCP server functionality"""
    
    def __init__(self):
        self.server_process: Optional[subprocess.Popen] = None
        self.test_results = []
        
    async def start_server(self, timeout: int = 10) -> bool:
        """Start MCP server for testing"""
        try:
            # Start the MCP server process
            self.server_process = subprocess.Popen(
                [sys.executable, "-m", "mujoco_mcp"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={
                    **os.environ,
                    "MUJOCO_GL": "osmesa",  # Use software rendering
                    "MUJOCO_MCP_LOG_LEVEL": "INFO"
                }
            )
            
            # Give server time to start
            await asyncio.sleep(2)
            
            # Check if process is running
            if self.server_process.poll() is None:
                return True
            else:
                print(f"Server failed to start. Return code: {self.server_process.returncode}")
                return False
                
        except Exception as e:
            print(f"Error starting server: {e}")
            return False
            
    async def stop_server(self):
        """Stop MCP server"""
        if self.server_process:
            self.server_process.terminate()
            await asyncio.sleep(1)
            if self.server_process.poll() is None:
                self.server_process.kill()
            self.server_process = None
            
    async def send_request(self, request: Dict, timeout: int = 5) -> Dict:
        """Send JSON-RPC request to MCP server"""
        if not self.server_process:
            return {"error": "Server not running"}
            
        try:
            # Send request
            request_json = json.dumps(request) + "\n"
            self.server_process.stdin.write(request_json)
            self.server_process.stdin.flush()
            
            # Read response with timeout
            response_line = ""
            start_time = time.time()
            while time.time() - start_time < timeout:
                if self.server_process.stdout.readable():
                    line = self.server_process.stdout.readline()
                    if line:
                        response_line = line.strip()
                        break
                await asyncio.sleep(0.1)
                
            if response_line:
                return json.loads(response_line)
            else:
                return {"error": "No response received"}
                
        except Exception as e:
            return {"error": f"Request failed: {e}"}
            
    def record_test(self, test_name: str, success: bool, details: str = ""):
        """Record test result"""
        self.test_results.append({
            "test": test_name,
            "success": success,
            "details": details,
            "timestamp": time.time()
        })
        
    async def test_server_info(self) -> bool:
        """Test get_server_info tool"""
        print("Testing get_server_info...")
        
        response = await self.send_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "get_server_info",
                "arguments": {}
            }
        })
        
        if "error" in response:
            self.record_test("get_server_info", False, f"Error: {response['error']}")
            return False
            
        result = response.get("result", {})
        content = result.get("content", [])
        
        if content and isinstance(content, list) and len(content) > 0:
            text_content = content[0].get("text", "")
            try:
                server_info = json.loads(text_content)
                if "name" in server_info and "version" in server_info:
                    self.record_test("get_server_info", True, f"Server: {server_info['name']} v{server_info['version']}")
                    return True
            except json.JSONDecodeError:
                pass
                
        self.record_test("get_server_info", False, "Invalid response format")
        return False
        
    async def test_list_tools(self) -> bool:
        """Test tools/list method"""
        print("Testing tools/list...")
        
        response = await self.send_request({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        })
        
        if "error" in response:
            self.record_test("tools/list", False, f"Error: {response['error']}")
            return False
            
        result = response.get("result", {})
        tools = result.get("tools", [])
        
        expected_tools = [
            "get_server_info",
            "create_scene", 
            "step_simulation",
            "get_state",
            "reset_simulation",
            "close_viewer"
        ]
        
        found_tools = [tool.get("name") for tool in tools]
        missing_tools = [tool for tool in expected_tools if tool not in found_tools]
        
        if not missing_tools:
            self.record_test("tools/list", True, f"Found {len(tools)} tools")
            return True
        else:
            self.record_test("tools/list", False, f"Missing tools: {missing_tools}")
            return False
            
    async def test_create_scene(self) -> bool:
        """Test create_scene tool"""
        print("Testing create_scene...")
        
        # Test with pendulum scene
        response = await self.send_request({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "create_scene",
                "arguments": {"scene_type": "pendulum"}
            }
        })
        
        if "error" in response:
            self.record_test("create_scene", False, f"Error: {response['error']}")
            return False
            
        result = response.get("result", {})
        content = result.get("content", [])
        
        if content and isinstance(content, list) and len(content) > 0:
            text_content = content[0].get("text", "")
            if "Created pendulum scene successfully" in text_content or "✅" in text_content:
                self.record_test("create_scene", True, "Pendulum scene created")
                return True
                
        self.record_test("create_scene", False, f"Unexpected response: {content}")
        return False
        
    async def test_invalid_scene(self) -> bool:
        """Test create_scene with invalid scene type"""
        print("Testing invalid scene type...")
        
        response = await self.send_request({
            "jsonrpc": "2.0", 
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "create_scene",
                "arguments": {"scene_type": "invalid_scene"}
            }
        })
        
        if "error" in response:
            self.record_test("invalid_scene_type", True, "Correctly rejected invalid scene")
            return True
            
        result = response.get("result", {})
        content = result.get("content", [])
        
        if content and isinstance(content, list) and len(content) > 0:
            text_content = content[0].get("text", "")
            if "Unknown scene type" in text_content or "❌" in text_content:
                self.record_test("invalid_scene_type", True, "Correctly rejected invalid scene")
                return True
                
        self.record_test("invalid_scene_type", False, "Should have rejected invalid scene")
        return False
        
    async def test_all_scene_types(self) -> bool:
        """Test all available scene types"""
        print("Testing all scene types...")
        
        scene_types = ["pendulum", "double_pendulum", "cart_pole"]
        success_count = 0
        
        for scene_type in scene_types:
            print(f"  Testing {scene_type}...")
            response = await self.send_request({
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call", 
                "params": {
                    "name": "create_scene",
                    "arguments": {"scene_type": scene_type}
                }
            })
            
            if "error" not in response:
                result = response.get("result", {})
                content = result.get("content", [])
                if content and "✅" in str(content):
                    success_count += 1
                    
        if success_count == len(scene_types):
            self.record_test("all_scene_types", True, f"All {len(scene_types)} scene types work")
            return True
        else:
            self.record_test("all_scene_types", False, f"Only {success_count}/{len(scene_types)} scene types work")
            return False
            
    async def test_step_simulation(self) -> bool:
        """Test step_simulation tool"""
        print("Testing step_simulation...")
        
        # First create a scene
        await self.send_request({
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "create_scene",
                "arguments": {"scene_type": "pendulum"}
            }
        })
        
        # Then try to step simulation
        response = await self.send_request({
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {
                "name": "step_simulation",
                "arguments": {"model_id": "pendulum", "steps": 10}
            }
        })
        
        if "error" in response:
            self.record_test("step_simulation", False, f"Error: {response['error']}")
            return False
            
        result = response.get("result", {})
        content = result.get("content", [])
        
        if content and isinstance(content, list) and len(content) > 0:
            text_content = content[0].get("text", "")
            if "Stepped simulation" in text_content or "⏩" in text_content:
                self.record_test("step_simulation", True, "Simulation stepped")
                return True
                
        self.record_test("step_simulation", False, f"Unexpected response: {content}")
        return False
        
    async def test_get_state(self) -> bool:
        """Test get_state tool"""
        print("Testing get_state...")
        
        response = await self.send_request({
            "jsonrpc": "2.0",
            "id": 8,
            "method": "tools/call",
            "params": {
                "name": "get_state", 
                "arguments": {"model_id": "pendulum"}
            }
        })
        
        if "error" in response:
            self.record_test("get_state", False, f"Error: {response['error']}")
            return False
            
        result = response.get("result", {})
        content = result.get("content", [])
        
        if content and isinstance(content, list) and len(content) > 0:
            text_content = content[0].get("text", "")
            try:
                # Try to parse as JSON state
                state_data = json.loads(text_content)
                if isinstance(state_data, dict):
                    self.record_test("get_state", True, "State retrieved successfully")
                    return True
            except json.JSONDecodeError:
                pass
                
        self.record_test("get_state", False, "Could not parse state data")
        return False
        
    async def test_reset_simulation(self) -> bool:
        """Test reset_simulation tool"""
        print("Testing reset_simulation...")
        
        response = await self.send_request({
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tools/call",
            "params": {
                "name": "reset_simulation",
                "arguments": {"model_id": "pendulum"}
            }
        })
        
        if "error" in response:
            self.record_test("reset_simulation", False, f"Error: {response['error']}")
            return False
            
        result = response.get("result", {})
        content = result.get("content", [])
        
        if content and isinstance(content, list) and len(content) > 0:
            text_content = content[0].get("text", "")
            if "reset" in text_content.lower() or "🔄" in text_content:
                self.record_test("reset_simulation", True, "Simulation reset")
                return True
                
        self.record_test("reset_simulation", False, f"Unexpected response: {content}")
        return False
        
    async def test_error_handling(self) -> bool:
        """Test error handling with invalid requests"""
        print("Testing error handling...")
        
        # Test invalid method
        response1 = await self.send_request({
            "jsonrpc": "2.0",
            "id": 10,
            "method": "invalid/method"
        })
        
        # Test invalid tool
        response2 = await self.send_request({
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {
                "name": "invalid_tool",
                "arguments": {}
            }
        })
        
        errors_handled = 0
        if "error" in response1:
            errors_handled += 1
        if "error" in response2 or ("result" in response2 and "Unknown tool" in str(response2["result"])):
            errors_handled += 1
            
        if errors_handled == 2:
            self.record_test("error_handling", True, "Errors handled correctly")
            return True
        else:
            self.record_test("error_handling", False, f"Only {errors_handled}/2 errors handled")
            return False
            
    async def run_all_tests(self) -> Dict:
        """Run comprehensive test suite"""
        print("🧪 Starting MuJoCo MCP comprehensive tests...")
        
        # Start server
        if not await self.start_server():
            return {"success": False, "error": "Could not start MCP server"}
            
        try:
            # Run all tests
            tests = [
                self.test_server_info,
                self.test_list_tools,
                self.test_create_scene,
                self.test_invalid_scene,
                self.test_all_scene_types,
                self.test_step_simulation,
                self.test_get_state,
                self.test_reset_simulation,
                self.test_error_handling
            ]
            
            total_tests = len(tests)
            passed_tests = 0
            
            for test_func in tests:
                try:
                    if await test_func():
                        passed_tests += 1
                    await asyncio.sleep(0.5)  # Brief pause between tests
                except Exception as e:
                    print(f"Test {test_func.__name__} failed with exception: {e}")
                    
        finally:
            await self.stop_server()
            
        # Generate report
        success_rate = passed_tests / total_tests
        
        return {
            "success": success_rate >= 0.8,  # 80% success rate required
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": success_rate,
            "test_results": self.test_results
        }

async def main():
    """Main test function"""
    tester = MCPTester()
    
    try:
        results = await tester.run_all_tests()
        
        # Print results
        print(f"\\n🔍 Test Results:")
        print(f"Total Tests: {results['total_tests']}")
        print(f"Passed: {results['passed_tests']}")
        print(f"Success Rate: {results['success_rate']:.1%}")
        
        if results['success']:
            print("✅ Overall: PASSED")
        else:
            print("❌ Overall: FAILED")
            
        # Detailed results
        print(f"\\n📊 Detailed Results:")
        for result in results['test_results']:
            status = "✅" if result['success'] else "❌"
            print(f"  {status} {result['test']}: {result['details']}")
            
        return 0 if results['success'] else 1
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
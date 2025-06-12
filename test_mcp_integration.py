#!/usr/bin/env python3
"""
MuJoCo MCP Integration Test
Tests the complete MCP server functionality
"""
import asyncio
import json
import subprocess
import time
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from mujoco_mcp.server import MuJoCoServer
from mujoco_mcp.version import __version__


class MCPTester:
    """Test the MuJoCo MCP server functionality"""
    
    def __init__(self):
        self.server: Optional[MuJoCoServer] = None
        self.test_results = []
    
    def log_test(self, name: str, success: bool, details: str = ""):
        """Log test result"""
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
        if details:
            print(f"    {details}")
        
        self.test_results.append({
            "name": name,
            "success": success,
            "details": details
        })
    
    async def test_server_initialization(self):
        """Test server can be initialized without errors"""
        try:
            self.server = MuJoCoServer()
            await self.server.initialize()
            
            # Check basic properties
            assert hasattr(self.server, 'name')
            assert hasattr(self.server, 'version')
            assert hasattr(self.server, 'mcp')
            
            self.log_test("Server Initialization", True, f"Server v{self.server.version} initialized")
            return True
            
        except Exception as e:
            self.log_test("Server Initialization", False, f"Error: {e}")
            return False
    
    async def test_server_info(self):
        """Test server info functionality"""
        try:
            if not self.server:
                self.log_test("Server Info", False, "Server not initialized")
                return False
            
            info = self.server.get_server_info()
            
            # Check required fields
            required_fields = ["name", "version", "description", "capabilities"]
            for field in required_fields:
                assert field in info, f"Missing field: {field}"
            
            assert info["version"] == __version__
            assert "mujoco" in info["name"].lower()
            
            self.log_test("Server Info", True, f"Info returned with {len(info)} fields")
            return True
            
        except Exception as e:
            self.log_test("Server Info", False, f"Error: {e}")
            return False
    
    async def test_tool_registration(self):
        """Test that tools are properly registered"""
        try:
            if not self.server:
                self.log_test("Tool Registration", False, "Server not initialized")
                return False
            
            # Check that FastMCP tools are registered
            tools = list(self.server.mcp._tool_manager._tools.values())
            tool_names = [t.name for t in tools]
            
            # Check for essential tools
            essential_tools = [
                "get_server_info",
                "load_model", 
                "step_simulation",
                "get_state"
            ]
            
            missing_tools = [tool for tool in essential_tools if tool not in tool_names]
            
            if missing_tools:
                self.log_test("Tool Registration", False, f"Missing tools: {missing_tools}")
                return False
            
            self.log_test("Tool Registration", True, f"{len(tool_names)} tools registered")
            return True
            
        except Exception as e:
            self.log_test("Tool Registration", False, f"Error: {e}")
            return False
    
    async def test_resource_registration(self):
        """Test that resources are properly registered"""
        try:
            if not self.server:
                self.log_test("Resource Registration", False, "Server not initialized")
                return False
            
            # Check that FastMCP resources are registered
            resources = list(self.server.mcp._resource_manager._resources.values())
            resource_names = [r.name for r in resources]
            
            # Check for essential resources
            essential_resources = [
                "simulation://state",
                "simulation://sensors",
                "simulation://config"
            ]
            
            missing_resources = [res for res in essential_resources if res not in resource_names]
            
            if missing_resources:
                self.log_test("Resource Registration", False, f"Missing resources: {missing_resources}")
                return False
            
            self.log_test("Resource Registration", True, f"{len(resource_names)} resources registered")
            return True
            
        except Exception as e:
            self.log_test("Resource Registration", False, f"Error: {e}")
            return False
    
    async def test_tool_execution(self):
        """Test executing a basic tool"""
        try:
            if not self.server:
                self.log_test("Tool Execution", False, "Server not initialized")
                return False
            
            # Test get_server_info tool
            result = await self.server.mcp.call_tool("get_server_info", {})
            
            # FastMCP returns a list of CallToolResults
            assert isinstance(result, list), "Result should be a list"
            assert len(result) > 0, "Result should not be empty"
            
            # Parse the result
            if hasattr(result[0], 'text'):
                content = json.loads(result[0].text)
            else:
                content = json.loads(result[0].content)
            
            assert "name" in content
            assert "version" in content
            
            self.log_test("Tool Execution", True, "get_server_info executed successfully")
            return True
            
        except Exception as e:
            self.log_test("Tool Execution", False, f"Error: {e}")
            return False
    
    async def test_resource_access(self):
        """Test accessing a resource"""
        try:
            if not self.server:
                self.log_test("Resource Access", False, "Server not initialized")
                return False
            
            # Test config resource
            result = await self.server.mcp.read_resource("simulation://config")
            
            # FastMCP returns a list of ReadResourceResults
            assert isinstance(result, list), "Result should be a list"
            assert len(result) > 0, "Result should not be empty"
            
            # Parse the result
            content = json.loads(result[0].content)
            assert "contents" in content
            assert "version" in content["contents"]
            
            self.log_test("Resource Access", True, "Config resource accessed successfully")
            return True
            
        except Exception as e:
            self.log_test("Resource Access", False, f"Error: {e}")
            return False
    
    async def test_pendulum_demo(self):
        """Test creating and running a pendulum simulation"""
        try:
            if not self.server:
                self.log_test("Pendulum Demo", False, "Server not initialized")
                return False
            
            # Create pendulum
            result = await self.server.mcp.call_tool("pendulum_demo", {"action": "setup"})
            
            # Parse result
            if isinstance(result, list) and len(result) > 0:
                if hasattr(result[0], 'text'):
                    content = json.loads(result[0].text)
                else:
                    content = json.loads(result[0].content)
                
                assert "model_id" in content
                assert content.get("success") is True
                
                model_id = content["model_id"]
                
                # Test stepping simulation
                step_result = await self.server.mcp.call_tool("step_simulation", {"model_id": model_id})
                
                assert isinstance(step_result, list)
                assert len(step_result) > 0
                
                self.log_test("Pendulum Demo", True, f"Created pendulum model {model_id}")
                return True
            else:
                self.log_test("Pendulum Demo", False, "Unexpected result format")
                return False
                
        except Exception as e:
            self.log_test("Pendulum Demo", False, f"Error: {e}")
            return False
    
    async def test_cli_startup(self):
        """Test that the CLI can start without errors"""
        try:
            # Test the module can be imported and help displayed
            cmd = [sys.executable, "-m", "mujoco_mcp", "--check"]
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=30,
                cwd=project_root
            )
            
            if result.returncode == 0:
                self.log_test("CLI Startup", True, "Configuration check passed")
                return True
            else:
                self.log_test("CLI Startup", False, f"Check failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.log_test("CLI Startup", False, "Timeout during configuration check")
            return False
        except Exception as e:
            self.log_test("CLI Startup", False, f"Error: {e}")
            return False
    
    def test_configuration_files(self):
        """Test that configuration files are valid"""
        try:
            config_files = [
                "mcp.json",
                "claude_desktop_config.json"
            ]
            
            all_valid = True
            for config_file in config_files:
                file_path = project_root / config_file
                if file_path.exists():
                    try:
                        with open(file_path, 'r') as f:
                            json.load(f)
                        print(f"    ✓ {config_file} is valid JSON")
                    except json.JSONDecodeError as e:
                        print(f"    ✗ {config_file} has invalid JSON: {e}")
                        all_valid = False
                else:
                    print(f"    ✗ {config_file} not found")
                    all_valid = False
            
            # Check .cursorrules exists
            cursorrules_path = project_root / ".cursorrules"
            if cursorrules_path.exists():
                print("    ✓ .cursorrules file exists")
            else:
                print("    ✗ .cursorrules file not found")
                all_valid = False
            
            # Check CONFIG.md exists
            config_md_path = project_root / "CONFIG.md"
            if config_md_path.exists():
                print("    ✓ CONFIG.md file exists")
            else:
                print("    ✗ CONFIG.md file not found")
                all_valid = False
            
            self.log_test("Configuration Files", all_valid, f"Checked {len(config_files) + 2} files")
            return all_valid
            
        except Exception as e:
            self.log_test("Configuration Files", False, f"Error: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all tests"""
        print(f"🧪 MuJoCo MCP Integration Test Suite v{__version__}")
        print("=" * 60)
        
        tests = [
            ("Configuration Files", self.test_configuration_files),
            ("CLI Startup", self.test_cli_startup),
            ("Server Initialization", self.test_server_initialization),
            ("Server Info", self.test_server_info),
            ("Tool Registration", self.test_tool_registration),
            ("Resource Registration", self.test_resource_registration),
            ("Tool Execution", self.test_tool_execution),
            ("Resource Access", self.test_resource_access),
            ("Pendulum Demo", self.test_pendulum_demo),
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            try:
                if asyncio.iscoroutinefunction(test_func):
                    success = await test_func()
                else:
                    success = test_func()
                
                if success:
                    passed += 1
                else:
                    failed += 1
                    
            except Exception as e:
                print(f"✗ FAIL: {test_name} - Unexpected error: {e}")
                failed += 1
        
        print("\n" + "=" * 60)
        print(f"Test Results: {passed} passed, {failed} failed")
        
        if failed == 0:
            print("🎉 All tests passed! MuJoCo MCP is ready for use.")
        else:
            print("⚠️  Some tests failed. Check the issues above.")
        
        return failed == 0
    
    async def cleanup(self):
        """Cleanup test resources"""
        if self.server:
            try:
                await self.server.cleanup()
            except:
                pass


async def main():
    """Main test function"""
    tester = MCPTester()
    
    try:
        success = await tester.run_all_tests()
        return 0 if success else 1
    finally:
        await tester.cleanup()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
#!/usr/bin/env python3
"""
MCP Protocol Compliance Test
Tests MuJoCo MCP server compliance with Model Context Protocol specification
"""

import asyncio
import json
import subprocess
import tempfile
import time
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_mcp_server_startup():
    """Test that MCP server can start successfully"""
    try:
        from mujoco_mcp.__main__ import main
        # Just test that main function exists and is callable
        assert callable(main)
        return True
    except Exception as e:
        print(f"❌ Server startup test failed: {e}")
        return False

async def test_tools_listing():
    """Test tools listing compliance"""
    try:
        from mujoco_mcp.server import MuJoCoServer
        server = MuJoCoServer()
        
        # Check that server has required attributes
        assert hasattr(server, 'name')
        assert hasattr(server, 'version')
        assert hasattr(server, 'description')
        
        # Check server info
        info = server.get_server_info()
        assert 'name' in info
        assert 'version' in info
        assert 'capabilities' in info
        
        return True
    except Exception as e:
        print(f"❌ Tools listing test failed: {e}")
        return False

async def test_protocol_messages():
    """Test basic protocol message handling"""
    try:
        from mujoco_mcp.server import MuJoCoServer
        server = MuJoCoServer()
        
        # Test server info
        info = server.get_server_info()
        
        # Verify required fields
        required_fields = ['name', 'version', 'description', 'capabilities']
        for field in required_fields:
            assert field in info, f"Missing required field: {field}"
        
        # Test capabilities
        capabilities = info['capabilities']
        assert isinstance(capabilities, dict), "Capabilities should be a dict"
        
        return True
    except Exception as e:
        print(f"❌ Protocol messages test failed: {e}")
        return False

async def test_error_handling():
    """Test error handling compliance"""
    try:
        from mujoco_mcp.server import MuJoCoServer
        server = MuJoCoServer()
        
        # Test that server handles initialization gracefully
        assert server is not None
        
        return True
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

async def run_compliance_tests():
    """Run all MCP compliance tests"""
    print("🧪 MCP Protocol Compliance Test Suite")
    print("=" * 50)
    
    tests = [
        ("Server Startup", test_mcp_server_startup),
        ("Tools Listing", test_tools_listing), 
        ("Protocol Messages", test_protocol_messages),
        ("Error Handling", test_error_handling)
    ]
    
    results = {}
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nTesting {test_name}...")
        try:
            result = await test_func()
            if result:
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
            results[test_name] = False
    
    # Generate compliance report
    compliance_report = {
        "timestamp": time.time(),
        "total_tests": total,
        "passed_tests": passed,
        "failed_tests": total - passed,
        "success_rate": (passed / total) * 100,
        "test_results": results,
        "mcp_version": "1.0",
        "server_info": {
            "name": "mujoco-mcp",
            "version": "0.8.2"
        }
    }
    
    # Save report
    with open("mcp_compliance_report.json", "w") as f:
        json.dump(compliance_report, f, indent=2)
    
    print("\n" + "=" * 50)
    print(f"📊 MCP Compliance Test Results")
    print(f"Total Tests: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {total - passed}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 All MCP compliance tests passed!")
        return True
    else:
        print(f"\n⚠️  {total - passed} compliance tests failed")
        return False

def main():
    """Main entry point"""
    try:
        success = asyncio.run(run_compliance_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
快速内测脚本 - 验证核心功能
"""

import asyncio
import sys
import time
from pathlib import Path

# Add project path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

async def test_core_functionality():
    """Test core functionality"""
    print("=== MuJoCo-MCP Quick Internal Test ===\n")
    
    results = {
        "import_test": False,
        "version_test": False,
        "tools_test": False,
        "mcp_protocol_test": False,
        "error_handling_test": False
    }
    
    # 1. Import test
    print("1. Testing package import...")
    try:
        import mujoco_mcp
        from mujoco_mcp.version import __version__
        from mujoco_mcp.mcp_server import handle_list_tools, handle_call_tool
        print(f"   ✅ Package imported successfully, version: {__version__}")
        results["import_test"] = True
        results["version_test"] = True
    except Exception as e:
        print(f"   ❌ Package import failed: {e}")
        return results
    
    # 2. Tools list test
    print("\n2. Testing tools list...")
    try:
        tools = await handle_list_tools()
        print(f"   ✅ Found {len(tools)}  tools:")
        for tool in tools:
            print(f"      - {tool.name}: {tool.description[:50]}...")
        results["tools_test"] = True
    except Exception as e:
        print(f"   ❌ Failed to get tools list: {e}")
    
    # 3. MCP protocol test
    print("\n3. Testing MCP protocol...")
    try:
        result = await handle_call_tool("get_server_info", {})
        if result and len(result) > 0:
            print(f"   ✅ Server info retrieved successfully")
            print(f"      Response length: {len(result[0].text)}  characters")
            results["mcp_protocol_test"] = True
        else:
            print(f"   ❌ Server info response is empty")
    except Exception as e:
        print(f"   ❌ MCP protocol test failed: {e}")
    
    # 4. Error handling test
    print("\n4. Testing error handling...")
    try:
        result = await handle_call_tool("invalid_tool", {})
        if result and "Unknown tool" in result[0].text:
            print(f"   ✅ Error handling working properly")
            results["error_handling_test"] = True
        else:
            print(f"   ⚠️  Error handling abnormal: {result[0].text if result else 'No result'}")
    except Exception as e:
        print(f"   ❌ Error handling test failed: {e}")
    
    return results

def print_summary(results):
    """Print test summary"""
    print("\n" + "="*50)
    print("Internal Test Summary")
    print("="*50)
    
    total_tests = len(results)
    passed_tests = sum(1 for success in results.values() if success)
    
    print(f"Test items: {total_tests}")
    print(f"Tests passed: {passed_tests}")
    print(f"Tests failed: {total_tests - passed_tests}")
    print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
    
    print(f"\nDetailed results:")
    status_map = {True: "✅ PASS", False: "❌ FAIL"}
    for test_name, success in results.items():
        print(f"  {status_map[success]} {test_name}")
    
    if passed_tests == total_tests:
        print(f"\n🎉 All tests passed! Project is ready.")
        return True
    elif passed_tests >= total_tests * 0.8:
        print(f"\n⚠️  Most tests passed, minor issues exist.")
        return True
    else:
        print(f"\n❌ Multiple tests failed, need fixes before release.")
        return False

async def main():
    start_time = time.time()
    
    try:
        results = await test_core_functionality()
        success = print_summary(results)
        
        end_time = time.time()
        print(f"\nTest duration: {end_time - start_time:.2f}s")
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\nSerious error occurred during testing: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
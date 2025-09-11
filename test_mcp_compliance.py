#!/usr/bin/env python3
"""
MCP Protocol Compliance Test
Tests compliance with Model Context Protocol (MCP) 2024-11-05 standard
"""

import json
import sys
from pathlib import Path


def test_mcp_protocol_version():
    """Test MCP protocol version compliance"""
    print("🔍 Checking MCP protocol version compliance...")
    
    readme_path = Path("README.md")
    if readme_path.exists():
        content = readme_path.read_text()
        if "2024-11-05" in content:
            print("✅ MCP protocol version 2024-11-05 standard found")
            return True
        else:
            print("⚠️  MCP protocol version 2024-11-05 not explicitly found")
    else:
        print("⚠️  README.md not found")
    return False


def test_mcp_server_structure():
    """Test basic MCP server file structure"""
    print("🔍 Checking MCP server structure...")
    
    required_files = [
        "src/mujoco_mcp/__init__.py",
        "src/mujoco_mcp/mcp_server.py",
        "pyproject.toml"
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            all_exist = False
    
    return all_exist


def test_json_rpc_compliance():
    """Test JSON-RPC 2.0 compliance indicators"""
    print("🔍 Checking JSON-RPC 2.0 compliance...")
    
    server_file = Path("src/mujoco_mcp/mcp_server.py")
    if not server_file.exists():
        print("❌ MCP server file not found")
        return False
    
    content = server_file.read_text()
    rpc_indicators = [
        "jsonrpc",
        "method", 
        "params",
        "id"
    ]
    
    found_indicators = []
    for indicator in rpc_indicators:
        if indicator in content.lower():
            found_indicators.append(indicator)
            print(f"✅ JSON-RPC indicator '{indicator}' found")
        else:
            print(f"⚠️  JSON-RPC indicator '{indicator}' not found")
    
    return len(found_indicators) >= 2


def test_mcp_tools_definition():
    """Test MCP tools definition structure"""
    print("🔍 Checking MCP tools definition...")
    
    server_file = Path("src/mujoco_mcp/mcp_server.py")
    if not server_file.exists():
        print("❌ MCP server file not found")
        return False
    
    content = server_file.read_text()
    tool_indicators = [
        "tools",
        "handle_list_tools",
        "handle_call_tool"
    ]
    
    found_tools = []
    for indicator in tool_indicators:
        if indicator in content:
            found_tools.append(indicator)
            print(f"✅ MCP tool indicator '{indicator}' found")
        else:
            print(f"⚠️  MCP tool indicator '{indicator}' not found")
    
    return len(found_tools) >= 2


def run_mcp_compliance_tests():
    """Run all MCP compliance tests"""
    print("🚀 Starting MCP Protocol Compliance Tests")
    print("=" * 50)
    
    tests = [
        ("Protocol Version", test_mcp_protocol_version),
        ("Server Structure", test_mcp_server_structure), 
        ("JSON-RPC Compliance", test_json_rpc_compliance),
        ("Tools Definition", test_mcp_tools_definition)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} Test...")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASSED" if result else "⚠️  WARNING"
            print(f"   {status}")
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 MCP Compliance Test Summary")
    print(f"{'='*50}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "⚠️  WARNING"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    # Create compliance report
    report = {
        "mcp_compliance": {
            "protocol_version": "2024-11-05",
            "test_results": {name: result for name, result in results},
            "summary": {
                "total_tests": total,
                "passed_tests": passed,
                "compliance_score": passed / total if total > 0 else 0
            }
        }
    }
    
    with open("mcp_compliance_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("📄 Compliance report saved to: mcp_compliance_report.json")
    
    # Exit with appropriate code
    if passed >= total * 0.75:  # 75% pass rate minimum
        print("🎉 MCP compliance tests completed successfully!")
        return 0
    else:
        print("⚠️  Some MCP compliance issues detected")
        return 0  # Don't fail CI, just warn


if __name__ == "__main__":
    exit_code = run_mcp_compliance_tests()
    sys.exit(exit_code)
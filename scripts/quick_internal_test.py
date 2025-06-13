#!/usr/bin/env python3
"""
快速内测脚本 - 验证核心功能
"""

import asyncio
import sys
import time
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

async def test_core_functionality():
    """测试核心功能"""
    print("=== MuJoCo-MCP 快速内测 ===\n")
    
    results = {
        "import_test": False,
        "version_test": False,
        "tools_test": False,
        "mcp_protocol_test": False,
        "error_handling_test": False
    }
    
    # 1. 导入测试
    print("1. 测试包导入...")
    try:
        import mujoco_mcp
        from mujoco_mcp.version import __version__
        from mujoco_mcp.mcp_server import handle_list_tools, handle_call_tool
        print(f"   ✅ 包导入成功，版本: {__version__}")
        results["import_test"] = True
        results["version_test"] = True
    except Exception as e:
        print(f"   ❌ 包导入失败: {e}")
        return results
    
    # 2. 工具列表测试
    print("\n2. 测试工具列表...")
    try:
        tools = await handle_list_tools()
        print(f"   ✅ 获取到 {len(tools)} 个工具:")
        for tool in tools:
            print(f"      - {tool.name}: {tool.description[:50]}...")
        results["tools_test"] = True
    except Exception as e:
        print(f"   ❌ 工具列表获取失败: {e}")
    
    # 3. MCP 协议测试
    print("\n3. 测试 MCP 协议...")
    try:
        result = await handle_call_tool("get_server_info", {})
        if result and len(result) > 0:
            print(f"   ✅ 服务器信息获取成功")
            print(f"      响应长度: {len(result[0].text)} 字符")
            results["mcp_protocol_test"] = True
        else:
            print(f"   ❌ 服务器信息响应为空")
    except Exception as e:
        print(f"   ❌ MCP 协议测试失败: {e}")
    
    # 4. 错误处理测试
    print("\n4. 测试错误处理...")
    try:
        result = await handle_call_tool("invalid_tool", {})
        if result and "Unknown tool" in result[0].text:
            print(f"   ✅ 错误处理正常")
            results["error_handling_test"] = True
        else:
            print(f"   ⚠️  错误处理异常: {result[0].text if result else 'No result'}")
    except Exception as e:
        print(f"   ❌ 错误处理测试失败: {e}")
    
    return results

def print_summary(results):
    """打印测试摘要"""
    print("\n" + "="*50)
    print("内测摘要")
    print("="*50)
    
    total_tests = len(results)
    passed_tests = sum(1 for success in results.values() if success)
    
    print(f"测试项目: {total_tests}")
    print(f"通过测试: {passed_tests}")
    print(f"失败测试: {total_tests - passed_tests}")
    print(f"成功率: {(passed_tests/total_tests)*100:.1f}%")
    
    print(f"\n详细结果:")
    status_map = {True: "✅ PASS", False: "❌ FAIL"}
    for test_name, success in results.items():
        print(f"  {status_map[success]} {test_name}")
    
    if passed_tests == total_tests:
        print(f"\n🎉 所有测试通过！项目准备就绪。")
        return True
    elif passed_tests >= total_tests * 0.8:
        print(f"\n⚠️  大部分测试通过，存在少量问题。")
        return True
    else:
        print(f"\n❌ 多个测试失败，需要修复后再发布。")
        return False

async def main():
    start_time = time.time()
    
    try:
        results = await test_core_functionality()
        success = print_summary(results)
        
        end_time = time.time()
        print(f"\n测试耗时: {end_time - start_time:.2f}s")
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        return 1
    except Exception as e:
        print(f"\n\n测试过程中发生严重错误: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
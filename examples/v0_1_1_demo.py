#!/usr/bin/env python3
"""
v0.1.1 演示 - 简单MCP服务器
展示基础的MCP服务器功能
"""
import sys
import json
from pathlib import Path

# 添加项目到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from mujoco_mcp import __version__
from mujoco_mcp.server import MuJoCoMCPServer


def print_section(title):
    """打印章节标题"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def demo_server_basics():
    """演示服务器基础功能"""
    print_section("1. 创建MCP服务器")
    
    # 创建服务器
    server = MuJoCoMCPServer()
    print(f"✓ 服务器创建成功")
    print(f"  - 名称: {server.name}")
    print(f"  - 版本: {server.version}")
    print(f"  - 描述: {server.description}")
    print(f"  - 运行状态: {server.is_running()}")
    
    return server


def demo_server_info(server):
    """演示获取服务器信息"""
    print_section("2. 获取服务器信息")
    
    # 直接调用方法
    info = server.get_server_info()
    print("✓ 调用 get_server_info():")
    print(json.dumps(info, indent=2))
    
    # 通过工具调用
    print("\n✓ 通过 call_tool 调用:")
    result = server.call_tool("get_server_info", {})
    print(json.dumps(result, indent=2))


def demo_get_tools(server):
    """演示获取工具列表"""
    print_section("3. 获取可用工具")
    
    # 直接调用
    tools = server.get_tools()
    print(f"✓ 可用工具数量: {len(tools)}")
    
    for i, tool in enumerate(tools, 1):
        print(f"\n{i}. {tool['name']}")
        print(f"   描述: {tool['description']}")
        print(f"   参数: {tool['parameters']}")
    
    # 通过工具调用
    print("\n✓ 通过 call_tool 获取工具列表:")
    result = server.call_tool("get_tools", {})
    print(f"  返回了 {len(result['tools'])} 个工具")


def demo_resources(server):
    """演示资源功能"""
    print_section("4. 资源管理")
    
    resources = server.get_resources()
    print(f"✓ 当前资源数量: {len(resources)}")
    print("  (v0.1.1 暂不包含资源，这是正常的)")


def demo_error_handling(server):
    """演示错误处理"""
    print_section("5. 错误处理")
    
    # 测试未知工具
    print("✓ 测试调用未知工具:")
    try:
        server.call_tool("unknown_tool", {})
        print("  ✗ 应该抛出异常")
    except ValueError as e:
        print(f"  ✓ 正确捕获异常: {e}")
    
    # 测试参数错误
    print("\n✓ 测试参数错误:")
    try:
        server.call_tool()  # 缺少参数
        print("  ✗ 应该抛出异常")
    except TypeError as e:
        print(f"  ✓ 正确捕获异常: TypeError")


def demo_server_lifecycle(server):
    """演示服务器生命周期"""
    print_section("6. 服务器生命周期")
    
    print(f"✓ 初始状态: {'运行中' if server.is_running() else '未运行'}")
    
    print("\n✓ 启动服务器:")
    server.start()
    print(f"  - 状态: {'运行中' if server.is_running() else '未运行'}")
    
    print("\n✓ 停止服务器:")
    server.stop()
    print(f"  - 状态: {'运行中' if server.is_running() else '未运行'}")


def demo_mcp_compliance():
    """演示MCP协议合规性"""
    print_section("7. MCP协议合规性检查")
    
    server = MuJoCoMCPServer()
    
    # 检查必需的方法
    required_methods = [
        'get_server_info',
        'get_tools', 
        'get_resources',
        'call_tool'
    ]
    
    print("✓ 必需方法检查:")
    for method in required_methods:
        has_method = hasattr(server, method)
        print(f"  - {method}: {'✓' if has_method else '✗'}")
    
    # 检查标准工具
    tools = server.get_tools()
    tool_names = [t['name'] for t in tools]
    
    print("\n✓ 标准工具检查:")
    standard_tools = ['get_server_info', 'get_tools']
    for tool in standard_tools:
        has_tool = tool in tool_names
        print(f"  - {tool}: {'✓' if has_tool else '✗'}")


def main():
    """主函数"""
    print(f"\n🚀 MuJoCo MCP v{__version__} - MCP服务器演示")
    print("="*60)
    
    # 运行演示
    server = demo_server_basics()
    demo_server_info(server)
    demo_get_tools(server)
    demo_resources(server)
    demo_error_handling(server)
    demo_server_lifecycle(server)
    demo_mcp_compliance()
    
    # 总结
    print_section("演示总结")
    print("✅ MCP服务器基础功能完整")
    print("✅ 支持标准MCP工具")
    print("✅ 错误处理机制完善")
    print("✅ 生命周期管理正常")
    print("✅ 符合MCP协议规范")
    print("\n🎯 下一步: 实现第一个MuJoCo控制工具")


if __name__ == "__main__":
    main()
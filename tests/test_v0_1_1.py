"""
测试 v0.1.1 - 简单MCP服务器
目标: 实现最简单的MCP服务器，能响应基本请求
"""
import pytest
import json
import asyncio
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


class TestSimpleMCPServer:
    """测试简单的MCP服务器功能"""
    
    def test_server_can_be_imported(self):
        """测试服务器模块可以导入"""
        from mujoco_mcp import server
        assert server is not None
        
    def test_create_mcp_server(self):
        """测试创建MCP服务器实例"""
        from mujoco_mcp.server import MuJoCoMCPServer
        
        server = MuJoCoMCPServer()
        assert server is not None
        assert hasattr(server, 'name')
        assert hasattr(server, 'version')
        
    def test_server_info(self):
        """测试服务器信息"""
        from mujoco_mcp.server import MuJoCoMCPServer
        
        server = MuJoCoMCPServer()
        assert server.name == "mujoco-mcp"
        assert server.version == "0.1.1"
        
    def test_get_server_info_method(self):
        """测试get_server_info方法"""
        from mujoco_mcp.server import MuJoCoMCPServer
        
        server = MuJoCoMCPServer()
        info = server.get_server_info()
        
        assert isinstance(info, dict)
        assert "name" in info
        assert "version" in info
        assert "description" in info
        assert "capabilities" in info
        
        assert info["name"] == "mujoco-mcp"
        assert info["version"] == "0.1.1"
        assert "MuJoCo" in info["description"]
        assert isinstance(info["capabilities"], list)
        
    def test_get_tools_method(self):
        """测试get_tools方法"""
        from mujoco_mcp.server import MuJoCoMCPServer
        
        server = MuJoCoMCPServer()
        tools = server.get_tools()
        
        assert isinstance(tools, list)
        assert len(tools) > 0
        
        # 检查工具格式
        for tool in tools:
            assert isinstance(tool, dict)
            assert "name" in tool
            assert "description" in tool
            assert "parameters" in tool
            
    def test_has_standard_mcp_tools(self):
        """测试是否包含标准MCP工具"""
        from mujoco_mcp.server import MuJoCoMCPServer
        
        server = MuJoCoMCPServer()
        tools = server.get_tools()
        
        tool_names = [tool["name"] for tool in tools]
        
        # 必须包含的标准工具
        assert "get_server_info" in tool_names
        assert "get_tools" in tool_names
        
    def test_tool_parameter_schema(self):
        """测试工具参数模式"""
        from mujoco_mcp.server import MuJoCoMCPServer
        
        server = MuJoCoMCPServer()
        tools = server.get_tools()
        
        # 找到get_server_info工具
        get_server_info_tool = next(
            (t for t in tools if t["name"] == "get_server_info"), 
            None
        )
        
        assert get_server_info_tool is not None
        assert get_server_info_tool["parameters"] == {}  # 无参数
        
    @pytest.mark.asyncio
    async def test_server_can_start(self):
        """测试服务器可以启动"""
        from mujoco_mcp.server import MuJoCoMCPServer
        
        server = MuJoCoMCPServer()
        
        # 测试服务器有启动方法
        assert hasattr(server, 'start')
        assert hasattr(server, 'stop')
        
    def test_call_tool_method(self):
        """测试call_tool方法"""
        from mujoco_mcp.server import MuJoCoMCPServer
        
        server = MuJoCoMCPServer()
        
        # 调用get_server_info工具
        result = server.call_tool("get_server_info", {})
        
        assert isinstance(result, dict)
        assert "name" in result
        assert result["name"] == "mujoco-mcp"
        
    def test_call_get_tools(self):
        """测试调用get_tools工具"""
        from mujoco_mcp.server import MuJoCoMCPServer
        
        server = MuJoCoMCPServer()
        
        result = server.call_tool("get_tools", {})
        
        assert isinstance(result, dict)
        assert "tools" in result
        assert isinstance(result["tools"], list)
        assert len(result["tools"]) > 0
        
    def test_call_unknown_tool(self):
        """测试调用未知工具"""
        from mujoco_mcp.server import MuJoCoMCPServer
        
        server = MuJoCoMCPServer()
        
        with pytest.raises(ValueError) as excinfo:
            server.call_tool("unknown_tool", {})
            
        assert "Unknown tool" in str(excinfo.value)
        
    def test_server_has_resources(self):
        """测试服务器有资源列表"""
        from mujoco_mcp.server import MuJoCoMCPServer
        
        server = MuJoCoMCPServer()
        
        assert hasattr(server, 'get_resources')
        resources = server.get_resources()
        
        assert isinstance(resources, list)
        # v0.1.1可以没有资源，但必须返回列表
        
    def test_mcp_protocol_compliance(self):
        """测试MCP协议合规性"""
        from mujoco_mcp.server import MuJoCoMCPServer
        
        server = MuJoCoMCPServer()
        
        # 检查必需的方法
        assert hasattr(server, 'get_server_info')
        assert hasattr(server, 'get_tools')
        assert hasattr(server, 'get_resources')
        assert hasattr(server, 'call_tool')
        
        # 检查方法签名
        import inspect
        
        # get_server_info应该不需要参数
        sig = inspect.signature(server.get_server_info)
        assert len(sig.parameters) == 0
        
        # call_tool应该接受tool_name和parameters
        sig = inspect.signature(server.call_tool)
        assert 'tool_name' in sig.parameters
        assert 'parameters' in sig.parameters


class TestServerIntegration:
    """服务器集成测试"""
    
    def test_server_lifecycle(self):
        """测试服务器生命周期"""
        from mujoco_mcp.server import MuJoCoMCPServer
        
        server = MuJoCoMCPServer()
        
        # 初始状态
        assert hasattr(server, 'is_running')
        assert not server.is_running()
        
        # 可以获取信息（即使未启动）
        info = server.get_server_info()
        assert info is not None
        
    def test_server_error_handling(self):
        """测试服务器错误处理"""
        from mujoco_mcp.server import MuJoCoMCPServer
        
        server = MuJoCoMCPServer()
        
        # 测试参数验证
        with pytest.raises(TypeError):
            server.call_tool()  # 缺少必需参数
            
        with pytest.raises(TypeError):
            server.call_tool("get_server_info")  # 缺少parameters参数
            
    def test_tool_registration(self):
        """测试工具注册机制"""
        from mujoco_mcp.server import MuJoCoMCPServer
        
        server = MuJoCoMCPServer()
        
        # 检查内部工具注册
        assert hasattr(server, '_tools') or hasattr(server, 'tools')
        
        # 获取已注册的工具
        tools = server.get_tools()
        registered_names = [t["name"] for t in tools]
        
        # 验证标准工具已注册
        assert "get_server_info" in registered_names
        assert "get_tools" in registered_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
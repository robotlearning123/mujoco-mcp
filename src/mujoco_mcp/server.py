"""
MuJoCo MCP Server
v0.1.2 - 简单服务器实现，包含load_model工具
"""
# 从simple_server导入所有内容
from .simple_server import MuJoCoMCPServer

# 导出
__all__ = ['MuJoCoMCPServer']
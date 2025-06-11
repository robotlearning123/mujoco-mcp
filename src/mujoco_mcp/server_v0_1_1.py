"""
MuJoCo MCP Server v0.1.1
简化版本，只包含基本功能
"""
from .simple_server import MuJoCoMCPServer

# 导出主要的服务器类
__all__ = ['MuJoCoMCPServer']
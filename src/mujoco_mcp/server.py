"""
MuJoCo MCP Server
v0.1.1 - 简单服务器实现
"""
# 从simple_server导入所有内容
from .simple_server import MuJoCoMCPServer, update_version

# 更新版本
update_version()

# 导出
__all__ = ['MuJoCoMCPServer']
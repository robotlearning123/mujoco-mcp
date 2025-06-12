"""
结构化日志配置模块
提供符合MCP规范的日志格式和JSON-RPC错误处理
"""

import logging
import json
import sys
import time
from typing import Dict, Any, Optional
from enum import Enum

class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO" 
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class MCPErrorCode(Enum):
    """MCP标准错误代码"""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    
    # MCP特定错误
    TOOL_NOT_FOUND = -32000
    TOOL_EXECUTION_ERROR = -32001
    RESOURCE_NOT_FOUND = -32002
    VIEWER_CONNECTION_ERROR = -32003
    SIMULATION_ERROR = -32004

class StructuredLogger:
    """结构化日志记录器"""
    
    def __init__(self, name: str, level: LogLevel = LogLevel.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.value))
        
        # 创建结构化格式处理器
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            formatter = StructuredFormatter()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def debug(self, message: str, **context):
        """调试日志"""
        self._log(LogLevel.DEBUG, message, context)
    
    def info(self, message: str, **context):
        """信息日志"""
        self._log(LogLevel.INFO, message, context)
    
    def warning(self, message: str, **context):
        """警告日志"""
        self._log(LogLevel.WARNING, message, context)
    
    def error(self, message: str, **context):
        """错误日志"""
        self._log(LogLevel.ERROR, message, context)
    
    def critical(self, message: str, **context):
        """严重错误日志"""
        self._log(LogLevel.CRITICAL, message, context)
    
    def _log(self, level: LogLevel, message: str, context: Dict[str, Any]):
        """内部日志方法"""
        log_entry = {
            "timestamp": time.time(),
            "level": level.value,
            "message": message,
            "context": context
        }
        
        # 调用标准logger
        getattr(self.logger, level.value.lower())(
            json.dumps(log_entry, ensure_ascii=False)
        )

class StructuredFormatter(logging.Formatter):
    """结构化日志格式器"""
    
    def format(self, record):
        try:
            # 尝试解析JSON日志
            log_data = json.loads(record.getMessage())
            
            # 格式化时间戳
            if 'timestamp' in log_data:
                timestamp = time.strftime(
                    "%Y-%m-%d %H:%M:%S", 
                    time.localtime(log_data['timestamp'])
                )
            else:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # 构建格式化字符串
            level = log_data.get('level', record.levelname)
            message = log_data.get('message', '')
            context = log_data.get('context', {})
            
            formatted = f"{timestamp} [{level}] {record.name}: {message}"
            
            # 添加上下文信息
            if context:
                context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
                formatted += f" | {context_str}"
            
            return formatted
            
        except (json.JSONDecodeError, KeyError):
            # 如果不是JSON格式，使用标准格式
            return super().format(record)

class MCPErrorHandler:
    """MCP错误处理器"""
    
    def __init__(self, logger: StructuredLogger):
        self.logger = logger
    
    def create_error_response(
        self, 
        code: MCPErrorCode, 
        message: str, 
        data: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """创建标准MCP错误响应"""
        
        error_response = {
            "success": False,
            "error": {
                "code": code.value,
                "message": message
            }
        }
        
        if data:
            error_response["error"]["data"] = data
        
        if request_id:
            error_response["id"] = request_id
        
        # 记录错误日志
        self.logger.error(
            f"MCP Error: {message}",
            error_code=code.value,
            error_data=data,
            request_id=request_id
        )
        
        return error_response
    
    def handle_tool_error(
        self, 
        tool_name: str, 
        error: Exception, 
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """处理工具执行错误"""
        
        # 确定错误类型
        if isinstance(error, KeyError):
            code = MCPErrorCode.TOOL_NOT_FOUND
            message = f"Tool '{tool_name}' not found"
        elif isinstance(error, (ValueError, TypeError)):
            code = MCPErrorCode.INVALID_PARAMS  
            message = f"Invalid parameters for tool '{tool_name}': {str(error)}"
        elif "viewer" in str(error).lower() or "connection" in str(error).lower():
            code = MCPErrorCode.VIEWER_CONNECTION_ERROR
            message = f"Viewer connection error in tool '{tool_name}': {str(error)}"
        elif "simulation" in str(error).lower() or "mujoco" in str(error).lower():
            code = MCPErrorCode.SIMULATION_ERROR
            message = f"Simulation error in tool '{tool_name}': {str(error)}"
        else:
            code = MCPErrorCode.TOOL_EXECUTION_ERROR
            message = f"Tool execution error in '{tool_name}': {str(error)}"
        
        return self.create_error_response(
            code=code,
            message=message,
            data={
                "tool_name": tool_name,
                "parameters": parameters,
                "exception_type": type(error).__name__,
                "exception_message": str(error)
            }
        )
    
    def handle_connection_error(self, error: Exception) -> Dict[str, Any]:
        """处理连接错误"""
        return self.create_error_response(
            code=MCPErrorCode.VIEWER_CONNECTION_ERROR,
            message="Failed to connect to MuJoCo Viewer Server",
            data={
                "suggestion": "Please start mujoco_viewer_server.py first",
                "error_details": str(error)
            }
        )
    
    def handle_simulation_error(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """处理仿真错误"""
        return self.create_error_response(
            code=MCPErrorCode.SIMULATION_ERROR,
            message="MuJoCo simulation error",
            data={
                "error_details": str(error),
                "context": context or {}
            }
        )

# 全局实例
mcp_logger = StructuredLogger("mujoco_mcp")
error_handler = MCPErrorHandler(mcp_logger)

def get_logger(name: str) -> StructuredLogger:
    """获取结构化日志器"""
    return StructuredLogger(name)

def configure_logging(level: LogLevel = LogLevel.INFO):
    """配置全局日志级别"""
    logging.basicConfig(
        level=getattr(logging, level.value),
        format='%(message)s',  # 使用结构化格式器
        stream=sys.stderr
    )

__all__ = [
    'StructuredLogger', 
    'MCPErrorHandler', 
    'MCPErrorCode', 
    'LogLevel',
    'mcp_logger', 
    'error_handler', 
    'get_logger', 
    'configure_logging'
]
"""
增强型授权管理器：提供更强大的安全特性和操作限制功能
"""
import logging
import json
import os
import time
import uuid
from typing import Dict, Set, Optional, List, Tuple, Any, Callable, Union

import dataclasses
from dataclasses import dataclass

@dataclass
class AuthRequest:
    """授权请求数据类"""
    client_id: str
    operation: str
    resource: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None

class AuthManager:
    """基础授权管理器"""
    
    def __init__(self):
        self.logger = logging.getLogger("mujoco_mcp.auth")
    
    def request_authorization(self, client_id: str, operation: str, 
                            resource: Optional[str] = None, 
                            parameters: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, str]:
        """请求授权 - 基础实现总是允许"""
        return True, "Authorized", str(uuid.uuid4())

class EnhancedAuthManager(AuthManager):
    """增强型授权管理器"""
    
    def __init__(self, config_file: Optional[str] = None, auto_approve_mode: bool = False):
        """
        初始化增强型授权管理器
        
        Args:
            config_file: 配置文件路径
            auto_approve_mode: 是否启用自动批准模式
        """
        # super().__init__(config_file, auto_approve_mode)  # AuthManager不接受参数
        self.logger = logging.getLogger("mujoco_mcp.enhanced_auth")
        
        # 安全限制
        self.action_limits: Dict[str, Dict[str, Tuple[float, float]]] = {}  # 操作参数限制
        self.velocity_limits: Dict[str, float] = {"default": 5.0}  # 速度限制
        self.force_limits: Dict[str, float] = {"default": 100.0}  # 力限制
        
        # 速率限制
        self.rate_limits: Dict[str, Dict[str, Union[int, Dict[str, int]]]] = {}  # 客户端操作频率限制
        self.operation_timestamps: Dict[str, Dict[str, List[float]]] = {}  # 操作时间戳记录
        
        # 加载安全配置
        self.security_config_file = None
        if config_file:
            base_dir = os.path.dirname(config_file)
            self.security_config_file = os.path.join(base_dir, "security_config.json")
            self._load_security_config()
    
    def _load_security_config(self) -> None:
        """加载安全配置"""
        if not self.security_config_file or not os.path.exists(self.security_config_file):
            self.logger.warning(f"安全配置文件不存在: {self.security_config_file}")
            return
        
        try:
            with open(self.security_config_file, 'r') as f:
                config = json.load(f)
            
            # 加载操作限制
            if "action_limits" in config:
                self.action_limits = config["action_limits"]
            
            # 加载速度限制
            if "velocity_limits" in config:
                self.velocity_limits = config["velocity_limits"]
            
            # 加载力限制
            if "force_limits" in config:
                self.force_limits = config["force_limits"]
            
            # 加载速率限制
            if "rate_limits" in config:
                self.rate_limits = config["rate_limits"]
            
            self.logger.info(f"已加载安全配置: {self.security_config_file}")
        except Exception as e:
            self.logger.error(f"加载安全配置失败: {str(e)}")
    
    def _save_security_config(self) -> None:
        """保存安全配置"""
        if not self.security_config_file:
            self.logger.warning("未指定安全配置文件，无法保存")
            return
        
        try:
            config = {
                "action_limits": self.action_limits,
                "velocity_limits": self.velocity_limits,
                "force_limits": self.force_limits,
                "rate_limits": self.rate_limits
            }
            
            with open(self.security_config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.logger.info(f"已保存安全配置: {self.security_config_file}")
        except Exception as e:
            self.logger.error(f"保存安全配置失败: {str(e)}")
    
    def set_action_limit(self, operation: str, parameter: str, min_value: float, max_value: float) -> None:
        """
        设置操作参数限制
        
        Args:
            operation: 操作名称
            parameter: 参数名称
            min_value: 最小允许值
            max_value: 最大允许值
        """
        if operation not in self.action_limits:
            self.action_limits[operation] = {}
        
        self.action_limits[operation][parameter] = (min_value, max_value)
        self._save_security_config()
    
    def set_velocity_limit(self, joint_name: str, max_velocity: float) -> None:
        """
        设置关节速度限制
        
        Args:
            joint_name: 关节名称 (使用 "default" 设置默认限制)
            max_velocity: 最大允许速度
        """
        self.velocity_limits[joint_name] = max_velocity
        self._save_security_config()
    
    def set_force_limit(self, body_name: str, max_force: float) -> None:
        """
        设置施加力限制
        
        Args:
            body_name: 刚体名称 (使用 "default" 设置默认限制)
            max_force: 最大允许力
        """
        self.force_limits[body_name] = max_force
        self._save_security_config()
    
    def set_rate_limit(self, client_id: str, operation: str, max_per_minute: int) -> None:
        """
        设置操作频率限制
        
        Args:
            client_id: 客户端ID (使用 "default" 设置默认限制)
            operation: 操作名称 (使用 "general" 设置通用限制)
            max_per_minute: 每分钟最大允许操作次数
        """
        if client_id not in self.rate_limits:
            self.rate_limits[client_id] = {}
        
        self.rate_limits[client_id][operation] = max_per_minute
        self._save_security_config()
    
    def _check_parameter_limits(self, operation: str, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """
        检查参数是否在限制范围内
        
        Args:
            operation: 操作名称
            parameters: 操作参数
            
        Returns:
            (是否通过, 错误消息)
        """
        if operation not in self.action_limits:
            return True, ""
        
        limits = self.action_limits[operation]
        for param, (min_val, max_val) in limits.items():
            if param in parameters:
                value = parameters[param]
                
                # 处理不同类型的参数
                if isinstance(value, (int, float)):
                    if value < min_val or value > max_val:
                        return False, f"参数 {param} 值 {value} 超出限制范围 [{min_val}, {max_val}]"
                elif isinstance(value, list) and all(isinstance(x, (int, float)) for x in value):
                    # 对于数值列表，检查所有元素
                    for i, x in enumerate(value):
                        if x < min_val or x > max_val:
                            return False, f"参数 {param}[{i}] 值 {x} 超出限制范围 [{min_val}, {max_val}]"
        
        # 特殊处理一些常见操作
        if operation == "set_joint_velocities" and "velocities" in parameters:
            velocities = parameters["velocities"]
            if isinstance(velocities, list):
                for i, vel in enumerate(velocities):
                    joint_name = f"joint_{i}"  # 假设关节名称
                    max_vel = self.velocity_limits.get(joint_name, self.velocity_limits.get("default", float('inf')))
                    if abs(vel) > max_vel:
                        return False, f"关节 {joint_name} 速度 {vel} 超出限制 {max_vel}"
        
        if operation == "apply_force" and "force" in parameters:
            force = parameters["force"]
            body_name = parameters.get("body_name", "default")
            max_force = self.force_limits.get(body_name, self.force_limits.get("default", float('inf')))
            
            if isinstance(force, list) and len(force) >= 3:
                magnitude = sum(x**2 for x in force) ** 0.5
                if magnitude > max_force:
                    return False, f"施加力大小 {magnitude} 超出限制 {max_force}"
            elif "magnitude" in parameters:
                magnitude = parameters["magnitude"]
                if magnitude > max_force:
                    return False, f"施加力大小 {magnitude} 超出限制 {max_force}"
        
        return True, ""
    
    def check_rate_limit(self, client_id: str, operation: str) -> Tuple[bool, str]:
        """Public wrapper for rate limit checking."""
        return self._check_rate_limit(client_id, operation)
    
    def _check_rate_limit(self, client_id: str, operation: str) -> Tuple[bool, str]:
        """
        检查操作频率是否超出限制
        
        Args:
            client_id: 客户端ID
            operation: 操作名称
            
        Returns:
            (是否通过, 错误消息)
        """
        # 获取适用的速率限制
        client_limits = self.rate_limits.get(client_id, self.rate_limits.get("default", {}))
        op_limit = client_limits.get(operation, client_limits.get("general", 0))
        if op_limit <= 0:
            return True, ""  # 无限制
        
        # 初始化时间戳记录
        if client_id not in self.operation_timestamps:
            self.operation_timestamps[client_id] = {}
        
        if operation not in self.operation_timestamps[client_id]:
            self.operation_timestamps[client_id][operation] = []
        
        # 获取最近一分钟的时间戳
        now = time.time()
        timestamps = self.operation_timestamps[client_id][operation]
        recent_timestamps = [ts for ts in timestamps if now - ts <= 60]
        
        # 更新时间戳记录
        self.operation_timestamps[client_id][operation] = recent_timestamps
        
        # 检查是否超出限制
        if len(recent_timestamps) >= op_limit:
            return False, f"操作 {operation} 频率超过限制 (每分钟 {op_limit} 次)"
        
        # 记录本次操作
        self.operation_timestamps[client_id][operation].append(now)
        return True, ""
    
    def validate_request(self, request: AuthRequest) -> Tuple[bool, str]:
        """
        验证请求的安全性
        
        Args:
            request: 授权请求
            
        Returns:
            (是否通过, 错误消息)
        """
        # 检查参数限制
        params_ok, params_msg = self._check_parameter_limits(request.operation, request.parameters)
        if not params_ok:
            self.logger.warning(f"请求参数验证失败: {params_msg}")
            return False, params_msg
        
        # 检查频率限制
        rate_ok, rate_msg = self._check_rate_limit(request.client_id, request.operation)
        if not rate_ok:
            self.logger.warning(f"请求频率验证失败: {rate_msg}")
            return False, rate_msg
        
        return True, ""
    
    def handle_auth_request(self, request: AuthRequest) -> Tuple[bool, Optional[str]]:
        """
        处理授权请求，包括安全验证
        
        Args:
            request: 授权请求
            
        Returns:
            (是否授权, 错误消息)
        """
        # 首先进行安全验证
        valid, msg = self.validate_request(request)
        if not valid:
            return False, msg
        
        # 然后执行标准授权流程
        result = super().handle_auth_request(request)
        return result, None 
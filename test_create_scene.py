#!/usr/bin/env python3
"""
直接测试场景创建和GUI显示
"""

import sys
import time
sys.path.insert(0, 'src')

from mujoco_mcp.remote_server import MuJoCoRemoteServer

print("🧪 直接测试场景创建...")

server = MuJoCoRemoteServer()

# 测试创建单摆场景
print("\n📍 创建单摆场景...")
result = server.call_tool('create_scene', {
    'scene_type': 'pendulum',
    'parameters': {'length': 0.6, 'mass': 0.8}
})

print(f"\n结果: {result}")

if result.get('success'):
    model_id = result.get('model_id')
    print(f"\n✅ 场景创建成功!")
    print(f"   Model ID: {model_id}")
    print("\n⚠️  检查是否有MuJoCo窗口弹出...")
    
    # 等待几秒让用户观察
    time.sleep(5)
    
    # 获取状态
    print("\n📊 获取仿真状态...")
    state_result = server.call_tool('get_state', {'model_id': model_id})
    if state_result.get('success'):
        print(f"   时间: {state_result.get('time', 0):.3f}s")
        print(f"   位置: {state_result.get('qpos', [])}")
else:
    print(f"\n❌ 创建失败: {result.get('error', 'Unknown error')}")
    
    # 检查连接诊断
    if 'viewer' in str(result.get('error', '')).lower():
        print("\n🔍 连接诊断:")
        from mujoco_mcp.viewer_client import get_system_diagnostics
        diagnostics = get_system_diagnostics()
        print(f"   活跃客户端: {diagnostics['viewer_manager']['active_clients']}")
        print(f"   默认端口: {diagnostics['viewer_manager']['default_port']}")
        
        print("\n💡 可能的解决方案:")
        print("1. 确保mujoco_viewer_server.py正在运行")
        print("2. 检查端口8888是否被占用")
        print("3. 尝试重启所有进程")
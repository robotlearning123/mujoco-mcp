#!/usr/bin/env python3
"""
MuJoCo MCP MVP 功能展示
展示当前版本(v0.1.0)的所有核心功能
"""
import sys
import time
import numpy as np
from pathlib import Path

# 添加项目到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from mujoco_mcp import MuJoCoSimulation, __version__
from mujoco_mcp.enhanced_auth_manager import EnhancedAuthManager

def print_section(title):
    """打印章节标题"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def showcase_basic_simulation():
    """展示基础仿真功能"""
    print_section("1. 基础仿真功能")
    
    # 创建仿真
    sim = MuJoCoSimulation()
    print(f"✓ 创建仿真实例")
    print(f"  - 仿真ID: {sim.sim_id}")
    print(f"  - 初始化状态: {sim.is_initialized()}")
    
    # 加载模型
    cartpole_xml = """
    <mujoco model="cartpole">
        <option timestep="0.002"/>
        
        <worldbody>
            <!-- 轨道 -->
            <body name="rail">
                <geom name="rail_geom" type="box" size="2 0.02 0.02" rgba="0.3 0.3 0.3 1"/>
            </body>
            
            <!-- 小车 -->
            <body name="cart" pos="0 0 0">
                <joint name="slider" type="slide" axis="1 0 0" limited="true" range="-1 1"/>
                <geom name="cart_geom" type="box" size="0.1 0.05 0.05" rgba="0.7 0.2 0.2 1"/>
                
                <!-- 摆杆 -->
                <body name="pole" pos="0 0 0.05">
                    <joint name="hinge" type="hinge" axis="0 1 0" limited="true" range="-90 90"/>
                    <geom name="pole_geom" type="capsule" fromto="0 0 0 0 0 0.4" size="0.02" rgba="0.2 0.7 0.2 1"/>
                    
                    <!-- 摆球 -->
                    <body name="mass" pos="0 0 0.4">
                        <geom name="mass_geom" type="sphere" size="0.05" rgba="0.2 0.2 0.7 1"/>
                    </body>
                </body>
            </body>
        </worldbody>
        
        <actuator>
            <motor name="slide_motor" joint="slider" gear="10" ctrllimited="true" ctrlrange="-1 1"/>
        </actuator>
    </mujoco>
    """
    
    sim.load_model_from_string(cartpole_xml)
    print(f"✓ 加载CartPole模型")
    
    # 获取模型信息
    info = sim.get_model_info()
    print(f"✓ 模型信息:")
    print(f"  - 自由度(nq): {info['nq']}")
    print(f"  - 速度维度(nv): {info['nv']}")
    print(f"  - 刚体数量: {info['nbody']}")
    print(f"  - 关节数量: {info['njoint']}")
    print(f"  - 执行器数量: {info['nu']}")
    
    return sim

def showcase_simulation_control(sim):
    """展示仿真控制功能"""
    print_section("2. 仿真控制功能")
    
    # 设置初始状态
    initial_pos = [0.0, 0.3]  # 小车位置0, 摆杆角度0.3弧度
    sim.set_joint_positions(initial_pos)
    print(f"✓ 设置初始状态: cart={initial_pos[0]:.2f}, pole={initial_pos[1]:.2f} rad")
    
    # 运行仿真
    print("\n✓ 运行仿真 (1秒):")
    print("  时间 | 小车位置 | 摆杆角度 | 小车速度 | 摆杆角速度")
    print("  " + "-"*50)
    
    for i in range(5):
        # 获取状态
        pos = sim.get_joint_positions()
        vel = sim.get_joint_velocities()
        time = sim.get_time()
        
        print(f"  {time:.2f}s | {pos[0]:+.4f} | {pos[1]:+.4f} | {vel[0]:+.4f} | {vel[1]:+.4f}")
        
        # 应用控制（简单的PD控制器）
        control = [-10.0 * pos[0] - 2.0 * vel[0]]  # 让小车回到中心
        sim.apply_control(control)
        
        # 步进仿真
        sim.step(100)  # 0.2秒
    
    # 重置仿真
    print("\n✓ 重置仿真")
    sim.reset()
    print(f"  - 时间: {sim.get_time()}")
    print(f"  - 位置: {sim.get_joint_positions()}")

def showcase_state_query(sim):
    """展示状态查询功能"""
    print_section("3. 状态查询功能")
    
    # 设置一个有趣的状态
    sim.set_joint_positions([0.2, 0.5])
    sim.set_joint_velocities([0.1, -0.2])
    sim.step(10)
    
    # 关节信息
    print("✓ 关节状态:")
    print(f"  - 位置: {sim.get_joint_positions()}")
    print(f"  - 速度: {sim.get_joint_velocities()}")
    print(f"  - 关节名称: {sim.get_joint_names()}")
    
    # 刚体状态
    print("\n✓ 刚体状态:")
    body_states = sim.get_rigid_body_states()
    for name, state in list(body_states.items())[:3]:  # 只显示前3个
        if name:
            print(f"  - {name}:")
            print(f"    位置: [{state['position'][0]:.3f}, {state['position'][1]:.3f}, {state['position'][2]:.3f}]")
    
    # 时间信息
    print(f"\n✓ 时间信息:")
    print(f"  - 当前时间: {sim.get_time():.3f}s")
    print(f"  - 时间步长: {sim.get_timestep():.3f}s")
    
    # 数量统计
    print(f"\n✓ 模型统计:")
    print(f"  - 关节数量: {sim.get_num_joints()}")
    print(f"  - 执行器数量: {sim.get_num_actuators()}")

def showcase_auth_manager():
    """展示认证管理器功能"""
    print_section("4. 安全认证功能")
    
    auth = EnhancedAuthManager()
    print("✓ 创建认证管理器")
    
    # 检查速率限制
    client_id = "demo_client"
    operation = "set_joint_positions"
    
    # 设置速率限制
    auth.set_rate_limit(client_id, operation, max_per_minute=10)
    print(f"✓ 设置速率限制: {operation} 最多 10次/分钟")
    
    # 测试速率限制
    print("\n✓ 测试速率限制:")
    for i in range(3):
        allowed, message = auth.check_rate_limit(client_id, operation)
        print(f"  - 请求 {i+1}: {'允许' if allowed else '拒绝'} {f'({message})' if message else ''}")
    
    # 验证请求
    print("\n✓ 请求验证功能已就绪")
    print("  - 支持参数范围验证")
    print("  - 支持速率限制")
    print("  - 支持操作权限控制")

def showcase_error_handling():
    """展示错误处理功能"""
    print_section("5. 错误处理功能")
    
    sim = MuJoCoSimulation()
    
    # 测试无效XML
    print("✓ 测试无效XML处理:")
    try:
        sim.load_model_from_string("invalid xml")
        print("  ✗ 应该抛出异常")
    except Exception as e:
        print(f"  ✓ 正确捕获异常: {type(e).__name__}")
    
    # 测试空模型
    print("\n✓ 测试空模型处理:")
    try:
        sim.load_model_from_string("<mujoco></mujoco>")
        print("  ✗ 应该抛出异常")
    except ValueError as e:
        print(f"  ✓ 正确捕获异常: {e}")
    
    # 测试未初始化操作
    print("\n✓ 测试未初始化操作:")
    sim2 = MuJoCoSimulation()
    try:
        sim2.step()
        print("  ✗ 应该抛出异常")
    except RuntimeError as e:
        print(f"  ✓ 正确捕获异常: {e}")

def main():
    """主函数"""
    print(f"\n🚀 MuJoCo MCP MVP 功能展示")
    print(f"版本: {__version__}")
    print(f"{'='*60}")
    
    # 1. 基础功能
    sim = showcase_basic_simulation()
    
    # 2. 控制功能
    showcase_simulation_control(sim)
    
    # 3. 状态查询
    showcase_state_query(sim)
    
    # 4. 认证功能
    showcase_auth_manager()
    
    # 5. 错误处理
    showcase_error_handling()
    
    # 总结
    print_section("MVP 功能总结")
    print("✅ 基础仿真框架完整")
    print("✅ 模型加载和管理")
    print("✅ 仿真控制和步进")
    print("✅ 状态查询和设置")
    print("✅ 安全认证机制")
    print("✅ 错误处理完善")
    print("\n🎯 下一步: 实现MCP服务器和工具")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
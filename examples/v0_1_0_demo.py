#!/usr/bin/env python3
"""
v0.1.0 演示 - 最小可运行版本
展示基础的MuJoCo仿真功能
"""
import sys
from pathlib import Path

# 添加项目到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from mujoco_mcp import MuJoCoSimulation, __version__, __author__, __email__

def main():
    print(f"MuJoCo MCP v{__version__}")
    print(f"作者: {__author__}")
    print(f"联系: {__email__}")
    print("-" * 50)
    
    # 1. 创建仿真实例
    print("\n1. 创建仿真实例...")
    sim = MuJoCoSimulation()
    print(f"   ✓ 仿真ID: {sim.sim_id}")
    print(f"   ✓ 初始化状态: {sim.is_initialized()}")
    
    # 2. 加载简单的单摆模型
    print("\n2. 加载单摆模型...")
    pendulum_xml = """
    <mujoco model="pendulum">
        <option timestep="0.001" gravity="0 0 -9.81"/>
        
        <worldbody>
            <!-- 固定基座 -->
            <body name="base" pos="0 0 1">
                <geom name="base_geom" type="cylinder" size="0.1 0.02" rgba="0.5 0.5 0.5 1"/>
                
                <!-- 摆杆 -->
                <body name="pendulum" pos="0 0 0">
                    <joint name="hinge" type="hinge" axis="0 1 0" damping="0.1"/>
                    <geom name="rod" type="capsule" fromto="0 0 0 0 0 -0.5" size="0.02" rgba="0.8 0.2 0.2 1"/>
                    
                    <!-- 摆球 -->
                    <body name="ball" pos="0 0 -0.5">
                        <geom name="ball_geom" type="sphere" size="0.05" rgba="0.2 0.2 0.8 1"/>
                    </body>
                </body>
            </body>
        </worldbody>
    </mujoco>
    """
    
    sim.load_model_from_string(pendulum_xml)
    print("   ✓ 模型加载成功")
    
    # 3. 获取模型信息
    print("\n3. 模型信息:")
    model_info = sim.get_model_info()
    for key, value in model_info.items():
        print(f"   - {key}: {value}")
    
    # 4. 设置初始角度
    print("\n4. 设置初始角度...")
    initial_angle = 0.5  # 弧度
    sim.set_joint_positions([initial_angle])
    print(f"   ✓ 设置角度: {initial_angle} rad")
    
    # 5. 运行仿真
    print("\n5. 运行仿真...")
    print("   时间 | 角度(rad) | 角速度(rad/s)")
    print("   " + "-" * 35)
    
    for i in range(5):
        # 记录当前状态
        time = sim.get_time()
        pos = sim.get_joint_positions()[0]
        vel = sim.get_joint_velocities()[0]
        print(f"   {time:.3f} | {pos:+.6f} | {vel:+.6f}")
        
        # 步进100步（0.1秒）
        sim.step(100)
    
    # 6. 重置仿真
    print("\n6. 重置仿真...")
    sim.reset()
    print(f"   ✓ 时间重置为: {sim.get_time()}")
    print(f"   ✓ 角度重置为: {sim.get_joint_positions()[0]}")
    
    # 7. 获取刚体状态
    print("\n7. 刚体状态:")
    body_states = sim.get_rigid_body_states()
    for body_name, state in body_states.items():
        if body_name:  # 跳过空名称
            print(f"   - {body_name}:")
            print(f"     位置: {state['position']}")
            print(f"     方向: {state['orientation']}")
    
    print("\n✅ v0.1.0 演示完成！")
    print("   基础功能正常工作")
    print("   可以加载模型、运行仿真、获取状态")

if __name__ == "__main__":
    main()
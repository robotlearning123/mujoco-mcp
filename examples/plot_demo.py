#!/usr/bin/env python3
"""
MuJoCo MCP 图表演示
生成仿真数据的可视化图表
"""
import sys
import numpy as np
from pathlib import Path

# 添加项目到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from mujoco_mcp import MuJoCoSimulation, __version__

def create_ascii_plot(data, width=60, height=20, title="", y_label=""):
    """创建ASCII图表"""
    if len(data) == 0:
        return
    
    # 数据范围
    min_val = min(data)
    max_val = max(data)
    data_range = max_val - min_val
    if data_range == 0:
        data_range = 1
    
    # 创建画布
    canvas = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Y轴刻度
    for i in range(height):
        y_val = max_val - (i / (height - 1)) * data_range
        label = f"{y_val:6.2f}"
        for j, char in enumerate(label):
            if j < width:
                canvas[i][j] = char
    
    # 绘制数据点
    x_scale = (width - 10) / (len(data) - 1) if len(data) > 1 else 1
    prev_y = None
    
    for i, value in enumerate(data):
        x = int(10 + i * x_scale)
        y = int((max_val - value) / data_range * (height - 1))
        
        if 0 <= x < width and 0 <= y < height:
            canvas[y][x] = '●'
            
            # 连接线
            if prev_y is not None and abs(y - prev_y) > 1:
                for j in range(min(y, prev_y) + 1, max(y, prev_y)):
                    if 0 <= j < height:
                        canvas[j][x] = '│'
            
            prev_y = y
    
    # 打印标题
    print(f"\n{title}")
    print("=" * width)
    
    # 打印画布
    for row in canvas:
        print(''.join(row))
    
    # X轴
    print("      " + "─" * (width - 6))
    print(f"      0{'时间(秒)':>{width-10}}{len(data)/100:.1f}")

def simulate_pendulum():
    """运行单摆仿真并收集数据"""
    print(f"\n🔬 MuJoCo MCP v{__version__} - 单摆仿真分析")
    
    # 创建仿真
    sim = MuJoCoSimulation()
    
    # 单摆模型
    pendulum_xml = """
    <mujoco model="pendulum">
        <option timestep="0.01" gravity="0 0 -9.81"/>
        <worldbody>
            <body name="pendulum" pos="0 0 1">
                <joint name="hinge" type="hinge" axis="0 1 0" damping="0.1"/>
                <geom name="rod" type="capsule" fromto="0 0 0 0 0 -1" size="0.02"/>
            </body>
        </worldbody>
    </mujoco>
    """
    
    sim.load_model_from_string(pendulum_xml)
    
    # 设置初始角度
    initial_angle = 1.0  # 约57度
    sim.set_joint_positions([initial_angle])
    
    # 收集数据
    times = []
    angles = []
    velocities = []
    energies = []
    
    # 运行仿真
    for i in range(300):  # 3秒
        # 记录数据
        t = sim.get_time()
        angle = sim.get_joint_positions()[0]
        velocity = sim.get_joint_velocities()[0]
        
        # 计算能量 (简化：只考虑势能和动能)
        # 势能 = mgh = mg(1-cos(θ))
        # 动能 = 0.5 * I * ω^2
        potential = 9.81 * (1 - np.cos(angle))
        kinetic = 0.5 * velocity**2
        total_energy = potential + kinetic
        
        times.append(t)
        angles.append(angle)
        velocities.append(velocity)
        energies.append(total_energy)
        
        # 步进
        sim.step()
    
    # 绘制图表
    print("\n📊 仿真结果:")
    
    # 角度图
    create_ascii_plot(angles, title="摆角随时间变化", y_label="角度(rad)")
    
    # 角速度图
    create_ascii_plot(velocities, title="角速度随时间变化", y_label="角速度(rad/s)")
    
    # 能量图
    create_ascii_plot(energies, title="系统总能量", y_label="能量(J)")
    
    # 相空间图（角度 vs 角速度）
    print("\n📈 相空间图 (角度 vs 角速度):")
    print("="*60)
    
    # 创建2D散点图
    angle_range = max(angles) - min(angles)
    vel_range = max(velocities) - min(velocities)
    
    canvas = [[' ' for _ in range(60)] for _ in range(20)]
    
    # 绘制轴
    for i in range(20):
        canvas[i][30] = '│'
    for j in range(60):
        canvas[10][j] = '─'
    canvas[10][30] = '┼'
    
    # 绘制轨迹
    for angle, vel in zip(angles[::5], velocities[::5]):  # 每5个点取一个
        x = int(30 + (angle / angle_range) * 25)
        y = int(10 - (vel / vel_range) * 8)
        if 0 <= x < 60 and 0 <= y < 20:
            canvas[y][x] = '●'
    
    for row in canvas:
        print(''.join(row))
    
    print(f"        {min(angles):.2f}  ← 角度(rad) →  {max(angles):.2f}")
    print(f"        {min(velocities):.2f} ← 角速度(rad/s) → {max(velocities):.2f}")
    
    # 统计信息
    print("\n📊 统计摘要:")
    print(f"  - 初始角度: {initial_angle:.3f} rad ({np.degrees(initial_angle):.1f}°)")
    print(f"  - 最大角度: {max(angles):.3f} rad ({np.degrees(max(angles)):.1f}°)")
    print(f"  - 最大角速度: {max(abs(v) for v in velocities):.3f} rad/s")
    print(f"  - 平均能量: {np.mean(energies):.3f} J")
    print(f"  - 能量标准差: {np.std(energies):.3f} J (能量守恒程度)")
    
    # 计算周期
    # 找到过零点
    zero_crossings = []
    for i in range(1, len(angles)):
        if angles[i-1] * angles[i] < 0 and velocities[i] > 0:  # 从负到正
            zero_crossings.append(times[i])
    
    if len(zero_crossings) > 1:
        periods = [zero_crossings[i+1] - zero_crossings[i] for i in range(len(zero_crossings)-1)]
        avg_period = np.mean(periods)
        print(f"  - 平均周期: {avg_period:.3f} 秒")
        print(f"  - 频率: {1/avg_period:.3f} Hz")

def simulate_cartpole_control():
    """运行CartPole控制仿真"""
    print(f"\n\n🎮 MuJoCo MCP v{__version__} - CartPole控制分析")
    
    # 创建仿真
    sim = MuJoCoSimulation()
    
    # CartPole模型
    cartpole_xml = """
    <mujoco model="cartpole">
        <option timestep="0.01"/>
        <worldbody>
            <body name="cart" pos="0 0 0">
                <joint name="slider" type="slide" axis="1 0 0" limited="true" range="-1 1"/>
                <geom name="cart" type="box" size="0.1 0.05 0.05"/>
                <body name="pole" pos="0 0 0.05">
                    <joint name="hinge" type="hinge" axis="0 1 0"/>
                    <geom name="pole" type="capsule" fromto="0 0 0 0 0 0.4" size="0.02"/>
                </body>
            </body>
        </worldbody>
        <actuator>
            <motor name="cart_motor" joint="slider" gear="10" ctrllimited="true" ctrlrange="-1 1"/>
        </actuator>
    </mujoco>
    """
    
    sim.load_model_from_string(cartpole_xml)
    
    # 设置初始状态 - 摆杆有一定倾斜
    sim.set_joint_positions([0.0, 0.3])
    
    # PD控制器参数
    kp_cart, kd_cart = 1.0, 0.5
    kp_pole, kd_pole = 15.0, 3.0
    
    # 收集数据
    times = []
    cart_positions = []
    pole_angles = []
    controls = []
    
    # 运行仿真
    for i in range(300):  # 3秒
        # 获取状态
        pos = sim.get_joint_positions()
        vel = sim.get_joint_velocities()
        
        # PD控制
        control = -kp_cart * pos[0] - kd_cart * vel[0] - kp_pole * pos[1] - kd_pole * vel[1]
        control = np.clip(control, -1.0, 1.0)
        
        # 记录数据
        times.append(sim.get_time())
        cart_positions.append(pos[0])
        pole_angles.append(pos[1])
        controls.append(control)
        
        # 应用控制
        sim.apply_control([control])
        sim.step()
    
    # 绘制结果
    print("\n📊 控制结果:")
    
    # 小车位置
    create_ascii_plot(cart_positions, title="小车位置", y_label="位置(m)")
    
    # 摆杆角度
    create_ascii_plot(pole_angles, title="摆杆角度", y_label="角度(rad)")
    
    # 控制输入
    create_ascii_plot(controls, title="控制输入", y_label="力(归一化)")
    
    # 性能指标
    print("\n📊 控制性能:")
    print(f"  - 初始摆杆角度: {0.3:.3f} rad ({np.degrees(0.3):.1f}°)")
    print(f"  - 最终摆杆角度: {pole_angles[-1]:.3f} rad ({np.degrees(pole_angles[-1]):.1f}°)")
    print(f"  - 最大偏移角度: {max(abs(a) for a in pole_angles):.3f} rad")
    print(f"  - 稳态误差: {abs(pole_angles[-1]):.4f} rad")
    print(f"  - 小车位移范围: [{min(cart_positions):.3f}, {max(cart_positions):.3f}] m")
    print(f"  - 控制能量: {sum(c**2 for c in controls) * 0.01:.3f} (归一化)")
    
    # 判断是否稳定
    last_100_angles = pole_angles[-100:]
    if all(abs(a) < 0.1 for a in last_100_angles):
        print("\n✅ 系统成功稳定！")
    else:
        print("\n⚠️  系统未完全稳定")

def main():
    """主函数"""
    print(f"{'='*60}")
    print(f"  MuJoCo MCP 仿真数据分析")
    print(f"  版本: {__version__}")
    print(f"{'='*60}")
    
    # 运行仿真
    simulate_pendulum()
    simulate_cartpole_control()
    
    print(f"\n{'='*60}")
    print("✅ 分析完成！")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
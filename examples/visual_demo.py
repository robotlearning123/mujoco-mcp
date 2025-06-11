#!/usr/bin/env python3
"""
MuJoCo MCP 可视化演示
生成ASCII艺术动画来展示仿真效果
"""
import sys
import time
import math
from pathlib import Path

# 添加项目到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from mujoco_mcp import MuJoCoSimulation, __version__

class ASCIIRenderer:
    """简单的ASCII渲染器"""
    
    def __init__(self, width=60, height=20):
        self.width = width
        self.height = height
        self.clear_screen = '\033[2J\033[H'  # ANSI转义序列清屏
        
    def render_pendulum(self, angle, time_val):
        """渲染单摆"""
        # 清屏
        print(self.clear_screen)
        
        # 标题
        print(f"🎯 MuJoCo MCP v{__version__} - 单摆仿真演示")
        print("="*self.width)
        
        # 计算摆球位置
        length = 15  # 摆长（字符单位）
        center_x = self.width // 2
        center_y = 5
        
        # 摆球位置
        ball_x = int(center_x + length * math.sin(angle))
        ball_y = int(center_y + length * math.cos(angle))
        
        # 创建画布
        canvas = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        
        # 绘制固定点
        canvas[center_y][center_x] = '█'
        
        # 绘制摆线
        steps = max(abs(ball_x - center_x), abs(ball_y - center_y))
        if steps > 0:
            for i in range(steps):
                t = i / steps
                x = int(center_x + t * (ball_x - center_x))
                y = int(center_y + t * (ball_y - center_y))
                if 0 <= x < self.width and 0 <= y < self.height:
                    canvas[y][x] = '│' if abs(angle) < 0.3 else '/'
        
        # 绘制摆球
        if 0 <= ball_x < self.width and 0 <= ball_y < self.height:
            canvas[ball_y][ball_x] = '●'
        
        # 打印画布
        for row in canvas:
            print(''.join(row))
        
        # 显示信息
        print("="*self.width)
        print(f"时间: {time_val:.2f}s | 角度: {angle:.3f} rad ({math.degrees(angle):.1f}°)")
        
    def render_cartpole(self, cart_pos, pole_angle, time_val):
        """渲染CartPole"""
        # 清屏
        print(self.clear_screen)
        
        # 标题
        print(f"🎯 MuJoCo MCP v{__version__} - CartPole仿真演示")
        print("="*self.width)
        
        # 参数
        ground_y = self.height - 5
        cart_width = 8
        cart_height = 3
        pole_length = 10
        
        # 计算位置
        cart_center_x = int(self.width // 2 + cart_pos * 20)  # 放大小车移动
        
        # 创建画布
        canvas = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        
        # 绘制地面
        for x in range(self.width):
            canvas[ground_y][x] = '─'
        
        # 绘制轨道
        for x in range(10, self.width-10):
            canvas[ground_y-1][x] = '═'
        
        # 绘制小车
        cart_left = max(0, cart_center_x - cart_width // 2)
        cart_right = min(self.width-1, cart_center_x + cart_width // 2)
        
        for y in range(ground_y-cart_height-1, ground_y-1):
            for x in range(cart_left, cart_right):
                if 0 <= x < self.width and 0 <= y < self.height:
                    if y == ground_y-cart_height-1 or y == ground_y-2:
                        canvas[y][x] = '─'
                    elif x == cart_left or x == cart_right-1:
                        canvas[y][x] = '│'
                    else:
                        canvas[y][x] = '█'
        
        # 绘制摆杆
        pole_end_x = int(cart_center_x + pole_length * math.sin(pole_angle))
        pole_end_y = int(ground_y - cart_height - 1 - pole_length * math.cos(pole_angle))
        
        # 画摆杆
        steps = pole_length
        for i in range(steps):
            t = i / steps
            x = int(cart_center_x + t * (pole_end_x - cart_center_x))
            y = int(ground_y - cart_height - 1 + t * (pole_end_y - (ground_y - cart_height - 1)))
            if 0 <= x < self.width and 0 <= y < self.height:
                canvas[y][x] = '║'
        
        # 绘制摆球
        if 0 <= pole_end_x < self.width and 0 <= pole_end_y < self.height:
            canvas[pole_end_y][pole_end_x] = '●'
        
        # 打印画布
        for row in canvas:
            print(''.join(row))
        
        # 显示信息
        print("="*self.width)
        print(f"时间: {time_val:.2f}s | 小车位置: {cart_pos:.3f} | 摆杆角度: {math.degrees(pole_angle):.1f}°")

def demo_pendulum():
    """单摆演示"""
    print("\n🎬 开始单摆演示...")
    time.sleep(1)
    
    # 创建仿真
    sim = MuJoCoSimulation()
    
    # 加载单摆模型
    pendulum_xml = """
    <mujoco model="pendulum">
        <option timestep="0.01" gravity="0 0 -9.81"/>
        <worldbody>
            <body name="pendulum" pos="0 0 1">
                <joint name="hinge" type="hinge" axis="0 1 0" damping="0.05"/>
                <geom name="rod" type="capsule" fromto="0 0 0 0 0 -1" size="0.02"/>
                <body name="ball" pos="0 0 -1">
                    <geom name="ball" type="sphere" size="0.1"/>
                </body>
            </body>
        </worldbody>
    </mujoco>
    """
    
    sim.load_model_from_string(pendulum_xml)
    
    # 设置初始角度
    sim.set_joint_positions([0.8])  # 约45度
    
    # 创建渲染器
    renderer = ASCIIRenderer()
    
    # 运行仿真
    for _ in range(150):  # 1.5秒
        # 获取状态
        angle = sim.get_joint_positions()[0]
        time_val = sim.get_time()
        
        # 渲染
        renderer.render_pendulum(angle, time_val)
        
        # 步进
        sim.step()
        
        # 控制帧率
        time.sleep(0.05)
    
    print("\n✅ 单摆演示完成！")
    time.sleep(1)

def demo_cartpole():
    """CartPole演示"""
    print("\n🎬 开始CartPole演示...")
    time.sleep(1)
    
    # 创建仿真
    sim = MuJoCoSimulation()
    
    # 加载CartPole模型
    cartpole_xml = """
    <mujoco model="cartpole">
        <option timestep="0.01"/>
        <worldbody>
            <body name="cart" pos="0 0 0">
                <joint name="slider" type="slide" axis="1 0 0" limited="true" range="-2 2"/>
                <geom name="cart" type="box" size="0.2 0.1 0.1"/>
                <body name="pole" pos="0 0 0.1">
                    <joint name="hinge" type="hinge" axis="0 1 0"/>
                    <geom name="pole" type="capsule" fromto="0 0 0 0 0 0.6" size="0.02"/>
                    <body name="mass" pos="0 0 0.6">
                        <geom name="mass" type="sphere" size="0.05"/>
                    </body>
                </body>
            </body>
        </worldbody>
        <actuator>
            <motor name="cart_motor" joint="slider" gear="10" ctrllimited="true" ctrlrange="-1 1"/>
        </actuator>
    </mujoco>
    """
    
    sim.load_model_from_string(cartpole_xml)
    
    # 设置初始状态
    sim.set_joint_positions([0.0, 0.2])  # 小车在中心，摆杆稍微倾斜
    
    # 创建渲染器
    renderer = ASCIIRenderer()
    
    # 简单的PD控制器参数
    kp_cart = 2.0
    kd_cart = 1.0
    kp_pole = 20.0
    kd_pole = 5.0
    
    # 运行仿真
    for _ in range(200):  # 2秒
        # 获取状态
        positions = sim.get_joint_positions()
        velocities = sim.get_joint_velocities()
        cart_pos = positions[0]
        pole_angle = positions[1]
        cart_vel = velocities[0]
        pole_vel = velocities[1]
        time_val = sim.get_time()
        
        # 简单的控制策略：保持小车在中心，摆杆竖直
        control = -kp_cart * cart_pos - kd_cart * cart_vel - kp_pole * pole_angle - kd_pole * pole_vel
        control = max(-1.0, min(1.0, control))  # 限制控制输入
        
        sim.apply_control([control])
        
        # 渲染
        renderer.render_cartpole(cart_pos, pole_angle, time_val)
        
        # 步进
        sim.step()
        
        # 控制帧率
        time.sleep(0.05)
    
    print("\n✅ CartPole演示完成！")

def demo_interactive():
    """交互式演示菜单"""
    print(f"\n🚀 MuJoCo MCP v{__version__} 可视化演示")
    print("="*60)
    print("\n请选择演示:")
    print("1. 单摆自由摆动")
    print("2. CartPole平衡控制")
    print("3. 运行所有演示")
    print("0. 退出")
    
    while True:
        choice = input("\n请输入选项 (0-3): ")
        
        if choice == '1':
            demo_pendulum()
        elif choice == '2':
            demo_cartpole()
        elif choice == '3':
            demo_pendulum()
            demo_cartpole()
            print("\n✅ 所有演示完成！")
        elif choice == '0':
            print("\n👋 再见！")
            break
        else:
            print("❌ 无效选项，请重试")

if __name__ == "__main__":
    # 检查终端是否支持ANSI
    import os
    if os.name == 'nt':  # Windows
        os.system('color')  # 启用ANSI支持
    
    try:
        demo_interactive()
    except KeyboardInterrupt:
        print("\n\n👋 演示中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
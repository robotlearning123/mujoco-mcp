#!/usr/bin/env python3
"""
MuJoCo MCP 动画演示
展示单摆的摆动动画
"""
import sys
import time
import math
from pathlib import Path

# 添加项目到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from mujoco_mcp import MuJoCoSimulation, __version__

def print_pendulum_frame(angle, time_val, frame_num):
    """打印单摆的一帧"""
    # 计算摆球位置
    length = 10
    x = int(20 + length * math.sin(angle))
    y = int(5 + length * math.cos(angle))
    
    # 创建20x40的画布
    canvas = []
    for i in range(15):
        canvas.append([' '] * 40)
    
    # 画固定点
    canvas[5][20] = 'O'
    
    # 画摆线
    steps = max(abs(x - 20), abs(y - 5))
    if steps > 0:
        for i in range(1, steps):
            t = i / steps
            lx = int(20 + t * (x - 20))
            ly = int(5 + t * (y - 5))
            if 0 <= lx < 40 and 0 <= ly < 15:
                canvas[ly][lx] = '|'
    
    # 画摆球
    if 0 <= x < 40 and 0 <= y < 15:
        canvas[y][x] = '●'
    
    # 打印帧
    print(f"\n帧 {frame_num:03d} | 时间: {time_val:.2f}s | 角度: {math.degrees(angle):+6.1f}°")
    print("=" * 40)
    for row in canvas:
        print(''.join(row))
    print("=" * 40)

def main():
    print(f"🎬 MuJoCo MCP v{__version__} - 单摆动画演示")
    print("=" * 40)
    
    # 创建仿真
    sim = MuJoCoSimulation()
    
    # 单摆模型
    pendulum_xml = """
    <mujoco model="pendulum">
        <option timestep="0.02" gravity="0 0 -9.81"/>
        <worldbody>
            <body name="pendulum" pos="0 0 1">
                <joint name="hinge" type="hinge" axis="0 1 0" damping="0.05"/>
                <geom name="rod" type="capsule" fromto="0 0 0 0 0 -1" size="0.02"/>
            </body>
        </worldbody>
    </mujoco>
    """
    
    sim.load_model_from_string(pendulum_xml)
    
    # 设置初始角度
    sim.set_joint_positions([0.8])
    
    print("\n🎬 开始播放动画（10帧展示）...\n")
    
    # 显示10帧
    for frame in range(10):
        # 获取状态
        angle = sim.get_joint_positions()[0]
        time_val = sim.get_time()
        
        # 打印这一帧
        print_pendulum_frame(angle, time_val, frame + 1)
        
        # 仿真前进
        sim.step(5)  # 每帧前进0.1秒
        
        # 等待一下，模拟动画效果
        time.sleep(0.5)
    
    # 最终统计
    print("\n📊 动画统计:")
    print(f"  - 初始角度: 45.8°")
    print(f"  - 最终角度: {math.degrees(sim.get_joint_positions()[0]):.1f}°")
    print(f"  - 仿真时间: {sim.get_time():.1f}秒")
    print(f"  - 演示帧数: 10帧")
    
    print("\n✅ 动画演示完成！")
    print("\n💡 提示: 实际应用中可以使用 mujoco-viewer 进行3D可视化")

if __name__ == "__main__":
    main()
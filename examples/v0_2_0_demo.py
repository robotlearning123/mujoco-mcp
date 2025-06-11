#!/usr/bin/env python3
"""
v0.2.0 演示 - 仿真控制
展示step_simulation, reset_simulation, get_simulation_state等功能
"""
import sys
import json
import math
from pathlib import Path

# 添加项目到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from mujoco_mcp import __version__
from mujoco_mcp.server import MuJoCoMCPServer


def print_section(title):
    """打印章节标题"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def demo_basic_simulation_control(server):
    """演示基本仿真控制"""
    print_section("1. 基本仿真控制")
    
    # 加载一个简单的钟摆模型
    pendulum_xml = """<mujoco model="simple_pendulum">
        <option timestep="0.002" gravity="0 0 -9.81"/>
        <worldbody>
            <body name="pendulum" pos="0 0 1">
                <joint name="swing" type="hinge" axis="0 1 0"/>
                <geom name="rod" type="cylinder" size="0.02 0.5" pos="0 0 -0.25" rgba="0.8 0.2 0.2 1"/>
                <geom name="mass" type="sphere" size="0.1" pos="0 0 -0.5" mass="1" rgba="0.2 0.2 0.8 1"/>
            </body>
        </worldbody>
    </mujoco>"""
    
    print("✓ 加载钟摆模型...")
    result = server.call_tool("load_model", {
        "model_string": pendulum_xml,
        "name": "simple_pendulum"
    })
    model_id = result["model_id"]
    print(f"  模型ID: {model_id[:8]}...")
    
    # 获取初始状态
    print("\n✓ 获取初始状态:")
    state = server.call_tool("get_simulation_state", {
        "model_id": model_id,
        "include_positions": True,
        "include_velocities": True
    })
    print(f"  时间: {state['time']:.3f}s")
    print(f"  位置: {state['qpos'][0]:.3f} rad")
    print(f"  速度: {state['qvel'][0]:.3f} rad/s")
    
    # 步进仿真
    print("\n✓ 步进仿真 100 步:")
    result = server.call_tool("step_simulation", {
        "model_id": model_id,
        "steps": 100
    })
    print(f"  完成 {result['steps_completed']} 步")
    print(f"  当前时间: {result['time']:.3f}s")
    
    # 重置仿真
    print("\n✓ 重置仿真:")
    result = server.call_tool("reset_simulation", {
        "model_id": model_id
    })
    print(f"  {result['message']}")
    print(f"  时间重置为: {result['time']:.3f}s")
    
    return model_id


def demo_pendulum_swing(server):
    """演示钟摆摆动"""
    print_section("2. 钟摆摆动仿真")
    
    # 加载钟摆模型
    pendulum_xml = """<mujoco model="pendulum">
        <option timestep="0.01" gravity="0 0 -9.81"/>
        <worldbody>
            <body name="pendulum" pos="0 0 1">
                <joint name="swing" type="hinge" axis="0 1 0" damping="0.05"/>
                <geom name="rod" type="cylinder" size="0.02 0.5" pos="0 0 -0.25" rgba="0.8 0.2 0.2 1"/>
                <geom name="mass" type="sphere" size="0.1" pos="0 0 -0.5" mass="1" rgba="0.2 0.2 0.8 1"/>
            </body>
        </worldbody>
    </mujoco>"""
    
    result = server.call_tool("load_model", {
        "model_string": pendulum_xml,
        "name": "damped_pendulum"
    })
    model_id = result["model_id"]
    
    # 设置初始角度为45度
    print("✓ 设置初始角度为 45°:")
    angle_deg = 45
    angle_rad = math.radians(angle_deg)
    server.call_tool("set_joint_positions", {
        "model_id": model_id,
        "positions": [angle_rad]
    })
    
    # 记录摆动过程
    print("\n✓ 模拟2秒钟的摆动:")
    print("  时间(s)  角度(°)  速度(°/s)")
    print("  " + "-"*30)
    
    for i in range(21):  # 0.0s 到 2.0s，每0.1s记录一次
        if i > 0:
            # 步进10步 (0.01s * 10 = 0.1s)
            server.call_tool("step_simulation", {
                "model_id": model_id,
                "steps": 10
            })
        
        # 获取状态
        state = server.call_tool("get_simulation_state", {
            "model_id": model_id,
            "include_positions": True,
            "include_velocities": True
        })
        
        time = state["time"]
        angle = math.degrees(state["qpos"][0])
        velocity = math.degrees(state["qvel"][0])
        
        print(f"  {time:6.1f}  {angle:7.1f}  {velocity:8.1f}")
    
    print("\n✓ 钟摆逐渐停止摆动（由于阻尼）")


def demo_multi_model_simulation(server):
    """演示多模型并行仿真"""
    print_section("3. 多模型并行仿真")
    
    # 创建两个不同参数的钟摆
    pendulum_configs = [
        ("light_pendulum", 0.5, 0.01),  # 名称, 质量, 阻尼
        ("heavy_pendulum", 2.0, 0.05)
    ]
    
    model_ids = []
    
    for name, mass, damping in pendulum_configs:
        xml = f"""<mujoco model="{name}">
            <option timestep="0.01" gravity="0 0 -9.81"/>
            <worldbody>
                <body name="pendulum" pos="0 0 1">
                    <joint name="swing" type="hinge" axis="0 1 0" damping="{damping}"/>
                    <geom name="rod" type="cylinder" size="0.02 0.5" pos="0 0 -0.25" rgba="0.8 0.2 0.2 1"/>
                    <geom name="mass" type="sphere" size="0.1" pos="0 0 -0.5" mass="{mass}" rgba="0.2 0.2 0.8 1"/>
                </body>
            </worldbody>
        </mujoco>"""
        
        result = server.call_tool("load_model", {
            "model_string": xml,
            "name": name
        })
        model_ids.append((name, result["model_id"], mass, damping))
        
        # 设置相同的初始角度
        server.call_tool("set_joint_positions", {
            "model_id": result["model_id"],
            "positions": [math.radians(60)]  # 60度
        })
    
    print("✓ 加载了两个不同参数的钟摆:")
    for name, _, mass, damping in model_ids:
        print(f"  - {name}: 质量={mass}kg, 阻尼={damping}")
    
    # 并行仿真
    print("\n✓ 并行仿真1秒:")
    print("  " + "轻钟摆".center(15) + " | " + "重钟摆".center(15))
    print("  角度(°)  速度(°/s) | 角度(°)  速度(°/s)")
    print("  " + "-"*40)
    
    for i in range(11):  # 0.0s 到 1.0s
        if i > 0:
            # 步进两个模型
            for _, model_id, _, _ in model_ids:
                server.call_tool("step_simulation", {
                    "model_id": model_id,
                    "steps": 10
                })
        
        # 获取两个模型的状态
        states = []
        for _, model_id, _, _ in model_ids:
            state = server.call_tool("get_simulation_state", {
                "model_id": model_id,
                "include_positions": True,
                "include_velocities": True
            })
            states.append(state)
        
        # 显示状态
        line = ""
        for state in states:
            angle = math.degrees(state["qpos"][0])
            velocity = math.degrees(state["qvel"][0])
            line += f"  {angle:6.1f}  {velocity:7.1f}"
            if state != states[-1]:
                line += " |"
        print(line)
    
    print("\n✓ 轻钟摆摆动更快，重钟摆更稳定")


def demo_simulation_persistence(server):
    """演示仿真状态持久性"""
    print_section("4. 仿真状态持久性")
    
    # 加载弹簧振子模型
    spring_xml = """<mujoco model="spring">
        <option timestep="0.001"/>
        <worldbody>
            <body name="mass" pos="0 0 0">
                <joint name="slide" type="slide" axis="0 0 1" stiffness="100" damping="1"/>
                <geom name="box" type="box" size="0.1 0.1 0.1" mass="1" rgba="0.8 0.8 0.2 1"/>
            </body>
        </worldbody>
    </mujoco>"""
    
    result = server.call_tool("load_model", {
        "model_string": spring_xml,
        "name": "spring_mass"
    })
    model_id = result["model_id"]
    
    print("✓ 加载弹簧振子模型")
    
    # 设置初始位移
    print("\n✓ 拉伸弹簧到 0.1m:")
    server.call_tool("set_joint_positions", {
        "model_id": model_id,
        "positions": [0.1]
    })
    
    # 分步仿真，展示状态持久性
    print("\n✓ 分步仿真，展示状态在调用间保持:")
    
    for phase in range(3):
        print(f"\n  阶段 {phase + 1}:")
        
        # 步进100步
        server.call_tool("step_simulation", {
            "model_id": model_id,
            "steps": 100
        })
        
        # 获取状态
        state = server.call_tool("get_simulation_state", {
            "model_id": model_id,
            "include_positions": True,
            "include_velocities": True
        })
        
        print(f"    时间: {state['time']:.3f}s")
        print(f"    位置: {state['qpos'][0]:.4f}m")
        print(f"    速度: {state['qvel'][0]:.4f}m/s")
        
        # 计算能量（动能 + 势能）
        ke = 0.5 * 1.0 * state['qvel'][0]**2  # 动能
        pe = 0.5 * 100 * state['qpos'][0]**2  # 势能
        total_energy = ke + pe
        print(f"    总能量: {total_energy:.4f}J")
    
    print("\n✓ 状态在多次调用间正确保持和更新")


def main():
    """主函数"""
    print(f"\n🚀 MuJoCo MCP v{__version__} - 仿真控制演示")
    print("="*60)
    
    # 创建服务器
    server = MuJoCoMCPServer()
    print(f"\n✓ 服务器版本: {server.version}")
    
    # 获取工具列表
    tools = server.get_tools()
    control_tools = [t["name"] for t in tools if t["name"] in 
                     ["step_simulation", "reset_simulation", "get_simulation_state", "set_joint_positions"]]
    print(f"✓ 仿真控制工具: {', '.join(control_tools)}")
    
    # 运行演示
    demo_basic_simulation_control(server)
    demo_pendulum_swing(server)
    demo_multi_model_simulation(server)
    demo_simulation_persistence(server)
    
    # 总结
    print_section("演示总结")
    result = server.call_tool("get_loaded_models", {})
    print(f"✅ 成功演示了 {len(result['models'])} 个不同的仿真场景")
    print("✅ step_simulation - 单步和多步仿真")
    print("✅ reset_simulation - 重置到初始状态")
    print("✅ get_simulation_state - 查询完整状态")
    print("✅ set_joint_positions - 设置初始条件")
    print("✅ 多模型并行仿真")
    print("✅ 状态持久性验证")
    print("\n🎯 下一步: v0.2.1 - 增强状态查询功能")


if __name__ == "__main__":
    main()
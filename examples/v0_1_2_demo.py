#!/usr/bin/env python3
"""
v0.1.2 演示 - 第一个MCP工具
展示load_model和get_loaded_models功能
"""
import sys
import json
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


def demo_load_simple_model(server):
    """演示加载简单模型"""
    print_section("1. 加载简单模型")
    
    # 定义一个简单的盒子模型
    simple_xml = """<mujoco>
        <worldbody>
            <body name="box">
                <geom type="box" size="0.1 0.1 0.1" rgba="1 0 0 1"/>
            </body>
        </worldbody>
    </mujoco>"""
    
    print("✓ 加载简单盒子模型:")
    print(f"  XML长度: {len(simple_xml)} 字符")
    
    # 调用load_model工具
    result = server.call_tool("load_model", {
        "model_string": simple_xml,
        "name": "simple_box"
    })
    
    print(f"\n✓ 加载结果:")
    print(json.dumps(result, indent=2))
    
    return result["model_id"]


def demo_load_pendulum_model(server):
    """演示加载钟摆模型"""
    print_section("2. 加载钟摆模型")
    
    # 定义一个钟摆模型
    pendulum_xml = """<mujoco model="pendulum">
        <option timestep="0.01" gravity="0 0 -9.81"/>
        <worldbody>
            <body name="cart" pos="0 0 1">
                <joint name="slider" type="slide" axis="1 0 0" limited="true" range="-1 1"/>
                <geom name="cart" type="box" size="0.2 0.2 0.1" rgba="0.7 0.7 0 1"/>
                <body name="pole" pos="0 0 0">
                    <joint name="hinge" type="hinge" axis="0 1 0" limited="true" range="-180 180"/>
                    <geom name="pole" type="cylinder" size="0.05 0.5" rgba="0 0.7 0.7 1" pos="0 0 0.5"/>
                    <site name="tip" pos="0 0 1" size="0.01"/>
                </body>
            </body>
        </worldbody>
        <actuator>
            <motor joint="slider" name="slide_motor" gear="10"/>
        </actuator>
    </mujoco>"""
    
    print("✓ 加载钟摆模型 (带滑块和驱动器):")
    print(f"  - 2个关节 (滑块 + 铰链)")
    print(f"  - 1个驱动器")
    
    # 调用load_model工具
    result = server.call_tool("load_model", {
        "model_string": pendulum_xml,
        "name": "cart_pendulum"
    })
    
    print(f"\n✓ 模型信息:")
    info = result["model_info"]
    print(f"  - 广义坐标数 (nq): {info['nq']}")
    print(f"  - 自由度数 (nv): {info['nv']}")
    print(f"  - 刚体数量 (nbody): {info['nbody']}")
    print(f"  - 关节数量 (njoint): {info['njoint']}")
    print(f"  - 几何体数量 (ngeom): {info['ngeom']}")
    print(f"  - 驱动器数量 (nu): {info['nu']}")
    print(f"  - 时间步长: {info['timestep']}")
    
    return result["model_id"]


def demo_get_loaded_models(server):
    """演示获取已加载的模型列表"""
    print_section("3. 获取已加载的模型")
    
    result = server.call_tool("get_loaded_models", {})
    
    print(f"✓ 当前已加载 {len(result['models'])} 个模型:\n")
    
    for i, model in enumerate(result['models'], 1):
        print(f"{i}. {model['name']}")
        print(f"   - ID: {model['model_id']}")
        print(f"   - 自由度: {model['nv']}")
        print(f"   - 刚体数: {model['nbody']}")
        print()


def demo_error_handling(server):
    """演示错误处理"""
    print_section("4. 错误处理演示")
    
    # 测试空模型字符串
    print("✓ 测试空模型字符串:")
    try:
        server.call_tool("load_model", {"model_string": ""})
    except ValueError as e:
        print(f"  ✓ 正确捕获错误: {e}")
    
    # 测试无效XML
    print("\n✓ 测试无效XML:")
    try:
        server.call_tool("load_model", {"model_string": "not xml"})
    except ValueError as e:
        print(f"  ✓ 正确捕获错误: {e}")
    
    # 测试缺少必需参数
    print("\n✓ 测试缺少必需参数:")
    try:
        server.call_tool("load_model", {})
    except ValueError as e:
        print(f"  ✓ 正确捕获错误: {e}")


def demo_multiple_models(server):
    """演示多模型管理"""
    print_section("5. 多模型管理")
    
    # 加载多个不同的模型
    models = [
        ("ball", """<mujoco>
            <worldbody>
                <body name="ball">
                    <geom type="sphere" size="0.1" rgba="0 1 0 1"/>
                </body>
            </worldbody>
        </mujoco>"""),
        
        ("double_pendulum", """<mujoco>
            <worldbody>
                <body name="link1" pos="0 0 1">
                    <joint name="joint1" type="hinge" axis="0 1 0"/>
                    <geom type="cylinder" size="0.05 0.3" pos="0 0 0.15" rgba="1 0 0 1"/>
                    <body name="link2" pos="0 0 0.3">
                        <joint name="joint2" type="hinge" axis="0 1 0"/>
                        <geom type="cylinder" size="0.05 0.3" pos="0 0 0.15" rgba="0 0 1 1"/>
                    </body>
                </body>
            </worldbody>
        </mujoco>"""),
    ]
    
    print("✓ 加载额外的模型:")
    for name, xml in models:
        result = server.call_tool("load_model", {
            "model_string": xml,
            "name": name
        })
        print(f"  - {name}: {result['model_id'][:8]}...")
    
    print("\n✓ 最终模型列表:")
    result = server.call_tool("get_loaded_models", {})
    print(f"  共 {len(result['models'])} 个模型已加载")


def main():
    """主函数"""
    print(f"\n🚀 MuJoCo MCP v{__version__} - load_model工具演示")
    print("="*60)
    
    # 创建服务器
    server = MuJoCoMCPServer()
    print(f"\n✓ 服务器版本: {server.version}")
    
    # 获取工具列表
    tools = server.get_tools()
    tool_names = [t["name"] for t in tools]
    print(f"✓ 可用工具: {', '.join(tool_names)}")
    
    # 运行演示
    model_id1 = demo_load_simple_model(server)
    model_id2 = demo_load_pendulum_model(server)
    demo_get_loaded_models(server)
    demo_error_handling(server)
    demo_multiple_models(server)
    
    # 最终统计
    print_section("演示总结")
    result = server.call_tool("get_loaded_models", {})
    print(f"✅ 成功加载 {len(result['models'])} 个模型")
    print("✅ load_model工具工作正常")
    print("✅ get_loaded_models工具工作正常")
    print("✅ 错误处理机制完善")
    print("✅ 支持多模型管理")
    print("\n🎯 下一步: v0.2.0 - 实现仿真控制 (step, reset)")


if __name__ == "__main__":
    main()
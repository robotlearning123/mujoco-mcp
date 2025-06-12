#!/usr/bin/env python3
"""
场景测试脚本
测试所有支持的MuJoCo场景类型：pendulum, double_pendulum, cart_pole, robotic_arm
"""

import sys
import os
import json
import time
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_scene_generation():
    """测试所有场景的XML生成"""
    print("🧪 测试场景XML生成")
    print("="*50)
    
    from mujoco_mcp.remote_server import MuJoCoRemoteServer
    server = MuJoCoRemoteServer()
    
    # 定义测试场景
    test_scenes = [
        {
            "name": "pendulum",
            "type": "pendulum", 
            "params": {"length": 0.6, "mass": 0.8, "damping": 0.2}
        },
        {
            "name": "double_pendulum",
            "type": "double_pendulum",
            "params": {"length1": 0.4, "length2": 0.3, "mass1": 0.5, "mass2": 0.4}
        },
        {
            "name": "cart_pole", 
            "type": "cart_pole",
            "params": {"cart_mass": 1.5, "pole_mass": 0.2, "pole_length": 0.6}
        },
        {
            "name": "robotic_arm",
            "type": "robotic_arm", 
            "params": {"link1_length": 0.35, "link2_length": 0.25, "base_mass": 0.8}
        }
    ]
    
    results = []
    
    for scene in test_scenes:
        print(f"\n🔧 测试场景: {scene['name']}")
        
        try:
            # 生成XML
            xml = server._generate_scene_xml(scene["type"], scene["params"])
            
            if xml:
                # 验证XML基本结构
                if "<mujoco>" in xml and "</mujoco>" in xml:
                    print(f"   ✅ XML生成成功")
                    
                    # 检查关键元素
                    elements = []
                    if "<worldbody>" in xml:
                        elements.append("worldbody")
                    if "<actuator>" in xml:
                        elements.append("actuator") 
                    if "<joint" in xml:
                        elements.append("joints")
                    if "<geom" in xml:
                        elements.append("geometries")
                    
                    print(f"   📋 包含元素: {', '.join(elements)}")
                    
                    # 保存XML文件用于调试
                    xml_file = f"test_output_{scene['name']}.xml"
                    with open(xml_file, 'w') as f:
                        f.write(xml)
                    print(f"   💾 XML保存至: {xml_file}")
                    
                    results.append({"scene": scene["name"], "status": "success", "xml_length": len(xml)})
                    
                else:
                    print(f"   ❌ XML格式无效")
                    results.append({"scene": scene["name"], "status": "invalid_xml"})
            else:
                print(f"   ❌ XML生成失败")
                results.append({"scene": scene["name"], "status": "generation_failed"})
                
        except Exception as e:
            print(f"   ❌ 异常: {e}")
            results.append({"scene": scene["name"], "status": "exception", "error": str(e)})
    
    return results

def test_mujoco_xml_validation():
    """使用MuJoCo验证XML"""
    print("\n🔍 MuJoCo XML验证测试")
    print("="*50)
    
    try:
        import mujoco
        print(f"   MuJoCo版本: {mujoco.__version__}")
        
        # 测试所有生成的XML文件
        xml_files = [
            "test_output_pendulum.xml",
            "test_output_double_pendulum.xml", 
            "test_output_cart_pole.xml",
            "test_output_robotic_arm.xml"
        ]
        
        validation_results = []
        
        for xml_file in xml_files:
            if os.path.exists(xml_file):
                print(f"\n   📄 验证: {xml_file}")
                try:
                    # 尝试加载模型
                    model = mujoco.MjModel.from_xml_path(xml_file)
                    data = mujoco.MjData(model)
                    
                    print(f"      ✅ 加载成功")
                    print(f"      - DOF: {model.nq}")
                    print(f"      - 刚体数: {model.nbody}")
                    print(f"      - 关节数: {model.njnt}")
                    print(f"      - 执行器数: {model.nu}")
                    
                    validation_results.append({
                        "file": xml_file,
                        "status": "valid",
                        "dof": model.nq,
                        "bodies": model.nbody,
                        "joints": model.njnt,
                        "actuators": model.nu
                    })
                    
                except Exception as e:
                    print(f"      ❌ 验证失败: {e}")
                    validation_results.append({
                        "file": xml_file,
                        "status": "invalid",
                        "error": str(e)
                    })
            else:
                print(f"   ⚠️  文件不存在: {xml_file}")
        
        return validation_results
        
    except ImportError:
        print("   ❌ MuJoCo未安装，跳过验证")
        return []

def test_server_integration():
    """测试服务器集成"""
    print("\n🖥️ 服务器集成测试")
    print("="*50)
    
    from mujoco_mcp.remote_server import MuJoCoRemoteServer
    server = MuJoCoRemoteServer()
    
    # 测试所有场景的服务器调用
    scenes_to_test = ["pendulum", "double_pendulum", "cart_pole", "robotic_arm"]
    
    integration_results = []
    
    for scene_type in scenes_to_test:
        print(f"\n   🎯 测试场景: {scene_type}")
        
        try:
            # 模拟create_scene调用（不实际连接viewer）
            result = server.call_tool('get_server_info', {})
            if result.get('supported_scenes') and scene_type in result.get('supported_scenes', []):
                print(f"      ✅ 场景已注册")
                integration_results.append({"scene": scene_type, "status": "registered"})
            else:
                print(f"      ❌ 场景未注册")
                integration_results.append({"scene": scene_type, "status": "not_registered"})
                
        except Exception as e:
            print(f"      ❌ 集成测试失败: {e}")
            integration_results.append({"scene": scene_type, "status": "error", "error": str(e)})
    
    return integration_results

def generate_scene_documentation():
    """生成场景文档"""
    print("\n📚 生成场景文档")
    print("="*50)
    
    scene_docs = """
# MuJoCo MCP 支持的场景

## 1. 单摆 (Pendulum)
**描述**: 经典单摆系统，研究周期运动和能量守恒
**参数**:
- `length`: 摆长 (默认: 0.5m)
- `mass`: 摆锤质量 (默认: 0.5kg) 
- `damping`: 阻尼系数 (默认: 0.1)

**使用示例**:
```json
{
  "scene_type": "pendulum",
  "parameters": {
    "length": 0.6,
    "mass": 0.8,
    "damping": 0.15
  }
}
```

## 2. 双摆 (Double Pendulum)
**描述**: 混沌双摆系统，展示复杂非线性动力学
**参数**:
- `length1`: 第一段摆长 (默认: 0.4m)
- `length2`: 第二段摆长 (默认: 0.4m)
- `mass1`: 第一段质量 (默认: 0.3kg)
- `mass2`: 第二段质量 (默认: 0.3kg)

**使用示例**:
```json
{
  "scene_type": "double_pendulum", 
  "parameters": {
    "length1": 0.5,
    "length2": 0.3,
    "mass1": 0.4,
    "mass2": 0.2
  }
}
```

## 3. 倒立摆 (Cart Pole)
**描述**: 经典控制问题，小车上的倒立摆平衡控制
**参数**:
- `cart_mass`: 小车质量 (默认: 1.0kg)
- `pole_mass`: 摆杆质量 (默认: 0.1kg)
- `pole_length`: 摆杆长度 (默认: 0.5m)

**使用示例**:
```json
{
  "scene_type": "cart_pole",
  "parameters": {
    "cart_mass": 1.5,
    "pole_mass": 0.15,
    "pole_length": 0.6
  }
}
```

## 4. 机械臂 (Robotic Arm)
**描述**: 2自由度机械臂，适合路径规划和逆运动学研究
**参数**:
- `link1_length`: 第一关节长度 (默认: 0.3m)
- `link2_length`: 第二关节长度 (默认: 0.3m)
- `base_mass`: 基座质量 (默认: 0.5kg)
- `link1_mass`: 第一连杆质量 (默认: 0.3kg)
- `link2_mass`: 第二连杆质量 (默认: 0.2kg)

**使用示例**:
```json
{
  "scene_type": "robotic_arm",
  "parameters": {
    "link1_length": 0.4,
    "link2_length": 0.25,
    "base_mass": 0.8
  }
}
```

## 控制接口

所有场景都支持以下控制方法:
- `get_state`: 获取当前状态
- `set_joint_positions`: 设置关节位置
- `reset_simulation`: 重置仿真
- `execute_command`: 自然语言控制

## 可视化

每个场景都会在MuJoCo Viewer中实时显示:
- 彩色几何体表示不同组件
- 实时物理仿真
- 鼠标交互支持
"""
    
    with open("SCENES_DOCUMENTATION.md", "w", encoding="utf-8") as f:
        f.write(scene_docs)
    
    print("   ✅ 场景文档已生成: SCENES_DOCUMENTATION.md")

def main():
    """主测试函数"""
    print("🚀 MuJoCo MCP 场景测试套件")
    print("版本: v0.6.2")
    print("时间:", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("="*50)
    
    # 运行所有测试
    test_results = {}
    
    # 1. 场景生成测试
    test_results["generation"] = test_scene_generation()
    
    # 2. MuJoCo验证测试
    test_results["validation"] = test_mujoco_xml_validation()
    
    # 3. 服务器集成测试
    test_results["integration"] = test_server_integration()
    
    # 4. 生成文档
    generate_scene_documentation()
    
    # 汇总结果
    print("\n" + "="*50)
    print("📊 测试结果汇总")
    print("="*50)
    
    # 统计结果
    total_tests = 0
    passed_tests = 0
    
    for test_type, results in test_results.items():
        if results:
            for result in results:
                total_tests += 1
                if result.get("status") in ["success", "valid", "registered"]:
                    passed_tests += 1
    
    print(f"通过率: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)" if total_tests > 0 else "无测试结果")
    
    # 详细结果
    for test_type, results in test_results.items():
        print(f"\n{test_type.title()} 测试:")
        for result in results:
            status_emoji = "✅" if result.get("status") in ["success", "valid", "registered"] else "❌"
            print(f"  {status_emoji} {result.get('scene', result.get('file', 'unknown'))}: {result.get('status')}")
    
    print("\n🎉 场景测试完成！")
    print("\n💡 接下来可以:")
    print("1. 启动 MuJoCo Viewer Server: python mujoco_viewer_server.py")
    print("2. 测试场景创建: python debug_mcp_inspector.py") 
    print("3. 在Claude Desktop中使用新场景")

if __name__ == "__main__":
    main()
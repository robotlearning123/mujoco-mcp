
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

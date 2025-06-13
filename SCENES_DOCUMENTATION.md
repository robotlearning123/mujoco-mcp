
# MuJoCo MCP Supported Scenes

## 1. Pendulum
**Description**: Classic pendulum system for studying periodic motion and energy conservation
**Parameters**:
- `length`: length (default: 0.5m)
- `mass`: pendulum mass (default: 0.5kg) 
- `damping`: damping (default: 0.1)

**Usage Example**:
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

## 2. Double Pendulum
**Description**: Chaotic double pendulum system demonstrating complex nonlinear dynamics
**Parameters**:
- `length1`: first segment length (default: 0.4m)
- `length2`: second segment length (default: 0.4m)
- `mass1`: first segment mass (default: 0.3kg)
- `mass2`: second segment mass (default: 0.3kg)

**Usage Example**:
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

## 3. Cart Pole
**Description**: Classic control problem, balancing an inverted pendulum on a cart
**Parameters**:
- `cart_mass`: cart mass (default: 1.0kg)
- `pole_mass`: pole mass (default: 0.1kg)
- `pole_length`: pole length (default: 0.5m)

**Usage Example**:
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

## 4. Robotic Arm
**Description**: 2-DOF manipulator, suitable for path planning and inverse kinematics research
**Parameters**:
- `link1_length`: first joint length (default: 0.3m)
- `link2_length`: second joint length (default: 0.3m)
- `base_mass`: base mass (default: 0.5kg)
- `link1_mass`: first link mass (default: 0.3kg)
- `link2_mass`: second link mass (default: 0.2kg)

**Usage Example**:
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

## Control Interface

All scenes support the following control methods:
- `get_state`: Get current state
- `set_joint_positions`: Set joint positions
- `reset_simulation`: Reset simulation
- `execute_command`: Natural language control

## Visualization

Each scene is displayed in real-time in MuJoCo Viewer:
- 彩色几何体表示不同组件
- 实时物理仿真
- 鼠标交互支持

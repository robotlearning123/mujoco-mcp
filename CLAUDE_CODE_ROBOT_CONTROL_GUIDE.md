# 🤖 **Complete Guide: Control Robots in MuJoCo via Claude Code MCP**

## 📚 **Step-by-Step Teaching Guide**

### **Prerequisites**
1. MuJoCo installed: `pip install mujoco`
2. MCP installed: `pip install mcp`
3. Claude Desktop configured with MCP server

---

## 🎯 **Step 1: Setup MCP Server Configuration**

### **1.1 Create the configuration file**
Location depends on your OS:
- **macOS/Linux**: `~/.config/claude-desktop/claude_desktop_config.json`
- **Windows**: `%APPDATA%/claude-desktop/claude_desktop_config.json`

### **1.2 Add robot control server configuration**
```json
{
  "mcpServers": {
    "mujoco-robot": {
      "command": "python",
      "args": ["-m", "mujoco_mcp.mcp_server_robot"],
      "cwd": "/path/to/mujoco-mcp",
      "env": {
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": "./src"
      }
    }
  }
}
```

### **1.3 Restart Claude Desktop**
After saving the configuration, restart Claude Desktop to load the MCP server.

---

## 🎮 **Step 2: Basic Robot Control Commands**

Once configured, you can control robots directly through conversation with Claude:

### **2.1 Load a Robot**
```
You: "Load a robot arm into the simulation"
Claude: [Uses load_robot tool with robot_type="arm"]
Response: "✅ Robot arm loaded with ID 'arm_1', 3 joints available"
```

### **2.2 Check Robot State**
```
You: "What's the current state of the robot arm?"
Claude: [Uses get_robot_state tool]
Response: "📊 Robot arm state:
  - Joint positions: [0.0, 0.0, 0.0] rad
  - Joint velocities: [0.0, 0.0, 0.0] rad/s
  - Simulation time: 0.0s"
```

### **2.3 Move Robot Joints**
```
You: "Move the robot arm to position [0.5, 1.0, 0.3]"
Claude: [Uses set_joint_positions tool]
        [Uses step_robot tool to simulate]
Response: "✅ Robot moved to target positions"
```

---

## 🔧 **Step 3: Advanced Control Modes**

### **3.1 Position Control**
```
You: "Set joint 1 to 45 degrees, joint 2 to 90 degrees"
Claude: [Converts to radians: 0.785, 1.571]
        [Uses set_joint_positions]
Response: "✅ Joints positioned at specified angles"
```

### **3.2 Velocity Control**
```
You: "Make the robot arm rotate joint 1 at 0.5 rad/s"
Claude: [Uses set_joint_velocities with [0.5, 0.0, 0.0]]
Response: "⚡ Joint 1 rotating at 0.5 rad/s"
```

### **3.3 Torque Control**
```
You: "Apply 2 Nm torque to joint 2"
Claude: [Uses set_joint_torques with [0.0, 2.0, 0.0]]
Response: "💪 Torque applied to joint 2"
```

---

## 🎪 **Step 4: Complex Robot Tasks**

### **4.1 Execute a Trajectory**
```
You: "Make the robot follow a circular path"
Claude: [Generates circular trajectory points]
        [Uses execute_trajectory tool]
Response: "🎯 Executing circular trajectory with 8 waypoints..."
```

### **4.2 Pick and Place Operation**
```
You: "Load a gripper and perform a pick operation"
Claude: [Uses load_robot with robot_type="gripper"]
        [Opens gripper: set_joint_positions [0.04, 0.04]]
        [Closes gripper: set_joint_positions [0.0, 0.0]]
Response: "🤏 Gripper opened → moved → closed (object grasped)"
```

### **4.3 Mobile Robot Navigation**
```
You: "Load a mobile robot and move it in a square pattern"
Claude: [Uses load_robot with robot_type="mobile"]
        [Creates square trajectory]
        [Uses execute_trajectory]
Response: "🚗 Mobile robot navigating square pattern..."
```

---

## 📊 **Step 5: Sensor Feedback and Control Loops**

### **5.1 Monitor Robot State**
```
You: "Continuously monitor the robot arm while it moves"
Claude: [Uses get_robot_state repeatedly]
Response: "📊 Monitoring:
  Time 0.1s: Position [0.1, 0.2, 0.05]
  Time 0.2s: Position [0.2, 0.4, 0.1]
  Time 0.3s: Position [0.3, 0.6, 0.15]"
```

### **5.2 Feedback Control**
```
You: "Move joint 1 to 1.0 rad and tell me when it reaches the target"
Claude: [Uses set_joint_positions]
        [Steps simulation]
        [Checks state]
Response: "✅ Joint 1 reached target position (error < 0.01 rad)"
```

---

## 🎯 **Step 6: Complete Robot Control Workflow**

### **Example: Robot Arm Pick and Place**

```
You: "Perform a complete pick and place operation with a robot arm"

Claude's Actions:
1. [load_robot: arm] → "Robot arm loaded"
2. [load_robot: gripper] → "Gripper loaded"
3. [set_joint_positions: arm to pick position]
4. [set_joint_positions: gripper open]
5. [step_robot: 50 steps]
6. [set_joint_positions: gripper close]
7. [step_robot: 30 steps]
8. [execute_trajectory: arm to place position]
9. [set_joint_positions: gripper open]
10. [reset_robot: both robots]

Response: "✅ Pick and place completed:
  - Object picked at position A
  - Moved through safe trajectory
  - Object placed at position B
  - Robots reset to initial state"
```

---

## 🧪 **Step 7: Testing Your Setup**

### **7.1 Test Server Connection**
```
You: "Check if the robot control server is connected"
Claude: [Lists available tools]
Response: "✅ Robot control server connected with 9 tools available"
```

### **7.2 Test Each Control Mode**
```
You: "Test all robot control modes"
Claude: [Systematically tests position, velocity, torque control]
Response: "✅ All control modes functional:
  - Position control ✓
  - Velocity control ✓
  - Torque control ✓
  - Trajectory execution ✓"
```

### **7.3 Run Full Demo**
```bash
# In terminal:
python demo_robot_control_mcp.py

# Or ask Claude:
You: "Run the complete robot control demonstration"
```

---

## 💡 **Pro Tips for Using Claude Code**

### **1. Natural Language Commands**
- "Move the robot arm up" → Claude interprets and sends appropriate commands
- "Rotate joint 2 slowly" → Claude sets appropriate velocity
- "Grab the object" → Claude coordinates gripper actions

### **2. Complex Sequences**
- "Pick up the red block and place it on the blue platform"
- "Navigate the mobile robot through the obstacle course"
- "Perform a welding motion with the arm"

### **3. Safety and Limits**
- Claude automatically checks joint limits
- Prevents unsafe velocities/torques
- Provides feedback on unreachable positions

### **4. Multi-Robot Coordination**
```
You: "Coordinate two robot arms to lift an object together"
Claude: [Controls both robots in synchronized motion]
```

---

## 🚨 **Troubleshooting**

### **Issue: "Server not found"**
✅ **Fix**: Check configuration file path and restart Claude Desktop

### **Issue: "Tool timeout"**
✅ **Fix**: MuJoCo might be slow to start, increase timeout in config

### **Issue: "Robot not responding"**
✅ **Fix**: Check if simulation is stepping with step_robot tool

### **Issue: "Position not reached"**
✅ **Fix**: Increase simulation steps or check joint limits

---

## 🎓 **Learning Exercises**

### **Exercise 1: Basic Control**
```
Task: Load a robot arm and move each joint individually
Expected: Understanding of joint indexing and position control
```

### **Exercise 2: Trajectory Planning**
```
Task: Create a figure-8 trajectory for the robot arm
Expected: Understanding of trajectory waypoints and timing
```

### **Exercise 3: Force Control**
```
Task: Apply specific torques to achieve a pushing motion
Expected: Understanding of torque control and dynamics
```

### **Exercise 4: Sensor Feedback**
```
Task: Move robot until end-effector reaches specific position
Expected: Understanding of forward kinematics and feedback
```

### **Exercise 5: Multi-Robot**
```
Task: Coordinate arm and gripper for object manipulation
Expected: Understanding of multi-robot coordination
```

---

## 🎉 **Summary**

You now know how to:
1. ✅ Configure MCP server for robot control
2. ✅ Load different robot types (arm, gripper, mobile, humanoid)
3. ✅ Control robots using position, velocity, and torque modes
4. ✅ Execute complex trajectories
5. ✅ Get sensor feedback and robot state
6. ✅ Coordinate multiple robots
7. ✅ Use natural language with Claude Code for robot control

**The key insight**: Claude Code acts as an intelligent interface between you and the MuJoCo simulation. You describe what you want in natural language, and Claude translates that into precise MCP tool calls to control the robots.

---

## 🚀 **Next Steps**

1. **Run the demo**: `python demo_robot_control_mcp.py`
2. **Configure Claude Desktop** with the robot control server
3. **Start experimenting** with natural language robot commands
4. **Build your own** robot control sequences
5. **Integrate with** real robot hardware (future extension)

**Remember**: MCP is the protocol that enables Claude Code to control your robots. You don't write code - you have conversations with Claude, and Claude handles the technical details!
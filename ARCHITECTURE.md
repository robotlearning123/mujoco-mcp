# MuJoCo MCP Architecture

## System Architecture and Design Documentation

This document provides a comprehensive overview of the MuJoCo MCP system architecture, design decisions, and implementation details.

---

## 📐 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Claude Desktop                              │
│                            (MCP Client Application)                      │
└────────────────────────────────┬───────────────────────────────────────┘
                                 │ JSON-RPC 2.0 over stdio
┌────────────────────────────────▼───────────────────────────────────────┐
│                           MCP Server Layer                              │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────────┐    │
│  │   mcp_server.py │  │ Protocol Handler │  │  Tool Registry    │    │
│  │                 │  │                  │  │                   │    │
│  │  - List Tools   │  │  - JSON-RPC 2.0  │  │ - 9 MCP Tools     │    │
│  │  - Call Tools   │  │  - Error Handling │  │ - Natural Lang    │    │
│  │  - Capabilities │  │  - Request Router │  │ - Validation      │    │
│  └─────────────────┘  └──────────────────┘  └───────────────────┘    │
└────────────────────────────────┬───────────────────────────────────────┘
                                 │ Socket IPC (localhost:8888)
┌────────────────────────────────▼───────────────────────────────────────┐
│                          Viewer Server Layer                            │
│  ┌─────────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │
│  │ mujoco_viewer_server│  │  Model Manager  │  │ Connection Pool  │   │
│  │                     │  │                 │  │                  │   │
│  │ - Socket Server     │  │ - Model Loading │  │ - Client Mgmt    │   │
│  │ - Command Handler   │  │ - State Mgmt    │  │ - Load Balancing │   │
│  │ - Thread Pool       │  │ - Memory Mgmt   │  │ - Health Monitor │   │
│  └─────────────────────┘  └─────────────────┘  └─────────────────┘   │
└────────────────────────────────┬───────────────────────────────────────┘
                                 │ MuJoCo Python API
┌────────────────────────────────▼───────────────────────────────────────┐
│                          MuJoCo Physics Engine                          │
│  ┌─────────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │
│  │  launch_passive()   │  │ Physics Solver  │  │   OpenGL Viewer  │   │
│  │                     │  │                 │  │                  │   │
│  │ - GUI Window        │  │ - Dynamics      │  │ - 3D Rendering   │   │
│  │ - User Interaction  │  │ - Collision     │  │ - Camera Control │   │
│  │ - Visualization     │  │ - Integration   │  │ - Debug Info     │   │
│  └─────────────────────┘  └─────────────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Component Architecture

### 1. MCP Server Components

```
src/mujoco_mcp/
├── __init__.py              # Package initialization
├── __main__.py              # Entry point: python -m mujoco_mcp
├── server.py                # Main server coordination
├── mcp_server.py            # MCP protocol implementation
├── viewer_client.py         # Socket client for viewer server
└── version.py               # Version management
```

#### Key Classes and Responsibilities

**`MuJoCoMCPServer` (server.py)**
- Main orchestrator for MCP functionality
- Manages simulation lifecycle
- Coordinates between MCP requests and viewer server

**`MCP Server` (mcp_server.py)**
- Implements MCP protocol specification
- Handles tool registration and invocation
- Manages JSON-RPC communication

**`MuJoCoViewerClient` (viewer_client.py)**
- Socket client for viewer server communication
- Connection management and retry logic
- Command serialization/deserialization

### 2. Advanced Feature Modules

```
src/mujoco_mcp/
├── advanced_controllers.py    # Control algorithms
├── multi_robot_coordinator.py # Multi-robot systems
├── sensor_feedback.py         # Sensor processing
├── rl_integration.py          # RL environments
└── visualization_tools.py     # Real-time plotting
```

#### Module Interactions

```
┌─────────────────────┐
│   User Request      │
└──────────┬──────────┘
           │
┌──────────▼──────────┐     ┌─────────────────┐
│   MCP Tool Call     │────▶│ Controller      │
└──────────┬──────────┘     └─────────────────┘
           │                           │
┌──────────▼──────────┐               │
│  Coordinator        │◀──────────────┘
└──────────┬──────────┘
           │
┌──────────▼──────────┐     ┌─────────────────┐
│  Viewer Server      │────▶│ Visualization   │
└─────────────────────┘     └─────────────────┘
```

### 3. Viewer Server Architecture

```
Enhanced Viewer Server
├── Connection Manager
│   ├── Connection Pool (max 50)
│   ├── Request Rate Limiting
│   └── Stale Connection Cleanup
├── Model Manager
│   ├── Model Loading/Unloading
│   ├── State Management
│   └── Resource Tracking
├── Performance Monitor
│   ├── CPU/Memory Tracking
│   ├── Request Statistics
│   └── Health Metrics
└── Command Processor
    ├── Command Validation
    ├── Error Handling
    └── Response Generation
```

---

## 🏗️ Design Patterns

### 1. External Application Pattern

Following successful MCP implementations (Blender, Figma), we use:
- **Process Separation**: Viewer runs as independent process
- **Socket Communication**: Reliable IPC via TCP
- **Graceful Degradation**: System functions even if viewer crashes

### 2. Command Pattern

All viewer operations use command objects:
```python
{
    "type": "command_name",
    "model_id": "target_model",
    "parameters": {...}
}
```

### 3. Factory Pattern

Component creation uses factories:
```python
# Controllers
controller = create_arm_controller("franka_panda")

# Environments
env = create_reaching_env("franka_panda")

# Sensor suites
sensors = create_robot_sensor_suite("franka_panda", n_joints=7)
```

### 4. Observer Pattern

Real-time monitoring uses observers:
```python
monitor = RobotStateMonitor(viewer_client)
monitor.start_monitoring("model_id")
# Automatically observes state changes
```

---

## 🔄 Data Flow

### 1. MCP Request Flow

```
1. User Input (Claude Desktop)
   └─> Natural language: "Create a pendulum simulation"

2. MCP Server Processing
   └─> Tool: create_scene
   └─> Arguments: {"scene_type": "pendulum"}

3. Scene Generation
   └─> Generate MuJoCo XML
   └─> Create model configuration

4. Viewer Server Command
   └─> {"type": "load_model", "model_xml": "..."}

5. MuJoCo Execution
   └─> Parse XML
   └─> Initialize physics
   └─> Launch viewer window

6. Response Chain
   └─> Viewer -> Socket -> MCP -> Claude Desktop
   └─> "✅ Created pendulum scene successfully!"
```

### 2. Simulation Loop

```
┌─────────────┐
│   Start     │
└──────┬──────┘
       │
┌──────▼──────┐     ┌─────────────┐
│ Load Model  │────▶│ Initialize  │
└──────┬──────┘     │   Physics   │
       │            └─────────────┘
┌──────▼──────┐
│   Step      │◀────┐
│ Simulation  │     │
└──────┬──────┘     │
       │            │
┌──────▼──────┐     │
│   Update    │     │
│   Viewer    │     │
└──────┬──────┘     │
       │            │
┌──────▼──────┐     │
│Check Control│─────┘
└─────────────┘
```

---

## 💾 State Management

### 1. Model State

Each model maintains:
```python
{
    "model_id": "unique_identifier",
    "model_type": "pendulum|arm|quadruped|etc",
    "created_time": timestamp,
    "physics_state": {
        "qpos": [...],  # Joint positions
        "qvel": [...],  # Joint velocities
        "time": 0.0     # Simulation time
    },
    "viewer_state": {
        "running": true,
        "camera_pos": [...],
        "render_options": {...}
    }
}
```

### 2. Connection State

Connection manager tracks:
```python
{
    "connection_id": "uuid",
    "created_time": timestamp,
    "last_activity": timestamp,
    "requests_handled": count,
    "bytes_transferred": {...},
    "errors": count
}
```

### 3. Task State (Multi-Robot)

Task allocation maintains:
```python
{
    "task_id": "unique_id",
    "task_type": "cooperative|formation|etc",
    "robots": ["robot1", "robot2"],
    "status": "pending|active|completed",
    "progress": 0.0-1.0
}
```

---

## 🔒 Concurrency Model

### 1. Thread Architecture

```
Main Thread
├── Socket Accept Loop
└── Shutdown Handler

Client Threads (1 per connection)
├── Command Processing
├── Response Generation
└── Error Handling

Model Threads (1 per model)
├── Physics Simulation Loop
├── Viewer Sync
└── State Updates

Monitor Thread
├── Performance Tracking
├── Resource Cleanup
└── Health Checks
```

### 2. Synchronization

- **Viewer Lock**: Thread-safe MuJoCo viewer access
- **Model Lock**: Protects model state modifications
- **Connection Lock**: Thread-safe connection management

---

## 🚀 Performance Optimizations

### 1. Connection Pooling
- Pre-allocated thread pool
- Connection reuse
- Automatic cleanup of idle connections

### 2. Memory Management
- Model reference counting
- Automatic garbage collection
- Resource limits enforcement

### 3. Communication Efficiency
- Binary protocol for large data
- Command batching support
- Async response handling

---

## 🛡️ Error Handling Strategy

### 1. Graceful Degradation
```
Viewer Crash -> Detect -> Cleanup -> Report -> Continue
Model Error -> Isolate -> Remove -> Report -> Continue
Network Error -> Retry -> Timeout -> Report -> Fallback
```

### 2. Error Categories
- **Recoverable**: Retry with exponential backoff
- **Model-Specific**: Isolate to single model
- **System-Wide**: Graceful shutdown sequence

### 3. User Feedback
- Clear error messages
- Suggested remediation
- Fallback options

---

## 🔌 Extension Points

### 1. Adding New MCP Tools

```python
@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    return existing_tools + [
        types.Tool(
            name="new_tool",
            description="Tool description",
            inputSchema={...}
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict):
    if name == "new_tool":
        return handle_new_tool(arguments)
```

### 2. Adding Robot Types

```python
# In robot_configs
"new_robot": {
    "joints": 6,
    "type": "manipulator",
    "home_position": [0, 0, 0, 0, 0, 0],
    "path": "manufacturer/model/scene.xml"
}
```

### 3. Custom Controllers

```python
class CustomController(RobotController):
    def custom_control_law(self, state, target):
        # Implementation
        return control_output
```

---

## 📊 Metrics and Monitoring

### 1. Performance Metrics
- Request latency (p50, p95, p99)
- Throughput (requests/second)
- Resource utilization (CPU, memory)
- Physics step time

### 2. Reliability Metrics
- Uptime percentage
- Error rates by type
- Recovery success rate
- Connection stability

### 3. Usage Metrics
- Active models count
- Tool usage frequency
- Feature adoption
- User patterns

---

## 🔮 Future Architecture Considerations

### 1. Distributed Simulation
- Multi-machine coordination
- Cloud deployment support
- Load balancing strategies

### 2. GPU Acceleration
- CUDA integration for physics
- Parallel simulation support
- ML model inference

### 3. Real Robot Integration
- Hardware abstraction layer
- Sensor data ingestion
- Control output mapping

---

## 📚 Architecture Principles

1. **Separation of Concerns**: Each component has clear responsibilities
2. **Loose Coupling**: Components interact through well-defined interfaces
3. **High Cohesion**: Related functionality grouped together
4. **Extensibility**: Easy to add new features without breaking existing ones
5. **Reliability**: Graceful handling of failures at all levels
6. **Performance**: Optimized for real-time simulation requirements
7. **Maintainability**: Clear code structure and comprehensive documentation

This architecture enables MuJoCo MCP to serve as a robust, scalable platform for robotics simulation and control research.
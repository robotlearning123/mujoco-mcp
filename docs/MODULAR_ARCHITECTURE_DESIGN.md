# MuJoCo MCP - Modular Architecture Design

## Core Principle

**MuJoCo MCP = Thin MCP↔MuJoCo Bridge**

Everything else is a pluggable module that extends functionality without touching the core.

## Architecture Layers

```
┌─────────────────────────────────────────────┐
│            Application Layer                 │
│  (Natural Language, Menagerie, MPC, etc.)   │
├─────────────────────────────────────────────┤
│             Module Layer                     │
│  (Robot Control, Vision, Planning, etc.)    │
├─────────────────────────────────────────────┤
│            Core MCP Layer                    │ ← THIS IS OUR CORE
│    (MCP Protocol ↔ MuJoCo Bridge)          │
├─────────────────────────────────────────────┤
│           MuJoCo Engine                      │
│    (Physics Simulation, Viewer)             │
└─────────────────────────────────────────────┘
```

## Core Layer (Minimal & Stable)

### Purpose
Pure MCP↔MuJoCo communication bridge. Nothing else.

### Core Interface (`core/mcp_bridge.py`)
```python
class MuJoCoMCPBridge:
    """Minimal core that NEVER changes"""
    
    def __init__(self):
        self.viewer_client = ViewerClient()
        self.modules = {}  # Plugin system
    
    # Core MCP Tools (minimal set)
    async def load_model(self, xml: str) -> Dict:
        """Load any MuJoCo XML model"""
        return self.viewer_client.load_model(xml)
    
    async def step_simulation(self, model_id: str, steps: int) -> Dict:
        """Step physics simulation"""
        return self.viewer_client.step(model_id, steps)
    
    async def get_state(self, model_id: str) -> Dict:
        """Get simulation state"""
        return self.viewer_client.get_state(model_id)
    
    async def set_control(self, model_id: str, control: List[float]) -> Dict:
        """Set control inputs"""
        return self.viewer_client.set_control(model_id, control)
    
    # Module System
    def register_module(self, name: str, module: 'BaseModule'):
        """Register a feature module"""
        self.modules[name] = module
        module.attach(self)
```

### Core Principles
1. **No Business Logic**: Just protocol translation
2. **No Model Knowledge**: Doesn't know about robots
3. **No Dependencies**: Only MuJoCo and MCP
4. **Never Changes**: API frozen after v1.0

## Module System

### Base Module Interface
```python
class BaseModule(ABC):
    """All modules inherit from this"""
    
    @abstractmethod
    def get_tools(self) -> List[MCPTool]:
        """Return MCP tools this module provides"""
        pass
    
    @abstractmethod
    def attach(self, bridge: MuJoCoMCPBridge):
        """Attach to core bridge"""
        pass
```

### Module Examples

#### 1. Menagerie Module (`modules/menagerie.py`)
```python
class MenagerieModule(BaseModule):
    """Adds Menagerie model loading"""
    
    def __init__(self, menagerie_path: str):
        self.models = self._scan_models(menagerie_path)
    
    def get_tools(self):
        return [
            MCPTool(
                name="load_menagerie_model",
                handler=self.load_menagerie_model
            ),
            MCPTool(
                name="list_menagerie_models",
                handler=self.list_models
            )
        ]
    
    async def load_menagerie_model(self, name: str):
        xml = self._load_model_xml(name)
        # Use core bridge to load
        return await self.bridge.load_model(xml)
```

#### 2. Natural Language Module (`modules/nlp.py`)
```python
class NaturalLanguageModule(BaseModule):
    """Natural language commands"""
    
    def get_tools(self):
        return [
            MCPTool(
                name="execute_command",
                handler=self.execute_command
            )
        ]
    
    async def execute_command(self, command: str):
        # Parse command
        # Call appropriate core functions
        # Can use other modules if available
        pass
```

#### 3. Robot Control Module (`modules/robot_control.py`)
```python
class RobotControlModule(BaseModule):
    """Advanced robot control"""
    
    def get_tools(self):
        return [
            MCPTool(
                name="inverse_kinematics",
                handler=self.compute_ik
            ),
            MCPTool(
                name="plan_trajectory", 
                handler=self.plan_trajectory
            )
        ]
```

#### 4. MPC Module (`modules/mpc.py`)
```python
class MPCModule(BaseModule):
    """Model Predictive Control integration"""
    
    def __init__(self):
        # Only loaded if needed
        self.mpc = None
    
    def get_tools(self):
        return [
            MCPTool(
                name="create_mpc_controller",
                handler=self.create_controller
            )
        ]
    
    async def create_controller(self, task_spec: Dict):
        if not self.mpc:
            # Lazy load MuJoCo MPC
            from mujoco_mpc import MPC
            self.mpc = MPC()
        # Use core bridge for physics
        pass
```

## Configuration System

### Modular Configuration (`config.yaml`)
```yaml
# Core configuration (minimal)
core:
  viewer_host: localhost
  viewer_port: 8888
  
# Modules (optional, can be added/removed)
modules:
  menagerie:
    enabled: true
    path: ~/.mujoco/menagerie
    
  natural_language:
    enabled: true
    
  robot_control:
    enabled: false  # Not ready yet
    
  mpc:
    enabled: false  # Future feature
    
  playground:
    enabled: false  # Community contribution
```

### Dynamic Module Loading
```python
class MuJoCoMCPServer:
    def __init__(self, config_path: str):
        # Load core
        self.bridge = MuJoCoMCPBridge()
        
        # Load modules based on config
        config = load_config(config_path)
        
        if config['modules']['menagerie']['enabled']:
            module = MenagerieModule(config['modules']['menagerie']['path'])
            self.bridge.register_module('menagerie', module)
            
        if config['modules']['natural_language']['enabled']:
            module = NaturalLanguageModule()
            self.bridge.register_module('nlp', module)
        
        # Modules can be added at runtime!
```

## Benefits of Modular Design

### 1. **Independent Development**
```bash
# Different developers can work on different modules
dev1: git checkout -b feature/menagerie-module
dev2: git checkout -b feature/mpc-module
# No conflicts!
```

### 2. **Easy Testing**
```python
def test_menagerie_module():
    # Test module in isolation
    module = MenagerieModule("/test/path")
    mock_bridge = Mock()
    module.attach(mock_bridge)
    # Test without full system
```

### 3. **Gradual Enhancement**
```python
# v0.8.0 - Just add Menagerie module
modules = ['menagerie']

# v0.9.0 - Add robustness module  
modules = ['menagerie', 'robustness']

# v1.0.0 - Add control module
modules = ['menagerie', 'robustness', 'control']
```

### 4. **User Choice**
```bash
# Minimal installation
pip install mujoco-mcp

# With specific features
pip install mujoco-mcp[menagerie,nlp]

# Everything
pip install mujoco-mcp[all]
```

### 5. **Community Friendly**
```python
# Easy for community to add modules
class CustomModule(BaseModule):
    """Community-contributed module"""
    pass

# Just drop in and register!
```

## Module Communication

### Via Core Bridge
```python
class ModuleA(BaseModule):
    async def do_something(self):
        # Use core functions
        state = await self.bridge.get_state(model_id)
        
        # Check if another module exists
        if 'menagerie' in self.bridge.modules:
            # Use it if available
            models = self.bridge.modules['menagerie'].list_models()
```

### Module Dependencies
```python
class MPCModule(BaseModule):
    def __init__(self):
        self.dependencies = ['robot_control']  # Needs IK
    
    def attach(self, bridge):
        # Check dependencies
        for dep in self.dependencies:
            if dep not in bridge.modules:
                raise ModuleError(f"MPC requires {dep} module")
```

## File Structure

```
mujoco-mcp/
├── src/mujoco_mcp/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── mcp_bridge.py      # Core bridge (minimal)
│   │   └── viewer_client.py   # Socket communication
│   │
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── base.py           # BaseModule interface
│   │   ├── menagerie.py      # Menagerie integration
│   │   ├── nlp.py            # Natural language
│   │   ├── robot_control.py  # Advanced control
│   │   └── mpc.py            # MPC integration
│   │
│   └── server.py             # Main entry point
│
├── config/
│   ├── default.yaml          # Default configuration
│   └── examples/             # Example configs
│
└── tests/
    ├── core/                 # Core tests
    └── modules/              # Module tests
```

## Development Workflow

### Adding a New Module
1. Create `modules/my_feature.py`
2. Inherit from `BaseModule`
3. Implement required methods
4. Add to config
5. Done! No core changes needed

### Example: Adding Vision Module
```python
# modules/vision.py
class VisionModule(BaseModule):
    """Add camera and vision tools"""
    
    def get_tools(self):
        return [
            MCPTool("add_camera", self.add_camera),
            MCPTool("get_image", self.get_image),
            MCPTool("detect_objects", self.detect_objects)
        ]
```

## Version Strategy

### Core Versions (Rare)
- v1.0.0 - Core API frozen
- v1.0.1 - Bug fixes only
- v2.0.0 - Major core change (avoid!)

### Module Versions (Frequent)
- menagerie-v1.0.0
- nlp-v1.2.0
- mpc-v0.1.0 (experimental)

Users can mix and match module versions!

## Conclusion

This modular architecture ensures:
1. **Core Stability**: MCP↔MuJoCo bridge never changes
2. **Feature Freedom**: Add anything without breaking core
3. **Independent Development**: Teams work separately
4. **User Choice**: Install only what you need
5. **Future Proof**: New features are just new modules

The core remains a thin, stable communication layer while all innovation happens in modules.
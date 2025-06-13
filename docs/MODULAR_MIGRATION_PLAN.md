# Modular Architecture Migration Plan

## Current State (v0.7.1)

Currently, everything is mixed in `remote_server.py`:
- Core MCP communication
- Scene generation (pendulum, cart-pole)
- Natural language parsing
- Business logic

## Target State (v0.8.0)

Clean separation:
- **Core**: Pure MCP↔MuJoCo bridge
- **Modules**: Everything else as plugins

## Migration Strategy

### Phase 1: Extract Core (1 week)

#### 1. Create Core Bridge
```python
# src/mujoco_mcp/core/bridge.py
class MuJoCoMCPBridge:
    """Minimal core - just 4 essential tools"""
    
    def __init__(self, viewer_client):
        self.viewer = viewer_client
        self._tools = {}
        self._register_core_tools()
    
    def _register_core_tools(self):
        # Only truly essential tools
        self._tools.update({
            "load_model": self._create_tool(
                "Load a MuJoCo XML model",
                ["xml"],
                self._load_model
            ),
            "step_simulation": self._create_tool(
                "Advance simulation",
                ["model_id", "steps"],
                self._step_simulation
            ),
            "get_state": self._create_tool(
                "Get simulation state",
                ["model_id"],
                self._get_state
            ),
            "set_control": self._create_tool(
                "Set control inputs",
                ["model_id", "control"],
                self._set_control
            )
        })
```

#### 2. Extract Current Features to Modules

**Built-in Scenes Module**:
```python
# src/mujoco_mcp/modules/builtin_scenes.py
class BuiltinScenesModule(BaseModule):
    """Current pendulum, cart-pole, etc."""
    
    def get_tools(self):
        return [{
            "name": "create_scene",
            "description": "Create built-in scenes",
            "parameters": ["scene_type"],
            "handler": self.create_scene
        }]
    
    def create_scene(self, scene_type: str):
        xml = self._generate_xml(scene_type)
        return self.bridge.load_model(xml)
```

**Natural Language Module**:
```python
# src/mujoco_mcp/modules/natural_language.py
class NaturalLanguageModule(BaseModule):
    """Current execute_command functionality"""
    
    def get_tools(self):
        return [{
            "name": "execute_command",
            "description": "Natural language control",
            "parameters": ["command"],
            "handler": self.execute_command
        }]
```

### Phase 2: Implement Module System (3 days)

#### 1. Module Registry
```python
# src/mujoco_mcp/modules/registry.py
class ModuleRegistry:
    def __init__(self):
        self.modules = {}
        self.tools = {}  # Flat list of all tools
    
    def register(self, name: str, module: BaseModule):
        self.modules[name] = module
        
        # Collect tools from module
        for tool in module.get_tools():
            self.tools[tool['name']] = {
                'module': name,
                'tool': tool
            }
    
    def get_all_tools(self):
        """For MCP list_tools"""
        return list(self.tools.values())
```

#### 2. Update Main Server
```python
# src/mujoco_mcp/server.py
class MuJoCoMCPServer:
    def __init__(self, config=None):
        # Core bridge
        self.bridge = MuJoCoMCPBridge()
        
        # Module system
        self.registry = ModuleRegistry()
        
        # Load default modules
        self._load_default_modules()
        
        # Load config modules
        if config:
            self._load_config_modules(config)
    
    def _load_default_modules(self):
        """Always-on modules"""
        # Keep current functionality
        self.registry.register(
            'builtin_scenes',
            BuiltinScenesModule()
        )
        self.registry.register(
            'natural_language',
            NaturalLanguageModule()
        )
```

### Phase 3: Add Menagerie Module (1 week)

```python
# src/mujoco_mcp/modules/menagerie.py
class MenagerieModule(BaseModule):
    def __init__(self, path=None):
        self.path = path or self._find_menagerie()
        self.models = self._scan_models()
    
    def get_tools(self):
        return [
            {
                "name": "load_menagerie_model",
                "description": "Load robot from Menagerie",
                "parameters": ["name"],
                "handler": self.load_model
            },
            {
                "name": "list_menagerie_models",
                "description": "List available robots",
                "parameters": ["category"],
                "handler": self.list_models
            }
        ]
```

## Backward Compatibility

### Keep Current API Working
```python
# In server.py
@server.call_tool()
async def handle_call_tool(name: str, arguments: dict):
    # Check core tools first
    if name in self.bridge._tools:
        return self.bridge._tools[name]['handler'](**arguments)
    
    # Check module tools
    if name in self.registry.tools:
        tool_info = self.registry.tools[name]
        module = self.registry.modules[tool_info['module']]
        return tool_info['tool']['handler'](**arguments)
    
    # Not found
    raise ValueError(f"Unknown tool: {name}")
```

### Gradual Migration
1. **v0.7.2**: Refactor code into modules (internal only)
2. **v0.8.0**: Add Menagerie module
3. **v0.9.0**: Public module API
4. **v1.0.0**: Freeze core API

## Testing Strategy

### Core Tests
```python
def test_core_bridge():
    """Test only core functionality"""
    bridge = MuJoCoMCPBridge(mock_viewer)
    
    # Test 4 core tools
    assert 'load_model' in bridge._tools
    assert 'step_simulation' in bridge._tools
    assert 'get_state' in bridge._tools
    assert 'set_control' in bridge._tools
    
    # No other tools in core!
    assert len(bridge._tools) == 4
```

### Module Tests
```python
def test_module_isolation():
    """Modules work independently"""
    module = MenagerieModule("/test/path")
    
    # Can test without full system
    tools = module.get_tools()
    assert len(tools) == 2
```

## Benefits After Migration

### 1. Clean Codebase
```
Before: remote_server.py (1000+ lines)
After:  
  - core/bridge.py (200 lines)
  - modules/builtin_scenes.py (300 lines)
  - modules/natural_language.py (200 lines)
  - modules/menagerie.py (150 lines)
```

### 2. Easy Feature Addition
```python
# Add new module without touching core
class VisionModule(BaseModule):
    def get_tools(self):
        return [{"name": "capture_image", ...}]

# Just register it!
server.registry.register('vision', VisionModule())
```

### 3. Community Friendly
```python
# Community can create modules
pip install mujoco-mcp-vision  # By community
pip install mujoco-mcp-ros     # By community

# Auto-discovery
from mujoco_mcp.modules import discover_modules
modules = discover_modules()  # Finds installed modules
```

## Success Criteria

1. **No Breaking Changes**: Existing code still works
2. **Clean Separation**: Core has no business logic
3. **Easy Testing**: Each module tests independently
4. **Documentation**: Clear module development guide

## Timeline

- **Week 1**: Core extraction, module system
- **Week 2**: Menagerie module, testing
- **Week 3**: Documentation, release v0.8.0

This migration sets us up for sustainable growth while keeping the core minimal and stable.
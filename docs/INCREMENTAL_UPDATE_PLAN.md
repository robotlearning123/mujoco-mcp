# MuJoCo MCP Incremental Update Plan

## Core Principle
**"If it works, don't break it. Add small features, test thoroughly, improve gradually."**

## Current State (v0.7.1)
- ✅ Working MCP server
- ✅ Basic physics simulation
- ✅ Natural language commands
- ✅ Stable architecture

**DO NOT**: Major refactoring, breaking changes, architectural overhauls  
**DO**: Small additions, backward compatibility, thorough testing

## Incremental Update Strategy

### v0.7.2 - Natural Language Improvements (1-2 days)
**Size**: Tiny  
**Risk**: Very Low  
**Changes**: Only in `execute_command` method

```python
# Just add more command patterns to existing method
def _handle_execute_command(self, command: str, context: Optional[Dict[str, Any]] = None):
    # Existing code stays...
    
    # Add new patterns:
    if "pause" in command_lower:
        # Pause simulation
    if "continue" in command_lower or "resume" in command_lower:
        # Resume simulation
    if "speed" in command_lower:
        # Adjust simulation speed
```

**Testing**:
- Run existing tests (must all pass)
- Add 5-10 new command tests
- Manual testing with Claude Desktop

### v0.7.3 - Model Info Enhancement (1-2 days)
**Size**: Tiny  
**Risk**: Very Low  
**Changes**: Add info to existing responses

```python
# In _handle_create_scene, just add more info
result = self._handle_create_scene("pendulum", context)
# Add extra info to response
result["model_info"] = {
    "total_mass": self._calculate_total_mass(model_id),
    "degrees_of_freedom": model.nq,
    "actuators": model.nu
}
```

### v0.7.4 - Better Error Messages (1 day)
**Size**: Tiny  
**Risk**: Zero  
**Changes**: Only error strings

```python
# Instead of:
return {"error": "Failed to load model"}

# Return:
return {
    "error": "Failed to load model", 
    "details": "MuJoCo viewer not connected. Please start viewer first.",
    "suggestion": "Run: python mujoco_viewer_server.py"
}
```

### v0.8.0 - Simple Menagerie Support (1 week)
**Size**: Small  
**Risk**: Low  
**Strategy**: Add without changing existing code

```python
# In _handle_create_scene, just add a check:
def _handle_create_scene(self, scene_type: str, context: Dict[str, Any]):
    # Try existing scenes first (no change)
    if scene_type in ["pendulum", "double_pendulum", "cart_pole"]:
        # Existing code unchanged
        return self._create_builtin_scene(scene_type)
    
    # NEW: Check if it's a Menagerie model
    menagerie_xml = self._try_load_menagerie(scene_type)
    if menagerie_xml:
        return self._load_xml_model(menagerie_xml)
    
    # Fallback
    return {"error": f"Unknown scene type: {scene_type}"}

def _try_load_menagerie(self, model_name: str) -> Optional[str]:
    """Try to load from Menagerie if available"""
    # Simple implementation - no complex registry
    menagerie_path = os.getenv("MENAGERIE_PATH")
    if not menagerie_path:
        return None
    
    # Look for model
    possible_paths = [
        f"{menagerie_path}/{model_name}/{model_name}.xml",
        f"{menagerie_path}/*/{model_name}.xml"
    ]
    # Return XML if found
```

**Testing**:
- All existing tests must pass
- Test with 2-3 Menagerie models
- Test without Menagerie (graceful fallback)

### v0.8.1 - List Available Models (2 days)
**Size**: Tiny  
**Risk**: Very Low  
**Changes**: Add one new tool

```python
# Just add to existing tools
self._tools["list_available_models"] = {
    "name": "list_available_models",
    "description": "List all available models",
    "handler": self._handle_list_models
}

def _handle_list_models(self):
    models = {
        "builtin": ["pendulum", "double_pendulum", "cart_pole"],
        "menagerie": []  # Scan if available
    }
    return {"models": models}
```

### v0.8.2 - Save/Load State (3 days)
**Size**: Small  
**Risk**: Low  
**Changes**: Add two new tools

```python
# New tools - don't touch existing ones
self._tools["save_state"] = {
    "name": "save_state",
    "handler": self._handle_save_state
}

self._tools["load_state"] = {
    "name": "load_state", 
    "handler": self._handle_load_state
}
```

## Testing Strategy for Each Update

### 1. Regression Testing
```bash
# Run ALL existing tests before release
python -m pytest tests/
# Must have 100% pass rate
```

### 2. New Feature Testing
```bash
# Test only the new feature
python tests/test_v0_x_x_new_feature.py
```

### 3. Integration Testing
```bash
# Test with Claude Desktop
# Test with MCP Inspector
npx @modelcontextprotocol/inspector python mcp_server_remote.py
```

### 4. Manual Testing Checklist
- [ ] Start viewer server
- [ ] Connect from Claude Desktop
- [ ] Test all existing commands still work
- [ ] Test new feature
- [ ] Check error cases
- [ ] Verify performance unchanged

## What We DON'T Do

### ❌ No Major Refactoring
- Don't reorganize file structure
- Don't change core architecture
- Don't rename existing methods
- Don't change existing APIs

### ❌ No Breaking Changes
- All existing commands must work
- All existing tools must work
- All existing tests must pass
- Configuration compatibility maintained

### ❌ No Complex Features Yet
- No multi-viewer support
- No advanced physics features
- No complex architectures
- No major dependencies

## Release Process

### For Each Version:
1. **Branch from main**
   ```bash
   git checkout -b feature/v0.x.x-description
   ```

2. **Make minimal changes**
   - Change as few files as possible
   - Add, don't modify when possible
   - Keep diff small

3. **Test thoroughly**
   ```bash
   # All tests must pass
   pytest tests/
   # Plus new feature tests
   pytest tests/test_new_feature.py
   ```

4. **Update version**
   ```python
   # src/mujoco_mcp/version.py
   __version__ = "0.x.x"
   # pyproject.toml
   version = "0.x.x"
   ```

5. **Update changelog**
   - List specific changes
   - Note any new dependencies
   - Highlight new features

6. **Create PR**
   - Small, focused PR
   - Clear description
   - Test results included

## Success Metrics

### For Each Release:
- ✅ All existing tests pass
- ✅ No breaking changes
- ✅ New feature works
- ✅ Performance unchanged
- ✅ Claude Desktop compatible
- ✅ Easy to rollback

## Long-term Vision (Still Applies)

We still aim for:
- Modular architecture
- Menagerie integration  
- Advanced features

But we get there through:
- Small increments
- Backward compatibility
- Thorough testing
- User feedback

## Example: Adding a Feature

### Good Approach ✅
```python
# v0.7.5 - Add simulation speed control
# Just add to existing execute_command:
if "slow" in command_lower:
    self.simulation_speed = 0.5
elif "fast" in command_lower:
    self.simulation_speed = 2.0
# That's it! Small, safe, testable
```

### Bad Approach ❌
```python
# Don't do this:
# "Let me refactor the entire command system while adding speed control"
class CommandParser:  # New abstraction
    def parse(self, command):  # New API
        # 500 lines of refactoring...
```

## Conclusion

**Small steps, frequent releases, always working.**

Each version should:
1. Add one small feature
2. Take 1-7 days max
3. Be fully tested
4. Maintain compatibility
5. Improve user experience

This approach ensures we maintain a stable, working system while gradually adding features users want.
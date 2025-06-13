# MuJoCo MCP Development Standards

## Core Principle

**Always check official documentation before implementation**

## Documentation Priority Order

### 1. MCP Documentation (Primary)
- **Specification**: https://modelcontextprotocol.io/specification
- **Concepts**: https://modelcontextprotocol.io/docs/concepts/
- **Implementation**: https://modelcontextprotocol.io/docs/server/
- **Check**: Before EVERY feature implementation

### 2. MuJoCo Documentation
- **Official Docs**: https://mujoco.readthedocs.io/
- **Python API**: https://mujoco.readthedocs.io/en/stable/python.html
- **GitHub**: https://github.com/google-deepmind/mujoco
- **Check**: For any physics/simulation features

### 3. Integration Documentation
- **Menagerie**: https://github.com/google-deepmind/mujoco_menagerie
- **Playground**: https://github.com/google-deepmind/mujoco_playground
- **MPC**: https://github.com/google-deepmind/mujoco_mpc
- **Check**: Before integrating these tools

## Development Workflow

### Before Starting ANY Feature:

1. **Check Official Docs**
```bash
# Example workflow
1. Read MCP specification for the feature area
2. Check if MuJoCo has relevant APIs
3. Look for existing examples
4. Verify version compatibility
```

2. **Validate Approach**
```python
# Wrong: Assume based on old knowledge
def my_feature():
    # Implementation based on memory
    
# Right: Check docs first
# After reading https://modelcontextprotocol.io/docs/concepts/tools
def my_feature():
    # Implementation following current spec
```

3. **Test Compliance**
```bash
# Always test with official tools
npx @modelcontextprotocol/inspector test
```

## Code Standards

### 1. MCP Protocol Compliance

**Every Tool Must Have:**
```python
{
    "name": "tool_name",              # Unique within server
    "description": "Clear description", # Human-readable
    "inputSchema": {                   # JSON Schema
        "type": "object",
        "properties": {
            "param": {
                "type": "string",
                "description": "Parameter description"
            }
        },
        "required": ["param"]
    },
    "annotations": {                   # Optional hints
        "readOnlyHint": false,
        "idempotentHint": true
    }
}
```

### 2. Error Handling

**Follow MCP Error Codes:**
```python
# Standard JSON-RPC 2.0 errors
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# Application errors (-32000 to -32099)
MODEL_NOT_FOUND = -32001
SIMULATION_ERROR = -32002
```

### 3. Message Format

**Always JSON-RPC 2.0:**
```json
{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
        "name": "load_model",
        "arguments": {"xml": "..."}
    },
    "id": "unique-id"
}
```

## Testing Standards

### 1. MCP Compliance Tests
```python
def test_mcp_compliance():
    # Test all required methods
    assert server.handles("initialize")
    assert server.handles("tools/list")
    assert server.handles("tools/call")
    
    # Test message format
    response = server.call("tools/list")
    assert response["jsonrpc"] == "2.0"
```

### 2. Integration Tests
```python
def test_with_mcp_inspector():
    # Must pass MCP Inspector
    result = run_mcp_inspector(server)
    assert result.passed
```

## Documentation Standards

### 1. Always Include References
```python
def load_menagerie_model(self, name: str):
    """
    Load a model from MuJoCo Menagerie.
    
    Based on:
    - https://github.com/google-deepmind/mujoco_menagerie#usage
    - MCP Tool spec: https://modelcontextprotocol.io/docs/concepts/tools
    
    Args:
        name: Model name as defined in Menagerie
    """
```

### 2. Version Documentation
```markdown
## Compatibility
- MCP Specification: 2024-11-05
- MuJoCo Version: >=3.0.0
- Python: >=3.8
- Last verified: 2025-01-13
```

## Review Checklist

### Before Every PR:
- [ ] Checked latest MCP specification
- [ ] Verified MuJoCo API usage
- [ ] Tested with MCP Inspector
- [ ] Updated compatibility docs
- [ ] Added reference links
- [ ] Validated error handling
- [ ] Ensured backward compatibility

### For New Features:
- [ ] Read ALL relevant official docs
- [ ] Checked for breaking changes
- [ ] Verified examples still work
- [ ] Updated version matrix
- [ ] Added integration tests

## Common Pitfalls

### 1. Using Outdated Patterns
```python
# Wrong: Old MCP pattern
async def my_handler(params):
    return {"result": "data"}

# Right: Current MCP pattern  
@server.setRequestHandler(CallToolRequestSchema)
async def handle_tool_call(request):
    return CallToolResult(...)
```

### 2. Assuming API Stability
```python
# Wrong: Hardcode based on old docs
MUJOCO_VERSION = "2.3.0"

# Right: Check at runtime
import mujoco
if version.parse(mujoco.__version__) < version.parse("3.0.0"):
    raise CompatibilityError("Requires MuJoCo 3.0.0+")
```

### 3. Ignoring Protocol Changes
```python
# Always check changelog
# https://modelcontextprotocol.io/changelog
# https://github.com/google-deepmind/mujoco/releases
```

## Continuous Monitoring

### Weekly Checks:
1. MCP specification updates
2. MuJoCo releases
3. Menagerie model additions
4. Community feedback

### Monthly Reviews:
1. Full documentation review
2. Compatibility testing
3. Performance benchmarks
4. Security updates

## Conclusion

**Golden Rule**: When in doubt, check the official documentation. Never assume based on past knowledge or patterns from other projects.
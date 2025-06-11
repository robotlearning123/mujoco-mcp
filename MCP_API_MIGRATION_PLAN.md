# MCP API Migration Plan

## Current State (v0.3.1)

Our current implementation (`simple_server.py`) is a custom MCP-like server that doesn't use the FastMCP framework. While it works for testing and development, it's not following the standard MCP protocol implementation.

## Issues with Current Implementation

1. **Not using FastMCP**: We're implementing our own tool registration and calling mechanism instead of using the standard FastMCP decorators and patterns.

2. **Missing MCP Features**:
   - No proper resource URIs (e.g., `simulation://list`)
   - No streaming support
   - No proper error handling according to MCP spec
   - No automatic schema generation

3. **Non-standard Tool Registration**: Our `_tools` dictionary approach doesn't follow MCP conventions.

## Proper MCP Implementation Pattern

According to the MCP specification and FastMCP documentation:

```python
from mcp.server import FastMCP

# Create server instance
mcp = FastMCP("MuJoCo Control Server")

# Resources use URI patterns
@mcp.resource("simulation://list")
def list_simulations() -> str:
    """List all active simulations."""
    return "..."

# Tools use decorators
@mcp.tool()
def load_model(model_string: str, name: Optional[str] = None) -> Dict[str, Any]:
    """Load a MuJoCo model from XML string."""
    return {"model_id": "..."}
```

## Migration Strategy

### Phase 1: Dual Implementation (v0.4.0)
- Keep `simple_server.py` for backward compatibility
- Create `fastmcp_server.py` using proper FastMCP implementation
- Migrate tools one by one to FastMCP decorators
- Add tests for both implementations

### Phase 2: Feature Parity (v0.5.0)
- Ensure all features work in FastMCP implementation
- Add MCP-specific features (streaming, resources)
- Update documentation

### Phase 3: Deprecation (v0.6.0)
- Mark `simple_server.py` as deprecated
- Update all examples to use FastMCP server
- Add migration guide

### Phase 4: Removal (v1.0.0)
- Remove `simple_server.py`
- Clean up codebase
- Full MCP compliance

## Benefits of Migration

1. **Standards Compliance**: Follow MCP protocol specification
2. **Better Tooling**: Automatic schema generation, validation
3. **Streaming Support**: For real-time simulation data
4. **Resource URIs**: Proper resource addressing
5. **Error Handling**: Standardized error responses
6. **Interoperability**: Works with any MCP client

## Example: Migrating load_model Tool

Current implementation:
```python
def _handle_load_model(self, model_string: str, name: Optional[str] = None):
    # ... implementation
```

FastMCP implementation:
```python
@mcp.tool()
def load_model(model_string: str, name: Optional[str] = None) -> Dict[str, Any]:
    """Load a MuJoCo model from XML string.
    
    Args:
        model_string: XML string containing the MuJoCo model
        name: Optional name for the loaded model
        
    Returns:
        Dictionary with model_id and model information
    """
    # ... implementation
```

## Next Steps

1. Review FastMCP documentation thoroughly
2. Create a prototype FastMCP server with basic tools
3. Compare performance and functionality
4. Plan migration timeline

This migration will ensure our MuJoCo MCP server is fully compliant with the Model Context Protocol specification and can interoperate with any MCP-compatible client.
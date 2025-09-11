# MuJoCo MCP Multi-Client Testing Guide

Complete guide for testing MuJoCo MCP server across different AI clients and platforms.

## Overview

This testing framework validates MuJoCo MCP functionality across:

- **Clients**: Claude Code, Cursor, Claude Desktop, OpenAI ChatGPT
- **Platforms**: macOS, Linux, Windows (WSL2)
- **Test Types**: Functionality, performance, client integration

## Quick Start

### 1. Setup Environment

```bash
# Choose your platform
./mcp-testing/scripts/setup-macos.sh     # macOS
./mcp-testing/scripts/setup-linux.sh    # Linux  
./mcp-testing/scripts/setup-windows.sh  # Windows/WSL2
```

### 2. Run Tests

```bash
# Run all tests
./mcp-testing/scripts/run-tests.sh --all

# Run specific test type
./mcp-testing/scripts/run-tests.sh --functionality
./mcp-testing/scripts/run-tests.sh --performance

# Test specific client
./mcp-testing/scripts/test-client.sh claude-desktop
./mcp-testing/scripts/test-client.sh cursor
```

### 3. Check Results

```bash
# View results
ls mcp-testing/results/

# Generate HTML report
./mcp-testing/scripts/run-tests.sh --report
open mcp-testing/results/test_report.html
```

## Test Types

### Functionality Tests

Validates core MCP server functionality:

- **Server Info**: Basic server information and status
- **Tool Listing**: Available tools and their schemas
- **Scene Creation**: All scene types (pendulum, double_pendulum, cart_pole)
- **Simulation Control**: Step, get state, reset operations
- **Error Handling**: Invalid inputs and edge cases

**Run**: `./mcp-testing/scripts/run-tests.sh --functionality`

### Performance Tests

Measures response times and resource usage:

- **Response Times**: Tool execution speed
- **Concurrent Requests**: Parallel request handling
- **Memory Usage**: Resource consumption during operations
- **Stress Testing**: High-load scenarios

**Run**: `./mcp-testing/scripts/run-tests.sh --performance`

### Client Integration Tests

Validates MCP configuration for each client:

- **Configuration Check**: Valid config files and paths
- **Connection Test**: MCP server connectivity
- **Tool Discovery**: Available tools in client
- **Basic Operations**: Create scene and step simulation

**Run**: `./mcp-testing/scripts/test-client.sh [client]`

## Client-Specific Setup

### Claude Code

```bash
# Install (when available)
# Installation method TBD

# Configure
cp mcp-testing/configs/claude-code/mcp.json ~/.claude/mcp.json

# Test
./mcp-testing/scripts/test-client.sh claude-code
```

### Cursor

```bash
# Install
# Download from https://cursor.sh

# Configure workspace
mkdir -p .cursor
cp mcp-testing/configs/cursor/settings.json .cursor/settings.json

# Test
./mcp-testing/scripts/test-client.sh cursor
```

### Claude Desktop

```bash
# Install
# Download from https://claude.ai/download

# Configure (macOS)
cp mcp-testing/configs/claude-desktop/claude_desktop_config.json \
   "$HOME/Library/Application Support/Claude/claude_desktop_config.json"

# Test
./mcp-testing/scripts/test-client.sh claude-desktop
```

### OpenAI ChatGPT Bridge

```bash
# Set API key
export OPENAI_API_KEY="your-key-here"

# Install dependencies
pip install -r mcp-testing/configs/chatgpt/requirements.txt

# Test
./mcp-testing/scripts/test-client.sh chatgpt

# Run bridge interactively
python mcp-testing/configs/chatgpt/mcp-bridge.py
```

## Platform-Specific Notes

### macOS

- **Graphics**: Uses osmesa backend by default
- **Dependencies**: Installed via Homebrew
- **Claude Desktop**: Native support
- **Architecture**: Intel and Apple Silicon supported

### Linux

- **Graphics**: EGL preferred, falls back to osmesa
- **Dependencies**: Package manager (apt/dnf/pacman)
- **Claude Desktop**: Not officially supported
- **Display**: X11 forwarding for GUI applications

### Windows (WSL2)

- **Graphics**: osmesa backend (software rendering)
- **Dependencies**: Ubuntu packages in WSL2
- **Claude Desktop**: Windows native app
- **X11**: VcXsrv or X410 for GUI support

## Troubleshooting

### Common Issues

1. **MCP Server Won't Start**
   ```bash
   # Check Python path
   which python3
   
   # Verify installation
   python3 -m mujoco_mcp --help
   
   # Check dependencies
   python3 -c "import mujoco, mcp"
   ```

2. **Graphics Backend Issues**
   ```bash
   # Test backends
   export MUJOCO_GL=osmesa  # Software rendering
   export MUJOCO_GL=egl     # Hardware rendering
   export MUJOCO_GL=glfw    # Window-based
   ```

3. **Client Connection Issues**
   ```bash
   # Check config paths
   ls ~/.claude/mcp.json                    # Claude Code
   ls ~/.config/Cursor/User/settings.json  # Cursor
   ls "$HOME/Library/Application Support/Claude/" # Claude Desktop
   ```

4. **Permission Issues**
   ```bash
   # Make scripts executable
   chmod +x mcp-testing/scripts/*.sh
   
   # Check file permissions
   ls -la mcp-testing/configs/
   ```

### Debug Mode

Enable detailed logging:

```bash
export MUJOCO_MCP_LOG_LEVEL=DEBUG
./mcp-testing/scripts/run-tests.sh --verbose
```

### Performance Issues

```bash
# Check system resources
htop
free -h

# Monitor during tests
./mcp-testing/scripts/run-tests.sh --performance --verbose
```

## Test Results Interpretation

### Functionality Test Results

- **✅ PASSED**: All core functionality working
- **❌ FAILED**: Critical functionality broken
- **⚠️ WARNINGS**: Non-critical issues

### Performance Benchmarks

- **Response Times**: <100ms for simple operations
- **Memory Usage**: <100MB increase during operations
- **Concurrent Requests**: 80%+ success rate
- **Stress Test**: 90%+ success rate for 50 operations

### Client Integration Results

- **Configuration**: Valid config files found
- **Connection**: MCP server responds
- **Tools**: All expected tools available
- **Basic Operations**: Scene creation works

## Continuous Integration

### GitHub Actions

Tests run automatically on:

- **Push**: Functionality tests
- **Pull Request**: Full test suite
- **Schedule**: Weekly performance tests

### Test Artifacts

Results uploaded as artifacts:

- `functionality_results.log`
- `performance_results.log`
- `client_test_results.log`
- `test_report.html`

## Development Workflow

### Adding New Tests

1. Create test in `mcp-testing/tests/`
2. Add to `run-tests.sh`
3. Update documentation
4. Test on all platforms

### Adding New Clients

1. Create config in `mcp-testing/configs/[client]/`
2. Add to `test-client.sh`
3. Update setup scripts
4. Document setup process

### Performance Monitoring

1. Run baseline tests
2. Make changes
3. Run tests again
4. Compare results
5. Investigate regressions

## Best Practices

### Test Environment

- Use virtual environments
- Pin dependency versions
- Test on clean systems
- Document OS versions

### Test Data

- Use deterministic test cases
- Verify all scene types
- Test edge cases
- Check error conditions

### Reporting

- Include system information
- Capture logs and errors
- Generate HTML reports
- Track performance over time

## Support

For issues with testing:

1. Check troubleshooting section
2. Review logs in `mcp-testing/results/`
3. Test on different platforms
4. Report issues with full system info
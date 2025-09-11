# MuJoCo MCP Multi-Client Testing Framework

Comprehensive testing framework for MuJoCo MCP server across multiple AI clients and platforms.

## Supported Clients
- **Claude Code** - Anthropic's CLI tool
- **Cursor** - AI code editor 
- **Claude Desktop** - Desktop application
- **OpenAI ChatGPT** - With MCP plugin support

## Supported Platforms
- **macOS** - Native and Homebrew installation
- **Linux** - Ubuntu/Debian with graphics backend support
- **Windows** - WSL2 environment

## Quick Start

```bash
# Run comprehensive test suite
./scripts/test-all-clients.sh

# Test specific client
./scripts/test-client.sh claude-code

# Run platform-specific tests
./scripts/test-client.sh --platform linux
```

## Directory Structure

```
mcp-testing/
├── configs/           # Client-specific configurations
│   ├── claude-code/   # Claude Code CLI config
│   ├── cursor/        # Cursor editor config
│   ├── claude-desktop/# Claude Desktop config
│   └── openai-chatgpt/# ChatGPT MCP config
├── scripts/           # Testing and setup scripts
├── tests/            # Automated test suites
└── docs/             # Documentation and guides
```

## Features

- Cross-platform compatibility testing
- Automated client configuration
- Performance benchmarking
- Graphics backend validation (osmesa, EGL, GLFW)
- MCP protocol compliance checking
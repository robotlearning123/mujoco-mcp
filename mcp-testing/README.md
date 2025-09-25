# MuJoCo MCP Multi-Client Testing Framework

Comprehensive testing framework for the MuJoCo MCP server across multiple AI clients and platforms.

## Supported Clients
- **Claude Code** – Anthropic CLI tooling
- **Cursor** – AI-assisted editor  
- **Claude Desktop** – Desktop application
- **OpenAI ChatGPT** – When MCP support is enabled

## Supported Platforms
- **macOS** – Native and Homebrew installs
- **Linux** – Ubuntu/Debian with OSMesa/EGL backends
- **Windows** – WSL2 environment

## Quick Start

```bash
# Run every client integration
./mcp-testing/scripts/test-client.sh all

# Exercise one client only
./mcp-testing/scripts/test-client.sh claude-code

# Target a specific platform profile
./mcp-testing/scripts/test-client.sh --platform linux
```

## Directory Structure

```
mcp-testing/
├── configs/             # Client-specific configuration files
│   ├── claude-code/     # Claude Code CLI config
│   ├── cursor/          # Cursor editor config
│   ├── claude-desktop/  # Claude Desktop config
│   └── openai-chatgpt/  # ChatGPT MCP config
├── scripts/             # Test harness and helpers
├── tests/               # Automated multi-client suites
└── docs/                # Additional guides and references
```

## Features
- Cross-platform compatibility coverage
- Automated client bootstrap and cleanup
- Performance and protocol benchmarking (MCP, graphics backend)
- Report generation for CI and manual validation

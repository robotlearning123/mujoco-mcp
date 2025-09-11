# MuJoCo MCP Multi-Client Testing

Comprehensive testing setup for MuJoCo MCP server across different AI clients and platforms.

## Supported Clients

- **Claude Code** - Anthropic's official CLI
- **Cursor** - AI-powered code editor  
- **Claude Desktop** - Official Anthropic desktop app
- **OpenAI ChatGPT** - Via MCP bridge/proxy solutions

## Supported Platforms

- **macOS** (Intel & Apple Silicon)
- **Linux** (Ubuntu 22.04+, Debian 11+)
- **Windows** (Windows 10+, WSL2)

## Quick Start

1. **Setup Environment**:
   ```bash
   ./setup.sh [platform]  # platform: macos|linux|windows
   ```

2. **Install MCP Server**:
   ```bash
   pip install -e .
   ```

3. **Test Client Integration**:
   ```bash
   ./test-client.sh [client]  # client: claude-code|cursor|claude-desktop|chatgpt
   ```

## Directory Structure

```
mcp-testing/
├── configs/           # Client-specific configurations
│   ├── claude-code/
│   ├── cursor/
│   ├── claude-desktop/
│   └── chatgpt/
├── scripts/          # Setup and testing scripts
│   ├── setup/        # Platform-specific setup
│   └── test/         # Testing utilities
├── docs/            # Documentation
└── results/         # Test results and logs
```

## Configuration Details

Each client has specific configuration requirements and capabilities. See individual client documentation for detailed setup instructions.
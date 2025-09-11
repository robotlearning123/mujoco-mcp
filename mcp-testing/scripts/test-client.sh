#!/bin/bash
# Test MuJoCo MCP with different clients
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

# Usage function
usage() {
    echo "Usage: $0 <client> [options]"
    echo ""
    echo "Clients:"
    echo "  claude-code     - Test with Claude Code CLI"
    echo "  cursor          - Test with Cursor editor"
    echo "  claude-desktop  - Test with Claude Desktop app"
    echo "  chatgpt         - Test with OpenAI ChatGPT bridge"
    echo "  all             - Test all available clients"
    echo ""
    exit 1
}

# Parse arguments
CLIENT=""
if [[ $# -gt 0 ]]; then
    CLIENT="$1"
else
    usage
fi

test_mcp_server() {
    print_header "Testing MCP Server"
    
    # Test server can start
    print_status "Testing server startup..."
    timeout 10 python -m mujoco_mcp --help > /dev/null 2>&1 || {
        print_error "MCP server failed to start"
        return 1
    }
    print_status "✅ MCP server startup OK"
    return 0
}

test_claude_code() {
    print_header "Testing Claude Code CLI"
    
    if ! command -v claude-code &> /dev/null; then
        print_warning "Claude Code CLI not found"
        print_status "Install from: https://github.com/anthropics/claude-code"
        return 1
    fi
    
    print_status "Found Claude Code: $(claude-code --version 2>/dev/null || echo 'version unknown')"
    
    # Check if MCP config exists
    if [ ! -f ~/.claude/mcp.json ]; then
        print_warning "MCP config not found at ~/.claude/mcp.json"
        print_status "Copying config..."
        mkdir -p ~/.claude
        cp mcp-testing/configs/claude-code/mcp.json ~/.claude/mcp.json
    fi
    
    print_status "✅ Claude Code setup OK"
    return 0
}

test_claude_desktop() {
    print_header "Testing Claude Desktop"
    
    # Platform-specific paths
    case "$(uname)" in
        Darwin)
            CONFIG_PATH="$HOME/Library/Application Support/Claude/claude_desktop_config.json"
            APP_PATH="/Applications/Claude.app"
            ;;
        Linux)
            print_warning "Claude Desktop not officially supported on Linux"
            return 1
            ;;
        CYGWIN*|MINGW*|MSYS*)
            CONFIG_PATH="$APPDATA/Claude/claude_desktop_config.json"
            ;;
    esac
    
    # Check configuration
    if [ ! -f "$CONFIG_PATH" ]; then
        print_warning "Claude Desktop MCP config not found"
        config_dir=$(dirname "$CONFIG_PATH")
        mkdir -p "$config_dir"
        cp mcp-testing/configs/claude-desktop/claude_desktop_config.json "$CONFIG_PATH"
        print_status "Config installed to: $CONFIG_PATH"
    fi
    
    print_status "✅ Claude Desktop setup OK"
    return 0
}

# Main testing function
run_tests() {
    local client="$1"
    
    print_header "Starting MuJoCo MCP tests for client: $client"
    
    # Always test MCP server first
    test_mcp_server || return 1
    
    # Test specific client
    case "$client" in
        claude-code)
            test_claude_code || return 1
            ;;
        claude-desktop)
            test_claude_desktop || return 1
            ;;
        all)
            print_header "Testing all clients..."
            test_claude_code || true
            test_claude_desktop || true
            ;;
        *)
            print_error "Unknown client: $client"
            return 1
            ;;
    esac
    
    print_status "🎉 Tests completed!"
    return 0
}

# Run the tests
run_tests "$CLIENT"
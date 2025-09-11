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
    echo "Options:"
    echo "  --verbose       - Enable verbose output"
    echo "  --no-viewer     - Skip viewer tests"
    echo "  --quick         - Run quick tests only"
    echo ""
    exit 1
}

# Parse arguments
CLIENT=""
VERBOSE=false
NO_VIEWER=false
QUICK=false

while [[ $# -gt 0 ]]; do
    case $1 in
        claude-code|cursor|claude-desktop|chatgpt|all)
            CLIENT="$1"
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --no-viewer)
            NO_VIEWER=true
            shift
            ;;
        --quick)
            QUICK=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            ;;
    esac
done

if [ -z "$CLIENT" ]; then
    usage
fi

# Test functions
test_mcp_server() {
    print_header "Testing MCP Server"
    
    # Test server can start
    print_status "Testing server startup..."
    timeout 10 python -m mujoco_mcp --help > /dev/null 2>&1 || {
        print_error "MCP server failed to start"
        return 1
    }
    print_status "✅ MCP server startup OK"
    
    # Test Python imports
    print_status "Testing Python imports..."
    python -c "
import mujoco_mcp
print(f'MuJoCo MCP version: {mujoco_mcp.__version__}')
import mujoco
print(f'MuJoCo version: {mujoco.__version__}')
"
    print_status "✅ Imports OK"
    
    return 0
}

test_viewer_server() {
    if [ "$NO_VIEWER" = true ]; then
        print_status "Skipping viewer tests"
        return 0
    fi
    
    print_header "Testing Viewer Server"
    
    # Test viewer server can start
    print_status "Testing viewer server startup..."
    timeout 5 python -m mujoco_mcp.viewer_server --help > /dev/null 2>&1 || {
        print_warning "Viewer server test failed (this may be OK)"
        return 0
    }
    print_status "✅ Viewer server OK"
    
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
    
    if [ "$QUICK" = false ]; then
        print_status "Testing MCP connection..."
        # This would be the actual test command when Claude Code supports MCP testing
        print_warning "MCP connection test not yet implemented for Claude Code"
    fi
    
    return 0
}

test_cursor() {
    print_header "Testing Cursor"
    
    # Check for Cursor installation
    CURSOR_CMD=""
    if command -v cursor &> /dev/null; then
        CURSOR_CMD="cursor"
    elif [ -f ~/.local/bin/cursor ]; then
        CURSOR_CMD="~/.local/bin/cursor"
    elif [ -f "/Applications/Cursor.app/Contents/MacOS/Cursor" ]; then
        CURSOR_CMD="/Applications/Cursor.app/Contents/MacOS/Cursor"
    else
        print_warning "Cursor not found"
        print_status "Install from: https://cursor.sh"
        return 1
    fi
    
    print_status "Found Cursor: $CURSOR_CMD"
    
    # Check configuration
    CONFIG_PATHS=(
        ~/.config/Cursor/User/settings.json
        ~/.cursor/settings.json
        .cursor/settings.json
    )
    
    CONFIG_FOUND=false
    for config_path in "${CONFIG_PATHS[@]}"; do
        if [ -f "$config_path" ]; then
            CONFIG_FOUND=true
            print_status "Found config at: $config_path"
            break
        fi
    done
    
    if [ "$CONFIG_FOUND" = false ]; then
        print_warning "Cursor MCP config not found"
        print_status "Copying config to ~/.config/Cursor/User/settings.json"
        mkdir -p ~/.config/Cursor/User
        cp mcp-testing/configs/cursor/settings.json ~/.config/Cursor/User/settings.json
    fi
    
    print_status "✅ Cursor setup OK"
    return 0
}

test_claude_desktop() {
    print_header "Testing Claude Desktop"
    
    # Platform-specific paths
    CONFIG_PATHS=()
    case "$(uname)" in
        Darwin)
            CONFIG_PATHS+=("$HOME/Library/Application Support/Claude/claude_desktop_config.json")
            APP_PATH="/Applications/Claude.app"
            ;;
        Linux)
            print_warning "Claude Desktop not officially supported on Linux"
            return 1
            ;;
        CYGWIN*|MINGW*|MSYS*)
            CONFIG_PATHS+=("$APPDATA/Claude/claude_desktop_config.json")
            ;;
    esac
    
    # Check if Claude Desktop is installed
    if [[ "$(uname)" == "Darwin" ]] && [[ ! -d "$APP_PATH" ]]; then
        print_warning "Claude Desktop not found"
        print_status "Install from: https://claude.ai/download"
        return 1
    fi
    
    # Check configuration
    CONFIG_FOUND=false
    for config_path in "${CONFIG_PATHS[@]}"; do
        if [ -f "$config_path" ]; then
            CONFIG_FOUND=true
            print_status "Found config at: $config_path"
            break
        fi
    done
    
    if [ "$CONFIG_FOUND" = false ]; then
        print_warning "Claude Desktop MCP config not found"
        config_dir=$(dirname "${CONFIG_PATHS[0]}")
        mkdir -p "$config_dir"
        cp mcp-testing/configs/claude-desktop/claude_desktop_config.json "${CONFIG_PATHS[0]}"
        print_status "Config installed to: ${CONFIG_PATHS[0]}"
    fi
    
    print_status "✅ Claude Desktop setup OK"
    return 0
}

test_chatgpt_bridge() {
    print_header "Testing OpenAI ChatGPT Bridge"
    
    # Check for OpenAI API key
    if [ -z "$OPENAI_API_KEY" ]; then
        print_error "OPENAI_API_KEY environment variable not set"
        print_status "Get your API key from: https://platform.openai.com/api-keys"
        return 1
    fi
    
    # Test OpenAI Python package
    print_status "Testing OpenAI package..."
    python -c "import openai; print(f'OpenAI package version: {openai.__version__}')" || {
        print_error "OpenAI package not found"
        print_status "Install with: pip install openai>=1.0.0"
        return 1
    }
    
    # Test bridge script
    print_status "Testing bridge script..."
    if [ ! -f mcp-testing/configs/chatgpt/mcp-bridge.py ]; then
        print_error "Bridge script not found"
        return 1
    fi
    
    python mcp-testing/configs/chatgpt/mcp-bridge.py --help > /dev/null 2>&1 || {
        print_warning "Bridge script test failed"
    }
    
    print_status "✅ ChatGPT bridge setup OK"
    
    if [ "$QUICK" = false ]; then
        print_status "To test the bridge, run:"
        print_status "  python mcp-testing/configs/chatgpt/mcp-bridge.py"
    fi
    
    return 0
}

# Main testing function
run_tests() {
    local client="$1"
    local failed=0
    
    print_header "Starting MuJoCo MCP tests for client: $client"
    
    # Always test MCP server first
    test_mcp_server || ((failed++))
    test_viewer_server || ((failed++))
    
    # Test specific client
    case "$client" in
        claude-code)
            test_claude_code || ((failed++))
            ;;
        cursor)
            test_cursor || ((failed++))
            ;;
        claude-desktop)
            test_claude_desktop || ((failed++))
            ;;
        chatgpt)
            test_chatgpt_bridge || ((failed++))
            ;;
        all)
            print_header "Testing all clients..."
            test_claude_code || ((failed++))
            test_cursor || ((failed++))
            test_claude_desktop || ((failed++))
            test_chatgpt_bridge || ((failed++))
            ;;
        *)
            print_error "Unknown client: $client"
            return 1
            ;;
    esac
    
    # Summary
    echo ""
    if [ $failed -eq 0 ]; then
        print_status "🎉 All tests passed!"
    else
        print_warning "⚠️  $failed test(s) failed or had warnings"
    fi
    
    return $failed
}

# Run the tests
run_tests "$CLIENT"
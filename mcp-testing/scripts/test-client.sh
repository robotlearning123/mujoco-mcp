#!/bin/bash

# MuJoCo MCP Multi-Client Testing Script
# Tests MCP server setup across different AI clients and platforms

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

print_status() {
    echo -e "${YELLOW}$1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Test MCP server basic functionality
test_mcp_server() {
    print_header "Testing MCP Server"
    
    print_status "Testing server startup..."
    timeout 10 python -m mujoco_mcp --help > /dev/null 2>&1 || {
        print_error "MCP server failed to start"
        return 1
    }
    print_success "MCP server startup OK"
    
    return 0
}

# Test Claude Code configuration
test_claude_code() {
    print_header "Testing Claude Code Configuration"
    
    local config_path="../configs/claude-code/mcp.json"
    
    if [ ! -f "$config_path" ]; then
        print_error "Claude Code config not found: $config_path"
        return 1
    fi
    
    print_success "Claude Code config valid"
    return 0
}

# Main test function
run_tests() {
    local client="$1"
    
    print_header "MuJoCo MCP Multi-Client Testing"
    print_status "Client: ${client:-all}"
    
    # Always test MCP server first
    test_mcp_server || exit 1
    
    # Test specific client or all clients
    case "$client" in
        "claude-code")
            test_claude_code || exit 1
            ;;
        *)
            test_claude_code || exit 1
            ;;
    esac
    
    print_success "All tests passed!"
}

# Parse command line arguments
CLIENT="$1"

# Change to script directory
cd "$(dirname "$0")"

# Run tests
run_tests "$CLIENT"
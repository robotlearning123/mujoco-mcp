#!/bin/bash
# Run comprehensive MCP tests
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

# Usage
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --functionality  - Run functionality tests only"
    echo "  --performance   - Run performance tests only"  
    echo "  --client <name> - Test specific client"
    echo "  --all           - Run all tests (default)"
    echo "  --verbose       - Enable verbose output"
    echo "  --report        - Generate HTML report"
    echo "  --help          - Show this help"
    echo ""
    exit 1
}

# Parse arguments
TEST_TYPE="all"
CLIENT=""
VERBOSE=false
GENERATE_REPORT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --functionality)
            TEST_TYPE="functionality"
            shift
            ;;
        --performance)
            TEST_TYPE="performance"
            shift
            ;;
        --client)
            CLIENT="$2"
            shift 2
            ;;
        --all)
            TEST_TYPE="all"
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --report)
            GENERATE_REPORT=true
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

# Create results directory
RESULTS_DIR="mcp-testing/results"
mkdir -p "$RESULTS_DIR"

# Test environment setup
check_environment() {
    print_header "Checking test environment..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 not found"
        exit 1
    fi
    
    print_status "Python: $(python3 --version)"
    
    # Check MuJoCo MCP
    if ! python3 -c "import mujoco_mcp" 2>/dev/null; then
        print_error "MuJoCo MCP not installed"
        print_status "Install with: pip install -e ."
        exit 1
    fi
    
    print_status "MuJoCo MCP: Available"
    
    # Check test dependencies
    python3 -c "import pytest, psutil" 2>/dev/null || {
        print_warning "Installing test dependencies..."
        pip install pytest psutil
    }
    
    # Set environment variables
    export MUJOCO_GL=osmesa
    export MUJOCO_MCP_LOG_LEVEL=INFO
    export PYTHONPATH="$PWD/src:$PYTHONPATH"
    
    print_status "✅ Environment ready"
}

# Run functionality tests
run_functionality_tests() {
    print_header "Running functionality tests..."
    
    cd mcp-testing/tests
    
    if [ "$VERBOSE" = true ]; then
        python3 test_mcp_functionality.py | tee ../../$RESULTS_DIR/functionality_results.log
    else
        python3 test_mcp_functionality.py > ../../$RESULTS_DIR/functionality_results.log 2>&1
    fi
    
    local exit_code=$?
    cd ../..
    
    if [ $exit_code -eq 0 ]; then
        print_status "✅ Functionality tests PASSED"
    else
        print_error "❌ Functionality tests FAILED"
    fi
    
    return $exit_code
}

# Run performance tests  
run_performance_tests() {
    print_header "Running performance tests..."
    
    cd mcp-testing/tests
    
    if [ "$VERBOSE" = true ]; then
        python3 test_performance.py | tee ../../$RESULTS_DIR/performance_results.log
    else
        python3 test_performance.py > ../../$RESULTS_DIR/performance_results.log 2>&1
    fi
    
    local exit_code=$?
    cd ../..
    
    if [ $exit_code -eq 0 ]; then
        print_status "✅ Performance tests PASSED"
    else
        print_error "❌ Performance tests FAILED"
    fi
    
    return $exit_code
}

# Run client tests
run_client_tests() {
    local client="$1"
    print_header "Running client tests for: $client"
    
    ./mcp-testing/scripts/test-client.sh "$client" --verbose > "$RESULTS_DIR/client_${client}_results.log" 2>&1
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        print_status "✅ Client tests ($client) PASSED"
    else
        print_error "❌ Client tests ($client) FAILED"
    fi
    
    return $exit_code
}

# Generate HTML report
generate_html_report() {
    print_header "Generating HTML report..."
    
    cat > "$RESULTS_DIR/test_report.html" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>MuJoCo MCP Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { color: #333; border-bottom: 2px solid #007acc; padding-bottom: 10px; }
        .section { margin: 20px 0; }
        .pass { color: green; font-weight: bold; }
        .fail { color: red; font-weight: bold; }
        .log { background: #f5f5f5; padding: 10px; border-left: 4px solid #007acc; font-family: monospace; white-space: pre-wrap; }
        .summary { background: #e7f3ff; padding: 15px; border-radius: 5px; }
    </style>
</head>
<body>
    <h1 class="header">MuJoCo MCP Test Report</h1>
    <div class="summary">
        <h2>Test Summary</h2>
        <p>Generated: $(date)</p>
        <p>Test Type: $TEST_TYPE</p>
EOF

    # Add functionality results if available
    if [ -f "$RESULTS_DIR/functionality_results.log" ]; then
        echo "<div class='section'>" >> "$RESULTS_DIR/test_report.html"
        echo "<h2>Functionality Tests</h2>" >> "$RESULTS_DIR/test_report.html"
        if grep -q "Overall: PASSED" "$RESULTS_DIR/functionality_results.log"; then
            echo "<p class='pass'>✅ PASSED</p>" >> "$RESULTS_DIR/test_report.html"
        else
            echo "<p class='fail'>❌ FAILED</p>" >> "$RESULTS_DIR/test_report.html"
        fi
        echo "<div class='log'>" >> "$RESULTS_DIR/test_report.html"
        cat "$RESULTS_DIR/functionality_results.log" >> "$RESULTS_DIR/test_report.html"
        echo "</div></div>" >> "$RESULTS_DIR/test_report.html"
    fi
    
    # Add performance results if available
    if [ -f "$RESULTS_DIR/performance_results.log" ]; then
        echo "<div class='section'>" >> "$RESULTS_DIR/test_report.html"
        echo "<h2>Performance Tests</h2>" >> "$RESULTS_DIR/test_report.html"
        if grep -q "Overall: PASSED" "$RESULTS_DIR/performance_results.log"; then
            echo "<p class='pass'>✅ PASSED</p>" >> "$RESULTS_DIR/test_report.html"
        else
            echo "<p class='fail'>❌ FAILED</p>" >> "$RESULTS_DIR/test_report.html"
        fi
        echo "<div class='log'>" >> "$RESULTS_DIR/test_report.html"
        cat "$RESULTS_DIR/performance_results.log" >> "$RESULTS_DIR/test_report.html"
        echo "</div></div>" >> "$RESULTS_DIR/test_report.html"
    fi
    
    echo "</body></html>" >> "$RESULTS_DIR/test_report.html"
    
    print_status "📄 Report generated: $RESULTS_DIR/test_report.html"
}

# Main execution
main() {
    print_header "MuJoCo MCP Test Suite"
    
    # Check environment
    check_environment
    
    # Track overall results
    overall_result=0
    
    # Run tests based on type
    case "$TEST_TYPE" in
        functionality)
            run_functionality_tests || overall_result=1
            ;;
        performance)
            run_performance_tests || overall_result=1
            ;;
        all)
            run_functionality_tests || overall_result=1
            run_performance_tests || overall_result=1
            ;;
    esac
    
    # Run client tests if specified
    if [ -n "$CLIENT" ]; then
        run_client_tests "$CLIENT" || overall_result=1
    fi
    
    # Generate report if requested
    if [ "$GENERATE_REPORT" = true ]; then
        generate_html_report
    fi
    
    # Final summary
    echo ""
    print_header "Test Summary"
    if [ $overall_result -eq 0 ]; then
        print_status "🎉 All tests completed successfully!"
    else
        print_error "⚠️  Some tests failed. Check logs in $RESULTS_DIR/"
    fi
    
    print_status "Results saved to: $RESULTS_DIR/"
    
    return $overall_result
}

# Run main function
main "$@"
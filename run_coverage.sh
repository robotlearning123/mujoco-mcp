#!/bin/bash
# Run tests with coverage reporting
# This script runs the full test suite and generates coverage reports

set -e  # Exit on error

echo "=========================================="
echo "MuJoCo MCP Coverage Report"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0.32m'
RED='\033[0.31m'
YELLOW='\033[0.33m'
NC='\033[0m' # No Color

# Clean previous coverage data
echo "1. Cleaning previous coverage data..."
rm -f .coverage .coverage.*
rm -rf htmlcov/
rm -f coverage.xml coverage.json
echo "   ✓ Cleaned"
echo ""

# Install test dependencies
echo "2. Checking test dependencies..."
python3 -m pip install --quiet pytest pytest-cov coverage hypothesis 2>/dev/null || true
echo "   ✓ Dependencies ready"
echo ""

# Run unit tests with coverage
echo "3. Running unit tests with coverage..."
python3 -m pytest tests/unit/ \
    --cov=src/mujoco_mcp \
    --cov-report=term-missing \
    --cov-report=html \
    --cov-report=xml \
    --cov-report=json \
    --cov-branch \
    -v \
    || { echo -e "${RED}Unit tests failed${NC}"; exit 1; }
echo ""

# Run integration tests (but don't fail on coverage)
echo "4. Running integration tests..."
python3 -m pytest tests/integration/ \
    --cov=src/mujoco_mcp \
    --cov-append \
    --cov-report= \
    -v \
    || echo -e "${YELLOW}Some integration tests failed (may require MuJoCo)${NC}"
echo ""

# Generate final coverage reports
echo "5. Generating final coverage reports..."
python3 -m coverage combine 2>/dev/null || true
python3 -m coverage report
python3 -m coverage html
python3 -m coverage xml
python3 -m coverage json
echo ""

# Display coverage summary
echo "=========================================="
echo "Coverage Summary"
echo "=========================================="
python3 -m coverage report --skip-covered

# Get coverage percentage
COVERAGE=$(python3 -m coverage report | tail -1 | awk '{print $(NF-0)}' | sed 's/%//')

echo ""
echo "=========================================="
if (( $(echo "$COVERAGE >= 95.0" | bc -l) )); then
    echo -e "${GREEN}✓ Coverage: ${COVERAGE}% (Target: 95%)${NC}"
    echo -e "${GREEN}✓ EXCELLENT COVERAGE${NC}"
elif (( $(echo "$COVERAGE >= 85.0" | bc -l) )); then
    echo -e "${YELLOW}⚠ Coverage: ${COVERAGE}% (Target: 95%)${NC}"
    echo -e "${YELLOW}⚠ GOOD COVERAGE - Aim for 95%${NC}"
else
    echo -e "${RED}✗ Coverage: ${COVERAGE}% (Target: 95%)${NC}"
    echo -e "${RED}✗ INSUFFICIENT COVERAGE${NC}"
fi
echo "=========================================="
echo ""

# Show where to find reports
echo "Coverage reports generated:"
echo "  - Terminal: (above)"
echo "  - HTML: open htmlcov/index.html"
echo "  - XML: coverage.xml"
echo "  - JSON: coverage.json"
echo ""

# Return appropriate exit code
if (( $(echo "$COVERAGE >= 85.0" | bc -l) )); then
    exit 0
else
    exit 1
fi

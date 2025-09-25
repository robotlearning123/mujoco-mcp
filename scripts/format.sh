#!/bin/bash
# Quick code formatting script

set -e

echo "🔧 Formatting code..."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${BLUE}📋 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# 1. Format with Black
print_step "Running Black formatter..."
black .
print_success "Black formatting completed"

# 2. Sort imports with isort
print_step "Sorting imports with isort..."
isort .
print_success "Import sorting completed"

# 3. Auto-fix with Ruff
print_step "Auto-fixing with Ruff..."
ruff check --fix .
print_success "Ruff auto-fixes applied"

# 4. Format with Ruff formatter
print_step "Running Ruff formatter..."
ruff format .
print_success "Ruff formatting completed"

echo ""
print_success "Code formatting completed! 🎉"
echo ""
echo "💡 Next steps:"
echo "  - Review the changes made"
echo "  - Run './scripts/quality-check.sh' for full quality analysis"
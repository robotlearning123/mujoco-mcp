#!/bin/bash
# Development environment setup script

set -e

echo "🚀 Setting up development environment..."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${BLUE}📋 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    print_warning "No virtual environment detected. Creating one..."
    python -m venv venv
    source venv/bin/activate
    print_success "Virtual environment created and activated"
else
    print_success "Virtual environment already activated: $VIRTUAL_ENV"
fi

# 1. Upgrade pip
print_step "Upgrading pip..."
python -m pip install --upgrade pip
print_success "Pip upgraded"

# 2. Install development dependencies
print_step "Installing development dependencies..."
pip install -e ".[dev]"
print_success "Development dependencies installed"

# 3. Install pre-commit hooks
print_step "Installing pre-commit hooks..."
pre-commit install
print_success "Pre-commit hooks installed"

# 4. Run initial pre-commit check
print_step "Running initial pre-commit check..."
if pre-commit run --all-files; then
    print_success "Pre-commit checks passed"
else
    print_warning "Pre-commit found issues. They have been auto-fixed where possible."
fi

# 5. Create necessary directories
print_step "Creating necessary directories..."
mkdir -p logs
mkdir -p reports
print_success "Directories created"

echo ""
print_success "Development environment setup completed! 🎉"
echo ""
echo "📚 Available scripts:"
echo "  - ./scripts/format.sh          - Quick code formatting"
echo "  - ./scripts/quality-check.sh   - Full quality analysis"
echo ""
echo "🔧 Development commands:"
echo "  - pre-commit run --all-files   - Run all pre-commit hooks"
echo "  - pytest --cov=src             - Run tests with coverage"
echo "  - mypy src/                     - Type checking"
echo "  - ruff check .                  - Linting"
echo "  - black .                       - Code formatting"
echo ""
echo "💡 Next steps:"
echo "  - Start coding with confidence!"
echo "  - Pre-commit hooks will automatically check your code"
echo "  - Run quality checks before pushing to ensure CI/CD success"
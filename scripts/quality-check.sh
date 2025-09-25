#!/bin/bash
# Quality check script for local development

set -e

echo "🔍 Running code quality checks..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${BLUE}📋 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    print_warning "No virtual environment detected. Consider activating one."
fi

# 1. Format code with Black
print_step "Formatting code with Black..."
if black --check --diff .; then
    print_success "Code formatting is correct"
else
    print_warning "Code formatting issues found. Running black to fix..."
    black .
    print_success "Code formatted successfully"
fi

# 2. Sort imports with isort
print_step "Sorting imports with isort..."
if isort --check-only --diff .; then
    print_success "Import sorting is correct"
else
    print_warning "Import sorting issues found. Running isort to fix..."
    isort .
    print_success "Imports sorted successfully"
fi

# 3. Lint with Ruff
print_step "Linting code with Ruff..."
if ruff check --fix .; then
    print_success "Linting passed"
else
    print_error "Linting issues found. Please review and fix manually."
fi

# 4. Type checking with MyPy
print_step "Type checking with MyPy..."
if mypy src/; then
    print_success "Type checking passed"
else
    print_error "Type checking issues found. Please review and fix."
fi

# 5. Security check with Bandit
print_step "Security check with Bandit..."
if bandit -r src/ -q; then
    print_success "Security check passed"
else
    print_warning "Security issues found. Please review bandit output."
fi

# 6. Dependency security check with Safety
print_step "Checking dependencies for security vulnerabilities..."
if safety check; then
    print_success "Dependency security check passed"
else
    print_warning "Security vulnerabilities found in dependencies."
fi

# 7. Complexity analysis
print_step "Analyzing code complexity..."
echo "Cyclomatic Complexity:"
radon cc src/ -s
echo ""
echo "Maintainability Index:"
radon mi src/ -s

# 8. Run tests with coverage
print_step "Running tests with coverage..."
if pytest --cov=src/mujoco_mcp --cov-report=term-missing --cov-report=html; then
    print_success "Tests passed with coverage report generated"
else
    print_error "Tests failed. Please review and fix."
fi

echo ""
print_success "Code quality check completed! 🎉"
echo ""
echo "📊 Reports generated:"
echo "  - Coverage report: htmlcov/index.html"
echo "  - Run 'open htmlcov/index.html' to view coverage"
echo ""
echo "💡 Next steps:"
echo "  - Review any warnings or errors above"
echo "  - Fix issues before committing"
echo "  - Run 'pre-commit install' to enable automatic checks"
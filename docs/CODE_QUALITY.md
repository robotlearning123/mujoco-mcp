# Code Quality Guide

This document outlines the code quality tools and practices implemented in the MuJoCo MCP project.

## 🚀 Quick Start

### Setting Up Development Environment

```bash
# Run the setup script to configure everything
./scripts/setup-dev.sh
```

This will:
- Create/activate virtual environment
- Install development dependencies
- Set up pre-commit hooks
- Run initial quality checks

### Daily Development Workflow

```bash
# Format code before committing
./scripts/format.sh

# Run full quality checks
./scripts/quality-check.sh

# Commit with automatic pre-commit hooks
git commit -m "your commit message"
```

## 🛠️ Tools Overview

### Code Formatting
- **Black**: Automatic code formatting with 100-character line length
- **isort**: Import sorting and organization
- **Ruff**: Fast Python formatter and fixer

### Linting and Analysis
- **Ruff**: Fast Python linter with comprehensive rule set
- **MyPy**: Static type checking with strict configuration
- **Bandit**: Security vulnerability scanning
- **Safety**: Dependency vulnerability checking

### Code Quality Metrics
- **Radon**: Cyclomatic complexity and maintainability analysis
- **Xenon**: Complexity threshold enforcement
- **Coverage**: Test coverage measurement and reporting

### Automation
- **Pre-commit**: Automatic code quality checks before commits
- **GitHub Actions**: Continuous integration with quality gates

## 📋 Configuration Files

### `.pre-commit-config.yaml`
Defines pre-commit hooks that run automatically:
- Code formatting (Black, isort)
- Linting (Ruff, flake8)
- Type checking (MyPy)
- Security scanning (Bandit)
- Spell checking (codespell)

### `pyproject.toml`
Contains configuration for:
- **MyPy**: Type checking settings
- **Black**: Code formatting preferences
- **isort**: Import sorting rules
- **Bandit**: Security scanning exclusions
- **pytest**: Test configuration
- **Coverage**: Coverage reporting settings

### `.ruff.toml`
Comprehensive linting configuration with:
- Enabled rule sets for code quality
- Selective rule ignoring for gradual improvement
- File-specific rule overrides
- Import conventions and styling preferences

## 🔧 Available Scripts

### `./scripts/setup-dev.sh`
- Sets up complete development environment
- Installs all dependencies
- Configures pre-commit hooks
- Runs initial quality checks

### `./scripts/format.sh`
Quick code formatting:
- Black formatting
- Import sorting with isort
- Ruff auto-fixes
- Ruff formatting

### `./scripts/quality-check.sh`
Comprehensive quality analysis:
- Code formatting verification
- Linting with Ruff
- Type checking with MyPy
- Security scanning with Bandit
- Dependency vulnerability checking
- Complexity analysis
- Test execution with coverage

## 📊 Quality Metrics

### Coverage Requirements
- **Minimum**: 80% line coverage
- **Target**: 90% line coverage
- **Reports**: Generated in `htmlcov/` directory

### Complexity Thresholds
- **Cyclomatic Complexity**: Max B grade per function
- **Maintainability Index**: Min A grade average
- **Enforcement**: Automated via Xenon in CI/CD

### Security Standards
- **Bandit**: No high-severity security issues
- **Safety**: No known vulnerabilities in dependencies
- **Regular Updates**: Automated dependency vulnerability checking

## 🎯 GitHub Actions Workflows

### Code Quality Workflow (`.github/workflows/code-quality.yml`)

Comprehensive CI/CD pipeline with parallel jobs:

1. **Pre-commit**: Runs all pre-commit hooks
2. **Lint and Format**: Multi-Python version testing
3. **Type Check**: MyPy static analysis
4. **Security**: Bandit and Safety vulnerability scanning
5. **Complexity**: Code complexity analysis
6. **Test Coverage**: Pytest with coverage reporting
7. **Quality Gate**: Final pass/fail determination

### Quality Gate Criteria
All checks must pass for successful CI/CD:
- ✅ Pre-commit hooks successful
- ✅ Formatting compliance (Black, isort)
- ✅ Linting compliance (Ruff)
- ✅ Type checking (MyPy)
- ✅ Security scanning (Bandit, Safety)
- ✅ Complexity thresholds (Xenon)
- ✅ Test coverage minimum (80%)

## 🔄 Development Workflow

### Before Starting Work
```bash
# Ensure development environment is set up
./scripts/setup-dev.sh

# Pull latest changes
git pull origin main
```

### During Development
```bash
# Format code regularly
./scripts/format.sh

# Run quality checks before committing
./scripts/quality-check.sh
```

### Before Committing
Pre-commit hooks will automatically run, but you can manually trigger:
```bash
pre-commit run --all-files
```

### Before Pushing
Ensure all quality checks pass:
```bash
./scripts/quality-check.sh
```

## 📈 Continuous Improvement

### Gradual Rule Enablement
The Ruff configuration includes many commented-out rules that can be gradually enabled:
1. Fix existing violations for a rule category
2. Uncomment the rule in `.ruff.toml`
3. Commit the improvement

### Metrics Tracking
Quality metrics are tracked over time:
- **Coverage**: Aim for incremental coverage improvements
- **Complexity**: Refactor high-complexity functions
- **Security**: Address all Bandit findings
- **Dependencies**: Keep dependencies up-to-date

### Tool Updates
Regular updates to maintain effectiveness:
- **Weekly**: Automated pre-commit hook updates
- **Monthly**: Review and update tool versions
- **Quarterly**: Evaluate new tools and practices

## 🆘 Troubleshooting

### Common Issues

#### Pre-commit hooks failing
```bash
# Update hooks
pre-commit autoupdate

# Run specific hook
pre-commit run black --all-files
```

#### MyPy type errors
```bash
# Check specific file
mypy src/mujoco_mcp/specific_file.py

# Generate type stubs
mypy --install-types
```

#### Coverage too low
```bash
# Run tests with detailed coverage
pytest --cov=src --cov-report=html --cov-report=term-missing

# View coverage report
open htmlcov/index.html
```

#### Security issues
```bash
# Detailed Bandit report
bandit -r src/ -f json

# Check specific vulnerability
safety check --json
```

### Getting Help
- Review tool documentation for specific error messages
- Check existing issues in the repository
- Run tools with verbose flags for detailed output
- Use `--help` flag with any tool for usage information

## 📚 Additional Resources

- [Black Documentation](https://black.readthedocs.io/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [MyPy Documentation](https://mypy.readthedocs.io/)
- [Pre-commit Documentation](https://pre-commit.com/)
- [Pytest Documentation](https://docs.pytest.org/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

---

**Remember**: Code quality is a journey, not a destination. These tools help maintain high standards while allowing for incremental improvements over time.
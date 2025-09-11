#!/bin/bash
# macOS Setup Script for MuJoCo MCP Multi-Client Testing
set -e

echo "🍎 Setting up MuJoCo MCP testing environment for macOS..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    print_error "This script is for macOS only"
    exit 1
fi

# Detect architecture
ARCH=$(uname -m)
if [[ "$ARCH" == "arm64" ]]; then
    print_status "Detected Apple Silicon Mac"
    CONDA_INSTALLER="Miniconda3-latest-MacOSX-arm64.sh"
else
    print_status "Detected Intel Mac"  
    CONDA_INSTALLER="Miniconda3-latest-MacOSX-x86_64.sh"
fi

# Install Homebrew if not present
if ! command -v brew &> /dev/null; then
    print_status "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    print_status "Homebrew already installed"
fi

# Install Python if needed
if ! command -v python3 &> /dev/null; then
    print_status "Installing Python via Homebrew..."
    brew install python
else
    print_status "Python already installed: $(python3 --version)"
fi

# Install pip if needed
if ! command -v pip3 &> /dev/null; then
    print_status "Installing pip..."
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python3 get-pip.py
    rm get-pip.py
else
    print_status "pip already installed: $(pip3 --version)"
fi

# Install system dependencies
print_status "Installing system dependencies..."
brew install cmake

# Install OpenGL libraries
print_status "Installing OpenGL libraries..."
brew install freeglut

# Create virtual environment
print_status "Creating Python virtual environment..."
python3 -m venv venv-mcp-testing
source venv-mcp-testing/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install MuJoCo MCP and dependencies
print_status "Installing MuJoCo MCP..."
pip install -e .

# Install additional testing dependencies
print_status "Installing testing dependencies..."
pip install pytest pytest-asyncio pytest-mock

# Install client-specific dependencies

# Claude Code (if available)
if command -v npm &> /dev/null; then
    print_status "Installing Claude Code CLI..."
    # Note: This would be the actual installation method when available
    print_warning "Claude Code CLI installation method TBD"
fi

# Cursor (manual install required)
print_warning "Cursor must be manually downloaded from https://cursor.sh"

# Claude Desktop (manual install required)
print_warning "Claude Desktop must be manually downloaded from https://claude.ai/download"

# OpenAI ChatGPT bridge dependencies
print_status "Installing OpenAI dependencies for ChatGPT bridge..."
pip install openai>=1.0.0

# Test MuJoCo installation
print_status "Testing MuJoCo installation..."
python3 -c "import mujoco; print('MuJoCo version:', mujoco.__version__)"

# Test MCP server
print_status "Testing MCP server..."
timeout 5 python3 -m mujoco_mcp --help || print_warning "MCP server test timed out (this is expected)"

# Setup configuration directories
print_status "Setting up configuration directories..."

# Claude Code config
mkdir -p ~/.claude
if [ ! -f ~/.claude/mcp.json ]; then
    cp mcp-testing/configs/claude-code/mcp.json ~/.claude/mcp.json
    print_status "Claude Code configuration installed"
fi

# Claude Desktop config
CLAUDE_DESKTOP_DIR="$HOME/Library/Application Support/Claude"
mkdir -p "$CLAUDE_DESKTOP_DIR"
if [ ! -f "$CLAUDE_DESKTOP_DIR/claude_desktop_config.json" ]; then
    cp mcp-testing/configs/claude-desktop/claude_desktop_config.json "$CLAUDE_DESKTOP_DIR/"
    print_status "Claude Desktop configuration installed"
fi

# Make scripts executable
chmod +x mcp-testing/scripts/*.sh

# Set environment variables
print_status "Setting up environment variables..."
echo "export MUJOCO_GL=osmesa" >> ~/.bashrc
echo "export MUJOCO_MCP_LOG_LEVEL=INFO" >> ~/.bashrc
echo "export PYTHONPATH=$PWD/src:\$PYTHONPATH" >> ~/.bashrc

# For zsh users
if [ "$SHELL" = "/bin/zsh" ] || [ "$SHELL" = "/usr/local/bin/zsh" ]; then
    echo "export MUJOCO_GL=osmesa" >> ~/.zshrc
    echo "export MUJOCO_MCP_LOG_LEVEL=INFO" >> ~/.zshrc  
    echo "export PYTHONPATH=$PWD/src:\$PYTHONPATH" >> ~/.zshrc
fi

print_status "✅ macOS setup complete!"
print_status ""
print_status "Next steps:"
print_status "1. Restart your terminal or run: source ~/.bashrc (or ~/.zshrc)"
print_status "2. Activate the virtual environment: source venv-mcp-testing/bin/activate"
print_status "3. Test with: ./mcp-testing/scripts/test-client.sh claude-desktop"
print_status ""
print_status "Manual installations required:"
print_status "- Cursor: https://cursor.sh"
print_status "- Claude Desktop: https://claude.ai/download"
print_status ""
print_warning "Remember to set OPENAI_API_KEY if using ChatGPT bridge"
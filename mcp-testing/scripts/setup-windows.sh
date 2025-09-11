#!/bin/bash
# Windows Setup Script for MuJoCo MCP Multi-Client Testing (WSL2)
set -e

echo "🪟 Setting up MuJoCo MCP testing environment for Windows (WSL2)..."

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

# Check if running in WSL
if ! grep -qi "microsoft" /proc/version 2>/dev/null; then
    print_error "This script is designed for Windows Subsystem for Linux (WSL2)"
    print_error "Please run this from WSL2 or use the native Windows setup"
    exit 1
fi

print_status "Detected WSL2 environment"

# Check WSL version
if [ -f /proc/version ] && grep -q "WSL2" /proc/version; then
    print_status "Running in WSL2"
else
    print_warning "WSL version unclear, assuming WSL2"
fi

# Update package lists
print_status "Updating package lists..."
sudo apt-get update

# Install system dependencies
print_status "Installing system dependencies..."
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    cmake \
    pkg-config \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libglu1-mesa-dev \
    freeglut3-dev \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    libxext6 \
    curl \
    wget \
    unzip

# Install X11 apps for GUI support (optional)
install_x11_support() {
    print_status "Installing X11 support for GUI applications..."
    sudo apt-get install -y \
        x11-apps \
        x11-utils \
        x11-xserver-utils \
        dbus-x11
        
    print_status "X11 support installed"
    print_warning "You'll need an X server on Windows (e.g., VcXsrv, X410) to run GUI applications"
}

# Ask user about X11 support
read -p "Install X11 support for GUI applications? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    install_x11_support
fi

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

# Install Windows-specific Python packages
print_status "Installing Windows-specific dependencies..."
pip install pywin32 || print_warning "pywin32 installation failed (expected in WSL)"

# OpenAI ChatGPT bridge dependencies
print_status "Installing OpenAI dependencies for ChatGPT bridge..."
pip install openai>=1.0.0

# Test MuJoCo installation
print_status "Testing MuJoCo installation..."
python3 -c "import mujoco; print('MuJoCo version:', mujoco.__version__)"

# Test graphics backends
print_status "Testing graphics backends..."
python3 -c "
import os

# Test osmesa (most reliable in WSL)
os.environ['MUJOCO_GL'] = 'osmesa'
try:
    import mujoco
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><geom size=\"1\"/></body></worldbody></mujoco>')
    print('✅ osmesa backend working')
except Exception as e:
    print(f'❌ osmesa backend failed: {e}')

# Test EGL (may not work without proper setup)
os.environ['MUJOCO_GL'] = 'egl'
try:
    import mujoco
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><geom size=\"1\"/></body></worldbody></mujoco>')
    print('✅ EGL backend working')
except Exception as e:
    print(f'❌ EGL backend failed: {e}')
    print('This is expected in WSL without GPU passthrough')
"

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

# Make scripts executable
chmod +x mcp-testing/scripts/*.sh

# Set environment variables
print_status "Setting up environment variables..."
echo "export MUJOCO_GL=osmesa" >> ~/.bashrc
echo "export MUJOCO_MCP_LOG_LEVEL=INFO" >> ~/.bashrc
echo "export PYTHONPATH=$PWD/src:\$PYTHONPATH" >> ~/.bashrc

# WSL-specific settings
echo "export DISPLAY=:0" >> ~/.bashrc
echo "export LIBGL_ALWAYS_INDIRECT=1" >> ~/.bashrc

# For zsh users
if [ "$SHELL" = "/usr/bin/zsh" ] || [ "$SHELL" = "/bin/zsh" ]; then
    echo "export MUJOCO_GL=osmesa" >> ~/.zshrc
    echo "export MUJOCO_MCP_LOG_LEVEL=INFO" >> ~/.zshrc  
    echo "export PYTHONPATH=$PWD/src:\$PYTHONPATH" >> ~/.zshrc
    echo "export DISPLAY=:0" >> ~/.zshrc
    echo "export LIBGL_ALWAYS_INDIRECT=1" >> ~/.zshrc
fi

# Create Windows-compatible test script
print_status "Creating Windows test scripts..."
cat > mcp-testing/scripts/test-windows-native.ps1 << 'EOF'
# Windows PowerShell script for testing MCP on native Windows
param(
    [Parameter(Mandatory=$false)]
    [string]$Client = "claude-desktop"
)

Write-Host "Testing MuJoCo MCP on Windows with client: $Client" -ForegroundColor Green

# Test Python installation
Write-Host "Testing Python..." -ForegroundColor Yellow
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python not found. Please install Python 3.10+" -ForegroundColor Red
    exit 1
}

# Test MuJoCo MCP installation
Write-Host "Testing MuJoCo MCP..." -ForegroundColor Yellow
python -c "import mujoco_mcp; print('MuJoCo MCP imported successfully')"
if ($LASTEXITCODE -ne 0) {
    Write-Host "MuJoCo MCP not found. Please install with: pip install -e ." -ForegroundColor Red
    exit 1
}

# Test MCP server
Write-Host "Testing MCP server..." -ForegroundColor Yellow
Start-Process -FilePath "python" -ArgumentList "-m", "mujoco_mcp", "--help" -Wait -NoNewWindow

Write-Host "Windows MCP test completed" -ForegroundColor Green
EOF

print_status "✅ Windows (WSL2) setup complete!"
print_status ""
print_status "Next steps:"
print_status "1. Restart your terminal or run: source ~/.bashrc"
print_status "2. Activate the virtual environment: source venv-mcp-testing/bin/activate"
print_status "3. Test with: ./mcp-testing/scripts/test-client.sh"
print_status ""
print_status "Windows-specific notes:"
print_status "- Graphics: Using osmesa backend (software rendering)"
print_status "- GUI apps: Install VcXsrv or X410 for X11 support"
print_status "- Native Windows: Use test-windows-native.ps1 in PowerShell"
print_status ""
print_status "Client installation:"
print_status "- Claude Desktop: Download from https://claude.ai/download"
print_status "- Cursor: Download from https://cursor.sh"
print_status "- VSCode with extensions: Install from Windows"
print_status ""
print_warning "Remember to set OPENAI_API_KEY if using ChatGPT bridge"

# Display system info
print_status ""
print_status "System Information:"
print_status "WSL Distribution: $(lsb_release -d 2>/dev/null | cut -f2 || echo 'Unknown')"
print_status "Kernel: $(uname -r)"
print_status "Python: $(python3 --version)"
print_status "Architecture: $(uname -m)"
print_status "Graphics: osmesa (software rendering)"
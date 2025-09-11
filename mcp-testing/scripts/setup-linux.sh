#!/bin/bash
# Linux Setup Script for MuJoCo MCP Multi-Client Testing
set -e

echo "🐧 Setting up MuJoCo MCP testing environment for Linux..."

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

# Detect Linux distribution
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
    VER=$VERSION_ID
else
    print_error "Cannot detect Linux distribution"
    exit 1
fi

print_status "Detected $OS $VER"

# Install system dependencies based on distribution
install_system_deps() {
    case $OS in
        ubuntu|debian)
            print_status "Installing system dependencies with apt..."
            sudo apt-get update
            sudo apt-get install -y \
                python3 \
                python3-pip \
                python3-venv \
                build-essential \
                cmake \
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
                wget
            ;;
        fedora|centos|rhel)
            print_status "Installing system dependencies with dnf/yum..."
            if command -v dnf &> /dev/null; then
                PKG_MGR="sudo dnf"
            else
                PKG_MGR="sudo yum"
            fi
            
            $PKG_MGR install -y \
                python3 \
                python3-pip \
                gcc \
                gcc-c++ \
                cmake \
                mesa-libGL-devel \
                mesa-libEGL-devel \
                mesa-libGLU-devel \
                freeglut-devel \
                libXrandr-devel \
                libXinerama-devel \
                libXcursor-devel \
                libXi-devel \
                curl \
                wget
            ;;
        arch|manjaro)
            print_status "Installing system dependencies with pacman..."
            sudo pacman -Syu --noconfirm \
                python \
                python-pip \
                base-devel \
                cmake \
                mesa \
                glu \
                freeglut \
                libxrandr \
                libxinerama \
                libxcursor \
                libxi \
                curl \
                wget
            ;;
        *)
            print_warning "Unsupported distribution: $OS"
            print_warning "Please install Python 3, pip, build tools, and OpenGL libraries manually"
            ;;
    esac
}

install_system_deps

# Update pip
print_status "Upgrading pip..."
python3 -m pip install --upgrade pip --user

# Create virtual environment
print_status "Creating Python virtual environment..."
python3 -m venv venv-mcp-testing
source venv-mcp-testing/bin/activate

# Install MuJoCo MCP and dependencies
print_status "Installing MuJoCo MCP..."
pip install -e .

# Install additional testing dependencies
print_status "Installing testing dependencies..."
pip install pytest pytest-asyncio pytest-mock

# Install Node.js for some clients (optional)
install_nodejs() {
    if ! command -v node &> /dev/null; then
        print_status "Installing Node.js..."
        case $OS in
            ubuntu|debian)
                curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
                sudo apt-get install -y nodejs
                ;;
            fedora|centos|rhel)
                curl -fsSL https://rpm.nodesource.com/setup_18.x | sudo bash -
                sudo dnf install -y nodejs || sudo yum install -y nodejs
                ;;
            arch|manjaro)
                sudo pacman -S --noconfirm nodejs npm
                ;;
            *)
                print_warning "Please install Node.js manually for additional client support"
                ;;
        esac
    else
        print_status "Node.js already installed: $(node --version)"
    fi
}

install_nodejs

# Install client-specific dependencies

# Cursor (AppImage)
install_cursor() {
    print_status "Downloading Cursor AppImage..."
    CURSOR_URL="https://downloader.cursor.sh/linux/appImage/x64"
    wget -O cursor.AppImage "$CURSOR_URL" || print_warning "Failed to download Cursor"
    
    if [ -f cursor.AppImage ]; then
        chmod +x cursor.AppImage
        mkdir -p ~/.local/bin
        mv cursor.AppImage ~/.local/bin/cursor
        print_status "Cursor installed to ~/.local/bin/cursor"
    fi
}

# Claude Desktop (not officially supported on Linux)
print_warning "Claude Desktop is not officially supported on Linux"
print_warning "Use Claude Code CLI or Cursor instead"

# Offer to install Cursor
read -p "Install Cursor AppImage? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    install_cursor
fi

# OpenAI ChatGPT bridge dependencies
print_status "Installing OpenAI dependencies for ChatGPT bridge..."
pip install openai>=1.0.0

# Test MuJoCo installation
print_status "Testing MuJoCo installation..."
python3 -c "import mujoco; print('MuJoCo version:', mujoco.__version__)"

# Test OpenGL
print_status "Testing OpenGL..."
python3 -c "
import os
os.environ['MUJOCO_GL'] = 'egl'
try:
    import mujoco
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><geom size=\"1\"/></body></worldbody></mujoco>')
    print('OpenGL test passed with EGL')
except Exception as e:
    print('EGL failed, trying osmesa...')
    os.environ['MUJOCO_GL'] = 'osmesa'
    try:
        import mujoco
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><geom size=\"1\"/></body></worldbody></mujoco>')
        print('OpenGL test passed with osmesa')
    except Exception as e:
        print(f'Graphics test failed: {e}')
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

# Cursor config (if installed)
if command -v cursor &> /dev/null || [ -f ~/.local/bin/cursor ]; then
    mkdir -p ~/.config/cursor/User
    if [ ! -f ~/.config/cursor/User/settings.json ]; then
        cp mcp-testing/configs/cursor/settings.json ~/.config/cursor/User/settings.json
        print_status "Cursor configuration installed"
    fi
fi

# Make scripts executable
chmod +x mcp-testing/scripts/*.sh

# Set environment variables
print_status "Setting up environment variables..."

# Determine best graphics backend
GRAPHICS_BACKEND="egl"
if ! ldconfig -p | grep -q libEGL; then
    GRAPHICS_BACKEND="osmesa"
    print_warning "EGL not found, using osmesa backend"
fi

echo "export MUJOCO_GL=$GRAPHICS_BACKEND" >> ~/.bashrc
echo "export MUJOCO_MCP_LOG_LEVEL=INFO" >> ~/.bashrc
echo "export PYTHONPATH=$PWD/src:\$PYTHONPATH" >> ~/.bashrc
echo "export PATH=\$HOME/.local/bin:\$PATH" >> ~/.bashrc

# For zsh users
if [ "$SHELL" = "/usr/bin/zsh" ] || [ "$SHELL" = "/bin/zsh" ]; then
    echo "export MUJOCO_GL=$GRAPHICS_BACKEND" >> ~/.zshrc
    echo "export MUJOCO_MCP_LOG_LEVEL=INFO" >> ~/.zshrc  
    echo "export PYTHONPATH=$PWD/src:\$PYTHONPATH" >> ~/.zshrc
    echo "export PATH=\$HOME/.local/bin:\$PATH" >> ~/.zshrc
fi

print_status "✅ Linux setup complete!"
print_status ""
print_status "Next steps:"
print_status "1. Restart your terminal or run: source ~/.bashrc"
print_status "2. Activate the virtual environment: source venv-mcp-testing/bin/activate"
print_status "3. Test with: ./mcp-testing/scripts/test-client.sh cursor"
print_status ""
print_status "Manual installations available:"
print_status "- Cursor: Already downloaded (if selected)"
print_status "- Claude Code CLI: Check https://github.com/anthropics/claude-code"
print_status ""
print_warning "Remember to set OPENAI_API_KEY if using ChatGPT bridge"

# Display system info
print_status ""
print_status "System Information:"
print_status "OS: $OS $VER"
print_status "Graphics Backend: $GRAPHICS_BACKEND"
print_status "Python: $(python3 --version)"
print_status "Architecture: $(uname -m)"
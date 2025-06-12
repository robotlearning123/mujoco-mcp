#!/usr/bin/env python3
"""
Verify Claude Desktop Configuration
Final validation of the MuJoCo MCP setup
"""
import json
import subprocess
import sys
from pathlib import Path


def verify_configuration():
    """Verify the current Claude Desktop configuration"""
    print("🔍 Verifying Claude Desktop MCP Configuration")
    print("=" * 60)
    
    # Read configuration
    config_path = Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("✅ Configuration file is valid JSON")
    except Exception as e:
        print(f"❌ Configuration file error: {e}")
        return False
    
    # Check MuJoCo MCP configuration
    if "mujoco-mcp" not in config.get("mcpServers", {}):
        print("❌ mujoco-mcp not found in configuration")
        return False
    
    mujoco_config = config["mcpServers"]["mujoco-mcp"]
    print("✅ mujoco-mcp configuration found")
    
    # Display configuration
    print(f"\n📋 Current Configuration:")
    print(f"   Command: {mujoco_config['command']}")
    print(f"   Args: {mujoco_config['args']}")
    print(f"   Working Directory: {mujoco_config['cwd']}")
    print(f"   Environment Variables: {len(mujoco_config.get('env', {}))}")
    
    # Check file existence
    command = mujoco_config["command"]
    script = mujoco_config["args"][0]
    cwd = Path(mujoco_config["cwd"])
    script_path = cwd / script
    
    if not Path(command).exists():
        print(f"❌ Python executable not found: {command}")
        return False
    print(f"✅ Python executable exists: {command}")
    
    if not script_path.exists():
        print(f"❌ Server script not found: {script_path}")
        return False
    print(f"✅ Server script exists: {script_path}")
    
    if not cwd.exists():
        print(f"❌ Working directory not found: {cwd}")
        return False
    print(f"✅ Working directory exists: {cwd}")
    
    return True


def test_basic_import():
    """Test basic import capabilities"""
    print(f"\n🐍 Testing Python Environment")
    print("=" * 40)
    
    command = "/opt/miniconda3/bin/python"
    env = {
        "PYTHONPATH": "/Users/robert/workspace/04-code/03-mujoco/mujoco-mcp/src",
        "PATH": "/opt/miniconda3/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
    }
    
    # Test imports
    imports_to_test = [
        ("import sys; print(f'Python {sys.version}')", "Python version"),
        ("import mujoco; print(f'MuJoCo {mujoco.__version__}')", "MuJoCo"),
        ("import mcp; print('MCP available')", "MCP"),
        ("import numpy; print(f'NumPy {numpy.__version__}')", "NumPy"),
    ]
    
    for test_code, description in imports_to_test:
        try:
            result = subprocess.run([command, '-c', test_code], 
                                  capture_output=True, text=True, timeout=10,
                                  env={**dict(subprocess.os.environ), **env})
            if result.returncode == 0:
                print(f"✅ {description}: {result.stdout.strip()}")
            else:
                print(f"❌ {description}: {result.stderr.strip()}")
                return False
        except Exception as e:
            print(f"❌ {description}: Error - {e}")
            return False
    
    return True


def print_next_steps():
    """Print next steps for user"""
    print(f"\n🚀 Next Steps")
    print("=" * 30)
    print("1. **Completely quit Claude Desktop**: Cmd+Q")
    print("2. **Wait 5 seconds**: Ensure all processes stop")
    print("3. **Reopen Claude Desktop**: Fresh start")
    print("4. **Wait for initialization**: May take 30-60 seconds")
    print("5. **Start new conversation**: Click 'New Chat'")
    print("6. **Test connection**: 'What MCP servers are available?'")
    
    print(f"\n🧪 Test Commands")
    print("=" * 20)
    print("• 'What MCP servers are connected?'")
    print("• 'Create a pendulum simulation'")
    print("• 'Show me the current simulation state'")
    print("• 'Move the pendulum to 45 degrees'")
    
    print(f"\n🔍 Troubleshooting")
    print("=" * 20)
    print("• Check logs: ~/Library/Logs/Claude/mcp-server-mujoco.log")
    print("• If issues persist, the server may need stdio mode adjustments")
    print("• MCP protocol requires exact stdio communication")


def main():
    """Main verification function"""
    print("🩺 Claude Desktop MuJoCo MCP Configuration Verification")
    print("=" * 70)
    
    # Step 1: Verify configuration
    if not verify_configuration():
        print("\n❌ Configuration verification failed!")
        return False
    
    # Step 2: Test Python environment
    if not test_basic_import():
        print("\n❌ Python environment test failed!")
        return False
    
    print(f"\n🎉 Configuration Verification Complete!")
    print("=" * 50)
    print("✅ Configuration file: Valid JSON")
    print("✅ MuJoCo MCP: Properly configured")
    print("✅ Python executable: Found and working")
    print("✅ Server script: Exists")
    print("✅ Dependencies: All available")
    print("✅ Environment: Properly set")
    
    print_next_steps()
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
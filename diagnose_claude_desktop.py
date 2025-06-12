#!/usr/bin/env python3
"""
Diagnose Claude Desktop MCP Issues
Identifies and fixes common Claude Desktop MCP configuration problems
"""
import json
import subprocess
import sys
import shutil
from pathlib import Path


def check_python_executable():
    """Check Python executable paths"""
    print("🐍 Python Executable Check")
    print("=" * 40)
    
    # Check different Python variants
    python_variants = ['python', 'python3', '/opt/miniconda3/bin/python', '/opt/homebrew/bin/python3']
    
    working_pythons = []
    
    for python_cmd in python_variants:
        try:
            if python_cmd.startswith('/'):
                # Absolute path
                if Path(python_cmd).exists():
                    result = subprocess.run([python_cmd, '--version'], capture_output=True, text=True)
                    if result.returncode == 0:
                        print(f"✅ {python_cmd}: {result.stdout.strip()}")
                        working_pythons.append(python_cmd)
                    else:
                        print(f"❌ {python_cmd}: Not working")
                else:
                    print(f"❌ {python_cmd}: File not found")
            else:
                # Command in PATH
                if shutil.which(python_cmd):
                    result = subprocess.run([python_cmd, '--version'], capture_output=True, text=True)
                    if result.returncode == 0:
                        full_path = shutil.which(python_cmd)
                        print(f"✅ {python_cmd} ({full_path}): {result.stdout.strip()}")
                        working_pythons.append(full_path)
                    else:
                        print(f"❌ {python_cmd}: Not working")
                else:
                    print(f"❌ {python_cmd}: Not found in PATH")
        except Exception as e:
            print(f"❌ {python_cmd}: Error - {e}")
    
    return working_pythons


def test_mujoco_import(python_cmd):
    """Test if MuJoCo can be imported with specific Python"""
    try:
        # Test MuJoCo import
        result = subprocess.run([
            python_cmd, '-c', 
            'import mujoco; import mcp; print(f"MuJoCo: {mujoco.__version__}, MCP available")'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print(f"✅ {python_cmd}: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ {python_cmd}: Import failed - {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ {python_cmd}: Error - {e}")
        return False


def test_mujoco_mcp_module(python_cmd):
    """Test if mujoco_mcp module can be loaded"""
    try:
        # Test mujoco_mcp module
        result = subprocess.run([
            python_cmd, '-m', 'mujoco_mcp', '--version'
        ], capture_output=True, text=True, timeout=10,
        cwd="/Users/robert/workspace/04-code/03-mujoco/mujoco-mcp",
        env={
            **dict(subprocess.os.environ),
            'PYTHONPATH': '/Users/robert/workspace/04-code/03-mujoco/mujoco-mcp/src'
        })
        
        if result.returncode == 0:
            print(f"✅ {python_cmd}: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ {python_cmd}: Module test failed - {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ {python_cmd}: Error - {e}")
        return False


def find_best_python():
    """Find the best Python executable for Claude Desktop"""
    print("\n🔍 Finding Best Python for Claude Desktop")
    print("=" * 50)
    
    working_pythons = check_python_executable()
    
    if not working_pythons:
        print("❌ No working Python installations found!")
        return None
    
    print(f"\n🧪 Testing MuJoCo/MCP imports...")
    print("=" * 40)
    
    best_python = None
    for python_cmd in working_pythons:
        if test_mujoco_import(python_cmd):
            print(f"\n🔧 Testing mujoco_mcp module with {python_cmd}...")
            if test_mujoco_mcp_module(python_cmd):
                best_python = python_cmd
                break
    
    if best_python:
        print(f"\n🎯 Best Python for Claude Desktop: {best_python}")
    else:
        print("\n❌ No Python installation can run MuJoCo MCP!")
    
    return best_python


def update_claude_desktop_config(python_path):
    """Update Claude Desktop configuration with correct Python path"""
    print(f"\n📝 Updating Claude Desktop Configuration")
    print("=" * 50)
    
    config_path = Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    
    if not config_path.exists():
        print(f"❌ Claude Desktop config not found: {config_path}")
        return False
    
    try:
        # Read current config
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Update mujoco-mcp configuration
        if "mcpServers" not in config:
            config["mcpServers"] = {}
        
        config["mcpServers"]["mujoco-mcp"] = {
            "command": python_path,
            "args": ["-m", "mujoco_mcp"],
            "cwd": "/Users/robert/workspace/04-code/03-mujoco/mujoco-mcp",
            "env": {
                "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": "/Users/robert/workspace/04-code/03-mujoco/mujoco-mcp/src",
                "MUJOCO_MCP_LOG_LEVEL": "INFO",
                "PATH": "/opt/miniconda3/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
            }
        }
        
        # Write updated config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Updated Claude Desktop config with Python: {python_path}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to update config: {e}")
        return False


def clear_claude_logs():
    """Clear Claude Desktop logs to get fresh start"""
    print(f"\n🧹 Clearing Claude Desktop Logs")
    print("=" * 40)
    
    log_path = Path.home() / "Library" / "Logs" / "Claude" / "mcp-server-mujoco.log"
    
    if log_path.exists():
        try:
            log_path.unlink()
            print("✅ Cleared old MuJoCo MCP logs")
        except Exception as e:
            print(f"⚠️  Could not clear logs: {e}")
    else:
        print("ℹ️  No existing logs found")


def test_final_configuration():
    """Test the final configuration"""
    print(f"\n🧪 Testing Final Configuration")
    print("=" * 40)
    
    config_path = Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        mujoco_config = config.get("mcpServers", {}).get("mujoco-mcp", {})
        
        if not mujoco_config:
            print("❌ mujoco-mcp not found in configuration")
            return False
        
        python_cmd = mujoco_config.get("command")
        cwd = mujoco_config.get("cwd")
        env = mujoco_config.get("env", {})
        
        print(f"Python command: {python_cmd}")
        print(f"Working directory: {cwd}")
        print(f"PYTHONPATH: {env.get('PYTHONPATH', 'Not set')}")
        
        # Test the exact command Claude Desktop would run
        test_env = {**dict(subprocess.os.environ), **env}
        
        result = subprocess.run([
            python_cmd, '-m', 'mujoco_mcp', '--check'
        ], capture_output=True, text=True, timeout=30, cwd=cwd, env=test_env)
        
        if result.returncode == 0:
            print("✅ Configuration test passed!")
            print(f"Output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Configuration test failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def main():
    """Main diagnostic function"""
    print("🩺 Claude Desktop MuJoCo MCP Diagnostic Tool")
    print("=" * 60)
    
    # Step 1: Find best Python
    best_python = find_best_python()
    
    if not best_python:
        print("\n❌ Cannot proceed - no working Python found!")
        print("\nTroubleshooting steps:")
        print("1. Install MuJoCo: pip install mujoco")
        print("2. Install MCP: pip install mcp")
        print("3. Install MuJoCo MCP: cd /path/to/mujoco-mcp && pip install -e .")
        return False
    
    # Step 2: Update configuration
    if not update_claude_desktop_config(best_python):
        print("\n❌ Failed to update configuration!")
        return False
    
    # Step 3: Clear old logs
    clear_claude_logs()
    
    # Step 4: Test final configuration
    if not test_final_configuration():
        print("\n❌ Final configuration test failed!")
        return False
    
    print("\n🎉 Diagnostic Complete - Claude Desktop Should Work Now!")
    print("=" * 60)
    print("Next steps:")
    print("1. Completely quit Claude Desktop (Cmd+Q)")
    print("2. Reopen Claude Desktop")
    print("3. Start a new conversation")
    print("4. Try: 'What MCP servers are connected?'")
    print("5. Try: 'Create a pendulum simulation'")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
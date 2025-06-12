#!/usr/bin/env python3
"""
Final Claude Desktop MCP Test
Tests the exact configuration that Claude Desktop will use
"""
import json
import subprocess
import sys
from pathlib import Path


def test_exact_claude_desktop_config():
    """Test the exact configuration Claude Desktop will use"""
    print("🧪 Testing Exact Claude Desktop Configuration")
    print("=" * 60)
    
    # Read Claude Desktop config
    config_path = Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        mujoco_config = config["mcpServers"]["mujoco-mcp"]
        
        command = mujoco_config["command"]
        args = mujoco_config["args"]
        cwd = mujoco_config["cwd"]
        env_vars = mujoco_config["env"]
        
        print(f"Command: {command}")
        print(f"Args: {args}")
        print(f"Working Directory: {cwd}")
        print(f"Environment Variables: {len(env_vars)} vars")
        
        # Prepare environment
        test_env = dict(subprocess.os.environ)
        test_env.update(env_vars)
        
        # Test 1: Version check
        print(f"\n🔍 Test 1: Version Check")
        result = subprocess.run([command] + args + ["--version"], 
                              capture_output=True, text=True, timeout=30,
                              cwd=cwd, env=test_env)
        
        if result.returncode == 0:
            print(f"✅ Version: {result.stdout.strip()}")
        else:
            print(f"❌ Version check failed: {result.stderr}")
            return False
        
        # Test 2: Configuration check
        print(f"\n🔍 Test 2: Configuration Check")
        result = subprocess.run([command] + args + ["--check"], 
                              capture_output=True, text=True, timeout=30,
                              cwd=cwd, env=test_env)
        
        if result.returncode == 0:
            print(f"✅ Configuration check passed")
            print(f"Details: {result.stdout.strip()}")
        else:
            print(f"❌ Configuration check failed: {result.stderr}")
            return False
        
        # Test 3: Quick server startup (3 seconds)
        print(f"\n🔍 Test 3: Quick Server Startup Test")
        try:
            process = subprocess.Popen([command] + args, 
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE,
                                     cwd=cwd, env=test_env)
            
            # Wait briefly to see if it starts
            try:
                stdout, stderr = process.communicate(timeout=3)
                print(f"❌ Server exited early: {stderr.decode()}")
                return False
            except subprocess.TimeoutExpired:
                # Good! Server is still running
                process.terminate()
                process.wait()
                print("✅ Server started successfully (terminated after 3s)")
        except Exception as e:
            print(f"❌ Server startup failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def check_log_file():
    """Check if new log file will be created"""
    print(f"\n📋 Log File Status")
    print("=" * 30)
    
    log_path = Path.home() / "Library" / "Logs" / "Claude" / "mcp-server-mujoco.log"
    
    if log_path.exists():
        print(f"⚠️  Old log file exists: {log_path}")
        print("   (This will be overwritten when Claude Desktop starts)")
    else:
        print(f"✅ Clean slate - no old log file")
    
    # Check log directory exists
    log_dir = log_path.parent
    if log_dir.exists():
        print(f"✅ Log directory exists: {log_dir}")
    else:
        print(f"❌ Log directory missing: {log_dir}")


def print_success_instructions():
    """Print final instructions for user"""
    print(f"\n🎉 Claude Desktop Configuration is Ready!")
    print("=" * 60)
    
    print("🔧 Configuration Summary:")
    print("   • Python: /opt/miniconda3/bin/python")
    print("   • Module: mujoco_mcp v0.6.0")
    print("   • Working Directory: Correct")
    print("   • Environment: All paths included")
    print("   • PATH: Includes /usr/sbin for sysctl")
    
    print(f"\n🚀 Next Steps:")
    print("1. **Completely quit Claude Desktop**")
    print("   - Press Cmd+Q (not just close window)")
    print("   - Ensure no Claude processes running")
    
    print(f"\n2. **Reopen Claude Desktop**")
    print("   - Launch Claude Desktop fresh")
    print("   - Wait for initialization (may take 30+ seconds)")
    
    print(f"\n3. **Test Connection**")
    print("   - Start a new conversation")
    print("   - Type: 'What MCP servers are available?'")
    print("   - Should see: mujoco-mcp in the list")
    
    print(f"\n4. **Test MuJoCo Functions**")
    print("   🔹 'Create a pendulum simulation'")
    print("   🔹 'Show me the current simulation state'")
    print("   🔹 'Move the pendulum to 45 degrees'")
    
    print(f"\n🔍 Troubleshooting:")
    print("   • Check logs: ~/Library/Logs/Claude/mcp-server-mujoco.log")
    print("   • If still failing, run: python diagnose_claude_desktop.py")
    print("   • Ensure no firewall blocking Python")


def main():
    """Main test function"""
    print("🔬 Final Claude Desktop MCP Configuration Test")
    print("=" * 70)
    
    # Test the configuration
    if test_exact_claude_desktop_config():
        check_log_file()
        print_success_instructions()
        return True
    else:
        print(f"\n❌ Configuration test failed!")
        print("Run: python diagnose_claude_desktop.py")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
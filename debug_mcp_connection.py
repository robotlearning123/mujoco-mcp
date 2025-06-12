#!/usr/bin/env python3
"""
MCP Connection Debugger for Claude Desktop
Based on MCP debugging documentation
"""
import sys
import os
import json
import subprocess
import time
import signal
from pathlib import Path

def test_server_directly():
    """Test the MCP server directly with JSON-RPC messages"""
    print("🔍 Testing MCP server directly...")
    
    # Test initialize
    init_msg = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "debug-client", "version": "1.0"}
        }
    }
    
    try:
        result = subprocess.run([
            "/opt/miniconda3/bin/python", 
            "mcp_server_stdio_corrected.py"
        ], 
        input=json.dumps(init_msg) + "\n",
        capture_output=True,
        text=True,
        timeout=10,
        cwd="/Users/robert/workspace/04-code/03-mujoco/mujoco-mcp"
        )
        
        if result.returncode == 0:
            print("✅ Server initialization: SUCCESS")
            try:
                response = json.loads(result.stdout.strip())
                print(f"   Protocol version: {response['result']['protocolVersion']}")
                print(f"   Server name: {response['result']['serverInfo']['name']}")
                print(f"   Server version: {response['result']['serverInfo']['version']}")
            except:
                print(f"   Raw response: {result.stdout[:200]}...")
        else:
            print("❌ Server initialization: FAILED")
            print(f"   Error: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("❌ Server initialization: TIMEOUT")
    except Exception as e:
        print(f"❌ Server initialization: ERROR - {e}")

def test_tools_list():
    """Test tools/list method"""
    print("\n🔍 Testing tools/list...")
    
    # Test tools list
    tools_msg = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list"
    }
    
    try:
        result = subprocess.run([
            "/opt/miniconda3/bin/python", 
            "mcp_server_stdio_corrected.py"
        ], 
        input=json.dumps(tools_msg) + "\n",
        capture_output=True,
        text=True,
        timeout=10,
        cwd="/Users/robert/workspace/04-code/03-mujoco/mujoco-mcp"
        )
        
        if result.returncode == 0:
            print("✅ Tools list: SUCCESS")
            try:
                response = json.loads(result.stdout.strip())
                tools = response['result']['tools']
                print(f"   Available tools: {len(tools)}")
                print(f"   First 3 tools: {[t['name'] for t in tools[:3]]}")
            except:
                print(f"   Raw response: {result.stdout[:200]}...")
        else:
            print("❌ Tools list: FAILED")
            print(f"   Error: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Tools list: ERROR - {e}")

def check_configuration():
    """Check Claude Desktop configuration"""
    print("\n🔍 Checking Claude Desktop configuration...")
    
    config_path = Path.home() / "Library/Application Support/Claude/claude_desktop_config.json"
    
    if not config_path.exists():
        print("❌ Configuration file not found")
        return
        
    try:
        with open(config_path) as f:
            config = json.load(f)
            
        if "mcpServers" not in config:
            print("❌ No mcpServers section in config")
            return
            
        if "mujoco-mcp" not in config["mcpServers"]:
            print("❌ mujoco-mcp not found in config")
            return
            
        mcp_config = config["mcpServers"]["mujoco-mcp"]
        print("✅ Configuration found:")
        print(f"   Command: {mcp_config['command']}")
        print(f"   Args: {mcp_config['args']}")
        print(f"   Working dir: {mcp_config.get('cwd', 'not set')}")
        
        # Check if files exist
        script_path = mcp_config['args'][0]
        if os.path.exists(script_path):
            print(f"✅ Server script exists: {script_path}")
        else:
            print(f"❌ Server script missing: {script_path}")
            
        if 'cwd' in mcp_config and os.path.exists(mcp_config['cwd']):
            print(f"✅ Working directory exists: {mcp_config['cwd']}")
        else:
            print(f"❌ Working directory missing: {mcp_config.get('cwd', 'not set')}")
            
    except Exception as e:
        print(f"❌ Configuration error: {e}")

def check_processes():
    """Check for running processes"""
    print("\n🔍 Checking running processes...")
    
    try:
        # Check Claude Desktop
        result = subprocess.run(["pgrep", "-f", "Claude"], capture_output=True, text=True)
        if result.stdout.strip():
            print("✅ Claude Desktop is running")
        else:
            print("❌ Claude Desktop is not running")
            
        # Check MCP server
        result = subprocess.run(["pgrep", "-f", "mcp_server"], capture_output=True, text=True)
        if result.stdout.strip():
            print("✅ MCP server process found")
            pids = result.stdout.strip().split('\n')
            print(f"   PIDs: {pids}")
        else:
            print("⚠️  No MCP server process found")
            
    except Exception as e:
        print(f"❌ Process check error: {e}")

def check_logs():
    """Check Claude Desktop logs"""
    print("\n🔍 Checking Claude Desktop logs...")
    
    log_dir = Path.home() / "Library/Logs/Claude"
    if not log_dir.exists():
        print("❌ Claude logs directory not found")
        return
        
    mcp_logs = list(log_dir.glob("mcp-server-*.log"))
    if not mcp_logs:
        print("⚠️  No MCP server logs found")
        return
        
    print(f"✅ Found {len(mcp_logs)} MCP log file(s):")
    for log_file in mcp_logs:
        print(f"   {log_file.name}")
        
        # Check last few lines
        try:
            with open(log_file) as f:
                lines = f.readlines()
                if lines:
                    print(f"   Last log entry: {lines[-1].strip()}")
        except Exception as e:
            print(f"   Error reading log: {e}")

def restart_claude_desktop():
    """Restart Claude Desktop completely"""
    print("\n🔄 Restarting Claude Desktop...")
    
    try:
        # Kill all Claude processes
        subprocess.run(["pkill", "-f", "Claude"], stderr=subprocess.DEVNULL)
        time.sleep(3)
        
        # Start Claude Desktop
        subprocess.Popen([
            "open", "-a", "Claude"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print("✅ Claude Desktop restart initiated")
        print("   Wait 10-15 seconds for full startup...")
        
    except Exception as e:
        print(f"❌ Restart error: {e}")

def main():
    """Main debugging routine"""
    print("🔧 MCP Connection Debugger for Claude Desktop")
    print("=" * 50)
    
    check_configuration()
    check_processes()
    test_server_directly()
    test_tools_list()
    check_logs()
    
    print("\n" + "=" * 50)
    print("🎯 Debugging Summary:")
    print("1. If server tests pass but Claude Desktop doesn't connect:")
    print("   - Restart Claude Desktop completely")
    print("   - Clear log files and try again")
    print("2. If server tests fail:")
    print("   - Check PYTHONPATH and dependencies")
    print("   - Verify MuJoCo installation")
    print("3. Check Chrome DevTools in Claude Desktop for more details")
    
    print("\n🚀 Would you like me to restart Claude Desktop? (y/n)")
    response = input().strip().lower()
    if response == 'y':
        restart_claude_desktop()

if __name__ == "__main__":
    main()
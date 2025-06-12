#!/usr/bin/env python3
"""
Setup script for Cursor MCP integration
Automatically configures Cursor to use MuJoCo MCP v0.6.0
"""
import json
import os
import shutil
from pathlib import Path


def get_cursor_config_path():
    """Get the Cursor MCP configuration file path"""
    home = Path.home()
    cursor_config = home / ".cursor" / "mcp.json"
    return cursor_config


def backup_existing_config(config_path):
    """Create a backup of existing configuration"""
    if config_path.exists():
        backup_path = config_path.with_suffix('.json.backup')
        shutil.copy2(config_path, backup_path)
        print(f"✅ Backed up existing config to: {backup_path}")
        return True
    return False


def load_current_config(config_path):
    """Load current Cursor MCP configuration"""
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("⚠️  Warning: Invalid JSON in existing config, creating new one")
            return {}
    return {}


def update_mujoco_config(config, project_path):
    """Update the mujoco-mcp configuration"""
    config["mujoco-mcp"] = {
        "command": "python",
        "args": ["-m", "mujoco_mcp"],
        "cwd": str(project_path),
        "env": {
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(project_path / "src"),
            "MUJOCO_MCP_LOG_LEVEL": "INFO"
        }
    }
    
    # Remove old mujoco config if it exists
    if "mujoco" in config:
        del config["mujoco"]
        print("🗑️  Removed old mujoco configuration")
    
    # Also check if it's nested in figma (malformed JSON structure)
    if "figma" in config and isinstance(config["figma"], dict):
        if "mujoco" in config["figma"]:
            del config["figma"]["mujoco"]
            print("🗑️  Removed nested mujoco configuration from figma")
    
    return config


def save_config(config_path, config):
    """Save the updated configuration"""
    # Ensure .cursor directory exists
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Updated Cursor MCP configuration: {config_path}")


def verify_installation():
    """Verify MuJoCo MCP installation"""
    try:
        import subprocess
        result = subprocess.run(
            ["python", "-m", "mujoco_mcp", "--check"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ MuJoCo MCP installation verified")
            return True
        else:
            print(f"❌ MuJoCo MCP check failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Installation check failed: {e}")
        return False


def main():
    """Main setup function"""
    print("🚀 Setting up Cursor MCP integration for MuJoCo MCP v0.6.0")
    print("=" * 60)
    
    # Get project path
    project_path = Path(__file__).parent.absolute()
    print(f"📁 Project path: {project_path}")
    
    # Get Cursor config path
    config_path = get_cursor_config_path()
    print(f"⚙️  Cursor config: {config_path}")
    
    # Verify installation first
    print("\n🔍 Verifying MuJoCo MCP installation...")
    if not verify_installation():
        print("\n❌ Installation verification failed!")
        print("Please run: pip install -e .")
        return False
    
    # Backup existing config
    print("\n💾 Managing configuration...")
    backup_existing_config(config_path)
    
    # Load and update configuration
    config = load_current_config(config_path)
    config = update_mujoco_config(config, project_path)
    
    # Save updated configuration
    save_config(config_path, config)
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Restart Cursor completely")
    print("2. Open a new chat/session")
    print("3. Try: 'Create a pendulum simulation using MuJoCo'")
    
    print(f"\n📖 For detailed setup guide, see: {project_path}/CURSOR_SETUP.md")
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
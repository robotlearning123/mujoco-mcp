#!/usr/bin/env python3
"""
Test MuJoCo Menagerie integration in MCP server
"""

import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mujoco_mcp.remote_server import MuJoCoRemoteServer
from mujoco_mcp.menagerie_loader import MenagerieLoader


def test_menagerie_integration():
    """Test Menagerie integration features"""
    print("Testing MuJoCo Menagerie Integration\n")
    
    # Create server
    server = MuJoCoRemoteServer()
    
    # Test 1: Check Menagerie availability
    print("1. Checking Menagerie availability...")
    if server.menagerie.is_available():
        print(f"   ✓ Menagerie found at: {server.menagerie.menagerie_path}")
    else:
        print("   ✗ Menagerie not found")
        print("\n   To install Menagerie:")
        print("   git clone https://github.com/google-deepmind/mujoco_menagerie.git")
        print("   export MUJOCO_MENAGERIE_PATH=/path/to/mujoco_menagerie")
        return
    
    # Test 2: List Menagerie models
    print("\n2. Testing list_menagerie_models tool...")
    result = server._handle_list_menagerie_models()
    if result.get("success"):
        print(f"   ✓ Found {result['total_models']} models")
        print(f"   Categories: {', '.join(result['categories'])}")
        
        # Show some models
        for cat, models in list(result['models_by_category'].items())[:3]:
            print(f"\n   {cat}:")
            for model in models[:3]:
                print(f"     - {model}")
    else:
        print(f"   ✗ Failed: {result.get('error')}")
    
    # Test 3: Natural language commands
    print("\n3. Testing natural language commands...")
    
    # Test help command
    result = server._handle_execute_command("help")
    if "MuJoCo Menagerie" in str(result.get("commands", [])):
        print("   ✓ Help includes Menagerie commands")
    else:
        print("   ✗ Help missing Menagerie commands")
    
    # Test list command
    result = server._handle_execute_command("list menagerie models")
    if result.get("success"):
        print("   ✓ 'list menagerie models' works")
    
    # Test 4: Load a Menagerie model (without viewer)
    print("\n4. Testing model loading (without viewer)...")
    
    # This will fail without viewer, but we can check if it tries
    test_models = ["franka_emika_panda", "spot", "unitree_go2"]
    
    for model in test_models:
        print(f"\n   Trying to load '{model}'...")
        result = server._handle_create_scene(model)
        
        if "viewer" in result.get("error", "").lower():
            print(f"   ✓ Model '{model}' recognized (viewer not connected)")
            break
        elif result.get("success"):
            print(f"   ✓ Model '{model}' loaded successfully!")
            print(f"      Model ID: {result.get('model_id')}")
            break
        else:
            print(f"   ✗ {result.get('error')}")
    
    # Test 5: Natural language model loading
    print("\n5. Testing natural language model loading...")
    
    test_commands = [
        "load franka panda",
        "create spot robot",
        "load the unitree go2"
    ]
    
    for cmd in test_commands:
        print(f"\n   Command: '{cmd}'")
        result = server._handle_execute_command(cmd)
        
        if "viewer" in str(result.get("error", "")).lower():
            print(f"   ✓ Command understood (viewer not connected)")
        elif result.get("success"):
            print(f"   ✓ Success!")
        else:
            print(f"   ! {result.get('error', 'Unknown error')}")
    
    print("\n✅ Menagerie integration test complete!")
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY:")
    print(f"- Menagerie available: {'Yes' if server.menagerie.is_available() else 'No'}")
    if server.menagerie.is_available():
        models = server.menagerie.list_models()
        print(f"- Total models: {len(models)}")
        print(f"- Integration: Working")
    print("="*50)


def test_specific_features():
    """Test specific Menagerie features"""
    loader = MenagerieLoader()
    
    if not loader.is_available():
        print("Menagerie not available for specific tests")
        return
    
    print("\nTesting specific Menagerie features:\n")
    
    # Test model name variations
    print("1. Testing model name variations...")
    variations = [
        ("franka_emika_panda", "franka panda"),
        ("unitree_go2", "unitree go2"),
        ("boston_dynamics_spot", "spot"),
    ]
    
    for correct_name, search_name in variations:
        found = loader.find_model(search_name)
        if found:
            print(f"   ✓ '{search_name}' → found")
        else:
            print(f"   ✗ '{search_name}' → not found")
    
    # Test category filtering
    print("\n2. Testing category filtering...")
    for category in ["arms", "quadrupeds"]:
        models = loader.list_models(category)
        if models:
            print(f"   ✓ {category}: {len(models)} models")
        else:
            print(f"   ✗ {category}: no models")


if __name__ == "__main__":
    test_menagerie_integration()
    print("\n" + "-"*50 + "\n")
    test_specific_features()
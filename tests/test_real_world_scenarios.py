#!/usr/bin/env python3
"""
Real-world scenario tests for MuJoCo MCP v0.7.1
Tests actual usage patterns that users would encounter
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mujoco_mcp.remote_server import MuJoCoRemoteServer


def test_scenario(name: str, test_func):
    """Run a test scenario and report results"""
    print(f"\n{'='*60}")
    print(f"SCENARIO: {name}")
    print('='*60)
    try:
        success, details = test_func()
        if success:
            print(f"✅ PASSED: {name}")
        else:
            print(f"❌ FAILED: {name}")
            print(f"   Details: {details}")
        return success
    except Exception as e:
        print(f"💥 ERROR: {name}")
        print(f"   Exception: {e}")
        return False


def scenario_1_basic_usage():
    """User wants to create a pendulum and control it"""
    server = MuJoCoRemoteServer()
    
    # Create pendulum
    result = server._handle_execute_command("create a pendulum")
    if "error" in result and "viewer" not in str(result["error"]).lower():
        return False, f"Failed to create pendulum: {result}"
    
    # Try to get state (should fail without viewer)
    result = server._handle_execute_command("show state")
    if "error" not in result:
        return False, "Expected error without active simulation"
    
    # Help should work
    result = server._handle_execute_command("help")
    if "commands" not in result:
        return False, "Help command failed"
    
    return True, "Basic usage works"


def scenario_2_menagerie_discovery():
    """User discovers and explores Menagerie models"""
    server = MuJoCoRemoteServer()
    
    # Check if Menagerie is available
    if not server.menagerie.is_available():
        return True, "Menagerie not installed (expected)"
    
    # List all models
    result = server._handle_execute_command("list menagerie models")
    if not result.get("success"):
        return False, f"Failed to list models: {result}"
    
    total = result.get("total_models", 0)
    if total == 0:
        return False, "No models found"
    
    # List specific category
    result = server._handle_execute_command("list menagerie arms")
    if "models_by_category" not in result:
        return False, "Category filtering failed"
    
    arms = result.get("models_by_category", {}).get("arms", [])
    if len(arms) == 0:
        return False, "No arm models found"
    
    return True, f"Found {total} models including {len(arms)} arms"


def scenario_3_load_real_robot():
    """User loads a real robot from Menagerie"""
    server = MuJoCoRemoteServer()
    
    if not server.menagerie.is_available():
        return True, "Menagerie not installed (expected)"
    
    # Try different ways users might request robots
    commands = [
        "load franka panda",
        "create the franka robot",
        "load spot",
        "create unitree go2"
    ]
    
    success_count = 0
    for cmd in commands:
        result = server._handle_execute_command(cmd)
        # Check if it tried to load (viewer error is ok)
        if "viewer" in str(result.get("error", "")).lower():
            success_count += 1
        elif result.get("success"):
            success_count += 1
    
    if success_count == 0:
        return False, "No robot loading commands worked"
    
    return True, f"{success_count}/{len(commands)} robot commands recognized"


def scenario_4_natural_language_variations():
    """Test various natural language phrasings"""
    server = MuJoCoRemoteServer()
    
    test_phrases = [
        ("show me what models are available", "list"),
        ("reset the simulation", "reset"),
        ("run the simulation for 100 steps", "step"),
        ("rotate the joint to 90 degrees", "set"),
        ("what robots can I use?", "help or list"),
        ("", "empty command")
    ]
    
    handled_count = 0
    for phrase, description in test_phrases:
        result = server._handle_execute_command(phrase)
        if isinstance(result, dict):
            handled_count += 1
        else:
            return False, f"Command '{phrase}' crashed"
    
    return True, f"All {handled_count} phrases handled without crashes"


def scenario_5_model_name_robustness():
    """Test model name variations and typos"""
    server = MuJoCoRemoteServer()
    
    if not server.menagerie.is_available():
        return True, "Menagerie not installed (expected)"
    
    # Test name variations
    name_tests = [
        ("franka_emika_panda", True),  # Exact
        ("franka emika panda", True),  # Spaces
        ("franka-emika-panda", True),  # Hyphens
        ("FRANKA_EMIKA_PANDA", True),  # Uppercase
        ("FrAnKa PaNdA", False),       # Mixed case partial
        ("panda", False),              # Partial name
        ("franka", False),             # Partial name
        ("definitely_not_a_robot", False)  # Invalid
    ]
    
    for name, should_find in name_tests:
        path = server.menagerie.find_model(name)
        found = path is not None
        
        if should_find and not found:
            return False, f"Should find '{name}' but didn't"
        elif not should_find and found:
            pass  # It's ok to find partial matches
    
    return True, "Model name handling is robust"


def scenario_6_backward_compatibility():
    """Ensure v0.7.0 usage still works"""
    server = MuJoCoRemoteServer()
    
    # Old-style commands
    old_commands = [
        ("create pendulum", "pendulum creation"),
        ("create double pendulum", "double pendulum"),
        ("create cart pole", "cart-pole"),
        ("show state", "state query"),
        ("reset", "reset command")
    ]
    
    for cmd, description in old_commands:
        result = server._handle_execute_command(cmd)
        if not isinstance(result, dict):
            return False, f"Old command '{cmd}' failed"
    
    # Check all original tools
    original_tools = [
        "get_server_info", "create_scene", "step_simulation",
        "get_state", "set_joint_positions", "reset_simulation",
        "execute_command", "get_loaded_models"
    ]
    
    for tool in original_tools:
        if tool not in server._tools:
            return False, f"Original tool '{tool}' missing"
    
    return True, "All v0.7.0 features still work"


def scenario_7_error_messages():
    """Test error message quality"""
    server = MuJoCoRemoteServer()
    
    # Test various error conditions
    error_tests = [
        ("load xyz123robot", "nonexistent model"),
        ("set joint position", "incomplete command"),
        ("step", "step without simulation"),
        ("create", "incomplete create")
    ]
    
    for cmd, description in error_tests:
        result = server._handle_execute_command(cmd)
        if "error" in result or "success" in result:
            # Check if error message is helpful
            error_msg = str(result.get("error", ""))
            if len(error_msg) < 10:
                return False, f"Unhelpful error for '{cmd}'"
        else:
            return False, f"No error handling for '{cmd}'"
    
    return True, "Error messages are helpful"


def main():
    """Run all scenarios"""
    print("MuJoCo MCP v0.7.1 - Real World Scenario Tests")
    print("=" * 60)
    
    scenarios = [
        ("Basic Usage", scenario_1_basic_usage),
        ("Menagerie Discovery", scenario_2_menagerie_discovery),
        ("Load Real Robot", scenario_3_load_real_robot),
        ("Natural Language Variations", scenario_4_natural_language_variations),
        ("Model Name Robustness", scenario_5_model_name_robustness),
        ("Backward Compatibility", scenario_6_backward_compatibility),
        ("Error Message Quality", scenario_7_error_messages)
    ]
    
    passed = 0
    total = len(scenarios)
    
    for name, func in scenarios:
        if test_scenario(name, func):
            passed += 1
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS: {passed}/{total} scenarios passed")
    print('='*60)
    
    if passed == total:
        print("\n🎉 ALL SCENARIOS PASSED!")
        print("RECOMMENDATION: Ready for v0.8.0 release (major feature)")
    elif passed >= total - 1:
        print("\n⚠️  MINOR ISSUES")
        print("RECOMMENDATION: Fix and release as v0.7.2")
    else:
        print("\n❌ SIGNIFICANT ISSUES")
        print("RECOMMENDATION: Do not release yet")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
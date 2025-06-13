#!/usr/bin/env python3
"""
Comprehensive test suite for v0.7.1 - MuJoCo Menagerie Integration
Tests all new features and ensures backward compatibility
"""

import sys
import os
import time
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mujoco_mcp.remote_server import MuJoCoRemoteServer
from mujoco_mcp.menagerie_loader import MenagerieLoader


class TestResults:
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def add_test(self, name: str, passed: bool, error: str = None):
        self.total += 1
        if passed:
            self.passed += 1
            print(f"✓ {name}")
        else:
            self.failed += 1
            print(f"✗ {name}")
            if error:
                print(f"  Error: {error}")
                self.errors.append((name, error))
    
    def print_summary(self):
        print("\n" + "="*60)
        print(f"TEST SUMMARY: {self.passed}/{self.total} passed")
        if self.failed > 0:
            print(f"\nFailed tests:")
            for name, error in self.errors:
                print(f"  - {name}: {error}")
        print("="*60)


def test_menagerie_loader():
    """Test the MenagerieLoader component"""
    results = TestResults()
    print("\n=== Testing MenagerieLoader ===")
    
    loader = MenagerieLoader()
    
    # Test 1: Availability
    results.add_test(
        "MenagerieLoader availability", 
        loader.is_available(),
        "Menagerie not found"
    )
    
    if not loader.is_available():
        print("Skipping remaining loader tests - Menagerie not installed")
        return results
    
    # Test 2: Path detection
    results.add_test(
        "Menagerie path exists",
        os.path.exists(loader.menagerie_path),
        f"Path not found: {loader.menagerie_path}"
    )
    
    # Test 3: List models
    models = loader.list_models()
    results.add_test(
        "List models returns results",
        len(models) > 0,
        f"No models found"
    )
    
    # Test 4: Category filtering
    arms = loader.list_models("arms")
    results.add_test(
        "Category filtering works",
        all(m['category'] == 'arms' for m in arms),
        "Category filtering failed"
    )
    
    # Test 5: Find specific models
    test_models = [
        ("franka_emika_panda", "franka panda"),
        ("unitree_go2", "go2"),
        ("shadow_hand", "shadow hand")
    ]
    
    for correct_name, search_name in test_models:
        found = loader.find_model(search_name)
        results.add_test(
            f"Find model '{search_name}'",
            found is not None,
            f"Model not found"
        )
    
    # Test 6: Load model XML
    xml = loader.load_model_xml("franka_emika_panda")
    results.add_test(
        "Load model XML",
        xml is not None and "<mujoco" in xml,
        "Failed to load XML"
    )
    
    # Test 7: Path fixing
    if xml:
        results.add_test(
            "XML paths are absolute",
            any(os.path.isabs(path) for path in xml.split('"') if '.xml' in path or '.stl' in path),
            "Paths not converted to absolute"
        )
    
    return results


def test_server_integration():
    """Test MuJoCo Remote Server integration"""
    results = TestResults()
    print("\n=== Testing Server Integration ===")
    
    server = MuJoCoRemoteServer()
    
    # Test 1: Server has Menagerie
    results.add_test(
        "Server has Menagerie loader",
        hasattr(server, 'menagerie'),
        "Menagerie not integrated"
    )
    
    # Test 2: List tool exists
    results.add_test(
        "list_menagerie_models tool registered",
        'list_menagerie_models' in server._tools,
        "Tool not found"
    )
    
    # Test 3: Tool handler works
    if server.menagerie.is_available():
        result = server._handle_list_menagerie_models()
        results.add_test(
            "list_menagerie_models returns success",
            result.get('success', False),
            result.get('error', 'Unknown error')
        )
        
        # Test 4: Category filtering in tool
        result = server._handle_list_menagerie_models("quadrupeds")
        quadrupeds = result.get('models_by_category', {}).get('quadrupeds', [])
        results.add_test(
            "Category filtering in tool",
            len(quadrupeds) > 0,
            "No quadrupeds found"
        )
    
    # Test 5: Create scene with Menagerie (without viewer)
    result = server._handle_create_scene("franka_emika_panda")
    is_menagerie_recognized = (
        "viewer" in str(result.get('error', '')).lower() or
        result.get('success', False)
    )
    results.add_test(
        "Create scene recognizes Menagerie models",
        is_menagerie_recognized,
        "Menagerie model not recognized"
    )
    
    # Test 6: Built-in scenes still work
    result = server._handle_create_scene("pendulum")
    is_builtin_recognized = (
        "viewer" in str(result.get('error', '')).lower() or
        result.get('success', False)
    )
    results.add_test(
        "Built-in scenes still work",
        is_builtin_recognized,
        "Built-in scenes broken"
    )
    
    return results


def test_natural_language():
    """Test natural language enhancements"""
    results = TestResults()
    print("\n=== Testing Natural Language ===")
    
    server = MuJoCoRemoteServer()
    
    # Test commands
    test_commands = [
        ("help", "commands", "Help command works"),
        ("list menagerie models", "success", "List Menagerie command"),
        ("load franka panda", None, "Load robot command"),
        ("create spot robot", None, "Create robot command"),
        ("show state", "error", "State command without model"),
        ("reset", "error", "Reset without model"),
        ("step 500", "error", "Step with count"),
        ("set angle to 45 degrees", "error", "Angle setting")
    ]
    
    for command, expected_key, test_name in test_commands:
        result = server._handle_execute_command(command)
        
        if expected_key:
            results.add_test(
                test_name,
                expected_key in result,
                f"Expected '{expected_key}' in result"
            )
        else:
            # Just check it doesn't crash
            results.add_test(
                test_name,
                isinstance(result, dict),
                "Command failed to return dict"
            )
    
    # Test help includes Menagerie
    help_result = server._handle_execute_command("help")
    commands_str = str(help_result.get('commands', []))
    results.add_test(
        "Help includes Menagerie info",
        "Menagerie" in commands_str,
        "Menagerie not in help"
    )
    
    return results


def test_backward_compatibility():
    """Ensure all v0.7.0 features still work"""
    results = TestResults()
    print("\n=== Testing Backward Compatibility ===")
    
    server = MuJoCoRemoteServer()
    
    # Test all original tools exist
    required_tools = [
        "get_server_info",
        "create_scene", 
        "step_simulation",
        "get_state",
        "set_joint_positions",
        "reset_simulation",
        "execute_command",
        "get_loaded_models"
    ]
    
    for tool in required_tools:
        results.add_test(
            f"Tool '{tool}' exists",
            tool in server._tools,
            "Tool missing"
        )
    
    # Test server info
    info = server._handle_get_server_info()
    results.add_test(
        "Server info returns version",
        "version" in info,
        "Version missing"
    )
    
    # Test get_loaded_models
    models = server._handle_get_loaded_models()
    results.add_test(
        "get_loaded_models works",
        "models" in models,
        "Models key missing"
    )
    
    return results


def test_error_handling():
    """Test error handling and edge cases"""
    results = TestResults()
    print("\n=== Testing Error Handling ===")
    
    server = MuJoCoRemoteServer()
    
    # Test 1: Invalid model name
    result = server._handle_create_scene("nonexistent_model_12345")
    results.add_test(
        "Invalid model returns error",
        "error" in result,
        "No error for invalid model"
    )
    
    # Test 2: Empty command
    result = server._handle_execute_command("")
    results.add_test(
        "Empty command handled",
        isinstance(result, dict),
        "Empty command crashed"
    )
    
    # Test 3: None parameters
    result = server._handle_list_menagerie_models(None)
    results.add_test(
        "None category parameter handled",
        isinstance(result, dict),
        "None parameter crashed"
    )
    
    # Test 4: Model name variations
    variations = ["franka-panda", "franka_panda", "FRANKA PANDA", "franka  panda"]
    for variant in variations:
        result = server._handle_execute_command(f"load {variant}")
        results.add_test(
            f"Name variation '{variant}'",
            isinstance(result, dict),
            "Name variation crashed"
        )
    
    return results


def test_performance():
    """Test performance metrics"""
    results = TestResults()
    print("\n=== Testing Performance ===")
    
    server = MuJoCoRemoteServer()
    
    if not server.menagerie.is_available():
        print("Skipping performance tests - Menagerie not available")
        return results
    
    # Test 1: List models speed
    start = time.time()
    server._handle_list_menagerie_models()
    elapsed = time.time() - start
    results.add_test(
        "List models < 1 second",
        elapsed < 1.0,
        f"Took {elapsed:.2f} seconds"
    )
    
    # Test 2: Find model speed
    start = time.time()
    server.menagerie.find_model("franka panda")
    elapsed = time.time() - start
    results.add_test(
        "Find model < 0.5 seconds",
        elapsed < 0.5,
        f"Took {elapsed:.2f} seconds"
    )
    
    # Test 3: Load XML speed
    start = time.time()
    server.menagerie.load_model_xml("franka_emika_panda")
    elapsed = time.time() - start
    results.add_test(
        "Load XML < 1 second",
        elapsed < 1.0,
        f"Took {elapsed:.2f} seconds"
    )
    
    return results


def main():
    """Run all tests"""
    print("MuJoCo MCP v0.7.1 Comprehensive Test Suite")
    print("==========================================")
    
    all_results = TestResults()
    
    # Run test suites
    test_suites = [
        ("Menagerie Loader", test_menagerie_loader),
        ("Server Integration", test_server_integration),
        ("Natural Language", test_natural_language),
        ("Backward Compatibility", test_backward_compatibility),
        ("Error Handling", test_error_handling),
        ("Performance", test_performance)
    ]
    
    for suite_name, test_func in test_suites:
        try:
            suite_results = test_func()
            all_results.total += suite_results.total
            all_results.passed += suite_results.passed
            all_results.failed += suite_results.failed
            all_results.errors.extend(suite_results.errors)
        except Exception as e:
            print(f"\n✗ {suite_name} suite crashed: {e}")
            all_results.failed += 1
            all_results.total += 1
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    all_results.print_summary()
    
    # Recommendation
    print("\nRECOMMENDATION:")
    if all_results.failed == 0:
        print("✅ All tests passed! Ready for release.")
        print("   Suggested version: 0.8.0 (major feature addition)")
    elif all_results.failed <= 3:
        print("⚠️  Minor issues found. Fix before release.")
        print("   Suggested version: 0.7.2 (after fixes)")
    else:
        print("❌ Significant issues found. Do not release.")
    
    return all_results.failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
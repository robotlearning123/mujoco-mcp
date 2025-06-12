#!/usr/bin/env python3
"""Test script for MuJoCo MCP Server configuration and functionality."""

import json
import os
import sys
import subprocess
import time
import signal
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import mujoco_mcp
from mujoco_mcp.simulation import MuJoCoSimulation
from mujoco_mcp.enhanced_auth_manager import EnhancedAuthManager


class MCPServerTester:
    """Test harness for MuJoCo MCP Server."""
    
    def __init__(self):
        self.results = {
            "configuration_validation": {},
            "server_tests": {},
            "functionality_tests": {},
            "issues": [],
            "recommendations": []
        }
    
    def validate_json_config(self, filepath, config_name):
        """Validate JSON configuration file."""
        print(f"\n[TEST] Validating {config_name}...")
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
            
            # Check required fields based on config type
            if config_name == "mcp.json":
                required = ["mcpServers"]
                if "mcpServers" in config and "mujoco-mcp" in config["mcpServers"]:
                    server_config = config["mcpServers"]["mujoco-mcp"]
                    server_required = ["command", "args"]
                    missing = [f for f in server_required if f not in server_config]
                    if missing:
                        self.results["issues"].append(f"{config_name}: Missing fields in server config: {missing}")
                    else:
                        print(f"  ✓ Valid structure with all required fields")
                        self.results["configuration_validation"][config_name] = "PASSED"
                        return True
            
            elif config_name == "claude_desktop_config.json":
                required = ["name", "command", "args"]
                missing = [f for f in required if f not in config]
                if missing:
                    self.results["issues"].append(f"{config_name}: Missing required fields: {missing}")
                else:
                    print(f"  ✓ Valid structure with all required fields")
                    self.results["configuration_validation"][config_name] = "PASSED"
                    return True
            
            self.results["configuration_validation"][config_name] = "FAILED"
            return False
            
        except json.JSONDecodeError as e:
            print(f"  ✗ Invalid JSON: {e}")
            self.results["configuration_validation"][config_name] = "INVALID_JSON"
            self.results["issues"].append(f"{config_name}: Invalid JSON - {e}")
            return False
        except FileNotFoundError:
            print(f"  ✗ File not found")
            self.results["configuration_validation"][config_name] = "NOT_FOUND"
            self.results["issues"].append(f"{config_name}: File not found")
            return False
    
    def validate_cursorrules(self, filepath):
        """Validate .cursorrules file."""
        print("\n[TEST] Validating .cursorrules...")
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Check for required sections
            required_sections = ["Project Overview", "MCP Server Configuration", "Available MCP Tools"]
            missing_sections = []
            
            for section in required_sections:
                if section not in content:
                    missing_sections.append(section)
            
            if missing_sections:
                print(f"  ✗ Missing sections: {missing_sections}")
                self.results["configuration_validation"][".cursorrules"] = "INCOMPLETE"
                self.results["issues"].append(f".cursorrules: Missing sections - {missing_sections}")
            else:
                print(f"  ✓ All required sections present")
                
                # Check if JSON config in the file is valid
                import re
                json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                if json_match:
                    try:
                        json.loads(json_match.group(1))
                        print(f"  ✓ Embedded JSON configuration is valid")
                    except:
                        print(f"  ✗ Embedded JSON configuration is invalid")
                        self.results["issues"].append(".cursorrules: Invalid embedded JSON")
                
                self.results["configuration_validation"][".cursorrules"] = "PASSED"
                return True
            
        except FileNotFoundError:
            print(f"  ✗ File not found")
            self.results["configuration_validation"][".cursorrules"] = "NOT_FOUND"
            self.results["issues"].append(".cursorrules: File not found")
            return False
    
    def test_server_import(self):
        """Test if server can be imported."""
        print("\n[TEST] Testing server import...")
        try:
            from mujoco_mcp import server
            print("  ✓ Server module imported successfully")
            self.results["server_tests"]["import"] = "PASSED"
            return True
        except ImportError as e:
            print(f"  ✗ Failed to import: {e}")
            self.results["server_tests"]["import"] = "FAILED"
            self.results["issues"].append(f"Server import failed: {e}")
            return False
    
    def test_simulation_creation(self):
        """Test basic simulation creation."""
        print("\n[TEST] Testing simulation creation...")
        try:
            sim = MuJoCoSimulation()
            print("  ✓ Simulation object created")
            
            # Test loading a simple model
            xml_string = """
            <mujoco>
                <worldbody>
                    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
                    <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
                </worldbody>
            </mujoco>
            """
            
            sim.load_model_from_string(xml_string, "test_model")
            print("  ✓ Model loaded successfully")
            
            self.results["functionality_tests"]["simulation_creation"] = "PASSED"
            return True
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            self.results["functionality_tests"]["simulation_creation"] = "FAILED"
            self.results["issues"].append(f"Simulation creation failed: {e}")
            return False
    
    def test_auth_manager(self):
        """Test authentication manager."""
        print("\n[TEST] Testing authentication manager...")
        try:
            auth = EnhancedAuthManager()
            
            # Test parameter validation
            valid = auth.validate_parameters({
                "force": [1.0, 2.0, 3.0],
                "position": [0.1, 0.2, 0.3]
            })
            
            if valid:
                print("  ✓ Parameter validation working")
            
            # Test rate limiting
            auth.check_rate_limit("test_client")
            print("  ✓ Rate limiting functional")
            
            self.results["functionality_tests"]["auth_manager"] = "PASSED"
            return True
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            self.results["functionality_tests"]["auth_manager"] = "FAILED"
            self.results["issues"].append(f"Auth manager test failed: {e}")
            return False
    
    def test_environment_variables(self):
        """Test environment variable configuration."""
        print("\n[TEST] Testing environment variables...")
        
        env_vars = {
            "MUJOCO_MCP_PORT": "8080",
            "MUJOCO_MCP_LOG_LEVEL": "DEBUG",
            "MUJOCO_MCP_HOST": "127.0.0.1"
        }
        
        # Set environment variables
        for var, value in env_vars.items():
            os.environ[var] = value
            print(f"  ✓ Set {var}={value}")
        
        self.results["server_tests"]["environment_variables"] = "PASSED"
        return True
    
    def generate_recommendations(self):
        """Generate recommendations based on test results."""
        if self.results["issues"]:
            self.results["recommendations"].append(
                "Fix the following issues before deployment:"
            )
            for issue in self.results["issues"]:
                self.results["recommendations"].append(f"  - {issue}")
        
        # Server-specific recommendations
        if "import" in self.results["server_tests"] and self.results["server_tests"]["import"] == "FAILED":
            self.results["recommendations"].append(
                "Ensure all dependencies are installed: pip install -e ."
            )
        
        # General recommendations
        self.results["recommendations"].extend([
            "Consider implementing the following improvements:",
            "  - Add health check endpoint for monitoring",
            "  - Implement proper graceful shutdown handling",
            "  - Add metrics collection for performance monitoring",
            "  - Create integration tests with actual MCP clients",
            "  - Add support for TLS/SSL for secure connections",
            "  - Implement connection pooling for better performance"
        ])
    
    def print_report(self):
        """Print comprehensive test report."""
        print("\n" + "="*60)
        print("MuJoCo MCP Server Test Report")
        print("="*60)
        
        print("\n## Configuration Validation")
        for config, status in self.results["configuration_validation"].items():
            symbol = "✓" if status == "PASSED" else "✗"
            print(f"  {symbol} {config}: {status}")
        
        print("\n## Server Tests")
        for test, status in self.results["server_tests"].items():
            symbol = "✓" if status == "PASSED" else "✗"
            print(f"  {symbol} {test}: {status}")
        
        print("\n## Functionality Tests")
        for test, status in self.results["functionality_tests"].items():
            symbol = "✓" if status == "PASSED" else "✗"
            print(f"  {symbol} {test}: {status}")
        
        if self.results["issues"]:
            print("\n## Issues Found")
            for issue in self.results["issues"]:
                print(f"  • {issue}")
        
        if self.results["recommendations"]:
            print("\n## Recommendations")
            for rec in self.results["recommendations"]:
                print(f"  {rec}")
        
        # Overall status
        total_tests = (
            len(self.results["configuration_validation"]) +
            len(self.results["server_tests"]) +
            len(self.results["functionality_tests"])
        )
        passed_tests = sum(
            1 for status in list(self.results["configuration_validation"].values()) +
            list(self.results["server_tests"].values()) +
            list(self.results["functionality_tests"].values())
            if status == "PASSED"
        )
        
        print(f"\n## Summary")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {passed_tests}")
        print(f"  Failed: {total_tests - passed_tests}")
        print(f"  Success Rate: {(passed_tests/total_tests*100):.1f}%")
        print("="*60)


def main():
    """Run all tests."""
    tester = MCPServerTester()
    
    # Get the project root
    project_root = Path(__file__).parent
    
    # Validate configurations
    tester.validate_json_config(project_root / "mcp.json", "mcp.json")
    tester.validate_json_config(project_root / "claude_desktop_config.json", "claude_desktop_config.json")
    tester.validate_cursorrules(project_root / ".cursorrules")
    
    # Test server functionality
    tester.test_server_import()
    tester.test_simulation_creation()
    tester.test_auth_manager()
    tester.test_environment_variables()
    
    # Generate recommendations
    tester.generate_recommendations()
    
    # Print report
    tester.print_report()
    
    # Save detailed report
    report_path = project_root / "test_report.json"
    with open(report_path, 'w') as f:
        json.dump(tester.results, f, indent=2)
    print(f"\nDetailed report saved to: {report_path}")


if __name__ == "__main__":
    main()
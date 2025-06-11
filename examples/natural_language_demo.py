#!/usr/bin/env python3
"""
Natural Language Interface Demo for MuJoCo MCP v0.3.2

This demo shows how to control MuJoCo simulations using natural language commands.
"""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from mujoco_mcp.simple_server import MuJoCoMCPServer


def print_section(title):
    """Print a section header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def main():
    """Run natural language interface demo"""
    # Create server
    server = MuJoCoMCPServer()
    print(f"MuJoCo MCP Server v{server.version} - Natural Language Interface Demo")
    
    # 1. Show capabilities
    print_section("1. Asking for Capabilities")
    result = server.call_tool("execute_command", {
        "command": "what can you do?"
    })
    print(f"Command: 'what can you do?'")
    print(f"Response capabilities:")
    for cap in result.get("capabilities", []):
        print(f"  - {cap}")
    
    # 2. Create a pendulum
    print_section("2. Creating a Pendulum")
    result = server.call_tool("execute_command", {
        "command": "create a pendulum"
    })
    model_id = result["model_id"]
    print(f"Command: 'create a pendulum'")
    print(f"Result: {result['interpretation']}")
    print(f"Model ID: {model_id}")
    
    # 3. Query state
    print_section("3. Querying State")
    result = server.call_tool("execute_command", {
        "command": "what is the current angle of the pendulum?",
        "context": {"model_id": model_id}
    })
    print(f"Command: 'what is the current angle of the pendulum?'")
    print(f"Answer: {result['answer']}")
    
    # 4. Control to position
    print_section("4. Controlling to Target Position")
    result = server.call_tool("execute_command", {
        "command": "move the pendulum to 45 degrees",
        "context": {"model_id": model_id}
    })
    print(f"Command: 'move the pendulum to 45 degrees'")
    print(f"Action: {result['action_taken']}")
    if "result" in result and "final_error" in result["result"]:
        print(f"Final error: {abs(result['result']['final_error']):.2f} degrees")
    
    # 5. Visualize
    print_section("5. Visualization")
    result = server.call_tool("execute_command", {
        "command": "show me the pendulum",
        "context": {"model_id": model_id}
    })
    print(f"Command: 'show me the pendulum'")
    print(f"Interpretation: {result['interpretation']}")
    if "visualization" in result and "ascii" in result["visualization"]:
        print("\nASCII Visualization:")
        print(result["visualization"]["ascii"])
    
    # 6. High-level scene creation
    print_section("6. High-Level Scene Creation")
    result = server.call_tool("create_scene", {
        "scene_type": "pendulum",
        "parameters": {
            "length": 0.7,
            "mass": 1.0,
            "damping": 0.05
        }
    })
    custom_model_id = result["model_id"]
    print(f"Created custom pendulum scene")
    print(f"Parameters: length=0.7m, mass=1.0kg, damping=0.05")
    print(f"Model ID: {custom_model_id}")
    
    # 7. Perform high-level task
    print_section("7. High-Level Task Execution")
    result = server.call_tool("perform_task", {
        "task": "balance",
        "model_id": custom_model_id,
        "parameters": {
            "target_angle": 90.0,
            "duration": 2.0
        }
    })
    print(f"Task: Balance pendulum at 90 degrees")
    print(f"Result: {result['message']}")
    
    # 8. Analyze behavior
    print_section("8. Behavior Analysis")
    # Step simulation a bit first
    server.call_tool("step_simulation", {
        "model_id": custom_model_id,
        "steps": 100
    })
    
    result = server.call_tool("analyze_behavior", {
        "model_id": custom_model_id,
        "analysis_type": "stability",
        "duration": 0.5
    })
    print(f"Stability Analysis:")
    if "analysis" in result and "stability" in result["analysis"]:
        analysis = result["analysis"]
        print(f"  - Stable: {analysis.get('is_stable', False)}")
        print(f"  - Stability Score: {analysis.get('stability_score', 0):.3f}")
        print(f"  - Std Deviation: {analysis.get('std_deviation', 0):.4f}")
    
    # 9. Complex natural language command
    print_section("9. Complex Multi-Step Command")
    result = server.call_tool("execute_command", {
        "command": "create a pendulum and swing it up to the top position"
    })
    print(f"Command: 'create a pendulum and swing it up to the top position'")
    print(f"Interpretation: {result['interpretation']}")
    if "steps_taken" in result:
        print("Steps taken:")
        for step in result["steps_taken"]:
            print(f"  - {step}")
    
    # 10. Invalid command handling
    print_section("10. Error Handling")
    result = server.call_tool("execute_command", {
        "command": "do something random"
    })
    print(f"Command: 'do something random'")
    print(f"Success: {result.get('success', False)}")
    if "clarification_needed" in result:
        print(f"Clarification: {result['clarification_needed']}")
        print("Examples provided:")
        for example in result.get("examples", []):
            print(f"  - {example}")
    
    print_section("Demo Complete!")
    print("\nThe natural language interface makes MuJoCo control accessible to LLMs")
    print("by accepting human-friendly commands and providing intelligent responses.")


if __name__ == "__main__":
    main()
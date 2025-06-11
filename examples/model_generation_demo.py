#!/usr/bin/env python3
"""
Model Generation Demo for MuJoCo MCP v0.4.0

This demo shows how to programmatically generate robots and environments.
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
    """Run model generation demo"""
    # Create server
    server = MuJoCoMCPServer()
    print(f"MuJoCo MCP Server v{server.version} - Model Generation Demo")
    
    # 1. Generate a simple robotic arm
    print_section("1. Generating a Simple Robotic Arm")
    result = server.call_tool("generate_robot", {
        "robot_type": "arm",
        "parameters": {
            "num_links": 3,
            "link_length": 0.3,
            "link_mass": 0.5,
            "joint_limits": [-90, 90],
            "max_torque": 10.0
        }
    })
    arm_model_id = result["model_id"]
    print(f"✓ Generated 3-link robotic arm")
    print(f"  Model ID: {arm_model_id}")
    print(f"  Features: {result['model_info']['nq']} joints, {result['model_info']['nu']} actuators")
    
    # 2. Generate a mobile robot
    print_section("2. Generating a Mobile Robot")
    result = server.call_tool("generate_robot", {
        "robot_type": "mobile",
        "parameters": {
            "base_size": [0.3, 0.2, 0.1],
            "wheel_radius": 0.05,
            "num_wheels": 4
        }
    })
    mobile_model_id = result["model_id"]
    print(f"✓ Generated 4-wheeled mobile robot")
    print(f"  Model ID: {mobile_model_id}")
    
    # 3. Generate a gripper
    print_section("3. Generating a Gripper")
    result = server.call_tool("generate_robot", {
        "robot_type": "gripper",
        "parameters": {
            "finger_length": 0.08,
            "max_opening": 0.1
        }
    })
    gripper_model_id = result["model_id"]
    print(f"✓ Generated parallel jaw gripper")
    print(f"  Model ID: {gripper_model_id}")
    
    # 4. Generate environments
    print_section("4. Generating Environments")
    
    # Flat ground
    result = server.call_tool("generate_environment", {
        "env_type": "flat_ground",
        "parameters": {
            "size": [10, 10],
            "friction": 1.0
        }
    })
    print(f"✓ Generated flat ground environment (10x10m)")
    
    # Obstacles
    result = server.call_tool("generate_environment", {
        "env_type": "obstacles",
        "parameters": {
            "ground_size": [5, 5],
            "num_obstacles": 5,
            "obstacle_size_range": [0.2, 0.8]
        }
    })
    print(f"✓ Generated environment with {result['obstacle_count']} obstacles")
    
    # Terrain (stairs)
    result = server.call_tool("generate_environment", {
        "env_type": "terrain",
        "parameters": {
            "terrain_type": "stairs",
            "num_steps": 5,
            "step_height": 0.1,
            "step_width": 0.3
        }
    })
    print(f"✓ Generated stairs terrain")
    
    # 5. Combine robot and environment
    print_section("5. Combining Robot and Environment")
    
    # Generate a test arena
    env_result = server.call_tool("generate_environment", {
        "env_type": "flat_ground",
        "parameters": {"size": [8, 8]}
    })
    
    # Generate a new arm
    robot_result = server.call_tool("generate_robot", {
        "robot_type": "arm",
        "parameters": {"num_links": 2}
    })
    
    # Combine them
    result = server.call_tool("combine_models", {
        "base_model_xml": env_result["xml"],
        "add_model_xml": robot_result["xml"],
        "position": [0, 0, 0.1]
    })
    combined_model_id = result["model_id"]
    print(f"✓ Combined robot arm with environment")
    print(f"  Combined Model ID: {combined_model_id}")
    
    # 6. Working with templates
    print_section("6. Template System")
    
    # List available templates
    result = server.call_tool("list_templates", {})
    print(f"Available templates:")
    print(f"  Robot templates: {len(result['robot_templates'])}")
    for template in result['robot_templates'][:3]:
        print(f"    - {template['name']}: {template['description']}")
    print(f"  Environment templates: {len(result['environment_templates'])}")
    
    # Generate from template
    result = server.call_tool("generate_from_template", {
        "template_name": "simple_arm",
        "parameters": {
            "num_links": 5,
            "link_length": 0.25
        }
    })
    print(f"\n✓ Generated 5-link arm from template")
    
    # Save custom template
    result = server.call_tool("save_as_template", {
        "model_id": arm_model_id,
        "template_name": "my_custom_arm",
        "description": "Custom 3-link arm with specific parameters",
        "parameterizable": ["link_length", "link_mass", "color"]
    })
    print(f"✓ Saved custom template: {result['template_name']}")
    
    # 7. XML Validation
    print_section("7. Model Validation")
    
    # Generate and validate
    robot_result = server.call_tool("generate_robot", {
        "robot_type": "arm",
        "parameters": {"num_links": 2}
    })
    
    validation = server.call_tool("validate_model_xml", {
        "xml_string": robot_result["xml"]
    })
    print(f"✓ Generated XML validation:")
    print(f"  Valid: {validation['is_valid']}")
    print(f"  Errors: {len(validation.get('errors', []))}")
    print(f"  Warnings: {len(validation.get('warnings', []))}")
    
    # 8. Safety constraints demo
    print_section("8. Safety Constraints")
    
    result = server.call_tool("generate_robot", {
        "robot_type": "arm",
        "parameters": {
            "num_links": 15,  # Too many
            "link_mass": 200,  # Too heavy
            "max_torque": 2000  # Too strong
        }
    })
    
    if "warnings" in result:
        print("Safety limits applied:")
        for warning in result["warnings"]:
            print(f"  ⚠️ {warning}")
    
    # 9. Visualize generated model
    print_section("9. Visualizing Generated Model")
    
    # Get ASCII visualization of the combined model
    result = server.call_tool("get_ascii_visualization", {
        "model_id": combined_model_id,
        "width": 50,
        "height": 20
    })
    print("\nCombined model visualization:")
    print(result["ascii_art"])
    
    # 10. Control generated model
    print_section("10. Controlling Generated Model")
    
    # Step simulation
    server.call_tool("step_simulation", {
        "model_id": combined_model_id,
        "steps": 100
    })
    
    # Get state
    state = server.call_tool("get_simulation_state", {
        "model_id": combined_model_id,
        "include_positions": True
    })
    print(f"Model state after 100 steps:")
    print(f"  Time: {state['time']:.3f}s")
    print(f"  Joint positions: {[f'{p:.2f}' for p in state['qpos']]}")
    
    print_section("Demo Complete!")
    print("\nModel generation enables:")
    print("- Dynamic robot creation with custom parameters")
    print("- Environment generation for testing")
    print("- Model combination and composition")
    print("- Template-based design patterns")
    print("- Safe parameter validation")


if __name__ == "__main__":
    main()
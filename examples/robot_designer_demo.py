#!/usr/bin/env python3
"""
Robot Designer Demo for MuJoCo MCP v0.4.2

This demo shows how to use AI-assisted robot design capabilities.
"""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from mujoco_mcp.simple_server import MuJoCoMCPServer


def print_section(title):
    """Print a section header"""
    print(f"\n{'='*70}")
    print(f" {title}")
    print('='*70)


def main():
    """Run robot designer demo"""
    # Create server
    server = MuJoCoMCPServer()
    print(f"MuJoCo MCP Server v{server.version} - Robot Designer Demo")
    
    # 1. Design a simple gripper
    print_section("1. Designing a Simple Gripper")
    print("Task: Design a gripper for picking up small objects")
    
    result = server.call_tool("design_robot", {
        "task_description": "I need a gripper that can pick up small electronic components",
        "constraints": {
            "max_size": [0.15, 0.15, 0.1],  # 15x15x10 cm
            "max_force": 5.0,  # 5N max grip force
            "precision": 0.001  # 1mm precision
        },
        "preferences": {
            "finger_type": "parallel",
            "actuation": "servo"
        }
    })
    
    print(f"✓ Gripper design completed")
    print(f"  Design ID: {result['design_id'][:8]}...")
    print(f"  Model saved as: {result['model_id']}")
    print(f"  Specifications:")
    specs = result['specifications']
    print(f"    - Type: {specs['type']}")
    print(f"    - Fingers: {specs.get('num_fingers', 2)}")
    print(f"    - Max grip force: {specs.get('max_grip_force', 0):.1f}N")
    print(f"    - Estimated precision: {specs['estimated_performance'].get('precision', 0)*1000:.1f}mm")
    
    gripper_id = result['design_id']
    
    # 2. Design a mobile robot
    print_section("2. Designing a Mobile Robot")
    print("Task: Design a robot for outdoor navigation")
    
    result = server.call_tool("design_robot", {
        "task_description": "Design a robot that can navigate rough outdoor terrain",
        "constraints": {
            "max_weight": 30.0,  # 30kg
            "terrain_type": "rough",
            "speed_requirement": 2.0,  # 2m/s
            "battery_life": 60  # 60 minutes
        },
        "preferences": {
            "wheel_type": "rugged",
            "stability": "high",
            "sensor_suite": "outdoor"
        },
        "optimize_for": ["stability", "energy_efficiency"]
    })
    
    print(f"✓ Mobile robot design completed")
    print(f"  Design ID: {result['design_id'][:8]}...")
    specs = result['specifications']
    print(f"  Specifications:")
    print(f"    - Type: {specs['type']}")
    print(f"    - Wheels: {specs['num_wheels']}")
    print(f"    - Suspension: {specs.get('suspension_type', 'N/A')}")
    print(f"    - Estimated speed: {specs['estimated_performance'].get('max_speed', 0):.1f} m/s")
    print(f"    - Stability score: {specs['estimated_performance'].get('stability', 0):.2f}")
    
    if "optimization_results" in result:
        print(f"  Optimization performed:")
        print(f"    - Stability score: {result['optimization_results'].get('stability_score', 0):.2f}")
        print(f"    - Energy efficiency: {result['optimization_results'].get('energy_efficiency_score', 0):.2f}")
    
    mobile_id = result['design_id']
    
    # 3. Design a manipulator arm
    print_section("3. Designing a Manipulator Arm")
    print("Task: Design a robotic arm for assembly tasks")
    
    result = server.call_tool("design_robot", {
        "task_description": "I need a robotic arm for precise electronic assembly",
        "constraints": {
            "workspace_radius": 0.6,  # 60cm reach
            "precision": 0.0005,  # 0.5mm precision
            "payload": 3.0,  # 3kg payload
            "cycle_time": 2.0  # 2 second cycle time
        },
        "use_components": True,
        "component_preferences": {
            "actuator_type": "servo",
            "sensor_types": ["position", "force", "vision"]
        }
    })
    
    print(f"✓ Manipulator arm design completed")
    specs = result['specifications']
    print(f"  Specifications:")
    print(f"    - Type: {specs['type']}")
    print(f"    - Joints: {specs['num_joints']} DOF")
    print(f"    - Reach: {specs.get('reach', 0)*100:.0f}cm")
    print(f"    - Estimated precision: {specs['estimated_performance'].get('precision', 0)*1000:.1f}mm")
    
    if "components_used" in result:
        print(f"  Components from library:")
        for comp in result['components_used'][:3]:
            print(f"    - {comp['type']}: {comp['name']}")
    
    arm_id = result['design_id']
    
    # 4. Design with cost estimation
    print_section("4. Educational Robot with Cost Estimation")
    print("Task: Design a low-cost robot for education")
    
    result = server.call_tool("design_robot", {
        "task_description": "Design a simple educational robot for teaching programming",
        "constraints": {
            "max_cost": 100.0,  # $100 budget
            "complexity": "simple",
            "safety_level": "educational"
        },
        "estimate_cost": True
    })
    
    print(f"✓ Educational robot design completed")
    specs = result['specifications']
    print(f"  Specifications:")
    print(f"    - Type: {specs['type']}")
    print(f"    - Complexity: {specs.get('complexity', 'N/A')}")
    
    if "cost_estimate" in result:
        cost = result['cost_estimate']
        print(f"  Cost Estimate:")
        print(f"    - Total: ${cost['total']:.2f}")
        print(f"    - Breakdown:")
        for item, price in cost['breakdown'].items():
            print(f"      · {item}: ${price:.2f}")
    
    edu_id = result['design_id']
    
    # 5. Compare designs
    print_section("5. Comparing Robot Designs")
    print("Comparing the mobile robot and manipulator arm designs")
    
    result = server.call_tool("compare_designs", {
        "design_ids": [mobile_id, arm_id],
        "metrics": ["complexity", "versatility", "cost", "performance"]
    })
    
    print(f"✓ Design comparison completed")
    print(f"  Winner: Design {result['winner'][:8]}...")
    print(f"  Detailed scores:")
    
    for i, (design_id, scores) in enumerate(zip([mobile_id, arm_id], result['detailed_scores'])):
        design_type = "Mobile Robot" if i == 0 else "Manipulator Arm"
        print(f"\n  {design_type} ({design_id[:8]}...):")
        total = 0
        for metric, score in scores.items():
            if isinstance(score, (int, float)):
                print(f"    - {metric}: {score:.2f}")
                total += score
        print(f"    - Total: {total:.2f}")
    
    # 6. Get improvement suggestions
    print_section("6. Design Improvement Suggestions")
    print("Getting suggestions for improving the gripper design")
    
    result = server.call_tool("suggest_improvements", {
        "model_id": gripper_id,
        "goals": ["increase_grip_force", "improve_precision", "reduce_weight"]
    })
    
    print(f"✓ Improvement suggestions generated")
    print(f"  Suggestions:")
    for i, suggestion in enumerate(result['suggestions'][:3], 1):
        print(f"\n  {i}. {suggestion['description']}")
        print(f"     Expected improvement: {suggestion['expected_improvement']}")
        if "implementation" in suggestion:
            print(f"     How to implement: {suggestion['implementation']}")
    
    # 7. Refine a design
    print_section("7. Refining a Design")
    print("Refining the gripper based on suggestions")
    
    result = server.call_tool("refine_design", {
        "design_id": gripper_id,
        "improvements": {
            "increase_grip_force": True,
            "add_force_sensing": True,
            "optimize_finger_geometry": True
        },
        "additional_constraints": {
            "max_weight": 0.3,  # 300g
            "sensor_accuracy": 0.01  # 10mN force sensing
        }
    })
    
    print(f"✓ Design refinement completed")
    print(f"  New design ID: {result['design_id'][:8]}...")
    print(f"  Version: {result.get('version', 'N/A')}")
    print(f"  Improvements applied:")
    for improvement in result['improvements_applied']:
        print(f"    - {improvement}")
    
    # 8. Explain design choices
    print_section("8. Understanding Design Decisions")
    print("Getting explanation for the manipulator arm design")
    
    result = server.call_tool("explain_design", {
        "design_id": arm_id
    })
    
    print(f"✓ Design explanation generated")
    print(f"\n  Key design choices:")
    for choice in result['design_choices'][:3]:
        print(f"    - {choice}")
    
    print(f"\n  Rationale:")
    for reason in result['rationale'][:3]:
        print(f"    - {reason}")
    
    # 9. List available components
    print_section("9. Component Library")
    print("Browsing available actuator components")
    
    result = server.call_tool("list_components", {
        "category": "actuator"
    })
    
    print(f"✓ Found {len(result['components'])} actuator components")
    print(f"  Sample components:")
    for comp in result['components'][:3]:
        print(f"\n  {comp['name']}:")
        print(f"    - Type: {comp['type']}")
        print(f"    - Specs: {comp['specifications']}")
        print(f"    - Compatible with: {', '.join(comp['compatible_with'])}")
    
    # 10. Check component compatibility
    print_section("10. Component Compatibility Check")
    print("Checking if components work together")
    
    result = server.call_tool("check_compatibility", {
        "component_a": {
            "type": "motor",
            "name": "High-torque servo",
            "torque": 15.0,
            "voltage": 24.0
        },
        "component_b": {
            "type": "controller",
            "name": "Precision servo controller",
            "voltage": 24.0,
            "channels": 6
        }
    })
    
    print(f"✓ Compatibility check completed")
    print(f"  Compatible: {'Yes' if result['compatible'] else 'No'}")
    if result['compatible']:
        print(f"  Compatibility score: {result.get('compatibility_score', 0):.2f}")
    else:
        print(f"  Reasons:")
        for reason in result['reasons']:
            print(f"    - {reason}")
    
    print_section("Demo Complete!")
    print("\nRobot Designer features demonstrated:")
    print("- Natural language robot design from task descriptions")
    print("- Constraint-based design with optimization")
    print("- Component library integration")
    print("- Cost estimation for budget-conscious projects")
    print("- Design comparison and improvement suggestions")
    print("- Iterative design refinement")
    print("- Explainable AI for understanding design decisions")
    print("\nThe Robot Designer makes complex robot design accessible to everyone!")


if __name__ == "__main__":
    main()
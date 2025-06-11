#!/usr/bin/env python3
"""
Parameter Optimization Demo for MuJoCo MCP v0.4.1

This demo shows how to optimize control parameters for better performance.
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
    """Run parameter optimization demo"""
    # Create server
    server = MuJoCoMCPServer()
    print(f"MuJoCo MCP Server v{server.version} - Parameter Optimization Demo")
    
    # 1. Create a pendulum for optimization
    print_section("1. Setting Up Pendulum Model")
    result = server.call_tool("pendulum_demo", {"action": "setup"})
    model_id = result["model_id"]
    print(f"✓ Created pendulum model")
    print(f"  Model ID: {model_id}")
    
    # 2. List available cost functions
    print_section("2. Available Cost Functions")
    result = server.call_tool("list_cost_functions", {})
    print("Cost functions for optimization:")
    for func in result["cost_functions"][:5]:
        print(f"  - {func['name']}: {func['description']}")
    
    # 3. Optimize for fastest swing time
    print_section("3. Optimizing for Fastest Swing Time")
    print("Objective: Reach 90° as quickly as possible")
    
    result = server.call_tool("optimize_parameters", {
        "model_id": model_id,
        "objective": "minimize_time",
        "target_state": {"angle": 90.0},
        "parameters_to_optimize": ["kp", "ki", "kd"],
        "parameter_bounds": {
            "kp": [0.1, 20.0],
            "ki": [0.0, 5.0],
            "kd": [0.0, 10.0]
        },
        "max_iterations": 15,
        "track_convergence": True
    })
    
    print(f"✓ Optimization completed")
    print(f"  Optimal parameters:")
    for param, value in result["optimal_parameters"].items():
        print(f"    {param}: {value:.3f}")
    print(f"  Optimal time: {result.get('optimal_time', 0):.3f}s")
    print(f"  Performance improvement: {result['performance_improvement']*100:.1f}%")
    
    # Show convergence
    if "convergence_history" in result:
        print("\n  Convergence (first 5 iterations):")
        for i, point in enumerate(result["convergence_history"][:5]):
            print(f"    Iter {i}: cost = {point['cost']:.3f}")
    
    # 4. Optimize for energy efficiency
    print_section("4. Optimizing for Energy Efficiency")
    print("Objective: Minimize control effort while maintaining position")
    
    result = server.call_tool("optimize_parameters", {
        "model_id": model_id,
        "objective": "minimize_energy",
        "target_state": {"angle": 45.0},
        "parameters_to_optimize": ["damping"],
        "parameter_bounds": {
            "damping": [0.01, 2.0]
        },
        "max_iterations": 10
    })
    
    print(f"✓ Energy optimization completed")
    print(f"  Optimal damping: {result['optimal_parameters']['damping']:.3f}")
    print(f"  Energy before: {result.get('energy_before', 0):.3f}")
    print(f"  Energy after: {result.get('energy_after', 0):.3f}")
    print(f"  Energy reduction: {(1 - result.get('energy_after', 1)/max(result.get('energy_before', 1), 0.001))*100:.1f}%")
    
    # 5. Optimize for stability
    print_section("5. Optimizing for Stability")
    print("Objective: Maximize stability at upright position")
    
    result = server.call_tool("optimize_parameters", {
        "model_id": model_id,
        "objective": "maximize_stability",
        "target_state": {"angle": 0.0},
        "parameters_to_optimize": ["kp", "kd"],
        "parameter_bounds": {
            "kp": [5.0, 50.0],
            "kd": [1.0, 20.0]
        }
    })
    
    print(f"✓ Stability optimization completed")
    print(f"  Optimal parameters:")
    for param, value in result["optimal_parameters"].items():
        print(f"    {param}: {value:.3f}")
    print(f"  Stability score: {result.get('stability_score', 0):.3f} (0-1 scale)")
    
    # 6. Multi-objective optimization
    print_section("6. Multi-Objective Optimization")
    print("Objectives: Balance speed and energy efficiency")
    
    result = server.call_tool("optimize_parameters", {
        "model_id": model_id,
        "objective": "multi_objective",
        "objectives": [
            {"type": "minimize_time", "weight": 0.5},
            {"type": "minimize_energy", "weight": 0.5}
        ],
        "target_state": {"angle": 90.0},
        "parameters_to_optimize": ["kp", "ki", "kd"],
        "parameter_bounds": {
            "kp": [0.1, 20.0],
            "ki": [0.0, 5.0],
            "kd": [0.0, 10.0]
        }
    })
    
    print(f"✓ Multi-objective optimization completed")
    if "pareto_front" in result and result["pareto_front"]:
        solution = result["pareto_front"][0]
        print(f"  Balanced solution:")
        for param, value in solution["parameters"].items():
            print(f"    {param}: {value:.3f}")
    
    # 7. Optimization with constraints
    print_section("7. Constrained Optimization")
    print("Objective: Minimize time with overshoot constraints")
    
    result = server.call_tool("optimize_parameters", {
        "model_id": model_id,
        "objective": "minimize_time",
        "target_state": {"angle": 90.0},
        "parameters_to_optimize": ["kp", "kd"],
        "parameter_bounds": {
            "kp": [0.1, 50.0],
            "kd": [0.0, 20.0]
        },
        "constraints": [
            {"type": "max_overshoot", "value": 10.0},
            {"type": "max_control_effort", "value": 1.5}
        ]
    })
    
    print(f"✓ Constrained optimization completed")
    print(f"  Constraints satisfied: {result.get('constraints_satisfied', False)}")
    print(f"  Optimal parameters (with constraints):")
    for param, value in result["optimal_parameters"].items():
        print(f"    {param}: {value:.3f}")
    
    # 8. Parameter sensitivity analysis
    print_section("8. Parameter Sensitivity Analysis")
    print("Analyzing which parameters most affect performance")
    
    result = server.call_tool("analyze_sensitivity", {
        "model_id": model_id,
        "parameters": ["kp", "ki", "kd"],
        "objective": "position_error",
        "target_state": {"angle": 45.0},
        "num_samples": 10
    })
    
    print(f"✓ Sensitivity analysis completed")
    print("  Parameter sensitivities (normalized):")
    for param, sensitivity in result["sensitivity"].items():
        print(f"    {param}: {sensitivity:.3f}")
    print(f"  Most sensitive parameter: {result.get('most_sensitive', 'N/A')}")
    
    # 9. Robustness analysis
    print_section("9. Robustness Analysis")
    print("Testing parameter robustness to perturbations")
    
    # Use some optimized parameters
    test_params = {"kp": 10.0}
    
    result = server.call_tool("analyze_robustness", {
        "model_id": model_id,
        "parameters": test_params,
        "perturbation_range": 0.1,  # ±10%
        "num_tests": 15
    })
    
    print(f"✓ Robustness analysis completed")
    print(f"  Robustness score: {result['robustness_score']:.3f} (0-1 scale)")
    print(f"  Mean performance: {result['mean_performance']:.3f}")
    print(f"  Performance std dev: {result['std_performance']:.3f}")
    print(f"  Worst case: {result['worst_case_performance']:.3f}")
    
    # 10. Save optimization results
    print_section("10. Saving Optimization Results")
    
    result = server.call_tool("optimize_parameters", {
        "model_id": model_id,
        "objective": "minimize_error",
        "target_state": {"angle": 45.0},
        "parameters_to_optimize": ["kp"],
        "parameter_bounds": {"kp": [1.0, 10.0]},
        "save_results": True,
        "results_name": "pendulum_45deg_optimal"
    })
    
    print(f"✓ Optimization results saved")
    print(f"  Results ID: {result.get('results_id', 'N/A')[:8]}...")
    print(f"  Saved as: {result.get('saved_results', {}).get('name', 'N/A')}")
    
    print_section("Demo Complete!")
    print("\nParameter optimization enables:")
    print("- Finding optimal control parameters automatically")
    print("- Multi-objective optimization for balanced solutions")
    print("- Constraint satisfaction for safety requirements")
    print("- Sensitivity and robustness analysis")
    print("- Gradient-free optimization for complex systems")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Performance Monitoring Demo for MuJoCo MCP v0.5.1

This demo shows the performance monitoring and optimization features.
"""

import asyncio
import time
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from mujoco_mcp.server import MuJoCoServer


def print_section(title):
    """Print a section header"""
    print(f"\n{'='*70}")
    print(f" {title}")
    print('='*70)


async def main():
    """Run performance monitoring demo"""
    # Create and initialize server
    server = MuJoCoServer()
    await server.initialize()
    
    print(f"MuJoCo MCP Server v{server.version} - Performance Monitoring Demo")
    
    # 1. Basic Performance Metrics
    print_section("1. Basic Performance Metrics")
    metrics = server._impl._handle_get_performance_metrics()
    print(f"Server uptime: {metrics['uptime']:.2f} seconds")
    print(f"Total requests: {metrics['total_requests']}")
    print(f"Active simulations: {metrics['active_simulations']}")
    print(f"Memory usage: {metrics['memory_usage']['rss_mb']:.1f} MB")
    print(f"CPU usage: {metrics['cpu_percent']:.1f}%")
    print(f"Tool count: {metrics['tool_count']}")
    
    # 2. Enable Performance Tracking
    print_section("2. Enabling Performance Tracking")
    server._impl.call_tool("enable_performance_tracking", {"enabled": True})
    print("✓ Performance tracking enabled")
    
    # 3. Create Some Load
    print_section("3. Creating Simulation Load")
    models = []
    for i in range(3):
        result = server._impl.call_tool("pendulum_demo", {"action": "setup"})
        models.append(result["model_id"])
        print(f"✓ Created simulation {i+1}: {result['model_id'][:8]}...")
    
    # 4. Run Simulations and Track Performance
    print_section("4. Running Simulations")
    print("Stepping each simulation 100 times...")
    
    for i, model_id in enumerate(models):
        start = time.time()
        server._impl.call_tool("step_simulation", {
            "model_id": model_id,
            "steps": 100
        })
        elapsed = time.time() - start
        print(f"  Model {i+1}: 100 steps in {elapsed:.3f}s ({100/elapsed:.1f} steps/s)")
    
    # 5. Get Simulation Metrics
    print_section("5. Simulation Performance Metrics")
    for i, model_id in enumerate(models):
        metrics = server._impl._handle_get_simulation_metrics(model_id)
        print(f"\nModel {i+1} ({model_id[:8]}...):")
        print(f"  Total steps: {metrics['total_steps']}")
        print(f"  Average step time: {metrics['average_step_time']*1000:.2f} ms")
        print(f"  Real-time factor: {metrics['real_time_factor']:.2f}x")
    
    # 6. Tool Performance Metrics
    print_section("6. Tool Performance Metrics")
    tool_metrics = server._impl._handle_get_tool_metrics()
    
    if tool_metrics["tool_calls"]:
        print("Tool call statistics:")
        for tool_name, stats in tool_metrics["tool_calls"].items():
            print(f"\n{tool_name}:")
            print(f"  Calls: {stats['count']}")
            print(f"  Average time: {stats['average_time']*1000:.2f} ms")
            print(f"  Max time: {stats['max_time']*1000:.2f} ms")
    
    # 7. Memory Usage Analysis
    print_section("7. Memory Usage Analysis")
    memory = server._impl._handle_get_memory_usage()
    print(f"Total memory: {memory['total_memory']:.1f} MB")
    print(f"Simulation memory: {memory['simulation_memory']:.1f} MB")
    print(f"Models loaded: {memory['model_count']}")
    print(f"Memory per model: {memory['memory_per_model']:.2f} MB")
    
    # 8. Performance Thresholds and Alerts
    print_section("8. Performance Thresholds and Alerts")
    
    # Set aggressive thresholds to trigger alerts
    server._impl.call_tool("set_performance_thresholds", {
        "max_step_time": 0.001,  # 1ms - very low to trigger alerts
        "max_memory_mb": 100,    # 100MB - might trigger
        "max_cpu_percent": 50    # 50% - might trigger
    })
    print("✓ Set performance thresholds")
    
    # Run more simulations to trigger alerts
    print("\nRunning intensive simulation...")
    for model_id in models:
        server._impl.call_tool("step_simulation", {
            "model_id": model_id,
            "steps": 50
        })
    
    # Check alerts
    alerts = server._impl._handle_get_performance_alerts()
    if alerts["alerts"]:
        print(f"\n⚠️  {len(alerts['alerts'])} performance alerts:")
        for alert in alerts["alerts"][-3:]:  # Show last 3
            print(f"  - {alert['type']}: {alert['message']}")
    else:
        print("\n✓ No performance alerts")
    
    # 9. Batch Operations
    print_section("9. Batch Operations for Better Performance")
    
    print("Running batch step operation...")
    start = time.time()
    batch_result = server._impl.call_tool("batch_step", {
        "model_ids": models,
        "steps": 50
    })
    batch_time = time.time() - start
    
    print(f"✓ Batch stepped {len(models)} models in {batch_time:.3f}s")
    print(f"  Average per model: {batch_time/len(models):.3f}s")
    
    # 10. Parallel Simulation
    print_section("10. Parallel Simulation Execution")
    
    print("Running simulations in parallel...")
    start = time.time()
    parallel_result = server._impl.call_tool("run_parallel", {
        "model_ids": models,
        "duration": 0.1,  # 100ms simulation time
        "timestep": 0.001
    })
    parallel_time = time.time() - start
    
    print(f"✓ Parallel execution completed in {parallel_time:.3f}s")
    for model_id, state in parallel_result["final_states"].items():
        print(f"  {model_id[:8]}...: {state['steps']} steps in {state['elapsed']:.3f}s")
    
    # 11. Performance History
    print_section("11. Performance History")
    
    history = server._impl.call_tool("get_performance_history", {
        "duration": 10,  # Last 10 seconds
        "resolution": "second"
    })
    
    if history["history"]:
        print(f"Performance over last {len(history['history'])} seconds:")
        # Show summary
        cpu_values = [h["metrics"]["cpu_percent"] for h in history["history"]]
        memory_values = [h["metrics"]["memory_mb"] for h in history["history"]]
        print(f"  CPU: min={min(cpu_values):.1f}%, max={max(cpu_values):.1f}%, avg={sum(cpu_values)/len(cpu_values):.1f}%")
        print(f"  Memory: min={min(memory_values):.1f}MB, max={max(memory_values):.1f}MB")
    
    # 12. Resource Management
    print_section("12. Resource Management")
    
    # Set resource limits
    server._impl.call_tool("set_resource_limits", {
        "max_models": 10,
        "max_memory_mb": 1000,
        "max_step_rate": 50000
    })
    
    # Get current usage
    usage = server._impl._handle_get_resource_usage()
    print(f"Resource usage vs limits:")
    print(f"  Models: {usage['model_count']}/{usage['limits']['max_models']}")
    print(f"  Memory: {usage['memory_mb']:.1f}/{usage['limits']['max_memory_mb']} MB")
    print(f"  Step rate: {usage['step_rate']:.0f}/{usage['limits']['max_step_rate']} steps/s")
    
    # 13. Auto Cleanup
    print_section("13. Automatic Resource Cleanup")
    
    # Enable auto cleanup
    server._impl.call_tool("enable_auto_cleanup", {
        "enabled": True,
        "idle_timeout": 30,  # 30 seconds
        "max_age": 300      # 5 minutes
    })
    print("✓ Auto cleanup enabled")
    print("  Idle timeout: 30 seconds")
    print("  Max model age: 5 minutes")
    
    # 14. Performance Summary
    print_section("14. Performance Summary")
    
    final_metrics = server._impl._handle_get_performance_metrics()
    print(f"Session statistics:")
    print(f"  Total requests: {final_metrics['total_requests']}")
    print(f"  Active simulations: {final_metrics['active_simulations']}")
    print(f"  Uptime: {final_metrics['uptime']:.1f} seconds")
    
    # Clear performance data
    server._impl.call_tool("clear_performance_data", {})
    print("\n✓ Performance data cleared")
    
    # Cleanup
    await server.cleanup()
    
    print_section("Demo Complete!")
    print("\nPerformance monitoring features demonstrated:")
    print("- Real-time performance metrics")
    print("- Tool execution tracking")
    print("- Memory usage analysis")
    print("- Performance alerts and thresholds")
    print("- Batch and parallel operations")
    print("- Resource limits and auto-cleanup")
    print("- Performance history tracking")


if __name__ == "__main__":
    asyncio.run(main())
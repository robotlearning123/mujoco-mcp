#!/usr/bin/env python3
"""
Multi-Agent Coordination Demo for MuJoCo MCP v0.5.2

This demo showcases the multi-agent coordination features including:
- Multi-agent world creation
- Agent communication
- Formation control
- Swarm behaviors
- Multi-agent learning
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
    """Run multi-agent coordination demo"""
    # Create and initialize server
    server = MuJoCoServer()
    await server.initialize()
    
    print(f"MuJoCo MCP Server v{server.version} - Multi-Agent Coordination Demo")
    
    # 1. Create Multi-Agent World
    print_section("1. Creating Multi-Agent World")
    
    world_result = server._impl._handle_create_multi_agent_world(
        world_type="collaborative",
        num_agents=4,
        world_size=[20.0, 20.0, 10.0],
        enable_collision_avoidance=True
    )
    
    world_id = world_result["world_id"]
    agent_ids = world_result["agent_ids"]
    
    print(f"✓ Created world: {world_id[:8]}...")
    print(f"✓ Agents: {len(agent_ids)} agents created")
    for i, agent_id in enumerate(agent_ids):
        print(f"  - Agent {i+1}: {agent_id[:16]}...")
    
    # 2. Agent Communication
    print_section("2. Agent Communication")
    
    # Send coordination message
    msg_result = server._impl._handle_send_agent_message(
        world_id=world_id,
        from_agent=agent_ids[0],
        to_agent=agent_ids[1],
        message_type="coordinate",
        content={"task": "patrol", "area": "north"}
    )
    print(f"✓ Message sent from Agent 1 to Agent 2")
    
    # Retrieve messages
    messages = server._impl._handle_get_agent_messages(
        world_id=world_id,
        agent_id=agent_ids[1]
    )
    print(f"✓ Agent 2 received {messages['count']} message(s)")
    if messages['messages']:
        msg = messages['messages'][0]
        print(f"  - Type: {msg['type']}")
        print(f"  - Content: {msg['content']}")
    
    # 3. Formation Control
    print_section("3. Formation Control")
    
    formations = ["line", "triangle", "circle"]
    for formation in formations:
        print(f"\nExecuting {formation} formation...")
        
        result = server._impl._handle_execute_formation(
            world_id=world_id,
            formation_type=formation,
            target_position=[10.0, 10.0, 0.0],
            maintain_spacing=True
        )
        
        print(f"✓ Formation '{formation}' executed")
        print(f"  - Agents moved: {result['agents_moved']}")
        print(f"  - Formation error: {result['formation_error']:.3f}")
        
        # Step simulation
        server._impl._handle_step_multi_agent_world(
            world_id=world_id,
            steps=10,
            sync_mode="lockstep"
        )
    
    # 4. Task Assignment
    print_section("4. Collective Task Assignment")
    
    tasks = [
        {"type": "pickup", "location": [5.0, 5.0, 0.0], "object": "box1"},
        {"type": "transport", "from": [5.0, 5.0, 0.0], "to": [15.0, 15.0, 0.0]},
        {"type": "inspect", "location": [10.0, 5.0, 0.0], "target": "area_a"},
        {"type": "patrol", "waypoints": [[0, 0, 0], [20, 0, 0], [20, 20, 0], [0, 20, 0]]}
    ]
    
    assignment = server._impl._handle_assign_tasks(
        world_id=world_id,
        tasks=tasks,
        assignment_strategy="optimal",
        allow_cooperation=True
    )
    
    print(f"✓ Assigned {assignment['total_tasks']} tasks to {assignment['agents_assigned']} agents")
    print(f"✓ Estimated completion time: {assignment['estimated_completion_time']:.1f}s")
    
    for agent_id, agent_tasks in assignment['assignments'].items():
        print(f"\nAgent {agent_id[:16]}...:")
        for task in agent_tasks:
            print(f"  - {task['type']}: {task.get('location', task.get('target', 'N/A'))}")
    
    # 5. Swarm Behavior
    print_section("5. Swarm Behavior")
    
    # Create a swarm
    swarm_result = server._impl._handle_create_swarm(
        swarm_size=10,
        swarm_type="homogeneous",
        behavior="flocking",
        spawn_area=[5.0, 5.0, 2.0],
        behavior_params={
            "cohesion": 1.0,
            "separation": 0.5,
            "alignment": 0.8
        }
    )
    
    swarm_id = swarm_result["swarm_id"]
    print(f"✓ Created swarm: {swarm_id[:8]}...")
    print(f"✓ Swarm size: {swarm_result['agent_count']} agents")
    print(f"✓ Behavior: {swarm_result['behavior']}")
    
    # Execute flocking behavior
    print("\nExecuting flocking behavior...")
    behavior_result = server._impl._handle_execute_swarm_behavior(
        swarm_id=swarm_id,
        duration=2.0,
        target_direction=[1.0, 0.0, 0.0]
    )
    
    print(f"✓ Flocking scores:")
    print(f"  - Cohesion: {behavior_result['cohesion_score']:.2f}")
    print(f"  - Alignment: {behavior_result['alignment_score']:.2f}")
    print(f"  - Separation: {behavior_result['separation_score']:.2f}")
    
    # Observe emergent behavior
    observation = server._impl._handle_observe_swarm_behavior(
        swarm_id=swarm_id,
        duration=3.0,
        metrics=["clustering", "velocity_variance", "spacing"]
    )
    
    print(f"\n✓ Emergent pattern detected: {observation['pattern_detected']}")
    print(f"✓ Confidence: {observation['confidence']:.2%}")
    print("✓ Behavior metrics:")
    for metric, value in observation['behavior_metrics'].items():
        print(f"  - {metric}: {value:.3f}")
    
    # 6. Multi-Agent Learning
    print_section("6. Multi-Agent Learning")
    
    # Create learning environment
    learning_env = server._impl._handle_create_learning_environment(
        env_type="cooperative",
        num_agents=3,
        shared_rewards=True,
        require_communication=False
    )
    
    env_id = learning_env["env_id"]
    learning_agents = learning_env["agent_ids"]
    
    print(f"✓ Created learning environment: {env_id[:8]}...")
    print(f"✓ Type: cooperative")
    print(f"✓ Agents: {len(learning_agents)}")
    
    # Add some experiences
    print("\nAdding learning experiences...")
    for i in range(5):
        for j, agent_id in enumerate(learning_agents):
            server._impl._handle_add_experience(
                env_id=env_id,
                agent_id=agent_id,
                state=[i * 0.1, j * 0.1],
                action=[0.1, 0.05],
                reward=0.5 + i * 0.1,
                next_state=[(i+1) * 0.1, j * 0.1]
            )
    
    # Train agents
    print("\nTraining agents...")
    training_result = server._impl._handle_train_agents(
        env_id=env_id,
        algorithm="multi_agent_ppo",
        episodes=10,
        coordination_mode="centralized_training"
    )
    
    print(f"✓ Training completed: {training_result['episodes_completed']} episodes")
    print(f"✓ Average improvement: {training_result['average_improvement']:.3f}")
    
    # Enable communication learning
    comm_result = server._impl._handle_enable_communication_learning(
        env_id=env_id,
        message_size=8,
        learning_rate=0.01
    )
    
    print(f"\n✓ Communication learning enabled")
    print(f"✓ Message size: {comm_result['message_size']} bits")
    
    # Test learned communication
    comm_test = server._impl._handle_test_learned_communication(
        env_id=env_id,
        test_scenarios=5
    )
    
    print(f"✓ Communication test results:")
    print(f"  - Information transfer: {comm_test['information_transfer']:.2%}")
    print(f"  - Protocol efficiency: {comm_test['protocol_efficiency']:.2%}")
    print(f"  - Vocabulary size: {comm_test['vocabulary_size']} symbols")
    
    # 7. Performance Summary
    print_section("7. Performance Summary")
    
    # Get collision statistics
    collision_stats = server._impl._handle_get_collision_statistics(world_id=world_id)
    print(f"Collision statistics:")
    print(f"  - Total collisions: {collision_stats['total_collisions']}")
    print(f"  - Near misses: {collision_stats['near_misses']}")
    print(f"  - Avoidance maneuvers: {collision_stats['avoidance_maneuvers']}")
    
    # Share observations
    obs_result = server._impl._handle_share_observations(
        world_id=world_id,
        sharing_mode="broadcast",
        include_processed=True
    )
    
    print(f"\nObservation sharing:")
    print(f"  - Mode: {obs_result['sharing_mode']}")
    print(f"  - Observations shared: {obs_result['observations_shared']}")
    
    # Cleanup
    await server.cleanup()
    
    print_section("Demo Complete!")
    print("\nMulti-agent coordination features demonstrated:")
    print("- Multi-agent world creation and management")
    print("- Agent-to-agent communication")
    print("- Formation control (line, triangle, circle)")
    print("- Task assignment and coordination")
    print("- Swarm behaviors (flocking)")
    print("- Emergent behavior observation")
    print("- Multi-agent learning with shared experiences")
    print("- Communication protocol learning")


if __name__ == "__main__":
    asyncio.run(main())
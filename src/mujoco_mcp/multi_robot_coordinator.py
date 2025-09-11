#!/usr/bin/env python3
"""
Multi-Robot Coordination System for MuJoCo MCP
Enables coordinated control of multiple robots with collision avoidance and task allocation
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor
import threading

from .viewer_client import MuJoCoViewerClient
from .advanced_controllers import RobotController, TrajectoryPlanner


class TaskType(Enum):
    """Types of coordinated tasks"""
    COOPERATIVE_MANIPULATION = "cooperative_manipulation"
    FORMATION_CONTROL = "formation_control"
    SEQUENTIAL_TASKS = "sequential_tasks"
    PARALLEL_TASKS = "parallel_tasks"
    COLLISION_AVOIDANCE = "collision_avoidance"


@dataclass
class RobotState:
    """Robot state information"""
    robot_id: str
    model_type: str
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    end_effector_pos: np.ndarray | None = None
    end_effector_vel: np.ndarray | None = None
    status: str = "idle"
    last_update: float = field(default_factory=time.time)
    
    def is_stale(self, timeout: float = 1.0) -> bool:
        """Check if state is stale"""
        return time.time() - self.last_update > timeout


@dataclass
class CoordinatedTask:
    """Coordinated task definition"""
    task_id: str
    task_type: TaskType
    robots: List[str]
    parameters: Dict[str, Any]
    priority: int = 1
    timeout: float = 30.0
    status: str = "pending"
    start_time: float | None = None
    completion_callback: Callable | None = None


class CollisionChecker:
    """Collision detection and avoidance for multi-robot systems"""
    
    def __init__(self, safety_margin: float = 0.1):
        self.safety_margin = safety_margin
        self.robot_bounding_boxes = {}
    
    def set_robot_bounds(self, robot_id: str, bounds: Dict[str, Tuple[float, float, float]]):
        """Set bounding box for robot"""
        self.robot_bounding_boxes[robot_id] = bounds
    
    def check_collision(self, robot1_state: RobotState, robot2_state: RobotState) -> bool:
        """Check if two robots are in collision"""
        if robot1_state.end_effector_pos is None or robot2_state.end_effector_pos is None:
            return False
        
        # Simple distance-based collision check
        distance = np.linalg.norm(robot1_state.end_effector_pos - robot2_state.end_effector_pos)
        return distance < self.safety_margin
    
    def find_collision_free_path(
        self,
        _robot_id: str,  # Unused but kept for API compatibility
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        other_robots: List[RobotState],
        num_waypoints: int = 10
    ) -> List[np.ndarray]:
        """Find collision-free path using simple potential field method"""
        
        waypoints = []
        current_pos = start_pos.copy()
        
        for i in range(num_waypoints):
            # Direct path point
            alpha = (i + 1) / num_waypoints
            target_pos = (1 - alpha) * start_pos + alpha * goal_pos
            
            # Apply repulsive forces from other robots
            repulsive_force = np.zeros_like(target_pos)
            
            for other_robot in other_robots:
                if other_robot.end_effector_pos is not None:
                    diff = target_pos - other_robot.end_effector_pos
                    distance = np.linalg.norm(diff)
                    
                    if distance < self.safety_margin * 2:
                        # Repulsive force inversely proportional to distance
                        force_magnitude = 1.0 / (distance + 0.01)
                        repulsive_force += force_magnitude * diff / distance
            
            # Combine attractive and repulsive forces
            adjusted_pos = target_pos + 0.1 * repulsive_force
            waypoints.append(adjusted_pos)
            current_pos = adjusted_pos
        
        return waypoints


class TaskAllocator:
    """Task allocation and scheduling for multi-robot systems"""
    
    def __init__(self):
        self.pending_tasks = []
        self.active_tasks = {}
        self.completed_tasks = []
        self.robot_capabilities = {}
    
    def register_robot(self, robot_id: str, capabilities: Dict[str, Any]):
        """Register robot capabilities"""
        self.robot_capabilities[robot_id] = capabilities
    
    def add_task(self, task: CoordinatedTask):
        """Add task to queue"""
        self.pending_tasks.append(task)
        self.pending_tasks.sort(key=lambda t: t.priority, reverse=True)
    
    def allocate_tasks(self, available_robots: List[str]) -> List[CoordinatedTask]:
        """Allocate tasks to available robots"""
        allocated_tasks = []
        
        for task in self.pending_tasks.copy():
            # Check if required robots are available
            if all(robot_id in available_robots for robot_id in task.robots):
                # Check robot capabilities
                if self._check_robot_capabilities(task):
                    self.pending_tasks.remove(task)
                    task.status = "allocated"
                    task.start_time = time.time()
                    allocated_tasks.append(task)
                    
                    # Remove allocated robots from available list
                    for robot_id in task.robots:
                        if robot_id in available_robots:
                            available_robots.remove(robot_id)
        
        return allocated_tasks
    
    def _check_robot_capabilities(self, task: CoordinatedTask) -> bool:
        """Check if robots have required capabilities for task"""
        for robot_id in task.robots:
            capabilities = self.robot_capabilities.get(robot_id, {})
            
            # Basic capability checking
            if task.task_type == TaskType.COOPERATIVE_MANIPULATION:
                if not capabilities.get("manipulation", False):
                    return False
            elif task.task_type == TaskType.FORMATION_CONTROL:
                if not capabilities.get("mobility", False):
                    return False
        
        return True


class MultiRobotCoordinator:
    """Main coordinator for multi-robot systems"""
    
    def __init__(self, viewer_client: MuJoCoViewerClient | None = None):
        self.viewer_client = viewer_client or MuJoCoViewerClient()
        
        # Robot management
        self.robots: Dict[str, RobotController] = {}
        self.robot_states: Dict[str, RobotState] = {}
        self.robot_configs = {}
        
        # Coordination components
        self.collision_checker = CollisionChecker()
        self.task_allocator = TaskAllocator()
        self.trajectory_planner = TrajectoryPlanner()
        
        # Synchronization
        self.state_lock = threading.Lock()
        self.task_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Control loop
        self.running = False
        self.control_frequency = 50.0  # Hz
        self.update_thread = None
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    def add_robot(self, robot_id: str, robot_type: str, capabilities: Dict[str, Any]):
        """Add robot to coordination system"""
        # Robot configurations (inline to avoid import issues)
        robot_configs = {
            "franka_panda": {"joints": 7, "type": "arm", "home_position": [0, -0.785, 0, -2.356, 0, 1.571, 0.785]},
            "ur5e": {"joints": 6, "type": "arm", "home_position": [0, -1.57, 1.57, -1.57, -1.57, 0]},
            "anymal_c": {"joints": 12, "type": "quadruped", "home_position": [0.0] * 12},
            "go2": {"joints": 12, "type": "quadruped", "home_position": [0.0] * 12}
        }
        
        if robot_type in robot_configs:
            config = robot_configs[robot_type]
            self.robot_configs[robot_id] = config
            
            # Create controller
            controller = RobotController(config)
            self.robots[robot_id] = controller
            
            # Initialize state
            initial_state = RobotState(
                robot_id=robot_id,
                model_type=robot_type,
                joint_positions=np.array(config.get("home_position", [0.0] * config["joints"])),
                joint_velocities=np.zeros(config["joints"])
            )
            self.robot_states[robot_id] = initial_state
            
            # Register with task allocator
            self.task_allocator.register_robot(robot_id, capabilities)
            
            self.logger.info(f"Added robot {robot_id} of type {robot_type}")
            return True
        else:
            self.logger.error(f"Unknown robot type: {robot_type}")
            return False
    
    def remove_robot(self, robot_id: str):
        """Remove robot from coordination system"""
        with self.state_lock:
            if robot_id in self.robots:
                del self.robots[robot_id]
                del self.robot_states[robot_id]
                del self.robot_configs[robot_id]
                self.logger.info(f"Removed robot {robot_id}")
    
    def update_robot_state(self, robot_id: str, joint_positions: np.ndarray, joint_velocities: np.ndarray):
        """Update robot state"""
        with self.state_lock:
            if robot_id in self.robot_states:
                state = self.robot_states[robot_id]
                state.joint_positions = joint_positions
                state.joint_velocities = joint_velocities
                state.last_update = time.time()
                
                # Update end effector position (simplified)
                if len(joint_positions) >= 3:
                    # Simple forward kinematics approximation
                    state.end_effector_pos = np.array([
                        joint_positions[0],
                        joint_positions[1],
                        sum(joint_positions[2:])
                    ])
    
    def start_coordination(self):
        """Start coordination control loop"""
        if not self.running:
            self.running = True
            self.update_thread = threading.Thread(target=self._coordination_loop)
            self.update_thread.start()
            self.logger.info("Started multi-robot coordination")
    
    def stop_coordination(self):
        """Stop coordination control loop"""
        self.running = False
        if self.update_thread:
            self.update_thread.join()
        self.logger.info("Stopped multi-robot coordination")
    
    def _coordination_loop(self):
        """Main coordination control loop"""
        dt = 1.0 / self.control_frequency
        
        while self.running:
            start_time = time.time()
            
            try:
                # Update robot states
                self._update_states()
                
                # Process tasks
                self._process_tasks()
                
                # Check collisions
                self._check_collisions()
                
                # Send control commands
                self._send_control_commands()
                
            except Exception as e:
                self.logger.error(f"Error in coordination loop: {e}")
            
            # Maintain control frequency
            elapsed = time.time() - start_time
            sleep_time = max(0, dt - elapsed)
            time.sleep(sleep_time)
    
    def _update_states(self):
        """Update all robot states"""
        # This would typically read from MuJoCo simulation
        # For now, we'll just mark stale states
        with self.state_lock:
            for _robot_id, state in self.robot_states.items():
                if state.is_stale():
                    state.status = "stale"
    
    def _process_tasks(self):
        """Process and allocate tasks"""
        with self.task_lock:
            # Get available robots
            available_robots = [
                robot_id for robot_id, state in self.robot_states.items()
                if state.status in ["idle", "ready"]
            ]
            
            # Allocate new tasks
            allocated_tasks = self.task_allocator.allocate_tasks(available_robots)
            
            for task in allocated_tasks:
                self.task_allocator.active_tasks[task.task_id] = task
                self._execute_task(task)
    
    def _execute_task(self, task: CoordinatedTask):
        """Execute coordinated task"""
        self.logger.info(f"Executing task {task.task_id} of type {task.task_type}")
        
        if task.task_type == TaskType.COOPERATIVE_MANIPULATION:
            self._execute_cooperative_manipulation(task)
        elif task.task_type == TaskType.FORMATION_CONTROL:
            self._execute_formation_control(task)
        elif task.task_type == TaskType.SEQUENTIAL_TASKS:
            self._execute_sequential_tasks(task)
        elif task.task_type == TaskType.PARALLEL_TASKS:
            self._execute_parallel_tasks(task)
    
    def _execute_cooperative_manipulation(self, task: CoordinatedTask):
        """Execute cooperative manipulation task"""
        robots = task.robots
        target_object = task.parameters.get("target_object")
        
        # Plan coordinated trajectories
        for robot_id in robots:
            controller = self.robots[robot_id]
            state = self.robot_states[robot_id]
            
            # Generate approach trajectory
            current_pos = state.joint_positions
            approach_pos = task.parameters.get(f"{robot_id}_approach", current_pos)
            
            waypoints = np.array([current_pos, approach_pos])
            times = np.array([0, 2.0])
            
            controller.set_trajectory(waypoints, times)
            state.status = "executing"
    
    def _execute_formation_control(self, task: CoordinatedTask):
        """Execute formation control task"""
        formation_type = task.parameters.get("formation", "line")
        spacing = task.parameters.get("spacing", 1.0)
        
        robots = task.robots
        n_robots = len(robots)
        
        # Generate formation positions
        if formation_type == "line":
            positions = []
            for i, _robot_id in enumerate(robots):
                x = (i - n_robots/2) * spacing
                positions.append([x, 0, 0])
        elif formation_type == "circle":
            radius = spacing
            positions = []
            for i, _robot_id in enumerate(robots):
                angle = 2 * np.pi * i / n_robots
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                positions.append([x, y, 0])
        
        # Set trajectories for each robot
        for i, robot_id in enumerate(robots):
            controller = self.robots[robot_id]
            state = self.robot_states[robot_id]
            
            current_pos = state.joint_positions[:3]  # Assume first 3 joints for position
            target_pos = np.array(positions[i])
            
            waypoints = np.array([current_pos, target_pos])
            times = np.array([0, 3.0])
            
            controller.set_trajectory(waypoints, times)
            state.status = "executing"
    
    def _execute_sequential_tasks(self, task: CoordinatedTask):
        """Execute tasks in sequence"""
        # Implementation for sequential task execution
        logging.info(f"Executing sequential task: {task.task_id}")
    
    def _execute_parallel_tasks(self, task: CoordinatedTask):
        """Execute tasks in parallel"""
        # Implementation for parallel task execution
        logging.info(f"Executing parallel task: {task.task_id}")
    
    def _check_collisions(self):
        """Check for potential collisions"""
        with self.state_lock:
            robot_ids = list(self.robot_states.keys())
            
            for i in range(len(robot_ids)):
                for j in range(i + 1, len(robot_ids)):
                    robot1_id = robot_ids[i]
                    robot2_id = robot_ids[j]
                    
                    state1 = self.robot_states[robot1_id]
                    state2 = self.robot_states[robot2_id]
                    
                    if self.collision_checker.check_collision(state1, state2):
                        self.logger.warning(f"Collision detected between {robot1_id} and {robot2_id}")
                        self._handle_collision(robot1_id, robot2_id)
    
    def _handle_collision(self, robot1_id: str, robot2_id: str):
        """Handle collision between robots"""
        # Emergency stop
        state1 = self.robot_states[robot1_id]
        state2 = self.robot_states[robot2_id]
        
        state1.status = "collision_stop"
        state2.status = "collision_stop"
        
        # Reset controllers
        self.robots[robot1_id].reset_controllers()
        self.robots[robot2_id].reset_controllers()
    
    def _send_control_commands(self):
        """Send control commands to robots"""
        for robot_id, controller in self.robots.items():
            state = self.robot_states[robot_id]
            
            if state.status == "executing":
                # Get trajectory command
                target_pos = controller.get_trajectory_command()
                
                if target_pos is not None:
                    # Send command to MuJoCo
                    if self.viewer_client and self.viewer_client.connected:
                        command = {
                            "type": "set_joint_positions",
                            "model_id": robot_id,
                            "positions": target_pos.tolist()
                        }
                        self.viewer_client.send_command(command)
                else:
                    # Trajectory complete
                    state.status = "idle"
    
    # High-level task interface
    def cooperative_manipulation(
        self,
        robots: List[str],
        target_object: str,
        approach_positions: Dict[str, np.ndarray]
    ) -> str:
        """Start cooperative manipulation task"""
        task = CoordinatedTask(
            task_id=f"coop_manip_{int(time.time())}",
            task_type=TaskType.COOPERATIVE_MANIPULATION,
            robots=robots,
            parameters={
                "target_object": target_object,
                **{f"{robot_id}_approach": pos for robot_id, pos in approach_positions.items()}
            }
        )
        
        self.task_allocator.add_task(task)
        return task.task_id
    
    def formation_control(
        self,
        robots: List[str],
        formation_type: str = "line",
        spacing: float = 1.0
    ) -> str:
        """Start formation control task"""
        task = CoordinatedTask(
            task_id=f"formation_{int(time.time())}",
            task_type=TaskType.FORMATION_CONTROL,
            robots=robots,
            parameters={
                "formation": formation_type,
                "spacing": spacing
            }
        )
        
        self.task_allocator.add_task(task)
        return task.task_id
    
    def get_task_status(self, task_id: str) -> str | None:
        """Get status of a task"""
        with self.task_lock:
            if task_id in self.task_allocator.active_tasks:
                return self.task_allocator.active_tasks[task_id].status
            
            for task in self.task_allocator.pending_tasks:
                if task.task_id == task_id:
                    return task.status
            
            for task in self.task_allocator.completed_tasks:
                if task.task_id == task_id:
                    return task.status
        
        return None
    
    def get_robot_status(self, robot_id: str) -> Dict[str, Any] | None:
        """Get status of a robot"""
        with self.state_lock:
            if robot_id in self.robot_states:
                state = self.robot_states[robot_id]
                return {
                    "robot_id": state.robot_id,
                    "model_type": state.model_type,
                    "status": state.status,
                    "joint_positions": state.joint_positions.tolist(),
                    "joint_velocities": state.joint_velocities.tolist(),
                    "last_update": state.last_update
                }
        return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        with self.state_lock, self.task_lock:
            return {
                "running": self.running,
                "num_robots": len(self.robots),
                "pending_tasks": len(self.task_allocator.pending_tasks),
                "active_tasks": len(self.task_allocator.active_tasks),
                "completed_tasks": len(self.task_allocator.completed_tasks),
                "robots": {robot_id: state.status for robot_id, state in self.robot_states.items()}
            }
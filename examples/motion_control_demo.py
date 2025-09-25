#!/usr/bin/env python3
"""
Motion Control Demo for MuJoCo MCP
Demonstrates advanced motion control with various Menagerie models

Features:
- Load different robot types from MuJoCo Menagerie
- Multiple control strategies (position, velocity, torque)
- Predefined motion patterns
- Interactive control via MCP
"""

import asyncio
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mujoco_mcp.viewer_client import MuJoCoViewerClient


class MotionControlDemo:
    """Motion control demonstrations for various robot models"""
    
    def __init__(self):
        self.viewer_client = MuJoCoViewerClient()
        self.current_model = None
        self.model_info = {}
        
        # Predefined robot configurations
        self.robot_configs = {
            # Robotic Arms
            "franka_panda": {
                "path": "franka_emika_panda/scene.xml",
                "type": "arm",
                "joints": 7,
                "home_position": [0, -0.785, 0, -2.356, 0, 1.571, 0.785],
                "demo_motions": ["wave", "pick_place", "circle"]
            },
            "ur5e": {
                "path": "universal_robots_ur5e/scene.xml", 
                "type": "arm",
                "joints": 6,
                "home_position": [0, -1.57, 1.57, -1.57, -1.57, 0],
                "demo_motions": ["wave", "pick_place", "spiral"]
            },
            "kuka_iiwa": {
                "path": "kuka_iiwa_14/scene.xml",
                "type": "arm",
                "joints": 7,
                "home_position": [0, 0.7, 0, -1.4, 0, 1.0, 0],
                "demo_motions": ["wave", "figure8", "reach"]
            },
            
            # Quadrupeds
            "anymal_c": {
                "path": "anybotics_anymal_c/scene.xml",
                "type": "quadruped",
                "joints": 12,
                "home_position": [0.0] * 12,
                "demo_motions": ["stand", "walk", "trot"]
            },
            "go2": {
                "path": "unitree_go2/scene.xml",
                "type": "quadruped", 
                "joints": 12,
                "home_position": [0.0] * 12,
                "demo_motions": ["stand", "walk", "jump"]
            },
            "spot": {
                "path": "google_barkour_vb/scene.xml",
                "type": "quadruped",
                "joints": 12,
                "home_position": [0.0] * 12,
                "demo_motions": ["stand", "walk", "dance"]
            },
            
            # Humanoids
            "g1": {
                "path": "unitree_g1/scene.xml",
                "type": "humanoid",
                "joints": 37,
                "home_position": None,  # Use default
                "demo_motions": ["stand", "wave_hand", "walk"]
            },
            "h1": {
                "path": "unitree_h1/scene.xml",
                "type": "humanoid",
                "joints": 25,
                "home_position": None,
                "demo_motions": ["stand", "balance", "squat"]
            },
            
            # Grippers/Hands
            "robotiq_2f85": {
                "path": "robotiq_2f85/scene.xml",
                "type": "gripper",
                "joints": 6,
                "home_position": [0.0] * 6,
                "demo_motions": ["open", "close", "pinch"]
            },
            "shadow_hand": {
                "path": "shadow_hand/scene_right.xml",
                "type": "hand",
                "joints": 24,
                "home_position": None,
                "demo_motions": ["open", "close", "wave", "grasp"]
            }
        }
    
    def get_menagerie_path(self) -> str | None:
        """Get MuJoCo Menagerie path"""
        # Check environment variable
        menagerie_path = os.environ.get('MUJOCO_MENAGERIE_PATH')
        if menagerie_path and Path(menagerie_path).exists():
            return menagerie_path
        
        # Check common locations
        possible_paths = [
            Path.home() / "mujoco_menagerie",
            Path.home() / "Documents" / "mujoco_menagerie",
            Path.home() / "repos" / "mujoco_menagerie",
            Path.cwd() / "mujoco_menagerie"
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
        
        return None
    
    async def connect(self) -> bool:
        """Connect to viewer server"""
        if self.viewer_client.connect():
            print("✅ Connected to MuJoCo viewer server")
            return True
        else:
            print("❌ Failed to connect to viewer server")
            print("   Please start the viewer server: python mujoco_viewer_server.py")
            return False
    
    def load_model(self, model_name: str) -> bool:
        """Load a model from Menagerie"""
        if model_name not in self.robot_configs:
            print(f"❌ Unknown model: {model_name}")
            print(f"   Available models: {', '.join(self.robot_configs.keys())}")
            return False
        
        config = self.robot_configs[model_name]
        menagerie_path = self.get_menagerie_path()
        
        if not menagerie_path:
            print("❌ MuJoCo Menagerie not found!")
            print("   Please set MUJOCO_MENAGERIE_PATH or install mujoco_menagerie")
            return False
        
        model_path = Path(menagerie_path) / config["path"]
        if not model_path.exists():
            print(f"❌ Model file not found: {model_path}")
            return False
        
        # Load model
        response = self.viewer_client.send_command({
            "type": "load_model",
            "model_xml": str(model_path),
            "model_id": model_name
        })
        
        if response.get("success"):
            self.current_model = model_name
            self.model_info = response.get("model_info", {})
            print(f"✅ Loaded {model_name} model")
            print(f"   Type: {config['type']}")
            print(f"   Joints: {config['joints']}")
            print(f"   Available demos: {', '.join(config['demo_motions'])}")
            return True
        else:
            print(f"❌ Failed to load model: {response.get('error')}")
            return False
    
    def get_state(self) -> Dict | None:
        """Get current robot state"""
        response = self.viewer_client.send_command({
            "type": "get_state",
            "model_id": self.current_model
        })
        
        if response.get("success"):
            return response
        return None
    
    def set_joint_positions(self, positions: List[float]) -> bool:
        """Set joint positions"""
        response = self.viewer_client.send_command({
            "type": "set_joint_positions",
            "model_id": self.current_model,
            "positions": positions
        })
        return response.get("success", False)
    
    def go_home(self) -> bool:
        """Move robot to home position"""
        if not self.current_model:
            print("❌ No model loaded")
            return False
        
        config = self.robot_configs[self.current_model]
        if config["home_position"]:
            print(f"🏠 Moving {self.current_model} to home position...")
            return self.set_joint_positions(config["home_position"])
        else:
            print("   Using default position")
            return True
    
    # ========== Motion Patterns ==========
    
    def wave_motion(self, duration: float = 5.0):
        """Wave motion for arms"""
        print("👋 Performing wave motion...")
        start_time = time.time()
        
        config = self.robot_configs[self.current_model]
        base_pos = config["home_position"] or [0.0] * config["joints"]
        
        while time.time() - start_time < duration:
            t = time.time() - start_time
            positions = base_pos.copy()
            
            # Modify specific joints for waving
            if config["type"] == "arm":
                # Wave with "elbow" and "wrist"
                if config["joints"] >= 4:
                    positions[1] = base_pos[1] + 0.5 * math.sin(2 * t)  # Shoulder
                    positions[3] = base_pos[3] + 0.3 * math.sin(2 * t + 0.5)  # Elbow
            elif config["type"] == "humanoid":
                # Wave right arm
                if self.current_model == "g1" and config["joints"] >= 37:
                    positions[19] = 0.5 * math.sin(2 * t)  # Right shoulder
                    positions[21] = 0.3 * math.sin(2 * t + 0.5)  # Right elbow
            
            self.set_joint_positions(positions)
            time.sleep(0.02)
    
    def circle_motion(self, radius: float = 0.2, duration: float = 10.0):
        """Circular end-effector motion for arms"""
        print("⭕ Performing circular motion...")
        start_time = time.time()
        
        config = self.robot_configs[self.current_model]
        base_pos = config["home_position"] or [0.0] * config["joints"]
        
        while time.time() - start_time < duration:
            t = time.time() - start_time
            angle = 2 * math.pi * t / 5.0  # One circle per 5 seconds
            
            positions = base_pos.copy()
            
            # Simple circular motion with first few joints
            if config["joints"] >= 3:
                positions[0] = base_pos[0] + radius * math.cos(angle)
                positions[1] = base_pos[1] + radius * math.sin(angle) * 0.5
                positions[2] = base_pos[2] + radius * math.sin(angle)
            
            self.set_joint_positions(positions)
            time.sleep(0.02)
    
    def walk_motion(self, duration: float = 10.0):
        """Walking motion for quadrupeds"""
        print("🚶 Performing walk motion...")
        start_time = time.time()
        
        config = self.robot_configs[self.current_model]
        base_pos = [0.0] * config["joints"]
        
        # Simple gait pattern
        phase_offset = [0, math.pi, math.pi, 0]  # Diagonal gait
        
        while time.time() - start_time < duration:
            t = time.time() - start_time
            
            positions = base_pos.copy()
            
            # Each leg has 3 joints (hip, knee, ankle)
            for leg in range(4):
                phase = 2 * math.pi * t / 2.0 + phase_offset[leg]
                
                # Hip joint - swing forward/backward
                positions[leg * 3] = 0.2 * math.sin(phase)
                
                # Knee joint - lift leg
                positions[leg * 3 + 1] = 0.1 + 0.1 * abs(math.sin(phase))
                
                # Ankle joint
                positions[leg * 3 + 2] = -0.1
            
            self.set_joint_positions(positions)
            time.sleep(0.02)
    
    def gripper_motion(self, duration: float = 5.0):
        """Open/close motion for grippers"""
        print("✋ Performing gripper motion...")
        start_time = time.time()
        
        config = self.robot_configs[self.current_model]
        
        while time.time() - start_time < duration:
            t = time.time() - start_time
            
            # Open and close gripper
            value = 0.5 + 0.5 * math.sin(t)
            positions = [value] * config["joints"]
            
            self.set_joint_positions(positions)
            time.sleep(0.02)
    
    def run_demo(self, demo_name: str, duration: float = 5.0):
        """Run a specific demo motion"""
        if not self.current_model:
            print("❌ No model loaded")
            return
        
        config = self.robot_configs[self.current_model]
        
        if demo_name not in config["demo_motions"]:
            print(f"❌ Demo '{demo_name}' not available for {self.current_model}")
            print(f"   Available: {', '.join(config['demo_motions'])}")
            return
        
        # Map demo names to functions
        demo_map = {
            "wave": self.wave_motion,
            "circle": self.circle_motion,
            "walk": self.walk_motion,
            "trot": self.walk_motion,  # Same as walk for now
            "open": lambda: self.gripper_motion(duration),
            "close": lambda: self.gripper_motion(duration),
            "stand": self.go_home,
        }
        
        if demo_name in demo_map:
            demo_map[demo_name](duration)
        else:
            print(f"⚠️  Demo '{demo_name}' not implemented yet")
    
    def interactive_control(self):
        """Interactive control mode"""
        print("\n🎮 Interactive Control Mode")
        print("Commands:")
        print("  list    - List available models")
        print("  load <model> - Load a model")
        print("  home    - Go to home position")
        print("  demo <name> - Run a demo motion")
        print("  state   - Show current state")
        print("  quit    - Exit")
        
        while True:
            try:
                cmd = input("\n> ").strip().lower().split()
                if not cmd:
                    continue
                
                if cmd[0] == "quit":
                    break
                
                elif cmd[0] == "list":
                    print("\n📋 Available models:")
                    for name, config in self.robot_configs.items():
                        print(f"  {name:<15} - {config['type']:<10} ({config['joints']} joints)")
                
                elif cmd[0] == "load" and len(cmd) > 1:
                    self.load_model(cmd[1])
                
                elif cmd[0] == "home":
                    self.go_home()
                
                elif cmd[0] == "demo" and len(cmd) > 1:
                    duration = float(cmd[2]) if len(cmd) > 2 else 5.0
                    self.run_demo(cmd[1], duration)
                
                elif cmd[0] == "state":
                    state = self.get_state()
                    if state:
                        print(f"Time: {state.get('time', 0):.2f}")
                        print(f"Positions: {[f'{p:.3f}' for p in state.get('qpos', [])]}")
                
                else:
                    print("❌ Unknown command")
                    
            except KeyboardInterrupt:
                print("\n👋 Exiting...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


async def main():
    """Main demo function"""
    print("🤖 MuJoCo Motion Control Demo")
    print("="*50)
    
    demo = MotionControlDemo()
    
    # Connect to viewer
    if not await demo.connect():
        return
    
    # Show some example sequences
    print("\n📖 Example sequences:")
    print("1. Franka Panda arm waving")
    print("2. Unitree Go2 walking")
    print("3. Shadow Hand grasping")
    
    try:
        choice = input("\nRun example (1-3) or interactive mode (i)? ").strip().lower()
        
        if choice == "1":
            # Franka Panda demo
            if demo.load_model("franka_panda"):
                demo.go_home()
                time.sleep(1)
                demo.wave_motion(5)
                time.sleep(1)
                demo.circle_motion(0.2, 10)
        
        elif choice == "2":
            # Unitree Go2 demo
            if demo.load_model("go2"):
                demo.go_home()
                time.sleep(1)
                demo.walk_motion(10)
        
        elif choice == "3":
            # Shadow Hand demo
            if demo.load_model("shadow_hand"):
                demo.gripper_motion(5)
        
        elif choice == "i":
            # Interactive mode
            demo.interactive_control()
        
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted")
    
    # Cleanup
    if demo.viewer_client.connected:
        demo.viewer_client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
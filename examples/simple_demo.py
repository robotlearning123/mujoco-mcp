#!/usr/bin/env python3
"""
MuJoCo Simplified Demo Script
Demonstrates basic MuJoCo functionality

Usage:
python simple_demo.py
"""

import sys
import os
import time
import numpy as np
import mujoco
import argparse
import logging
from typing import Dict, Any, List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mujoco_simple_demo')

# Example MuJoCo model XML (simple manipulator and some objects)
EXAMPLE_MODEL_XML = """
<mujoco>
    <worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
        <geom name="floor" type="plane" size="5 5 0.1" rgba=".9 .9 .9 1"/>
        
        <!-- Simple robot -->
        <body name="robot1" pos="0 0 0.5">
            <joint name="robot1_base_rot" type="ball"/>
            <geom name="robot1_base" type="cylinder" size="0.1 0.1" rgba="0.7 0.7 0.7 1"/>
            <body name="robot1_arm" pos="0 0 0.1">
                <joint name="robot1_shoulder" axis="0 1 0" range="-180 180"/>
                <geom name="robot1_arm_geom" type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" rgba="0.7 0.7 0.7 1"/>
                <body name="robot1_forearm" pos="0.5 0 0">
                    <joint name="robot1_elbow" axis="0 1 0" range="-90 90"/>
                    <geom name="robot1_forearm_geom" type="capsule" size="0.04" fromto="0 0 0 0.5 0 0" rgba="0.7 0.7 0.7 1"/>
                    <body name="robot1_wrist" pos="0.5 0 0">
                        <joint name="robot1_wrist_rot" axis="1 0 0" range="-180 180"/>
                        <geom name="robot1_wrist_geom" type="sphere" size="0.05" rgba="0.5 0.5 0.5 1"/>
                        <site name="robot1_grasp_site" pos="0 0 0.1" size="0.02"/>
                    </body>
                </body>
            </body>
        </body>
        
        <!-- 场景中的物体 -->
        <body name="red_cube" pos="1 0 0.1">
            <joint type="free"/>
            <geom type="box" size="0.1 0.1 0.1" rgba="1 0 0 1"/>
        </body>
        
        <body name="blue_cube" pos="0 1 0.1">
            <joint type="free"/>
            <geom type="box" size="0.1 0.1 0.1" rgba="0 0 1 1"/>
        </body>
        
        <body name="green_sphere" pos="-1 0 0.1">
            <joint type="free"/>
            <geom type="sphere" size="0.1" rgba="0 1 0 1"/>
        </body>
    </worldbody>
    
    <actuator>
        <motor name="robot1_base_rot_motor" joint="robot1_base_rot"/>
        <motor name="robot1_shoulder_motor" joint="robot1_shoulder"/>
        <motor name="robot1_elbow_motor" joint="robot1_elbow"/>
        <motor name="robot1_wrist_motor" joint="robot1_wrist_rot"/>
    </actuator>
</mujoco>
"""

class MuJoCoSimulation:
    """MuJoCo模拟类"""
    
    def __init__(self, model_xml: str):
        """初始化MuJoCo模拟"""
        logger.info("初始化MuJoCo模拟...")
        self.model = mujoco.MjModel.from_xml_string(model_xml)
        self.data = mujoco.MjData(self.model)
        
        # 查找物体和机器人部件
        self._find_bodies()
        
        # 创建渲染上下文
        self.create_renderer()
        
        logger.info("模拟初始化完成")
    
    def _find_bodies(self):
        """Find objects and robot parts in the scene"""
        self.body_names = {}
        self.robot_parts = {}
        
        # Get all body indices and names
        for i in range(self.model.nbody):
            name = self.model.body(i).name
            if name:
                self.body_names[name] = i
                
                # Identify robot parts
                if name.startswith("robot1_"):
                    self.robot_parts[name] = i
        
        logger.info(f"找到物体: {list(self.body_names.keys())}")
    
    def create_renderer(self):
        """创建渲染上下文"""
        try:
            # 创建一个离屏渲染上下文
            self.renderer = mujoco.Renderer(self.model, 640, 480)
            logger.info("渲染上下文创建成功")
        except Exception as e:
            logger.error(f"创建渲染上下文失败: {str(e)}")
            self.renderer = None
    
    def step(self, num_steps: int = 1):
        """步进模拟"""
        for _ in range(num_steps):
            mujoco.mj_step(self.model, self.data)
    
    def reset(self):
        """重置模拟"""
        mujoco.mj_resetData(self.model, self.data)
        logger.info("模拟已重置")
    
    def render(self):
        """渲染当前场景"""
        if self.renderer:
            self.renderer.update_scene(self.data)
            return self.renderer.render()
        return None
    
    def set_joint_positions(self, positions: List[float]):
        """设置关节位置"""
        for i, pos in enumerate(positions):
            if i < self.model.nq:
                self.data.qpos[i] = pos
        mujoco.mj_forward(self.model, self.data)
    
    def get_joint_positions(self) -> List[float]:
        """获取关节位置"""
        return self.data.qpos.copy()
    
    def apply_control(self, control: List[float]):
        """应用控制信号"""
        for i, ctrl in enumerate(control):
            if i < self.model.nu:
                self.data.ctrl[i] = ctrl
    
    def get_body_position(self, body_name: str) -> Optional[List[float]]:
        """获取刚体位置"""
        if body_name in self.body_names:
            body_id = self.body_names[body_name]
            return self.data.body(body_id).xpos.copy()
        return None
    
    def apply_force(self, body_name: str, force: List[float]):
        """对刚体应用力"""
        if body_name in self.body_names:
            body_id = self.body_names[body_name]
            self.data.xfrc_applied[body_id, :3] = force
            logger.info(f"对 {body_name} 施加力 {force}")
    
    def move_robot_arm(self, shoulder_angle: float, elbow_angle: float, wrist_angle: float):
        """移动机器人手臂"""
        # 找到相关关节的索引
        shoulder_idx = -1
        elbow_idx = -1
        wrist_idx = -1
        
        for i in range(self.model.njnt):
            name = self.model.joint(i).name
            if name == "robot1_shoulder":
                shoulder_idx = self.model.joint(i).qposadr[0]
            elif name == "robot1_elbow":
                elbow_idx = self.model.joint(i).qposadr[0]
            elif name == "robot1_wrist_rot":
                wrist_idx = self.model.joint(i).qposadr[0]
        
        # 设置关节角度
        if shoulder_idx >= 0:
            self.data.qpos[shoulder_idx] = np.deg2rad(shoulder_angle)
        if elbow_idx >= 0:
            self.data.qpos[elbow_idx] = np.deg2rad(elbow_angle)
        if wrist_idx >= 0:
            self.data.qpos[wrist_idx] = np.deg2rad(wrist_angle)
        
        # 更新模拟
        mujoco.mj_forward(self.model, self.data)
        logger.info(f"移动机器人手臂到 肩膀={shoulder_angle}°, 肘部={elbow_angle}°, 手腕={wrist_angle}°")
    
    def move_to_target(self, target_pos: List[float], steps: int = 100):
        """移动机器人手臂到目标位置"""
        # 这是一个非常简化的运动规划器，真实情况下需要更复杂的逆运动学
        
        # 获取末端执行器位置
        wrist_pos = self.get_body_position("robot1_wrist")
        if wrist_pos is None:
            logger.error("找不到机器人手腕")
            return
        
        # 计算到目标的方向向量
        direction = np.array(target_pos) - wrist_pos
        distance = np.linalg.norm(direction)
        
        logger.info(f"开始移动到目标位置 {target_pos}, 距离 {distance:.2f}")
        
        # 逐步移动
        for i in range(steps):
            # 获取当前关节角度
            qpos = self.get_joint_positions()
            
            # 简单的基于梯度的控制
            # 注意: 这不是真正的逆运动学，只是一个简化的演示
            shoulder_idx = -1
            elbow_idx = -1
            
            for j in range(self.model.njnt):
                name = self.model.joint(j).name
                if name == "robot1_shoulder":
                    shoulder_idx = self.model.joint(j).qposadr[0]
                elif name == "robot1_elbow":
                    elbow_idx = self.model.joint(j).qposadr[0]
            
            # 使用简单的启发式方法调整关节角度
            wrist_pos = self.get_body_position("robot1_wrist")
            direction = np.array(target_pos) - wrist_pos
            
            # 调整肩部角度
            if shoulder_idx >= 0:
                qpos[shoulder_idx] += np.sign(direction[0]) * 0.01
            
            # 调整肘部角度
            if elbow_idx >= 0:
                qpos[elbow_idx] += np.sign(direction[2]) * 0.01
            
            # 更新位置
            self.set_joint_positions(qpos)
            
            # 步进模拟
            self.step(5)
            
            # 检查是否接近目标
            wrist_pos = self.get_body_position("robot1_wrist")
            distance = np.linalg.norm(np.array(target_pos) - wrist_pos)
            
            if distance < 0.2:
                logger.info(f"已接近目标位置, 剩余距离 {distance:.2f}")
                break
            
            if i % 10 == 0:
                logger.info(f"移动中... 步骤 {i}, 剩余距离 {distance:.2f}")
    
    def grasp_object(self, object_name: str):
        """抓取物体(简化版)"""
        # 获取物体位置
        object_pos = self.get_body_position(object_name)
        if object_pos is None:
            logger.error(f"找不到物体 {object_name}")
            return
        
        logger.info(f"尝试抓取 {object_name} 在位置 {object_pos}")
        
        # 移动到物体位置上方
        target_pos = object_pos.copy()
        target_pos[2] += 0.2  # 稍微在物体上方
        
        self.move_to_target(target_pos)
        
        # 模拟抓取操作(在真实MuJoCo中，这通常涉及创建约束)
        logger.info(f"已抓取 {object_name}")


def run_demo():
    """运行演示"""
    logger.info("开始MuJoCo简化演示")
    
    # 创建模拟
    sim = MuJoCoSimulation(EXAMPLE_MODEL_XML)
    
    # 重置模拟
    sim.reset()
    
    # 显示场景中的物体
    for name in sim.body_names:
        pos = sim.get_body_position(name)
        if pos is not None:
            logger.info(f"物体 {name}: 位置 {pos}")
    
    # 步骤1: 移动机器人手臂
    logger.info("步骤1: 移动机器人手臂")
    sim.move_robot_arm(30, 45, 0)
    sim.step(100)  # 步进模拟让动作生效
    
    # 步骤2: 接近红色立方体
    logger.info("步骤2: 移动到红色立方体")
    red_cube_pos = sim.get_body_position("red_cube")
    if red_cube_pos is not None:
        # 移动到立方体上方
        target_pos = red_cube_pos.copy()
        target_pos[2] += 0.3  # 立方体上方0.3单位
        sim.move_to_target(target_pos)
    
    # 步骤3: 模拟抓取
    logger.info("步骤3: 抓取红色立方体")
    sim.grasp_object("red_cube")
    
    # 步骤4: 应用一些控制信号
    logger.info("步骤4: 应用控制信号")
    for i in range(5):
        # 应用随机控制信号
        control = np.random.uniform(-1, 1, sim.model.nu)
        sim.apply_control(control.tolist())
        sim.step(20)
        logger.info(f"应用控制 {control}")
    
    # 步骤5: 对绿色球施加力
    logger.info("步骤5: 对绿色球施加力")
    sim.apply_force("green_sphere", [10.0, 0.0, 5.0])
    for _ in range(5):
        sim.step(20)
        pos = sim.get_body_position("green_sphere")
        logger.info(f"绿色球位置: {pos}")
    
    logger.info("演示完成")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="MuJoCo简化演示")
    parser.add_argument("--verbose", "-v", action="store_true", help="输出详细信息")
    args = parser.parse_args()
    
    # Setup logging级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 运行演示
    run_demo()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n演示已终止")
    except Exception as e:
        print(f"错误: {str(e)}")
    finally:
        print("演示已结束") 
#!/usr/bin/env python3
"""
Basic example: Demonstrates how to start and interact with MuJoCo MCP server
"""

import sys
import time
import threading
import logging
from model_context_protocol import MCPClient

# Import mujoco_mcp module
try:
    import mujoco_mcp
except ImportError:
    print("mujoco_mcp module not found. Please ensure the module is installed.")
    print("Try running: pip install -e .")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_example")

# MuJoCo model XML - simple pendulum
PENDULUM_XML = """
<mujoco model="pendulum">
  <option timestep="0.01" />
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="1 1 0.1" rgba=".9 .9 .9 1"/>
    <body name="pendulum" pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.1"/>
      <geom name="pole" type="capsule" size="0.045 0.5" rgba="0 0.7 0.7 1" fromto="0 0 0 0 0 -1"/>
      <site name="tip" pos="0 0 -1" size="0.01"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="torque" joint="hinge" gear="1" ctrllimited="true" ctrlrange="-2.0 2.0"/>
  </actuator>
  <sensor>
    <jointpos name="joint_pos" joint="hinge"/>
    <jointvel name="joint_vel" joint="hinge"/>
  </sensor>
</mujoco>
"""


def start_server():
    """在单独的线程中启动MCP服务器"""
    logger.info("正在启动MuJoCo MCP服务器...")
    server_thread = threading.Thread(
        target=mujoco_mcp.start,
        kwargs={"host": "localhost", "port": 8000, "blocking": True},
        daemon=True,
    )
    server_thread.start()
    time.sleep(1)  # 给服务器一些启动时间
    return server_thread


def run_client():
    """运行MCP客户端示例"""
    logger.info("正在连接到MuJoCo MCP服务器...")
    client = MCPClient("http://localhost:8000")

    # 启动模拟
    logger.info("正在启动新的模拟...")
    result = client.call_tool("start_simulation", {"model_xml": PENDULUM_XML})
    sim_id = result["simulation_id"]
    logger.info(f"模拟ID: {sim_id}")

    # 获取模拟信息
    sim_info = client.get_resource("simulation_info", {"simulation_id": sim_id})
    logger.info(f"模拟信息: {sim_info}")

    # 重置模拟
    client.call_tool("reset_simulation", {"simulation_id": sim_id})

    # 进行控制循环
    logger.info("开始控制循环, 运行50步...")
    for i in range(50):
        # 获取关节位置和速度
        positions = client.get_resource("joint_positions", {"simulation_id": sim_id})
        velocities = client.get_resource("joint_velocities", {"simulation_id": sim_id})

        # 应用简单的控制 - 向下拉动摆锤
        control = [1.0 * (i % 10)]  # 每10步改变方向
        client.call_tool("apply_control", {"simulation_id": sim_id, "control": control})

        # 获取传感器数据
        sensors = client.get_resource("sensor_data", {"simulation_id": sim_id})

        # 打印状态
        if i % 5 == 0:  # 每5步打印一次
            logger.info(f"步骤 {i}: 位置={positions}, 速度={velocities}, 控制={control}")
            logger.info(f"传感器: {sensors}")

        # 向前推进模拟
        client.call_tool("step_simulation", {"simulation_id": sim_id, "num_steps": 1})
        time.sleep(0.01)  # 稍微减慢循环以便观察

    # 清理
    logger.info("正在Delete模拟...")
    client.call_tool("delete_simulation", {"simulation_id": sim_id})
    logger.info("示例完成!")


def main():
    """主程序"""
    logger.info("启动基本MuJoCo MCP示例")

    server_thread = start_server()

    try:
        run_client()
    except KeyboardInterrupt:
        logger.info("用户中断，正在关闭...")
    except Exception as e:
        logger.exception(f"发生错误: {str(e)}")
    finally:
        # 停止服务器
        logger.info("正在停止服务器...")
        mujoco_mcp.stop()
        # 等待服务器线程结束
        server_thread.join(timeout=2)
        logger.info("程序退出")


if __name__ == "__main__":
    main()

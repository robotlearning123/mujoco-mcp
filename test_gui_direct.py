#!/usr/bin/env mjpython
"""
直接测试MuJoCo GUI显示
"""

import mujoco
import mujoco.viewer
import time

# 创建一个简单的单摆模型
pendulum_xml = """
<mujoco>
    <option gravity="0 0 -9.81" timestep="0.01"/>
    <worldbody>
        <body name="pendulum" pos="0 0 1">
            <joint name="hinge" type="hinge" axis="1 0 0"/>
            <geom name="rod" type="cylinder" size="0.02 0.3" 
                  pos="0 0 -0.3" rgba="0.8 0.4 0.4 1"/>
            <geom name="mass" type="sphere" size="0.08" 
                  pos="0 0 -0.6" mass="0.5" rgba="0.4 0.4 0.8 1"/>
        </body>
    </worldbody>
    <actuator>
        <motor joint="hinge" gear="1"/>
    </actuator>
</mujoco>
"""

print("🧪 测试MuJoCo GUI显示...")

try:
    # 创建模型和数据
    model = mujoco.MjModel.from_xml_string(pendulum_xml)
    data = mujoco.MjData(model)
    
    print("✅ 模型创建成功")
    print(f"   DOF: {model.nq}")
    print(f"   Bodies: {model.nbody}")
    
    # 启动被动查看器
    print("🖥️ 启动GUI查看器...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("✅ GUI已启动！")
        print("   - 你应该看到一个MuJoCo窗口")
        print("   - 窗口中有一个红色的单摆")
        print("   - 可以用鼠标拖动视角")
        print("\n⏰ 运行30秒...")
        
        start_time = time.time()
        while viewer.is_running() and time.time() - start_time < 30:
            step_start = time.time()
            
            # 步进仿真
            mujoco.mj_step(model, data)
            
            # 同步到查看器
            viewer.sync()
            
            # 控制帧率
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        
        print("\n✅ 测试完成！")
        
except Exception as e:
    print(f"\n❌ 错误: {e}")
    print("\n可能的原因：")
    print("1. 没有使用mjpython运行")
    print("2. macOS权限问题")
    print("3. 显示设备问题")
    
print("\n💡 正确的运行方式：")
print("   mjpython test_gui_direct.py")
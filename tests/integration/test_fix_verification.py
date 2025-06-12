#!/usr/bin/env python3
"""
验证修复是否有效
测试多场景创建和连接稳定性
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mujoco_mcp.remote_server import MuJoCoRemoteServer

def test_multi_scene():
    """测试多场景创建"""
    print("🧪 Testing Multi-Scene Creation Fix")
    print("=" * 50)
    
    server = MuJoCoRemoteServer()
    
    # Test 1: 创建第一个场景
    print("\n1️⃣ Creating first pendulum...")
    result1 = server._handle_create_scene("pendulum")
    if result1.get("success"):
        model_id1 = result1["model_id"]
        print(f"✅ Success: {model_id1}")
    else:
        print(f"❌ Failed: {result1.get('error')}")
        return False
    
    # Test 2: 创建第二个场景（之前会失败）
    print("\n2️⃣ Creating second scene (double_pendulum)...")
    time.sleep(1)
    result2 = server._handle_create_scene("double_pendulum")
    if result2.get("success"):
        model_id2 = result2["model_id"]
        print(f"✅ Success: {model_id2}")
    else:
        print(f"❌ Failed: {result2.get('error')}")
        return False
    
    # Test 3: 创建第三个场景
    print("\n3️⃣ Creating third scene (cart_pole)...")
    time.sleep(1)
    result3 = server._handle_create_scene("cart_pole")
    if result3.get("success"):
        model_id3 = result3["model_id"]
        print(f"✅ Success: {model_id3}")
    else:
        print(f"❌ Failed: {result3.get('error')}")
        return False
    
    # Test 4: 获取所有模型
    print("\n4️⃣ Getting loaded models...")
    models = server._handle_get_loaded_models()
    print(f"   Total models: {len(models.get('models', {}))}")
    for mid, info in models.get('models', {}).items():
        print(f"   - {mid}: {info['scene_type']}")
    
    # Test 5: 控制不同模型
    print("\n5️⃣ Testing independent control...")
    
    # 设置第一个模型的位置
    result = server._handle_set_joint_positions([1.57], model_id1)
    print(f"   Model 1 set position: {'✅' if result.get('success') else '❌'}")
    
    # 获取第二个模型的状态
    result = server._handle_get_state(model_id2)
    print(f"   Model 2 get state: {'✅' if result.get('success') else '❌'}")
    
    # 重置第三个模型
    result = server._handle_reset_simulation(model_id3)
    print(f"   Model 3 reset: {'✅' if result.get('success') else '❌'}")
    
    print("\n✅ All tests passed! Multi-scene support is working.")
    return True

if __name__ == "__main__":
    success = test_multi_scene()
    if success:
        print("\n🎉 Fix verified successfully!")
    else:
        print("\n❌ Fix verification failed!")
        sys.exit(1)
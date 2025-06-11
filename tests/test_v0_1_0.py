"""
测试 v0.1.0 - 最小可运行版本
目标: 修复现有问题，确保基础功能可运行
"""
import pytest
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


class TestBasicFunctionality:
    """测试基础功能是否正常"""
    
    def test_import_all_modules(self):
        """确保所有模块可以正确导入"""
        # 这些导入应该不会抛出异常
        import mujoco_mcp
        import mujoco_mcp.server
        import mujoco_mcp.simulation
        import mujoco_mcp.enhanced_auth_manager
        
        # 验证模块确实被导入
        assert mujoco_mcp is not None
        assert mujoco_mcp.server is not None
        assert mujoco_mcp.simulation is not None
        assert mujoco_mcp.enhanced_auth_manager is not None
        
    def test_version_exists(self):
        """测试版本信息是否存在"""
        from mujoco_mcp import __version__
        assert __version__ is not None
        assert isinstance(__version__, str)
        assert len(__version__) > 0
        
    def test_create_simulation(self):
        """测试创建基础仿真实例"""
        from mujoco_mcp.simulation import MuJoCoSimulation
        
        # 应该能创建实例
        sim = MuJoCoSimulation()
        assert sim is not None
        
        # 初始状态应该是未加载模型
        assert sim.model is None
        assert sim.data is None
        
    def test_load_simple_model_from_string(self):
        """测试从字符串加载简单的XML模型"""
        from mujoco_mcp.simulation import MuJoCoSimulation
        
        # 最简单的有效MuJoCo模型
        xml_string = """
        <mujoco>
            <worldbody>
                <body name="box">
                    <geom name="box_geom" type="box" size="0.1 0.1 0.1"/>
                </body>
            </worldbody>
        </mujoco>
        """
        
        sim = MuJoCoSimulation()
        sim.load_model_from_string(xml_string)
        
        # 验证模型已加载
        assert sim.model is not None
        assert sim.data is not None
        
        # 验证基本属性
        assert sim.model.nbody > 0  # 至少有一个body
        assert sim.model.ngeom > 0  # 至少有一个geom
        
    def test_simulation_step(self):
        """测试仿真步进功能"""
        from mujoco_mcp.simulation import MuJoCoSimulation
        
        xml_string = """
        <mujoco>
            <option timestep="0.002"/>
            <worldbody>
                <body name="box">
                    <freejoint/>
                    <geom name="box_geom" type="box" size="0.1 0.1 0.1"/>
                </body>
            </worldbody>
        </mujoco>
        """
        
        sim = MuJoCoSimulation()
        sim.load_model_from_string(xml_string)
        
        # 记录初始时间
        initial_time = sim.data.time
        assert initial_time == 0.0
        
        # 执行一步仿真
        sim.step()
        
        # 验证时间已前进
        assert sim.data.time > initial_time
        assert sim.data.time == pytest.approx(0.002)  # 一个时间步长
        
    def test_simulation_reset(self):
        """测试仿真重置功能"""
        from mujoco_mcp.simulation import MuJoCoSimulation
        
        xml_string = """
        <mujoco>
            <worldbody>
                <body name="box" pos="0 0 1">
                    <freejoint/>
                    <geom name="box_geom" type="box" size="0.1 0.1 0.1"/>
                </body>
            </worldbody>
        </mujoco>
        """
        
        sim = MuJoCoSimulation()
        sim.load_model_from_string(xml_string)
        
        # 执行几步仿真
        for _ in range(10):
            sim.step()
            
        # 验证时间已前进
        assert sim.data.time > 0
        
        # 重置仿真
        sim.reset()
        
        # 验证时间回到0
        assert sim.data.time == 0.0
        
    def test_get_model_info(self):
        """测试获取模型信息"""
        from mujoco_mcp.simulation import MuJoCoSimulation
        
        xml_string = """
        <mujoco model="test_model">
            <worldbody>
                <body name="link1">
                    <joint name="joint1" type="hinge" axis="0 0 1"/>
                    <geom name="geom1" type="box" size="0.1 0.1 0.1"/>
                </body>
            </worldbody>
        </mujoco>
        """
        
        sim = MuJoCoSimulation()
        sim.load_model_from_string(xml_string)
        
        info = sim.get_model_info()
        
        # 验证返回的信息
        assert isinstance(info, dict)
        assert info["nq"] == 1  # 一个关节
        assert info["nv"] == 1  # 一个速度
        assert info["nbody"] == 2  # world + link1
        assert info["njoint"] == 1  # 一个关节
        
    def test_enhanced_auth_manager_import(self):
        """测试增强认证管理器可以导入和使用"""
        from mujoco_mcp.enhanced_auth_manager import EnhancedAuthManager
        
        # 创建实例
        auth_manager = EnhancedAuthManager()
        assert auth_manager is not None
        
        # 测试基本功能
        assert hasattr(auth_manager, 'validate_request')
        assert hasattr(auth_manager, 'check_rate_limit')
        
    def test_error_handling_invalid_xml(self):
        """测试无效XML的错误处理"""
        from mujoco_mcp.simulation import MuJoCoSimulation
        
        sim = MuJoCoSimulation()
        
        # 应该抛出异常
        with pytest.raises(Exception):
            sim.load_model_from_string("invalid xml")
            
    def test_error_handling_empty_model(self):
        """测试空模型的错误处理"""
        from mujoco_mcp.simulation import MuJoCoSimulation
        
        sim = MuJoCoSimulation()
        
        # 空的mujoco标签应该失败
        with pytest.raises(Exception):
            sim.load_model_from_string("<mujoco></mujoco>")


class TestProjectStructure:
    """测试项目结构是否正确"""
    
    def test_required_files_exist(self):
        """测试必需的文件是否存在"""
        required_files = [
            "src/mujoco_mcp/__init__.py",
            "src/mujoco_mcp/server.py",
            "src/mujoco_mcp/simulation.py",
            "src/mujoco_mcp/enhanced_auth_manager.py",
            "src/mujoco_mcp/version.py",
        ]
        
        for file_path in required_files:
            full_path = project_root / file_path
            assert full_path.exists(), f"Missing required file: {file_path}"
            
    def test_package_metadata(self):
        """测试包元数据是否正确"""
        import mujoco_mcp
        
        # 应该有这些属性
        assert hasattr(mujoco_mcp, '__version__')
        assert hasattr(mujoco_mcp, '__author__')
        assert hasattr(mujoco_mcp, '__email__')
        

if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])
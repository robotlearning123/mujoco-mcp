"""
Simplified conftest.py for v0.8 tests
避免复杂的依赖导入，专注于基础测试
"""

import pytest


@pytest.fixture(autouse=True)
def simple_setup():
    """简化的测试设置，不导入复杂模块"""
    # 不做任何复杂导入，只是确保测试环境清洁
    yield
    # 测试后清理
    pass


@pytest.fixture
def mock_viewer():
    """模拟viewer，避免GUI依赖"""
    class MockViewer:
        def close(self):
            pass
        
        def sync(self):
            pass
    
    return MockViewer()
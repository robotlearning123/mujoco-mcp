"""
Test package structure and basic functionality
"""
import pytest
import sys
from pathlib import Path

def test_package_import():
    """Test that the package can be imported"""
    import mujoco_mcp
    assert mujoco_mcp is not None

def test_version_import():
    """Test that version can be imported"""
    from mujoco_mcp.version import __version__
    assert __version__ is not None
    assert isinstance(__version__, str)
    assert len(__version__) > 0

def test_main_module():
    """Test that main module exists"""
    from mujoco_mcp import __main__
    assert __main__ is not None
    assert hasattr(__main__, 'main')

def test_mcp_server_module():
    """Test that MCP server module exists"""
    from mujoco_mcp import mcp_server
    assert mcp_server is not None

def test_viewer_client_module():
    """Test that viewer client module exists"""
    from mujoco_mcp import viewer_client
    assert viewer_client is not None

@pytest.mark.unit
def test_cli_entry_point():
    """Test CLI entry point without running it"""
    from mujoco_mcp.__main__ import main, parse_args, check_configuration
    
    # Test that functions exist
    assert callable(main)
    assert callable(parse_args)
    assert callable(check_configuration)

@pytest.mark.unit
def test_dependencies():
    """Test that required dependencies can be imported"""
    # Test MuJoCo
    try:
        import mujoco
        print(f"✓ MuJoCo version: {mujoco.__version__}")
    except ImportError:
        pytest.skip("MuJoCo not installed")
    
    # Test MCP
    try:
        import mcp
        print("✓ MCP package available")
    except ImportError:
        pytest.fail("MCP package not installed")
    
    # Test NumPy
    try:
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
    except ImportError:
        pytest.fail("NumPy not installed")

@pytest.mark.unit
def test_package_structure():
    """Test that package has expected structure"""
    package_root = Path(__file__).parent.parent / "src" / "mujoco_mcp"
    
    expected_files = [
        "__init__.py",
        "__main__.py", 
        "version.py",
        "mcp_server.py",
        "viewer_client.py"
    ]
    
    for file_name in expected_files:
        file_path = package_root / file_name
        assert file_path.exists(), f"Missing required file: {file_name}"
"""
Simplified conftest.py for v0.8 tests
Avoid complex dependency imports, focus on basic testing
"""

import pytest


@pytest.fixture(autouse=True)
def simple_setup():
    """Simplified test setup, no complex module imports"""
    # No complex imports, just ensure clean test environment
    return
    # Cleanup after tests


@pytest.fixture
def mock_viewer():
    """Mock viewer, avoid GUI dependencies"""

    class MockViewer:
        def close(self):
            pass

        def sync(self):
            pass

    return MockViewer()

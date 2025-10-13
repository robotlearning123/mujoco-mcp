"""Pytest configuration for Mujoco MCP tests."""

from __future__ import annotations

import asyncio
import inspect

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Item) -> bool | None:
    """Allow ``async def`` tests to run without pytest-asyncio."""
    test_func = getattr(pyfuncitem, "obj", None)
    if inspect.iscoroutinefunction(test_func):
        asyncio.run(test_func(**pyfuncitem.funcargs))
        return True
    return None

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
        # Filter out pytest-asyncio specific arguments
        funcargs = {
            k: v for k, v in pyfuncitem.funcargs.items()
            if k not in ('event_loop', 'event_loop_policy')
        }
        asyncio.run(test_func(**funcargs))
        return True
    return None

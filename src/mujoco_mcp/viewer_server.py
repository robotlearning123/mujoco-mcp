"""MuJoCo Viewer Server access helpers.

This module exposes the interactive viewer server class used by integration and
RL tests while keeping a simple CLI entry point for package builds. The actual
implementation lives in the top-level ``mujoco_viewer_server`` module so that it
can be launched both from source checkouts and from installed environments.
"""

from __future__ import annotations

import importlib
import logging
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional

_LOGGER = logging.getLogger("mujoco_mcp.viewer_server")


def _load_viewer_module() -> ModuleType:
    """Return the runtime viewer server module, importing it on demand.

    The viewer server lives at the repository root (``mujoco_viewer_server.py``).
    When the package is installed in editable mode this file sits alongside the
    package sources, so we extend ``sys.path`` with the project root before
    importing. If the module cannot be imported we surface a helpful error message
    instead of failing silently.
    """

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.append(str(root))

    try:
        return importlib.import_module("mujoco_viewer_server")
    except Exception as exc:  # pragma: no cover - executed only when missing deps
        raise ImportError(
            "Unable to import mujoco_viewer_server. Ensure the viewer server script "
            "is available in the project root or installed alongside the package"
        ) from exc


def get_viewer_class() -> type:
    """Fetch ``MuJoCoViewerServer`` from the runtime module."""

    module = _load_viewer_module()
    try:
        return getattr(module, "MuJoCoViewerServer")
    except AttributeError as exc:  # pragma: no cover - defensive guard
        raise ImportError(
            "mujoco_viewer_server does not define MuJoCoViewerServer"
        ) from exc


class MuJoCoViewerServer(get_viewer_class()):  # type: ignore[misc]
    """Proxy subclass so imports continue to work unchanged.

    ``get_viewer_class`` is evaluated at import time and returns the concrete
    implementation. Subclassing keeps backward compatibility for user code that
    expects ``mujoco_mcp.viewer_server.MuJoCoViewerServer`` to be instantiable.
    """

    pass


def _resolve_script_path() -> Path:
    """Locate the standalone viewer server script used by the CLI."""

    module = _load_viewer_module()
    path = Path(getattr(module, "__file__", ""))
    if not path:
        raise FileNotFoundError("Unable to resolve mujoco_viewer_server.py path")
    return path


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point that spawns the viewer server in a child process."""

    logging.basicConfig(level=logging.INFO)
    script_path = _resolve_script_path()
    cmd = [sys.executable, str(script_path)]

    try:
        completed = subprocess.run(cmd, check=True)
        return completed.returncode
    except subprocess.CalledProcessError as exc:
        _LOGGER.exception("Viewer server exited with error")
        return exc.returncode
    except KeyboardInterrupt:  # pragma: no cover - interactive convenience
        _LOGGER.info("Viewer server interrupted by user")
        return 130
    except Exception as exc:  # pragma: no cover - defensive
        _LOGGER.exception("Unexpected viewer server failure: %s", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover - CLI shim
    sys.exit(main())

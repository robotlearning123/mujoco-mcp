#!/usr/bin/env python3
"""
MuJoCo Viewer Server Launcher
Simplified viewer server for package distribution
"""

import sys
import subprocess
import logging
from pathlib import Path

def main():
    """Main entry point for viewer server"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("mujoco-mcp-viewer")

    # Find the viewer server script
    script_path = Path(__file__).parent.parent.parent / "mujoco_viewer_server.py"

    if not script_path.exists():
        # Try relative to current working directory
        script_path = Path("mujoco_viewer_server.py")

    if not script_path.exists():
        logger.error("Could not find mujoco_viewer_server.py")
        logger.error("Please run from the mujoco-mcp directory or ensure the viewer server is in your PATH")
        sys.exit(1)

    logger.info(f"Starting MuJoCo Viewer Server from {script_path}")

    try:
        # Launch the viewer server
        subprocess.run([sys.executable, str(script_path)], check=True)
    except KeyboardInterrupt:
        logger.info("Viewer server stopped by user")
    except subprocess.CalledProcessError as e:
        logger.exception(f"Viewer server failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

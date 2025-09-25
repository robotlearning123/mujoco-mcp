#!/usr/bin/env python3
"""
MuJoCo MCP Server CLI Entry Point
Handles proper asyncio event loop management
"""
import asyncio
import sys
import logging
import argparse
import os

from .version import __version__


def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description=f"MuJoCo MCP Server v{__version__}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m mujoco_mcp                    # Start with defaults
  python -m mujoco_mcp --port 8080       # Custom port
  python -m mujoco_mcp --debug           # Enable debug logging
  python -m mujoco_mcp --host 0.0.0.0    # Listen on all interfaces
        """
    )

    parser.add_argument(
        "--host",
        default=os.getenv("MUJOCO_MCP_HOST", "localhost"),
        help="Host to bind to (default: localhost)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("MUJOCO_MCP_PORT", "8000")),
        help="Port to bind to (default: 8000)"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=os.getenv("MUJOCO_MCP_LOG_LEVEL", "INFO"),
        help="Logging level (default: INFO)"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (equivalent to --log-level DEBUG)"
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"MuJoCo MCP Server v{__version__}"
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Check configuration and exit"
    )

    return parser.parse_args()




def check_configuration():
    """Check configuration and dependencies"""
    print(f"MuJoCo MCP Server v{__version__}")
    print("Configuration Check:")

    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"✓ Python version: {python_version}")

    # Check dependencies
    try:
        import mujoco
        print(f"✓ MuJoCo version: {mujoco.__version__}")
    except ImportError:
        print("✗ MuJoCo not installed")
        return False

    try:
        import mcp
        print("✓ MCP package available")
    except ImportError:
        print("✗ MCP package not installed")
        return False

    try:
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
    except ImportError:
        print("✗ NumPy not installed")
        return False

    # Check environment variables
    print("\nEnvironment Variables:")
    env_vars = ["MUJOCO_MCP_HOST", "MUJOCO_MCP_PORT", "MUJOCO_MCP_LOG_LEVEL"]
    for var in env_vars:
        value = os.getenv(var, "not set")
        print(f"  {var}: {value}")

    print("\n✓ Configuration check passed")
    return True


def main():
    """Main entry point for CLI - runs MCP server via stdio"""
    args = parse_args()

    # Handle debug flag
    if args.debug:
        args.log_level = "DEBUG"

    # Configuration check
    if args.check:
        success = check_configuration()
        sys.exit(0 if success else 1)

    # Import and run MCP server
    try:
        from .mcp_server import main as mcp_main
        asyncio.run(mcp_main())
    except KeyboardInterrupt:
        print("\nMCP server stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"MCP server failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

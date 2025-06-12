from setuptools import setup, find_packages

setup(
    name="mujoco-mcp",
    version="0.6.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "mujoco>=2.3.0",
        "mcp",
        "mcp-server-fastmcp>=0.1.0",
        "numpy>=1.22.0",
        "pydantic>=2.0.0",
    ],
    description="MuJoCo Model Context Protocol server implementation",
    author="MuJoCo MCP Team",
    python_requires=">=3.8",
) 
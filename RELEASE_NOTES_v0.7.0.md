# MuJoCo MCP v0.7.0 Release Notes

## 🎉 Major Release: Enhanced Stability & Multi-Model Architecture

We're excited to announce the release of MuJoCo MCP v0.7.0, featuring significant improvements to connection stability, a new multi-model architecture, and enhanced error handling.

### 🚀 What's New

#### Multi-Model Architecture
- Introduced `ModelViewer` class for managing individual simulations
- Support for multiple concurrent models (with MuJoCo GUI limitations)
- Each model now has its own unique ID for independent control

#### Enhanced Socket Communication
- **Multi-threaded server**: Handle multiple client connections simultaneously
- **Dynamic buffer management**: Support for models up to 1MB in size
- **Improved connection handling**: From 1 to 10 concurrent connections
- **Better error recovery**: Automatic reconnection and graceful degradation

#### Protocol Improvements
- All MCP tools now accept `model_id` parameter
- Standardized response format with proper line endings
- Enhanced error messages for better debugging

### 🛠️ Technical Improvements

- **Connection Stability**: Fixed socket timeout issues that caused frequent disconnections
- **Buffer Management**: Dynamic receive buffers prevent data truncation
- **Thread Safety**: Proper locking mechanisms for concurrent operations
- **Performance**: Response time improved from <100ms to <50ms typical

### 📊 Test Results

- Created comprehensive test suite with 20+ test scenarios
- Connection success rate improved from ~15% to ~85%
- Multi-scene creation now partially supported (GUI limitations remain)

### ⚠️ Known Limitations

- **MuJoCo GUI Constraint**: One viewer window per process (MuJoCo limitation)
- **Multi-Model Visualization**: Full support requires process separation (planned for v0.8.0)

### 📦 Installation

```bash
# Clone the repository
git clone https://github.com/robotlearning123/mujoco-mcp.git
cd mujoco-mcp

# Install dependencies
pip install -e .

# Start the viewer server
mjpython mujoco_viewer_server.py  # macOS
python mujoco_viewer_server.py     # Linux/Windows
```

### 🔄 Upgrade Guide

From v0.6.x:
1. Replace `mujoco_viewer_server.py` with the new version
2. Update API calls to include `model_id` parameter
3. No changes needed to Claude Desktop configuration

### 👥 Contributors

Thanks to everyone who contributed to this release through testing, bug reports, and feedback!

### 📚 Documentation

- [README.md](README.md) - Updated quick start guide
- [CHANGELOG.md](CHANGELOG.md) - Detailed change history
- [FIX_REPORT_v062.md](FIX_REPORT_v062.md) - Technical details of improvements

### 🔮 What's Next

Version 0.8.0 will focus on:
- Full multi-window support using process separation
- Advanced physics constraints
- Performance monitoring tools
- Integration with MuJoCo MPC

### 🐛 Bug Reports

Please report any issues on our [GitHub Issues](https://github.com/robotlearning123/mujoco-mcp/issues) page.

---

**Full Changelog**: https://github.com/robotlearning123/mujoco-mcp/compare/v0.6.2...v0.7.0
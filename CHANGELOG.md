# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.2] - 2025-06-14

### Major Transformation - Enterprise Robotics Platform

#### Added
- **Advanced Control Algorithms**
  - PID controllers with anti-windup
  - Minimum jerk trajectory planning
  - Spline-based path planning
  - Model predictive control (MPC)
  - Adaptive and force control strategies

- **Multi-Robot Coordination**
  - Formation control (line, circle, custom)
  - Cooperative manipulation tasks
  - Task allocation and scheduling
  - Collision detection and avoidance
  - Distributed control architecture

- **Sensor Feedback Systems**
  - Multi-modal sensor processing (IMU, force/torque, joints)
  - Sensor fusion with weighted averaging
  - Closed-loop feedback control
  - Low-pass and complementary filters
  - Automatic sensor calibration

- **Reinforcement Learning Integration**
  - Gymnasium-compatible environments
  - Multiple task types (reaching, balancing, walking)
  - Configurable reward functions
  - Training utilities and baselines
  - Real-time visualization during training

- **Physics Benchmarking Suite**
  - Performance testing (FPS, CPU, memory)
  - Accuracy validation against analytical solutions
  - Stability analysis with energy conservation
  - Scalability testing with multiple objects
  - Automated benchmark reporting

- **Enhanced Viewer Server**
  - Connection pooling (50+ concurrent connections)
  - Performance monitoring and diagnostics
  - Automatic resource cleanup
  - Graceful shutdown handling
  - Memory leak prevention

- **Real-time Visualization Tools**
  - Interactive Plotly dashboards
  - Real-time joint position plotting
  - 3D trajectory visualization
  - Performance analytics
  - Data export (JSON/NPZ)

#### Documentation
- Created comprehensive DOCUMENTATION_INDEX.md
- Added complete API_REFERENCE.md
- Created detailed ARCHITECTURE.md
- Updated ADVANCED_FEATURES_GUIDE.md
- Enhanced README.md with professional layout

#### Technical Improvements
- Production-ready error handling
- Comprehensive logging system
- Thread-safe operations
- Resource management
- Connection stability

## [0.8.1] - 2025-06-13

### Security and Localization Update

#### Changed
- Removed all Chinese text and replaced with English
- Fixed security issues with private keys in configs
- Removed absolute local paths (/Users/robert, /opt/miniconda3)
- Simplified project structure from 200+ files to ~25 core files

#### Fixed
- Import errors in server.py
- Test failures in comprehensive_test.py
- Configuration path issues

## [0.8.0] - 2025-06-13

### MuJoCo Menagerie Integration

#### Added
- Support for loading models from MuJoCo Menagerie
- Motion control demo with 9 robot types
- Natural language control for Menagerie models
- Comprehensive motion patterns (wave, walk, circle, grasp)

## [0.7.4] - 2025-06-13

### Fixed
- **Hand Model Scene Loading**: Fixed Shadow Hand and Leap Hand models to load with ground planes
  - Added special handling for models with `scene_left.xml` and `scene_right.xml`
  - Shadow Hand now loads with ground plane and manipulation objects
  - Leap Hand now loads with proper ground plane
  - Automatically selects scene_left.xml as default for hand models

## [0.7.3] - 2025-06-13

### Added
- **Render Capture Feature**: Fetch rendered images from MuJoCo simulations
  - New `capture_render` MCP tool for screenshot capture
  - Real-time offscreen rendering using MuJoCo's Renderer class
  - Configurable resolution (default: 640x480)
  - Base64-encoded PNG output for easy transport
  - Works with all models and scenes
- **Video Recording Feature**: Create videos from simulations
  - New `record_video` MCP tool for video capture
  - MP4 video output with H.264 codec
  - GIF animation support
  - Configurable frame rate (default: 30 fps)
  - Real-time physics capture during recording
  - Automatic format detection from file extension

### Fixed
- **Menagerie Model Loading**: Critical fix for path resolution
  - Changed from `from_xml_string()` to `from_xml_path()` for proper relative path handling
  - All mesh files and assets now load correctly
  - Scene.xml files load with complete environments including ground planes
  - No modification of relative paths needed
  - Preserves original XML structure and paths

### Enhanced
- **Model Loading System**: Improved file handling
  - `menagerie_loader.py` now returns file paths instead of XML content
  - `mujoco_viewer_server.py` detects and uses appropriate loading method
  - Better support for complex scenes with multiple includes
  - Automatic scene.xml prioritization for complete environments

### Technical
- New `video_recorder.py` module for video capture functionality
- Updated `viewer_client.py` with capture_render() method
- Modified `ModelViewer` class to support both XML strings and file paths
- Added PIL and ffmpeg-python as optional dependencies for video features
- Comprehensive test suite for render capture and video recording

### Tested
- ✅ All Menagerie models load with proper ground planes
- ✅ Render capture works at various resolutions
- ✅ Video recording produces valid MP4 and GIF files
- ✅ Path resolution handles complex nested includes
- ✅ Backwards compatibility maintained for existing code

## [0.7.2] - 2025-06-13

### Added
- **UI Close Functionality**: Complete control over MuJoCo GUI lifecycle
  - New `close_viewer` MCP tool for direct GUI window closure
  - Natural language command support: "close viewer", "close ui", "exit", "quit"
  - Automatic resource cleanup: model state, client connections, memory
  - Enhanced help menu with UI control commands
- **Resource Management**: Improved system resource handling
  - Proper GUI window closure with MuJoCo viewer.close()
  - Client connection cleanup and disconnection
  - Model state synchronization between server and GUI
  - Memory leak prevention and system resource release

### Enhanced
- **Natural Language Processing**: Extended command recognition
  - Multiple close command variations supported
  - Intelligent command parsing for UI control actions
  - Consistent response format across all close operations
- **User Experience**: Complete simulation environment control
  - Create simulations → Control simulations → Close GUI seamlessly
  - Perfect for interactive sessions and resource management
  - Programmatic and conversational UI control options

### Technical
- New `close_viewer` command in MuJoCo Viewer Server
- Enhanced `viewer_client.py` with close_viewer() and shutdown_server() methods  
- Improved `remote_server.py` with _handle_close_viewer() implementation
- Comprehensive test suite for UI closure functionality
- Thread-safe viewer state management

### Tested
- ✅ No UI close scenarios handled gracefully
- ✅ Active GUI closure with complete cleanup verification
- ✅ Natural language command processing for all close variations
- ✅ Model state consistency after closure operations
- ✅ Client connection management and resource release

## [0.8.0] - 2025-01-06

### Added
- **MuJoCo Menagerie Integration**: Full support for Google DeepMind's robot model library
  - Access to 56+ high-quality robot models through MCP
  - New `list_menagerie_models` tool with category filtering
  - Automatic model discovery and loading
  - Support for all model variants (with/without hands, MJX-compatible)
- **Enhanced Natural Language**: Load Menagerie models with natural commands
  - "load franka panda", "create spot robot", "show unitree go2"
  - "list menagerie models", "show available robots"
  - "list menagerie arms", "show quadruped robots"
- **Smart Model Resolution**: Intelligent model name matching
  - Handles spaces, underscores, and hyphens
  - Automatic fallback from built-in to Menagerie models
  - Robust XML path fixing for all asset types

### Technical
- New `MenagerieLoader` class for modular model loading
- Environment variable `MUJOCO_MENAGERIE_PATH` support
- Comprehensive path resolution for meshes, textures, and includes
- Full backward compatibility with v0.7.x maintained

### Performance
- Model discovery completes in < 1 second
- Model loading optimized for < 1 second response
- Efficient caching of model information

## [0.7.1] - 2025-01-13

### Fixed
- **Enhanced Natural Language Commands**: Significantly improved `execute_command` functionality
  - Added support for double pendulum creation: "create double pendulum"
  - Added support for robotic arm creation: "create robotic arm"
  - Improved command parsing for step simulation with custom step counts
  - Added angle setting with degree-to-radian conversion
  - Added "help" command to list all available commands
  - Better error messages when no active simulation exists

### Added
- **Help Command**: New "help" or "?" command shows all available natural language commands
- **Improved Command Recognition**: More flexible parsing for various command phrasings

### Changed
- Natural language commands now provide clearer feedback and suggestions
- Better default model_id handling for commands when multiple models exist


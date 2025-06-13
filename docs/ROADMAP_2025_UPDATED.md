# MuJoCo MCP Roadmap 2025 - Updated with Official Sources

## Current Status: v0.7.1
- ✅ Basic physics simulation (pendulum, cart-pole)
- ✅ Natural language control
- ✅ External viewer architecture
- ❌ Real robot models
- ❌ GPU acceleration
- ❌ Training capabilities

## Roadmap Overview

### Phase 1: MuJoCo Menagerie Integration (v0.8.0) - Q1 2025
**Priority**: CRITICAL - Foundation for everything else  
**Timeline**: 5-6 weeks

#### Why Menagerie First?
- Provides the robot models needed by other components
- MuJoCo Playground downloads Menagerie during installation
- Essential for moving beyond toy physics problems

#### Implementation Plan
```python
# Based on official Menagerie usage pattern
from robot_descriptions import panda_mj_description
model = mujoco.MjModel.from_xml_path(panda_mj_description.MJCF_PATH)
```

#### Key Features
1. **Model Loading**
   - Support both `robot_descriptions` package method
   - Direct XML/asset loading from cloned repo
   - Handle model quality grades (A+ to C)

2. **Model Categories** (from official repo)
   - Arms (Franka Panda, UR5e, etc.)
   - Bipeds (Cassie, etc.)
   - Quadrupeds (Spot, Go2, A1, etc.)
   - Humanoids (H1, Apollo, etc.)
   - End-effectors (Allegro Hand, Shadow Hand)
   - Mobile bases and manipulators

3. **Technical Requirements**
   - Check minimum MuJoCo version per model
   - Handle asset paths correctly
   - Support both `<model>.xml` and `scene.xml`

### Phase 2: Documentation Auto-Update System (v0.9.0) - Q2 2025
**Priority**: HIGH - Maintain compatibility  
**Timeline**: 3 weeks

#### Monitoring Targets
```python
sources = {
    "mujoco": "https://github.com/google-deepmind/mujoco",
    "menagerie": "https://github.com/google-deepmind/mujoco_menagerie", 
    "playground": "https://github.com/google-deepmind/mujoco_playground",
    "mcp": "https://github.com/anthropics/mcp"
}
```

#### Features
- Git commit monitoring
- Breaking change detection
- Auto-generate compatibility reports
- Version requirement updates

### Phase 3: MuJoCo Playground Integration (v1.0.0) - Q3 2025
**Priority**: HIGH - Advanced capabilities  
**Timeline**: 8-10 weeks

#### Official Information (2025)
- **Purpose**: "GPU-accelerated robot learning and sim-to-real transfer"
- **Installation**: `pip install playground`
- **Built with**: MJX (GPU-accelerated MuJoCo)
- **Supports**: "quadrupeds, humanoids, dexterous hands, and robotic arms"

#### Integration Challenges
1. **GPU Requirement**
   - Playground uses MJX for GPU acceleration
   - Our current setup is CPU-based
   - Need to handle GPU/CPU fallback

2. **Training vs Control**
   - Playground focuses on RL training
   - MCP focuses on direct control
   - Need to bridge paradigms

3. **Dependencies**
   - Playground downloads Menagerie on install
   - Need to coordinate model management

#### Architecture Proposal
```python
class PlaygroundIntegration:
    def __init__(self):
        # Check GPU availability
        self.device = "gpu" if mjx_available else "cpu"
        
        # Initialize playground environments
        import mujoco_playground
        
    def create_training_env(self, env_name: str):
        """Create Playground training environment"""
        # Use Playground's env creation
        pass
        
    def extract_trained_policy(self):
        """Get trained policy for MCP control"""
        pass
```

### Phase 4: Advanced Features (v1.1.0+) - Q4 2025

#### Based on Playground Capabilities
1. **Sim-to-Real Transfer**
   - "Zero-shot sim-to-real transfer"
   - Support both state and pixel inputs

2. **Multi-Robot Support**
   - Playground tested on "5+ robot platforms"
   - Coordinate multiple simulations

3. **Vision-Based Control**
   - Playground supports pixel inputs
   - Add camera/vision tools to MCP

## Technical Stack Evolution

### Current (v0.7.1)
```
Claude → MCP Server → Socket → MuJoCo Viewer
         (Python)     (TCP)     (CPU-based)
```

### Target (v1.0.0)
```
Claude → MCP Server → Model Manager → MuJoCo Core
         (Python)     (Menagerie)    (CPU/GPU)
                           ↓
                      Playground
                      (Training)
```

## Dependencies Update

```toml
[project.dependencies]
# Current
mujoco = ">=3.0.0"
numpy = ">=1.24.0"
pydantic = ">=2.0.0"

[project.optional-dependencies]
# v0.8.0 - Menagerie
menagerie = [
    "robot-descriptions>=1.0.0",  # Easy model loading
]

# v1.0.0 - Playground  
playground = [
    "playground>=0.1.0",          # Official package
    "jax>=0.4.0",                # Required by MJX
    "jaxlib>=0.4.0",             # GPU support
]
```

## Key Decisions

### 1. Model Management Strategy
- **Option A**: Use `robot-descriptions` package (easier)
- **Option B**: Clone Menagerie repo (more control)
- **Recommendation**: Support both methods

### 2. GPU Support
- **Option A**: Require GPU for Playground features
- **Option B**: CPU fallback with reduced functionality
- **Recommendation**: Graceful degradation

### 3. Training Integration
- **Option A**: Full RL training in MCP
- **Option B**: Load pre-trained policies only
- **Recommendation**: Start with Option B, evolve to A

## Success Metrics

### v0.8.0 (Menagerie)
- Load 20+ robot models successfully
- Control real robots via natural language
- Maintain <100ms response time

### v0.9.0 (Auto-Update)
- Detect breaking changes within 24 hours
- Auto-generate migration guides
- 95% compatibility maintained

### v1.0.0 (Playground)
- Train policies on single GPU
- Transfer to real robots
- Support state and vision inputs

## Next Steps

1. **Immediate** (This Week)
   - Clone Menagerie repository
   - Test model loading patterns
   - Design model registry

2. **Short Term** (Next Month)
   - Implement Menagerie integration
   - Create robot-specific tools
   - Update documentation

3. **Medium Term** (Q2 2025)
   - Research Playground architecture
   - Design GPU support strategy
   - Plan training integration

---

**Key Insight**: The proper sequence is Menagerie → Auto-Update → Playground, as each builds on the previous. Menagerie provides models, Auto-Update maintains compatibility, and Playground adds advanced training capabilities.
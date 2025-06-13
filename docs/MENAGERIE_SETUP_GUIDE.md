# MuJoCo Menagerie Setup Guide

## What is MuJoCo Menagerie?

MuJoCo Menagerie is a collection of high-quality robot models maintained by Google DeepMind. It includes models of real robots like:
- Robotic arms (Franka Panda, UR5e, etc.)
- Quadrupeds (Boston Dynamics Spot, Unitree robots)
- Humanoids (Unitree H1, Apptronik Apollo)
- Grippers and hands (Shadow Hand, Allegro Hand)

## Installation

### Option 1: Clone to Default Location

```bash
# Create MuJoCo directory if it doesn't exist
mkdir -p ~/.mujoco

# Clone Menagerie
cd ~/.mujoco
git clone https://github.com/google-deepmind/mujoco_menagerie.git menagerie
```

### Option 2: Clone to Custom Location

```bash
# Clone anywhere you prefer
cd /your/preferred/path
git clone https://github.com/google-deepmind/mujoco_menagerie.git

# Set environment variable
export MUJOCO_MENAGERIE_PATH=/your/preferred/path/mujoco_menagerie

# Add to your shell profile for persistence
echo 'export MUJOCO_MENAGERIE_PATH=/your/preferred/path/mujoco_menagerie' >> ~/.bashrc
# or for macOS:
echo 'export MUJOCO_MENAGERIE_PATH=/your/preferred/path/mujoco_menagerie' >> ~/.zshrc
```

## Using with MuJoCo MCP

Once Menagerie is installed, you can use it through MuJoCo MCP:

### 1. List Available Models

```python
# Using MCP tool
"list_menagerie_models"

# Using natural language
"list menagerie models"
"show me all robotic arms"
"list quadruped robots"
```

### 2. Load a Model

```python
# Using create_scene tool
"create_scene": "franka_emika_panda"

# Using natural language
"load franka panda"
"create spot robot"
"load the shadow hand"
```

### 3. Common Model Names

#### Robotic Arms
- `franka_emika_panda` - 7-DOF collaborative robot
- `universal_robots_ur5e` - Universal Robots UR5e
- `kuka_iiwa_14` - KUKA LBR iiwa

#### Quadrupeds
- `boston_dynamics_spot` - Spot robot
- `unitree_go2` - Unitree Go2
- `unitree_a1` - Unitree A1
- `anybotics_anymal_c` - ANYmal C

#### End Effectors
- `shadow_hand` - Shadow Dexterous Hand
- `allegro_hand` - Allegro Hand
- `robotiq_2f85` - Robotiq 2-Finger Gripper

## Troubleshooting

### Models Not Found

If models aren't found, check:

1. **Path is correct**:
   ```bash
   echo $MUJOCO_MENAGERIE_PATH
   ls $MUJOCO_MENAGERIE_PATH
   ```

2. **Repository is cloned**:
   ```bash
   ls ~/.mujoco/menagerie
   # Should show directories like: anybotics, boston_dynamics, etc.
   ```

3. **Using correct model names**:
   ```python
   # Use underscores, not spaces
   "franka_emika_panda"  # ✓ Correct
   "franka emika panda"  # ✗ Wrong (but natural language handles this)
   ```

### Asset Path Issues

If you see errors about missing meshes or textures:
- The integration automatically fixes relative paths to absolute
- Make sure the Menagerie repository is complete (includes all subdirectories)

## Examples

### Example 1: Load and Control Franka Panda

```python
# Load the robot
"load franka panda"

# Check state
"show state"

# Control (once we know joint configuration)
"set joint 0 to 45 degrees"
```

### Example 2: Explore Quadrupeds

```python
# List all quadrupeds
"list menagerie quadrupeds"

# Load Spot
"create spot robot"

# Control simulation
"step 1000"
"reset"
```

## Integration Details

The MuJoCo MCP integration:
- Automatically searches for models by name
- Handles name variations (underscores, hyphens, spaces)
- Fixes asset paths in XML files
- Provides helpful error messages
- Works with natural language commands

## Next Steps

1. Install Menagerie following the instructions above
2. Restart MuJoCo MCP server
3. Try loading some models!

For the full list of available models, visit:
https://github.com/google-deepmind/mujoco_menagerie
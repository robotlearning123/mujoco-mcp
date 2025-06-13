# MuJoCo Menagerie Integration - Implementation Plan

## Goal
Enable loading any MuJoCo Menagerie model through MCP with minimal code changes.

## Implementation Approach

### 1. Minimal Code Addition (No Refactoring)

#### Step 1: Add Menagerie Detection to `_handle_create_scene`
```python
# In remote_server.py, modify _handle_create_scene
def _handle_create_scene(self, scene_type: str, context: Dict[str, Any] = None):
    """Create a physics scene"""
    context = context or {}
    
    # First, try existing built-in scenes (unchanged)
    if scene_type in ["pendulum", "double_pendulum", "cart_pole", "robotic_arm"]:
        # Existing code path - no changes
        return self._create_builtin_scene(scene_type, context)
    
    # NEW: Try to load from Menagerie
    menagerie_result = self._try_load_menagerie_model(scene_type)
    if menagerie_result:
        return menagerie_result
    
    # Fallback error
    return {
        "success": False,
        "error": f"Unknown scene type: {scene_type}",
        "available": ["pendulum", "double_pendulum", "cart_pole", "robotic_arm", 
                     "or any MuJoCo Menagerie model name"]
    }
```

#### Step 2: Add Menagerie Loader Method
```python
def _try_load_menagerie_model(self, model_name: str) -> Optional[Dict[str, Any]]:
    """Try to load a model from MuJoCo Menagerie"""
    
    # Check if Menagerie is available
    menagerie_path = os.getenv("MUJOCO_MENAGERIE_PATH")
    if not menagerie_path:
        # Try default locations
        possible_paths = [
            os.path.expanduser("~/.mujoco/menagerie"),
            os.path.expanduser("~/mujoco_menagerie"),
            "./mujoco_menagerie"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                menagerie_path = path
                break
    
    if not menagerie_path or not os.path.exists(menagerie_path):
        return None
    
    # Search for the model
    model_xml_path = self._find_menagerie_model(menagerie_path, model_name)
    if not model_xml_path:
        return None
    
    # Load the XML
    try:
        with open(model_xml_path, 'r') as f:
            xml_content = f.read()
        
        # Fix relative paths in the XML
        xml_content = self._fix_menagerie_paths(xml_content, model_xml_path)
        
        # Use existing model loading mechanism
        model_id = str(uuid.uuid4())
        
        # Ensure viewer connection
        if not self._ensure_viewer_connection(model_id):
            return {"success": False, "error": "Failed to connect to viewer"}
        
        # Load model in viewer
        client = self.viewer_manager.get_client(model_id)
        result = client.load_model(model_id, xml_content)
        
        if result.get("success", False):
            # Store model info
            self._models[model_id] = {
                "id": model_id,
                "name": model_name,
                "source": "menagerie",
                "xml_path": model_xml_path,
                "created_at": time.time()
            }
            
            return {
                "success": True,
                "model_id": model_id,
                "message": f"Loaded {model_name} from MuJoCo Menagerie",
                "model_info": result.get("model_info", {})
            }
        else:
            return {"success": False, "error": result.get("error", "Unknown error")}
            
    except Exception as e:
        return {"success": False, "error": f"Failed to load Menagerie model: {str(e)}"}
```

#### Step 3: Helper Methods
```python
def _find_menagerie_model(self, menagerie_path: str, model_name: str) -> Optional[str]:
    """Find model XML file in Menagerie directory structure"""
    
    # Common patterns for Menagerie models
    search_patterns = [
        f"{model_name}/{model_name}.xml",
        f"*/{model_name}/{model_name}.xml",
        f"{model_name}/scene.xml",
        f"*/{model_name}/scene.xml",
        f"*/{model_name}.xml"
    ]
    
    import glob
    for pattern in search_patterns:
        matches = glob.glob(os.path.join(menagerie_path, pattern))
        if matches:
            return matches[0]
    
    # Try case-insensitive search
    for root, dirs, files in os.walk(menagerie_path):
        for file in files:
            if file.lower() == f"{model_name.lower()}.xml":
                return os.path.join(root, file)
    
    return None

def _fix_menagerie_paths(self, xml_content: str, xml_path: str) -> str:
    """Fix relative paths in Menagerie XML files"""
    
    import xml.etree.ElementTree as ET
    from pathlib import Path
    
    xml_dir = Path(xml_path).parent
    
    # Parse XML
    root = ET.fromstring(xml_content)
    
    # Fix mesh paths
    for mesh in root.findall(".//mesh"):
        if "file" in mesh.attrib:
            rel_path = mesh.attrib["file"]
            if not os.path.isabs(rel_path):
                abs_path = (xml_dir / rel_path).resolve()
                mesh.attrib["file"] = str(abs_path)
    
    # Fix texture paths
    for texture in root.findall(".//texture"):
        if "file" in texture.attrib:
            rel_path = texture.attrib["file"]
            if not os.path.isabs(rel_path):
                abs_path = (xml_dir / rel_path).resolve()
                texture.attrib["file"] = str(abs_path)
    
    return ET.tostring(root, encoding='unicode')
```

### 2. Add List Menagerie Models Tool

```python
# Add to __init__ method
self._tools["list_menagerie_models"] = {
    "name": "list_menagerie_models",
    "description": "List available MuJoCo Menagerie models",
    "parameters": {
        "category": "(optional) Filter by category"
    },
    "handler": self._handle_list_menagerie_models
}

def _handle_list_menagerie_models(self, category: Optional[str] = None) -> Dict[str, Any]:
    """List available Menagerie models"""
    
    menagerie_path = self._get_menagerie_path()
    if not menagerie_path:
        return {
            "success": False,
            "error": "MuJoCo Menagerie not found. Set MUJOCO_MENAGERIE_PATH or install in default location"
        }
    
    models = []
    categories = set()
    
    # Scan Menagerie directory
    for item in os.listdir(menagerie_path):
        item_path = os.path.join(menagerie_path, item)
        if os.path.isdir(item_path):
            # Check if it's a model directory
            xml_files = glob.glob(os.path.join(item_path, "*.xml"))
            if xml_files:
                # Determine category from path structure
                cat = self._guess_category(item)
                categories.add(cat)
                
                if category is None or cat == category:
                    models.append({
                        "name": item,
                        "category": cat,
                        "xml_files": [os.path.basename(f) for f in xml_files]
                    })
    
    return {
        "success": True,
        "models": sorted(models, key=lambda x: x["name"]),
        "categories": sorted(list(categories)),
        "total": len(models)
    }
```

### 3. Natural Language Updates

```python
# In _handle_execute_command, add Menagerie support
if "load" in command_lower:
    # Extract model name (everything after "load")
    parts = command.lower().split("load")
    if len(parts) > 1:
        model_name = parts[1].strip()
        # Clean common phrases
        model_name = model_name.replace("the", "").replace("robot", "").strip()
        
        # Try common variations
        variations = [
            model_name,
            model_name.replace(" ", "_"),
            model_name.replace(" ", "-"),
            model_name.replace("_", " ")
        ]
        
        for variant in variations:
            result = self._handle_create_scene(variant)
            if result.get("success"):
                return result
        
        # If failed, provide helpful error
        return {
            "success": False,
            "error": f"Could not find model '{model_name}'",
            "suggestion": "Try 'list menagerie models' to see available models"
        }
```

## Testing Plan

### 1. Unit Tests
```python
# tests/test_menagerie_integration.py
def test_menagerie_path_detection():
    """Test finding Menagerie installation"""
    
def test_model_loading():
    """Test loading common models"""
    
def test_path_fixing():
    """Test XML path correction"""
    
def test_graceful_fallback():
    """Test behavior without Menagerie"""
```

### 2. Integration Tests
```python
# Test with real Menagerie models
test_models = [
    "franka_emika_panda",  # Arm
    "unitree_go2",         # Quadruped  
    "agility_cassie",      # Biped
    "shadow_hand"          # End-effector
]
```

### 3. Manual Testing Checklist
- [ ] Install Menagerie in test environment
- [ ] Start viewer server
- [ ] Test loading each category of robot
- [ ] Test natural language loading
- [ ] Test listing models
- [ ] Test without Menagerie (fallback)
- [ ] Test with incorrect model names
- [ ] Verify asset paths work

## Installation Instructions

### For Users
```bash
# Option 1: Set environment variable
export MUJOCO_MENAGERIE_PATH=/path/to/mujoco_menagerie

# Option 2: Install in default location
cd ~
git clone https://github.com/google-deepmind/mujoco_menagerie.git
# or
mkdir -p ~/.mujoco
cd ~/.mujoco
git clone https://github.com/google-deepmind/mujoco_menagerie.git menagerie
```

### In Documentation
```markdown
## Using MuJoCo Menagerie Models

MuJoCo MCP now supports loading models from MuJoCo Menagerie!

### Setup
1. Clone Menagerie: `git clone https://github.com/google-deepmind/mujoco_menagerie.git`
2. Set path: `export MUJOCO_MENAGERIE_PATH=/path/to/menagerie`

### Usage
- Load a model: `create scene franka_emika_panda`
- List models: `list menagerie models`
- Natural language: `load the panda robot`
```

## Success Criteria

1. **Zero Breaking Changes**
   - All existing commands work
   - All tests pass
   - No API changes

2. **Menagerie Support**
   - Can load 10+ different models
   - Handles missing Menagerie gracefully
   - Clear error messages

3. **User Experience**
   - Natural language works
   - Fast model loading (<3 seconds)
   - Helpful error messages

## Timeline

- Day 1: Implement basic loading
- Day 2: Add path fixing and error handling
- Day 3: Natural language integration
- Day 4: Testing and documentation
- Day 5: Final testing and release

Total: 5 days for v0.8.0
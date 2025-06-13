"""
MuJoCo Menagerie model loader utility.
Minimal, focused implementation for loading Menagerie models.
"""

import os
import glob
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Dict, List, Tuple


class MenagerieLoader:
    """Handles loading models from MuJoCo Menagerie"""
    
    def __init__(self, menagerie_path: Optional[str] = None):
        self.menagerie_path = menagerie_path or self._find_menagerie()
        
    def _find_menagerie(self) -> Optional[str]:
        """Find MuJoCo Menagerie installation"""
        # Check environment variable first
        env_path = os.getenv("MUJOCO_MENAGERIE_PATH")
        if env_path and os.path.exists(env_path):
            return env_path
            
        # Check common locations
        possible_paths = [
            os.path.expanduser("~/.mujoco/menagerie"),
            os.path.expanduser("~/mujoco_menagerie"),
            os.path.expanduser("~/MuJoCo/menagerie"),
            "./mujoco_menagerie",
            "../mujoco_menagerie"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return os.path.abspath(path)
                
        return None
    
    def is_available(self) -> bool:
        """Check if Menagerie is available"""
        return self.menagerie_path is not None
    
    def find_model(self, model_name: str) -> Optional[str]:
        """Find model XML file by name"""
        if not self.menagerie_path:
            return None
            
        # Clean model name
        model_name = model_name.lower().strip()
        
        # Direct directory check first
        for item in os.listdir(self.menagerie_path):
            if item.lower() == model_name or item.lower().replace("_", " ") == model_name:
                # Prefer main model file over scene.xml
                xml_path = os.path.join(self.menagerie_path, item, f"{item}.xml")
                if os.path.exists(xml_path):
                    return xml_path
                # For some models, try without hand
                nohand_path = os.path.join(self.menagerie_path, item, f"{item}_nohand.xml")
                if os.path.exists(nohand_path):
                    return nohand_path
                # Last resort: scene.xml (has includes)
                scene_path = os.path.join(self.menagerie_path, item, "scene.xml")
                if os.path.exists(scene_path):
                    return scene_path
        
        # Common variations
        variations = [
            model_name,
            model_name.replace(" ", "_"),
            model_name.replace(" ", "-"),
            model_name.replace("-", "_"),
            model_name.replace("_", "-")
        ]
        
        # Search patterns
        for variant in variations:
            patterns = [
                f"{variant}/{variant}.xml",
                f"*/{variant}/{variant}.xml",
                f"{variant}/scene.xml",
                f"*/{variant}/scene.xml",
                f"**/{variant}.xml",
                f"**/{variant}_*.xml"
            ]
            
            for pattern in patterns:
                matches = glob.glob(
                    os.path.join(self.menagerie_path, pattern),
                    recursive=True
                )
                if matches:
                    return matches[0]
        
        # Case-insensitive fallback
        for root, dirs, files in os.walk(self.menagerie_path):
            for file in files:
                if file.endswith('.xml'):
                    file_lower = file.lower()
                    for variant in variations:
                        if variant in file_lower or file_lower.startswith(variant):
                            return os.path.join(root, file)
        
        return None
    
    def load_model_xml(self, model_name: str) -> Optional[str]:
        """Load model XML with fixed paths"""
        xml_path = self.find_model(model_name)
        if not xml_path:
            return None
            
        try:
            with open(xml_path, 'r') as f:
                xml_content = f.read()
            
            # Fix relative paths
            xml_content = self._fix_paths(xml_content, xml_path)
            return xml_content
            
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            return None
    
    def _fix_paths(self, xml_content: str, xml_path: str) -> str:
        """Fix relative paths in XML to absolute paths"""
        try:
            xml_dir = Path(xml_path).parent
            root = ET.fromstring(xml_content)
            
            # Fix include files first
            for include in root.findall(".//include"):
                if "file" in include.attrib:
                    rel_path = include.attrib["file"]
                    if not os.path.isabs(rel_path):
                        abs_path = (xml_dir / rel_path).resolve()
                        include.attrib["file"] = str(abs_path)
            
            # Fix compiler include paths
            compiler = root.find(".//compiler")
            if compiler is not None:
                for attr in ["meshdir", "texturedir", "assetdir"]:
                    if attr in compiler.attrib:
                        rel_path = compiler.attrib[attr]
                        if not os.path.isabs(rel_path):
                            abs_path = (xml_dir / rel_path).resolve()
                            compiler.attrib[attr] = str(abs_path)
            
            # Fix mesh paths
            for mesh in root.findall(".//mesh"):
                if "file" in mesh.attrib:
                    self._fix_path_attribute(mesh, "file", xml_dir)
            
            # Fix texture paths
            for texture in root.findall(".//texture"):
                if "file" in texture.attrib:
                    self._fix_path_attribute(texture, "file", xml_dir)
            
            # Fix heightfield paths
            for hfield in root.findall(".//hfield"):
                if "file" in hfield.attrib:
                    self._fix_path_attribute(hfield, "file", xml_dir)
            
            # Fix other asset paths
            for asset in root.findall(".//asset"):
                if "file" in asset.attrib:
                    self._fix_path_attribute(asset, "file", xml_dir)
            
            return ET.tostring(root, encoding='unicode')
            
        except Exception as e:
            print(f"Warning: Could not fix paths in XML: {e}")
            return xml_content
    
    def _fix_path_attribute(self, element: ET.Element, attr: str, base_dir: Path):
        """Fix a single path attribute"""
        path = element.attrib[attr]
        if not os.path.isabs(path):
            abs_path = (base_dir / path).resolve()
            if abs_path.exists():
                element.attrib[attr] = str(abs_path)
    
    def list_models(self, category: Optional[str] = None) -> List[Dict[str, str]]:
        """List available models"""
        if not self.menagerie_path:
            return []
        
        models = []
        
        # Scan directory structure
        for root, dirs, files in os.walk(self.menagerie_path):
            # Skip hidden directories and assets
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'assets']
            
            # Look for XML files
            xml_files = [f for f in files if f.endswith('.xml')]
            if xml_files:
                rel_path = os.path.relpath(root, self.menagerie_path)
                model_name = os.path.basename(root)
                
                # Guess category
                path_parts = rel_path.split(os.sep)
                model_category = self._guess_category(path_parts, model_name)
                
                if category is None or model_category == category:
                    models.append({
                        "name": model_name,
                        "category": model_category,
                        "path": rel_path,
                        "xml_files": xml_files
                    })
        
        return sorted(models, key=lambda x: (x["category"], x["name"]))
    
    def _guess_category(self, path_parts: List[str], model_name: str) -> str:
        """Guess model category from path or name"""
        # Check path for category keywords
        for part in path_parts:
            part_lower = part.lower()
            if "arm" in part_lower:
                return "arms"
            elif "hand" in part_lower or "gripper" in part_lower:
                return "end_effectors"
            elif "quadruped" in part_lower:
                return "quadrupeds"
            elif "biped" in part_lower:
                return "bipeds"
            elif "humanoid" in part_lower:
                return "humanoids"
            elif "mobile" in part_lower:
                return "mobile_manipulators"
        
        # Check model name
        name_lower = model_name.lower()
        if any(x in name_lower for x in ["ur", "panda", "kuka", "iiwa"]):
            return "arms"
        elif any(x in name_lower for x in ["hand", "gripper", "allegro", "shadow"]):
            return "end_effectors"
        elif any(x in name_lower for x in ["spot", "go1", "go2", "a1", "anymal"]):
            return "quadrupeds"
        elif any(x in name_lower for x in ["cassie", "digit"]):
            return "bipeds"
        elif any(x in name_lower for x in ["h1", "atlas", "apollo"]):
            return "humanoids"
        
        return "other"
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, any]]:
        """Get information about a specific model"""
        xml_path = self.find_model(model_name)
        if not xml_path:
            return None
        
        info = {
            "name": model_name,
            "xml_path": xml_path,
            "directory": os.path.dirname(xml_path),
            "category": self._guess_category(
                os.path.relpath(xml_path, self.menagerie_path).split(os.sep),
                model_name
            )
        }
        
        # Check for README
        readme_path = os.path.join(os.path.dirname(xml_path), "README.md")
        if os.path.exists(readme_path):
            info["has_readme"] = True
            
        # Check for LICENSE
        license_path = os.path.join(os.path.dirname(xml_path), "LICENSE")
        if os.path.exists(license_path):
            info["has_license"] = True
            
        return info
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
            item_lower = item.lower()
            # Check exact match or with underscores replaced by spaces
            if item_lower == model_name or item_lower.replace("_", " ") == model_name:
                # Found exact match
                pass
            # Check if search term matches part of the model name
            elif all(part in item_lower for part in model_name.split()):
                # All parts of search term are in model name (e.g., "franka panda" matches "franka_emika_panda")
                pass
            else:
                continue
                
            model_dir = os.path.join(self.menagerie_path, item)
            
            # ALWAYS prefer scene.xml first as it contains the complete scene setup
            scene_path = os.path.join(model_dir, "scene.xml")
            if os.path.exists(scene_path):
                return scene_path
            
            # Special handling for models with scene variants
            # Priority order: scene.xml > scene_left.xml > scene_right.xml > scene_*.xml
            scene_variants = []
            
            # Hand models often have left/right variants
            if "hand" in item.lower() or "allegro" in item.lower():
                scene_variants.extend(["scene_left.xml", "scene_right.xml"])
                
            # Some models have actuation variants
            scene_variants.extend([
                "scene_motor.xml", "scene_position.xml", "scene_velocity.xml",
                "scene_mjx.xml", "scene_external.xml"
            ])
            
            # Check all scene variants
            for variant in scene_variants:
                variant_path = os.path.join(model_dir, variant)
                if os.path.exists(variant_path):
                    return variant_path
            
            # Fallback to main model file if no scene.xml
            xml_path = os.path.join(model_dir, f"{item}.xml")
            if os.path.exists(xml_path):
                return xml_path
            
            # Try alternative naming (e.g., panda.xml for franka_emika_panda)
            if "_" in item:
                # Try last part of name (e.g., panda from franka_emika_panda)
                last_part = item.split("_")[-1]
                alt_path = os.path.join(model_dir, f"{last_part}.xml")
                if os.path.exists(alt_path):
                    return alt_path
            
            # For some models, try without hand
            nohand_path = os.path.join(model_dir, f"{item}_nohand.xml")
            if os.path.exists(nohand_path):
                return nohand_path
            
            # Last resort: any XML file
            xml_files = [f for f in os.listdir(model_dir) if f.endswith('.xml')]
            if xml_files:
                # Prefer files that match model name parts
                for xml_file in xml_files:
                    if any(part in xml_file.lower() for part in item.split('_')):
                        return os.path.join(model_dir, xml_file)
                # Return first XML if no match
                return os.path.join(model_dir, xml_files[0])
        
        # Common variations
        variations = [
            model_name,
            model_name.replace(" ", "_"),
            model_name.replace(" ", "-"),
            model_name.replace("-", "_"),
            model_name.replace("_", "-")
        ]
        
        # Search patterns - prioritize scene.xml
        for variant in variations:
            patterns = [
                f"{variant}/scene.xml",
                f"*/{variant}/scene.xml",
                f"{variant}/{variant}.xml",
                f"*/{variant}/{variant}.xml",
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
        """Load model XML - return path instead of content for proper relative path handling"""
        xml_path = self.find_model(model_name)
        if not xml_path:
            return None
            
        # 重要：返回XML文件路径而不是内容
        # 这样MuJoCo可以从正确的目录加载，正确处理所有相对路径
        return xml_path
    
    def _fix_paths(self, xml_content: str, xml_path: str) -> str:
        """Fix relative paths in XML - keep relative structure but ensure base path is correct"""
        try:
            # 对于scene.xml，我们需要确保工作目录是正确的
            # MuJoCo会从XML文件所在的目录解析相对路径
            xml_dir = os.path.dirname(os.path.abspath(xml_path))
            
            # 创建一个修改过的XML，添加基础路径信息
            root = ET.fromstring(xml_content)
            
            # 检查是否已经有compiler标签
            compiler = root.find(".//compiler")
            if compiler is None:
                # 如果没有compiler标签，创建一个
                compiler = ET.SubElement(root, "compiler")
            
            # 如果这是一个scene.xml，确保include路径正确
            for include in root.findall(".//include"):
                if "file" in include.attrib:
                    include_file = include.attrib["file"]
                    # 保持相对路径，MuJoCo会从XML所在目录解析
                    # 不需要改成绝对路径
            
            # 重要：对于包含meshdir的模型，mesh文件路径是相对于meshdir的
            # 我们不应该修改这些路径，让MuJoCo自己处理
            
            return ET.tostring(root, encoding='unicode')
            
        except Exception as e:
            print(f"Warning: Could not process XML: {e}")
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
            
        # Check for available scene files
        model_dir = os.path.dirname(xml_path)
        scene_files = [f for f in os.listdir(model_dir) if f.startswith("scene") and f.endswith(".xml")]
        if scene_files:
            info["scene_files"] = scene_files
            
        return info
    
    def get_all_models(self) -> Dict[str, List[str]]:
        """Get all available models organized by category"""
        if not self.menagerie_path:
            return {}
            
        models_by_category = {
            "robotic_arms": [],
            "humanoids": [],
            "quadrupeds": [],
            "grippers_hands": [],
            "mobile_robots": [],
            "aerial_vehicles": [],
            "special_purpose": [],
            "soft_robots": [],
            "sensors": []
        }
        
        # Scan all directories
        for item in os.listdir(self.menagerie_path):
            if item.startswith('.') or item == 'assets':
                continue
                
            item_path = os.path.join(self.menagerie_path, item)
            if os.path.isdir(item_path):
                # Check if it contains XML files
                xml_files = [f for f in os.listdir(item_path) if f.endswith('.xml')]
                if xml_files:
                    category = self._guess_category([item], item)
                    
                    # Map to standardized categories
                    if category == "arms":
                        models_by_category["robotic_arms"].append(item)
                    elif category == "humanoids":
                        models_by_category["humanoids"].append(item)
                    elif category == "quadrupeds":
                        models_by_category["quadrupeds"].append(item)
                    elif category in ["end_effectors", "grippers"]:
                        models_by_category["grippers_hands"].append(item)
                    elif category == "mobile_manipulators":
                        models_by_category["mobile_robots"].append(item)
                    elif category == "bipeds":
                        models_by_category["humanoids"].append(item)
                    else:
                        # Determine by name patterns
                        if "drone" in item or "crazyflie" in item or "skydio" in item:
                            models_by_category["aerial_vehicles"].append(item)
                        elif "soft" in item:
                            models_by_category["soft_robots"].append(item)
                        elif "realsense" in item:
                            models_by_category["sensors"].append(item)
                        else:
                            models_by_category["special_purpose"].append(item)
                            
        # Sort each category
        for category in models_by_category:
            models_by_category[category].sort()
            
        return models_by_category
#!/usr/bin/env python3
"""
MuJoCo Menagerie Model Loader
Handles downloading and loading of MuJoCo Menagerie models with include resolution
"""

import logging
import os
from pathlib import Path
import tempfile
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
import urllib.request
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


class MenagerieLoader:
    """Load MuJoCo Menagerie models with automatic include resolution"""

    BASE_URL = "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main"

    def __init__(self, cache_dir: Optional[str] = None, root_path: Optional[str] = None):
        self.cache_dir = (
            Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "mujoco_menagerie"
        )
        self.cache_dir.mkdir(exist_ok=True)

        env_path = os.environ.get("MUJOCO_MENAGERIE_PATH")
        base_path = root_path or env_path
        self.root_path: Optional[Path] = Path(base_path).expanduser() if base_path else None
        if self.root_path and not self.root_path.exists():
            logger.warning(
                "Configured Menagerie path '%s' does not exist. Falling back to remote repository.",
                self.root_path,
            )
            self.root_path = None

    def _read_local_file(self, model_name: str, file_path: str, is_binary: bool) -> Optional[str]:
        """Attempt to read a Menagerie asset from a local checkout."""

        if not self.root_path:
            return None

        local_file = self.root_path / model_name / file_path
        if not local_file.exists():
            return None

        if is_binary:
            return str(local_file)

        return local_file.read_text(encoding="utf-8")

    def download_file(self, model_name: str, file_path: str) -> str:
        """Retrieve a Menagerie asset, preferring local files when available."""

        # Determine if file is binary based on extension
        binary_extensions = {".stl", ".obj", ".png", ".jpg", ".jpeg", ".dae", ".mtl"}
        is_binary = any(file_path.lower().endswith(ext) for ext in binary_extensions)

        # Try local checkout first
        local_result = self._read_local_file(model_name, file_path, is_binary)
        if local_result is not None:
            return local_result

        url = f"{self.BASE_URL}/{model_name}/{file_path}"

        # Check cache before hitting the network
        cache_file = self.cache_dir / model_name / file_path

        if cache_file.exists():
            if is_binary:
                return str(cache_file)
            return cache_file.read_text(encoding="utf-8")

        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                if response.getcode() != 200:
                    raise Exception(f"HTTP {response.getcode()}")

                content_bytes = response.read()
                cache_file.parent.mkdir(parents=True, exist_ok=True)

                if is_binary:
                    cache_file.write_bytes(content_bytes)
                    return str(cache_file)

                content = content_bytes.decode("utf-8")
                cache_file.write_text(content, encoding="utf-8")
                return content
        except Exception as e:
            raise Exception(f"Failed to download {url}: {e}")

    def resolve_includes(
        self, xml_content: str, model_name: str, visited: Optional[set] = None
    ) -> str:
        """Resolve XML include directives recursively"""
        if visited is None:
            visited = set()

        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            logger.warning(f"XML parse error: {e}")
            return xml_content

        # Build parent map for efficient parent lookup
        parent_map = {child: parent for parent in root.iter() for child in parent}

        # Find all include elements
        includes = root.findall(".//include")

        for include in includes:
            file_attr = include.get("file")
            if not file_attr:
                continue

            # Avoid circular includes
            if file_attr in visited:
                logger.warning(f"Circular include detected: {file_attr}")
                continue

            visited.add(file_attr)

            try:
                # Download included file
                included_content = self.download_file(model_name, file_attr)

                # Recursively resolve includes in the included file
                included_content = self.resolve_includes(
                    included_content, model_name, visited.copy()
                )

                # Parse included content
                included_root = ET.fromstring(included_content)

                # Find parent of include element
                parent = parent_map.get(include)
                if parent is not None:
                    # Find position of include in parent
                    include_idx = list(parent).index(include)
                    parent.remove(include)

                    # Insert all children of included root at the same position
                    for i, child in enumerate(included_root):
                        parent.insert(include_idx + i, child)

            except Exception as e:
                logger.warning(f"Failed to resolve include {file_attr}: {e}")
                # Keep the include element as-is if we can't resolve it
                continue

        # Return modified XML
        return ET.tostring(root, encoding="unicode")

    def _ensure_absolute_asset_dirs(self, xml_content: str, model_name: str) -> str:
        """Rewrite compiler directories so MuJoCo can find cached assets."""
        asset_root = self.cache_dir / model_name
        asset_assets = asset_root / "assets"
        meshdir_path = asset_assets if asset_assets.exists() else asset_root
        asset_root_str = str(meshdir_path)
        texture_dir_path = asset_root / "assets"
        texturedir_str = str(texture_dir_path if texture_dir_path.exists() else asset_root)

        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError:
            # If we can't parse the XML, return it unchanged
            return xml_content

        compiler = root.find("compiler")
        if compiler is None:
            compiler = ET.Element("compiler")
            root.insert(0, compiler)

        compiler.set("meshdir", asset_root_str)
        compiler.set("texturedir", texturedir_str)
        if "angle" not in compiler.attrib:
            compiler.set("angle", "radian")

        return ET.tostring(root, encoding="unicode")

    def _download_all_assets(self, model_name: str) -> None:
        """Pre-download all assets from the assets/ directory.

        This ensures all mesh, texture, and other asset files are available
        before MuJoCo tries to load them.
        """
        # Common asset file extensions
        asset_extensions = ['.stl', '.obj', '.png', '.jpg', '.jpeg', '.dae', '.mtl', '.xml']

        # Try to download common asset files
        common_assets = []

        # Add numbered variations that might exist (like link_base_0_00.stl through link_base_0_19.stl)
        for i in range(50):
            for prefix in ['link_base_0_', 'link_base_1_', 'link_']:
                for ext in ['.stl', '.obj']:
                    common_assets.append(f"assets/{prefix}{i:02d}{ext}")

        # Add common named assets
        common_names = [
            'link_base', 'link_bicep', 'link_elbow', 'link_forearm', 'link_shoulder',
            'link_wrist', 'link_finger_base', 'link_finger_tip', 'link_gripper',
            'link_head_pan', 'link_head_tilt', 'link_torso', 'link_wheel',
            'robot_texture', 'finger_base_texture', 'finger_tip_texture'
        ]

        for name in common_names:
            for ext in asset_extensions:
                common_assets.append(f"assets/{name}{ext}")
                common_assets.append(f"assets/{name}_v{ext}")

        # Try downloading each asset (failures are expected and ignored)
        for asset_path in common_assets:
            try:
                self.download_file(model_name, asset_path)
            except Exception:
                pass  # Asset doesn't exist, which is fine

    def _download_asset_references(self, model_name: str, xml_content: str) -> None:
        """Download asset files referenced inside the resolved XML."""
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as exc:
            logger.warning(f"Could not parse XML while downloading assets: {exc}")
            return

        referenced_files = set()

        def collect(tag: str, attribute: str, *, skip_builtin: bool = False) -> None:
            for elem in root.findall(f".//{tag}"):
                file_attr = elem.get(attribute)
                if not file_attr:
                    continue
                if skip_builtin and elem.get("builtin"):
                    continue
                referenced_files.add(file_attr.strip())

        collect("mesh", "file")
        collect("texture", "file", skip_builtin=True)
        collect("hfield", "file")
        collect("heightfield", "file")
        collect("skin", "file")

        for file_path in referenced_files:
            candidates = [file_path]
            # Provide common fallbacks when files live under assets/ or textures/
            if "/" not in file_path:
                candidates.extend(
                    [
                        f"assets/{file_path}",
                        f"meshes/{file_path}",
                        f"textures/{file_path}",
                    ]
                )

            downloaded = False
            for candidate in candidates:
                try:
                    self.download_file(model_name, candidate)
                    downloaded = True
                    break
                except Exception:
                    logger.debug(
                        "Failed to download asset '%s' (candidate '%s') for model '%s'",
                        file_path,
                        candidate,
                        model_name,
                    )
            if not downloaded:
                logger.warning(
                    "Unable to download asset '%s' for model '%s'; simulation may fail",
                    file_path,
                    model_name,
                )

    def get_model_xml(self, model_name: str) -> str:
        """Get complete XML for a Menagerie model with includes resolved"""

        # Pre-download assets to ensure they're available
        try:
            self._download_all_assets(model_name)
        except Exception as e:
            logger.debug(f"Asset pre-download had issues (continuing anyway): {e}")

        # Try different common file patterns (expanded for edge cases)
        possible_files = [
            f"{model_name}.xml",           # Standard: franka_emika_panda.xml
            "scene.xml",                    # Standard scene
            f"{model_name}_mjx.xml",       # MJX variant
            "scene_mjx.xml",                # MJX scene
            "scene_left.xml",               # Shadow hand left
            "scene_right.xml",              # Shadow hand right
            "left_hand.xml",                # Direct hand reference
            "right_hand.xml",               # Direct hand reference
            f"{model_name}_left.xml",      # Generic left variant
            f"{model_name}_right.xml",     # Generic right variant
        ]

        for xml_file in possible_files:
            try:
                # Download main XML file
                xml_content = self.download_file(model_name, xml_file)

                # Resolve includes
                resolved_xml = self.resolve_includes(xml_content, model_name)

                # Ensure all asset references are absolute so MuJoCo can locate them
                resolved_xml = self._ensure_absolute_asset_dirs(resolved_xml, model_name)

                # Download referenced assets required by the resolved XML
                self._download_asset_references(model_name, resolved_xml)

                logger.info(f"Successfully loaded {model_name} from {xml_file}")
                return resolved_xml

            except Exception as e:
                logger.debug(f"Failed to load {model_name} from {xml_file}: {e}")
                continue

        raise Exception(f"Could not load any XML files for model {model_name}")

    def get_model_path(self, model_name: str) -> str:
        """Download model and all assets, return path to main XML file"""
        # Pre-download assets
        try:
            self._download_all_assets(model_name)
        except Exception as e:
            logger.debug(f"Asset pre-download had issues (continuing anyway): {e}")

        # Try different common file patterns (same as get_model_xml)
        possible_files = [
            f"{model_name}.xml",
            "scene.xml",
            f"{model_name}_mjx.xml",
            "scene_mjx.xml",
            "scene_left.xml",
            "scene_right.xml",
            "left_hand.xml",
            "right_hand.xml",
            f"{model_name}_left.xml",
            f"{model_name}_right.xml",
        ]

        for xml_file in possible_files:
            try:
                # Download main XML file to cache
                xml_content = self.download_file(model_name, xml_file)

                # Resolve includes to get complete XML
                resolved_xml = self.resolve_includes(xml_content, model_name)

                # Ensure absolute asset paths before parsing
                resolved_xml = self._ensure_absolute_asset_dirs(resolved_xml, model_name)

                # Download referenced assets prior to returning path
                self._download_asset_references(model_name, resolved_xml)

                # Parse resolved XML to find asset references
                try:
                    root = ET.fromstring(resolved_xml)
                except ET.ParseError as e:
                    logger.warning(f"Could not parse resolved XML: {e}")
                    continue

                # Check compiler meshdir attribute to determine asset path prefix
                compiler = root.find(".//compiler")
                meshdir = ""
                if compiler is not None:
                    meshdir = compiler.get("meshdir", "")
                    if meshdir and not meshdir.endswith("/"):
                        meshdir += "/"

                # Download all mesh assets
                for mesh in root.findall(".//mesh"):
                    file_attr = mesh.get("file")
                    if file_attr:
                        mesh_path = meshdir + file_attr if meshdir else file_attr
                        try:
                            self.download_file(model_name, mesh_path)
                        except Exception as e:
                            logger.debug(f"Could not download mesh {mesh_path}: {e}")

                # Download all texture assets
                for texture in root.findall(".//texture"):
                    file_attr = texture.get("file")
                    if file_attr and not texture.get("builtin"):  # Skip builtin textures
                        try:
                            self.download_file(model_name, file_attr)
                        except Exception as e:
                            logger.debug(f"Could not download texture {file_attr}: {e}")

                # Return path to cached XML file
                xml_path = self.cache_dir / model_name / xml_file
                logger.info(f"Successfully downloaded {model_name} to {xml_path}")
                return str(xml_path)

            except Exception as e:
                logger.debug(f"Failed to prepare {model_name} from {xml_file}: {e}")
                continue

        raise Exception(f"Could not prepare model files for {model_name}")

    def get_asset_root(self, model_name: str) -> Path:
        """Return the cache directory where model assets are stored."""
        return self.cache_dir / model_name

    def get_available_models(self) -> Dict[str, List[str]]:
        """Get list of available models grouped by category.

        Returns models organized by robot type (arms, quadrupeds, humanoids, etc.)
        based on the MuJoCo Menagerie repository structure.
        """

        if self.root_path and self.root_path.exists():
            local_models = sorted(
                entry.name
                for entry in self.root_path.iterdir()
                if entry.is_dir() and not entry.name.startswith(".")
            )
            return {"local": local_models}

        # Comprehensive model catalog from MuJoCo Menagerie (60+ models)
        # Organized by category for easy filtering
        return {
            "arms": [
                "franka_emika_panda",
                "universal_robots_ur5e",
                "universal_robots_ur10e",
                "ufactory_xarm7",
                "kinova_gen3",
                "kuka_iiwa_14",
                "rethink_sawyer",
                "trossen_vx300s",
                "trossen_wx250s",
                "ufactory_lite6",
                "unitree_z1",
            ],
            "quadrupeds": [
                "unitree_go2",
                "unitree_go1",
                "unitree_a1",
                "anybotics_anymal_b",
                "anybotics_anymal_c",
                "boston_dynamics_spot",
                "google_barkour_v0",
                "google_barkour_vb",
            ],
            "humanoids": [
                "unitree_g1",
                "unitree_h1",
                "pal_talos",
                "agility_cassie",
            ],
            "mobile_manipulators": [
                "google_robot",
                "hello_robot_stretch",
                "hello_robot_stretch_2",
                "hello_robot_stretch_3",
            ],
            "grippers": [
                "robotiq_2f85",
                "shadow_hand",
                "wonik_allegro",
            ],
            "drones": [
                "bitcraze_crazyflie_2",
                "skydio_x2",
            ],
        }

    def validate_model(self, model_name: str) -> Dict[str, Any]:
        """Validate that a model can be loaded and return info"""
        try:
            xml_content = self.get_model_xml(model_name)

            # Basic validation
            if not xml_content.strip():
                return {"valid": False, "error": "Empty XML content"}

            # Try to parse XML
            try:
                root = ET.fromstring(xml_content)
                if root.tag != "mujoco":
                    return {
                        "valid": False,
                        "error": "Not a valid MuJoCo XML (root is not 'mujoco')",
                    }
            except ET.ParseError as e:
                return {"valid": False, "error": f"XML parse error: {e}"}

            # Try MuJoCo loading if available
            try:
                import mujoco

                with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as tmp:
                    tmp.write(xml_content)
                    tmp_path = tmp.name

                try:
                    model = mujoco.MjModel.from_xml_path(tmp_path)
                    result = {
                        "valid": True,
                        "n_bodies": model.nbody,
                        "n_joints": model.njnt,
                        "n_actuators": model.nu,
                        "xml_size": len(xml_content),
                    }
                finally:
                    os.unlink(tmp_path)

                return result

            except ImportError:
                # MuJoCo not available, just return basic validation
                return {
                    "valid": True,
                    "xml_size": len(xml_content),
                    "note": "MuJoCo validation skipped (not installed)",
                }

        except Exception as e:
            return {"valid": False, "error": str(e)}

    def create_scene_xml(self, model_name: str, scene_name: Optional[str] = None) -> str:
        """Create a complete scene XML for a Menagerie model"""
        model_xml = self.get_model_xml(model_name)

        # If the model XML already contains a Mujoco root/worldbody, return as-is
        if "<worldbody" in model_xml and "<mujoco" in model_xml:
            return model_xml

        # Otherwise, wrap it in a scene template
        scene_template = f"""
        <mujoco model="{scene_name or model_name}_scene">
          <compiler angle="radian" meshdir="." texturedir="."/>
          <option timestep="0.002" integrator="RK4"/>
          
          <default>
            <joint damping="0.1"/>
            <geom contype="1" conaffinity="1"/>
          </default>
          
          <asset>
            <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
            <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance="0"/>
          </asset>
          
          <worldbody>
            <geom name="floor" size="0 0 0.05" type="plane" material="grid"/>
            <light name="light" pos="0 0 1"/>
            
            {model_xml}
          </worldbody>
        </mujoco>
        """

        scene_xml = scene_template.strip()
        return self._ensure_absolute_asset_dirs(scene_xml, model_name)

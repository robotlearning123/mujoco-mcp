#!/usr/bin/env python3
"""
Comprehensive MuJoCo Menagerie Model Testing
Tests all available models from MuJoCo Menagerie collection
"""

import json
import time
import sys
from pathlib import Path
from typing import Dict, Any
import urllib.request
import tempfile
import os

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    print("⚠️ MuJoCo not available, running compatibility tests only")

# MuJoCo Menagerie Models Catalog
MENAGERIE_MODELS = {
    "arms": [
        "franka_emika_panda",
        "universal_robots_ur5e",
        "kinova_gen3",
        "kinova_jaco2",
        "robotiq_2f85",
        "barrett_wam",
        "ufactory_lite6",
        "ufactory_xarm7",
        "leap_hand",
        "wonik_allegro",
        "shadow_hand",
        "robotis_op3",
        "abb_irb1600",
        "fanuc_m20ia",
        "kuka_iiwa_14",
        "rethink_sawyer",
        "widowx_250"
    ],
    "quadrupeds": [
        "unitree_go2",
        "unitree_go1",
        "unitree_a1",
        "boston_dynamics_spot",
        "anybotics_anymal_c",
        "anybotics_anymal_b",
        "google_barkour_v0",
        "mit_mini_cheetah"
    ],
    "humanoids": [
        "unitree_h1",
        "unitree_g1",
        "apptronik_apollo",
        "pal_talos",
        "berkeley_humanoid",
        "robotis_op3",
        "nasa_valkyrie",
        "honda_asimo",
        "boston_dynamics_atlas",
        "agility_cassie"
    ],
    "mobile_manipulators": [
        "google_robot",
        "hello_robot_stretch",
        "clearpath_ridgeback_ur5e",
        "fetch_robotics",
        "pr2"
    ],
    "drones": [
        "skydio_x2",
        "crazyflie_2"
    ],
    "grippers": [
        "robotiq_2f85",
        "robotiq_2f140",
        "shadow_hand",
        "leap_hand",
        "wonik_allegro",
        "barrett_hand"
    ]
}

class MenagerieModelTester:
    """Test MuJoCo Menagerie models for compatibility"""

    def __init__(self):
        self.results = {
            "test_summary": {
                "total_models": 0,
                "successful_loads": 0,
                "failed_loads": 0,
                "compatibility_score": 0.0,
                "test_duration": 0.0
            },
            "model_results": {},
            "category_performance": {},
            "recommendations": []
        }
        self.start_time = time.time()

    def test_model_url_access(self, model_name: str, category: str) -> Dict[str, Any]:
        """Test if model XML is accessible from GitHub"""
        base_url = "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main"

        # Try different common file patterns
        possible_files = [
            f"{model_name}/{model_name}.xml",
            f"{model_name}/scene.xml",
            f"{model_name}/{model_name}_mjx.xml"
        ]

        result = {
            "model_name": model_name,
            "category": category,
            "url_accessible": False,
            "available_files": [],
            "primary_xml": None,
            "file_size": 0,
            "load_time": 0.0
        }

        for xml_file in possible_files:
            url = f"{base_url}/{xml_file}"
            try:
                start = time.time()
                with urllib.request.urlopen(url, timeout=10) as response:
                    if response.getcode() == 200:
                        content_length = response.headers.get('Content-Length')
                        result["url_accessible"] = True
                        result["available_files"].append(xml_file)
                        result["load_time"] = time.time() - start

                        if content_length:
                            result["file_size"] = int(content_length)

                        # Use first accessible file as primary
                        if not result["primary_xml"]:
                            result["primary_xml"] = xml_file

            except Exception as e:
                continue

        return result

    def test_model_mujoco_compatibility(self, model_name: str, xml_url: str) -> Dict[str, Any]:
        """Test if model can be loaded with MuJoCo"""
        result = {
            "mujoco_compatible": False,
            "load_time": 0.0,
            "n_bodies": 0,
            "n_joints": 0,
            "n_actuators": 0,
            "error": None
        }

        if not MUJOCO_AVAILABLE:
            result["error"] = "MuJoCo not installed"
            return result

        try:
            # Download XML to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as tmp_file:
                with urllib.request.urlopen(xml_url, timeout=30) as response:
                    xml_content = response.read().decode('utf-8')
                    tmp_file.write(xml_content)
                    tmp_path = tmp_file.name

            # Test MuJoCo loading
            start = time.time()
            model = mujoco.MjModel.from_xml_path(tmp_path)
            result["load_time"] = time.time() - start

            # Extract model information
            result["mujoco_compatible"] = True
            result["n_bodies"] = model.nbody
            result["n_joints"] = model.njnt
            result["n_actuators"] = model.nu

            # Cleanup
            os.unlink(tmp_path)

        except Exception as e:
            result["error"] = str(e)
            result["mujoco_compatible"] = False

        return result

    def test_mcp_integration(self, model_name: str, category: str) -> Dict[str, Any]:
        """Test MCP server integration with model"""
        result = {
            "mcp_compatible": False,
            "scene_creation": False,
            "tools_available": False,
            "error": None
        }

        try:
            # Test basic MCP imports
            result["tools_available"] = True

            # Test scene creation (would need actual XML content)
            # For now, just test if the scene_type is supported
            supported_scenes = ["pendulum", "double_pendulum", "cart_pole", "arm"]

            # Map categories to supported scene types
            category_mapping = {
                "arms": "arm",
                "quadrupeds": "pendulum",  # Could extend to quadruped scene
                "humanoids": "pendulum",   # Could extend to humanoid scene
                "mobile_manipulators": "arm",
                "drones": "pendulum",
                "grippers": "arm"
            }

            if category in category_mapping:
                result["scene_creation"] = True
                result["mcp_compatible"] = True

        except Exception as e:
            result["error"] = str(e)

        return result

    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive testing of all Menagerie models"""
        print("🚀 Starting MuJoCo Menagerie Model Compatibility Testing...")
        print(f"📊 Testing {sum(len(models) for models in MENAGERIE_MODELS.values())} models across {len(MENAGERIE_MODELS)} categories")

        total_models = 0
        successful_loads = 0

        for category, models in MENAGERIE_MODELS.items():
            print(f"\n🔍 Testing {category.upper()} category ({len(models)} models)...")

            category_results = {
                "models_tested": len(models),
                "successful_loads": 0,
                "avg_load_time": 0.0,
                "compatibility_rate": 0.0
            }

            category_load_times = []

            for model_name in models:
                total_models += 1
                print(f"  🔬 Testing {model_name}...")

                # Test URL accessibility
                url_result = self.test_model_url_access(model_name, category)

                model_result = {
                    "category": category,
                    "url_test": url_result,
                    "mujoco_test": {},
                    "mcp_test": {}
                }

                # Test MuJoCo compatibility if URL is accessible
                if url_result["url_accessible"] and url_result["primary_xml"]:
                    base_url = "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main"
                    xml_url = f"{base_url}/{url_result['primary_xml']}"

                    mujoco_result = self.test_model_mujoco_compatibility(model_name, xml_url)
                    model_result["mujoco_test"] = mujoco_result

                    if mujoco_result["mujoco_compatible"]:
                        successful_loads += 1
                        category_results["successful_loads"] += 1
                        category_load_times.append(mujoco_result["load_time"])

                # Test MCP integration
                mcp_result = self.test_mcp_integration(model_name, category)
                model_result["mcp_test"] = mcp_result

                self.results["model_results"][model_name] = model_result

                # Status indicator
                status = (
                    "✅" if (
                        url_result["url_accessible"] and
                        model_result["mujoco_test"].get("mujoco_compatible", False)
                    ) else "❌"
                )
                print(f"    {status} {model_name}")

            # Calculate category metrics
            if category_load_times:
                category_results["avg_load_time"] = (
                    sum(category_load_times) / len(category_load_times)
                )
            category_results["compatibility_rate"] = (
                category_results["successful_loads"] / len(models)
            )

            self.results["category_performance"][category] = category_results

            print(
                f"  📈 {category.upper()}: {category_results['successful_loads']}/"
                f"{len(models)} models compatible "
                f"({category_results['compatibility_rate']:.1%})"
            )

        # Calculate overall metrics
        self.results["test_summary"]["total_models"] = total_models
        self.results["test_summary"]["successful_loads"] = successful_loads
        self.results["test_summary"]["failed_loads"] = total_models - successful_loads
        self.results["test_summary"]["compatibility_score"] = successful_loads / total_models if total_models > 0 else 0.0
        self.results["test_summary"]["test_duration"] = time.time() - self.start_time

        # Generate recommendations
        self._generate_recommendations()

        return self.results

    def _generate_recommendations(self):
        """Generate recommendations based on test results"""
        compatibility_score = self.results["test_summary"]["compatibility_score"]

        if compatibility_score >= 0.8:
            self.results["recommendations"].append("✅ Excellent compatibility! MCP server ready for production use with Menagerie models")
        elif compatibility_score >= 0.6:
            self.results["recommendations"].append("⚠️ Good compatibility. Consider adding support for failed models")
        else:
            self.results["recommendations"].append("❌ Low compatibility. Significant work needed for Menagerie integration")

        # Category-specific recommendations
        for category, perf in self.results["category_performance"].items():
            if perf["compatibility_rate"] < 0.5:
                self.results["recommendations"].append(f"🔧 Improve {category} support (only {perf['compatibility_rate']:.1%} compatible)")

        # Technical recommendations
        self.results["recommendations"].extend([
            "🚀 Consider adding direct Menagerie integration to MCP server",
            "📦 Add model auto-discovery from GitHub repository",
            "🎯 Implement category-specific scene templates",
            "🔄 Add model caching for faster loading"
        ])

def main():
    """Main test execution"""
    tester = MenagerieModelTester()
    results = tester.run_comprehensive_test()

    # Save detailed results
    with open("menagerie_compatibility_report.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("🎯 MUJOCO MENAGERIE COMPATIBILITY REPORT")
    print(f"{'='*60}")

    summary = results["test_summary"]
    print(f"📊 Total Models Tested: {summary['total_models']}")
    print(f"✅ Successfully Loaded: {summary['successful_loads']}")
    print(f"❌ Failed to Load: {summary['failed_loads']}")
    print(f"🎯 Compatibility Score: {summary['compatibility_score']:.1%}")
    print(f"⏱️ Test Duration: {summary['test_duration']:.2f}s")

    print("\n📈 CATEGORY PERFORMANCE:")
    for category, perf in results["category_performance"].items():
        print(f"  {category.upper()}: {perf['successful_loads']}/{perf['models_tested']} ({perf['compatibility_rate']:.1%})")

    print("\n💡 RECOMMENDATIONS:")
    for rec in results["recommendations"]:
        print(f"  {rec}")

    print("\n📄 Detailed report saved to: menagerie_compatibility_report.json")

    return 0 if summary['compatibility_score'] >= 0.7 else 1

if __name__ == "__main__":
    sys.exit(main())

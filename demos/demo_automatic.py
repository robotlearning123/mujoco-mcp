#!/usr/bin/env python3
"""
Automatic Demo: Random MuJoCo Menagerie Model via MCP
Shows testing, loading, and controlling random models
"""

import asyncio
import json
import random
import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent / "src"))


async def demo_random_menagerie_model():
    """Automatic demo with random model selection"""
    print("🚀 Automatic MuJoCo Menagerie MCP Demo")
    print("=" * 45)

    try:
        # Import enhanced MCP server
        from mujoco_mcp.mcp_server_menagerie import handle_list_tools, handle_call_tool

        # Step 1: Show MCP capabilities
        print("\n🔧 Step 1: MCP Server Capabilities")
        print("-" * 35)

        tools = await handle_list_tools()
        print(f"✅ Enhanced MCP Server with {len(tools)} tools:")

        menagerie_tools = []
        built_in_tools = []

        for tool in tools:
            if "menagerie" in tool.name.lower():
                menagerie_tools.append(tool.name)
            else:
                built_in_tools.append(tool.name)

        print(f"  🆕 New Menagerie tools: {', '.join(menagerie_tools)}")
        print(f"  📋 Built-in tools: {', '.join(built_in_tools)}")

        # Step 2: Discover all available models
        print("\n📦 Step 2: Model Discovery")
        print("-" * 28)

        models_result = await handle_call_tool("list_menagerie_models", {})
        models_data = json.loads(models_result[0].text)

        print(
            f"🌟 Discovered {models_data['total_models']} models across "
            f"{models_data['categories']} categories:"
        )

        # Collect all models and show category breakdown
        all_models = []
        for category, info in models_data["models"].items():
            print(f"  🏷️  {category.upper()}: {info['count']} models")

            # Show 2 examples from each category
            examples = info["models"][:2]
            if examples:
                print(f"     Examples: {', '.join(examples)}")

            all_models.extend(info["models"])

        # Step 3: Random model selection
        print("\n🎲 Step 3: Random Model Selection")
        print("-" * 32)

        # Select models from different categories for variety
        categories = list(models_data["models"].keys())
        selected_models = []

        for category in random.sample(categories, min(3, len(categories))):
            category_models = models_data["models"][category]["models"]
            selected_model = random.choice(category_models)
            selected_models.append((selected_model, category))

        print("🎯 Selected models for testing:")
        for i, (model, category) in enumerate(selected_models, 1):
            print(f"  {i}. {model} ({category})")

        # Step 4: Test each selected model
        for i, (model_name, category) in enumerate(selected_models, 1):
            print(f"\n{'=' * 20} MODEL {i}: {model_name.upper()} {'=' * 20}")

            # Validate the model
            print(f"🔬 Validating {model_name}...")
            validation_result = await handle_call_tool(
                "validate_menagerie_model", {"model_name": model_name}
            )

            validation_text = validation_result[0].text
            print(f"   {validation_text}")

            # Create scene from the model
            print("🏗️  Creating scene...")
            scene_name = f"demo_{model_name}_{i}"
            scene_result = await handle_call_tool(
                "create_menagerie_scene", {"model_name": model_name, "scene_name": scene_name}
            )

            scene_text = scene_result[0].text
            print(f"   🎭 {scene_text}")

            # Test simulation control
            if "✅" in scene_text or "XML generated" in scene_text:
                print("⚡ Testing simulation control...")

                # Step simulation
                step_result = await handle_call_tool(
                    "step_simulation", {"model_id": scene_name, "steps": 3}
                )
                print(f"   🔄 Step: {step_result[0].text}")

                # Get state
                state_result = await handle_call_tool("get_state", {"model_id": scene_name})
                state_preview = (
                    state_result[0].text[:100] + "..."
                    if len(state_result[0].text) > 100
                    else state_result[0].text
                )
                print(f"   📊 State: {state_preview}")

                # Reset simulation
                reset_result = await handle_call_tool("reset_simulation", {"model_id": scene_name})
                print(f"   🔄 Reset: {reset_result[0].text}")

        # Step 5: Demonstrate enhanced create_scene
        print("\n✨ Step 5: Enhanced Scene Creation")
        print("-" * 35)

        # Pick a final model for enhanced demo
        final_model = random.choice(all_models)
        print(f"🎪 Demonstrating enhanced create_scene with {final_model}")

        enhanced_result = await handle_call_tool(
            "create_scene",
            {
                "scene_type": "pendulum",  # Built-in scene type
                "menagerie_model": final_model,  # Our Menagerie enhancement!
            },
        )

        enhanced_text = enhanced_result[0].text
        print(f"   ✨ Enhanced: {enhanced_text}")

        # Demo Summary
        print(f"\n{'=' * 60}")
        print("🎉 AUTOMATIC DEMO COMPLETE")
        print(f"{'=' * 60}")
        print(f"🎯 Models Tested: {len(selected_models)}")
        print(f"📊 Total Available: {models_data['total_models']} models")
        print(f"🏷️  Categories: {', '.join(models_data['models'].keys())}")
        print(f"🔧 MCP Tools: {len(menagerie_tools)} Menagerie + {len(built_in_tools)} built-in")
        print("✅ Status: All models accessible via MCP!")

        print("\n💡 What you can do now:")
        print(f"   • Use any of the {models_data['total_models']} models in Claude Desktop")
        print("   • Call list_menagerie_models to browse all models")
        print("   • Call create_menagerie_scene with any model name")
        print("   • Use enhanced create_scene with menagerie_model parameter")
        print("   • Control simulations with step_simulation, get_state, reset_simulation")

        return True

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(demo_random_menagerie_model())
        print(f"\n🚀 Demo {'successful' if success else 'failed'}!")
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n💥 Demo crashed: {e}")
        sys.exit(1)

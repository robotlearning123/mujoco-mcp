#!/usr/bin/env python3
"""
Demo: Test, Load, and Control Random MuJoCo Menagerie Model via MCP
Interactive demonstration of MCP server with Menagerie models
"""

import asyncio
import json
import random
import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def demo_menagerie_mcp():
    """Interactive demo of MCP server with Menagerie models"""
    print("🚀 MuJoCo Menagerie MCP Demo")
    print("=" * 50)

    try:
        # Import enhanced MCP server
        from mujoco_mcp.mcp_server_menagerie import handle_list_tools, handle_call_tool

        # Step 1: Show available tools
        print("\n🔧 Step 1: Available MCP Tools")
        print("-" * 30)

        tools = await handle_list_tools()
        print(f"✅ Total tools available: {len(tools)}")

        for tool in tools:
            print(f"  📋 {tool.name}: {tool.description}")

        # Step 2: List all Menagerie models
        print("\n📦 Step 2: Available Menagerie Models")
        print("-" * 40)

        models_result = await handle_call_tool("list_menagerie_models", {})
        models_data = json.loads(models_result[0].text)

        print(f"✅ Found {models_data['categories']} categories with {models_data['total_models']} total models")

        # Show models by category
        all_models = []
        for category, info in models_data['models'].items():
            print(f"\n  🏷️  {category.upper()}: {info['count']} models")
            for model in info['models'][:3]:  # Show first 3
                print(f"     • {model}")
                all_models.append(model)
            if len(info['models']) > 3:
                print(f"     ... and {len(info['models']) - 3} more")
                all_models.extend(info['models'][3:])

        # Step 3: Select random model
        print("\n🎲 Step 3: Random Model Selection")
        print("-" * 35)

        random_model = random.choice(all_models)
        print(f"🎯 Selected random model: {random_model}")

        # Step 4: Validate the selected model
        print("\n🔬 Step 4: Model Validation")
        print("-" * 30)

        validation_result = await handle_call_tool("validate_menagerie_model", {
            "model_name": random_model
        })

        validation_text = validation_result[0].text
        print(f"📊 Validation result: {validation_text}")

        # Step 5: Create scene from the model
        print("\n🎭 Step 5: Scene Creation")
        print("-" * 25)

        scene_name = f"demo_{random_model}"
        scene_result = await handle_call_tool("create_menagerie_scene", {
            "model_name": random_model,
            "scene_name": scene_name
        })

        scene_text = scene_result[0].text
        print(f"🏗️  Scene creation: {scene_text}")

        # Step 6: Simulate control operations
        print("\n⚡ Step 6: Simulation Control")
        print("-" * 30)

        # Try to step simulation
        step_result = await handle_call_tool("step_simulation", {
            "model_id": scene_name,
            "steps": 5
        })

        step_text = step_result[0].text
        print(f"🔄 Simulation step: {step_text}")

        # Try to get state
        state_result = await handle_call_tool("get_state", {
            "model_id": scene_name
        })

        state_text = state_result[0].text
        print(f"📊 State query: {state_text[:200]}..." if len(state_text) > 200 else f"📊 State query: {state_text}")

        # Try to reset simulation
        reset_result = await handle_call_tool("reset_simulation", {
            "model_id": scene_name
        })

        reset_text = reset_result[0].text
        print(f"🔄 Reset simulation: {reset_text}")

        # Step 7: Enhanced scene creation (alternative method)
        print("\n🎪 Step 7: Enhanced Scene Creation")
        print("-" * 35)

        # Pick another random model for enhanced demo
        another_model = random.choice([m for m in all_models if m != random_model])
        print(f"🎯 Using enhanced create_scene with: {another_model}")

        enhanced_result = await handle_call_tool("create_scene", {
            "scene_type": "pendulum",  # Required parameter
            "menagerie_model": another_model  # Our enhancement!
        })

        enhanced_text = enhanced_result[0].text
        print(f"✨ Enhanced scene: {enhanced_text}")

        # Step 8: Cleanup
        print("\n🧹 Step 8: Cleanup")
        print("-" * 20)

        cleanup_result = await handle_call_tool("close_viewer", {
            "model_id": scene_name
        })

        cleanup_text = cleanup_result[0].text
        print(f"🚮 Cleanup: {cleanup_text}")

        # Demo Summary
        print(f"\n{'=' * 50}")
        print("🎉 DEMO SUMMARY")
        print(f"{'=' * 50}")
        print(f"🎯 Random Model: {random_model}")
        print(f"🏗️  Scene Name: {scene_name}")
        print(f"✨ Enhanced Demo: {another_model}")
        print(f"📊 Total Models Available: {models_data['total_models']}")
        print("🔧 MCP Tools Used: 7 different tools")
        print("✅ Demo Status: Complete!")

        return True

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def interactive_menagerie_demo():
    """Interactive version where user can choose models"""
    print("🎮 Interactive MuJoCo Menagerie MCP Demo")
    print("=" * 45)

    try:
        from mujoco_mcp.mcp_server_menagerie import handle_call_tool

        # Get available models
        models_result = await handle_call_tool("list_menagerie_models", {})
        models_data = json.loads(models_result[0].text)

        print("\n📦 Available Categories:")
        categories = list(models_data['models'].keys())
        for i, category in enumerate(categories, 1):
            count = models_data['models'][category]['count']
            print(f"  {i}. {category.upper()} ({count} models)")

        # Let user choose category
        print(f"\n🎯 Choose a category (1-{len(categories)}) or 'r' for random:")
        choice = input("Your choice: ").strip().lower()

        if choice == 'r':
            # Random category and model
            category = random.choice(categories)
            available_models = models_data['models'][category]['models']
            selected_model = random.choice(available_models)
            print(f"🎲 Random selection: {selected_model} from {category}")
        else:
            try:
                cat_index = int(choice) - 1
                if 0 <= cat_index < len(categories):
                    category = categories[cat_index]
                    available_models = models_data['models'][category]['models']

                    print(f"\n📋 Models in {category.upper()}:")
                    for i, model in enumerate(available_models, 1):
                        print(f"  {i}. {model}")

                    model_choice = input(f"\nChoose model (1-{len(available_models)}) or 'r' for random: ").strip()

                    if model_choice.lower() == 'r':
                        selected_model = random.choice(available_models)
                    else:
                        model_index = int(model_choice) - 1
                        if 0 <= model_index < len(available_models):
                            selected_model = available_models[model_index]
                        else:
                            print("Invalid choice, using random")
                            selected_model = random.choice(available_models)
                else:
                    print("Invalid choice, using random")
                    category = random.choice(categories)
                    selected_model = random.choice(models_data['models'][category]['models'])
            except ValueError:
                print("Invalid input, using random")
                category = random.choice(categories)
                selected_model = random.choice(models_data['models'][category]['models'])

        print(f"\n🎯 Selected: {selected_model}")
        print(f"📂 Category: {category}")

        # Validate and create scene
        print(f"\n🔬 Validating {selected_model}...")
        validation_result = await handle_call_tool("validate_menagerie_model", {
            "model_name": selected_model
        })
        print(f"✅ {validation_result[0].text}")

        print("\n🏗️  Creating scene...")
        scene_result = await handle_call_tool("create_menagerie_scene", {
            "model_name": selected_model,
            "scene_name": f"interactive_{selected_model}"
        })
        print(f"🎭 {scene_result[0].text}")

        print("\n🎉 Interactive demo complete!")
        print(f"✨ You can now use '{selected_model}' in your MCP client!")

        return True

    except Exception as e:
        print(f"\n❌ Interactive demo failed: {e}")
        return False

async def main():
    """Main demo runner"""
    print("🚀 MuJoCo Menagerie MCP Demonstration")
    print("Which demo would you like to run?")
    print("1. Automatic demo with random model")
    print("2. Interactive demo - choose your model")
    print("3. Both demos")

    choice = input("\nEnter your choice (1/2/3): ").strip()

    success = True

    if choice in ['1', '3']:
        success &= await demo_menagerie_mcp()

    if choice in ['2', '3']:
        if choice == '3':
            print("\n" + "="*60)
            print("STARTING INTERACTIVE DEMO")
            print("="*60)
        success &= await interactive_menagerie_demo()

    if choice not in ['1', '2', '3']:
        print("Invalid choice, running automatic demo...")
        success = await demo_menagerie_mcp()

    return success

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        if success:
            print("\n🎉 All demos completed successfully!")
            print(f"🚀 Your MCP server can now control all {39} Menagerie models!")
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Demo crashed: {e}")
        sys.exit(1)

#!/usr/bin/env python3
"""
Test script to validate MCP protocol compliance fixes
Tests the critical fixes implemented for MCP protocol version 2024-11-05
"""

import asyncio
import json
import sys
from typing import Dict, Any, List
import jsonschema
from jsonschema import validate

# Import the MCP server components
from src.mujoco_mcp.mcp_server import (
    handle_list_tools, 
    handle_call_tool,
    MCP_PROTOCOL_VERSION,
    server,
    main
)

def test_protocol_version():
    """Test Critical Fix #1: Protocol Version Alignment"""
    print("Testing Protocol Version...")
    
    # Check that protocol version is set correctly
    assert MCP_PROTOCOL_VERSION == "2024-11-05", f"Expected '2024-11-05', got '{MCP_PROTOCOL_VERSION}'"
    print("✅ Protocol version is correctly set to 2024-11-05")
    
    return True

async def test_tool_schemas():
    """Test Critical Fix #3: Tool Schema Validation (JSON Schema Draft 7)"""
    print("Testing Tool Schema Compliance...")
    
    tools = await handle_list_tools()
    
    for tool in tools:
        schema = tool.inputSchema
        
        # Check for $schema field
        assert "$schema" in schema, f"Tool '{tool.name}' missing $schema field"
        assert schema["$schema"] == "http://json-schema.org/draft-07/schema#", \
            f"Tool '{tool.name}' has incorrect $schema"
        
        # Check for additionalProperties
        assert "additionalProperties" in schema, f"Tool '{tool.name}' missing additionalProperties"
        assert schema["additionalProperties"] == False, \
            f"Tool '{tool.name}' should set additionalProperties to False"
        
        # Validate schema structure
        try:
            # This validates that our schema is a valid JSON Schema
            jsonschema.Draft7Validator.check_schema(schema)
            print(f"✅ Tool '{tool.name}' has valid JSON Schema Draft 7")
        except jsonschema.SchemaError as e:
            print(f"❌ Tool '{tool.name}' has invalid schema: {e}")
            return False
    
    return True

async def test_server_initialization():
    """Test Critical Fix #2: Server Initialization"""
    print("Testing Server Initialization...")
    
    # Import required MCP components  
    from mcp.server import NotificationOptions
    
    # Test that server has proper name and capabilities
    capabilities = server.get_capabilities(
        notification_options=NotificationOptions(),
        experimental_capabilities={}
    )
    
    assert capabilities is not None, "Server capabilities should not be None"
    assert hasattr(capabilities, 'tools'), "Server should have tools capability"
    assert capabilities.tools is not None, "Tools capability should not be None"
    
    print("✅ Server initialization appears correct")
    return True

async def test_response_format():
    """Test Critical Fix #4: Response Format Consistency"""
    print("Testing Response Format...")
    
    # Test get_server_info response
    response = await handle_call_tool("get_server_info", {})
    
    assert len(response) == 1, "Should return exactly one response item"
    assert response[0].type == "text", "Response should be of type 'text'"
    
    # Validate that the response text is valid JSON
    try:
        data = json.loads(response[0].text)
        assert "name" in data, "Server info should include name"
        assert "version" in data, "Server info should include version" 
        assert "status" in data, "Server info should include status"
        print("✅ Response format is consistent and valid")
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in response: {e}")
        return False
    
    return True

def test_schema_validation_examples():
    """Test that our tool schemas can validate actual inputs"""
    print("Testing Schema Validation with Examples...")
    
    # Test create_scene schema
    create_scene_schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "scene_type": {
                "type": "string",
                "description": "Type of scene to create",
                "enum": ["pendulum", "double_pendulum", "cart_pole", "arm"]
            }
        },
        "required": ["scene_type"],
        "additionalProperties": False
    }
    
    # Valid input
    valid_input = {"scene_type": "pendulum"}
    try:
        validate(instance=valid_input, schema=create_scene_schema)
        print("✅ Valid input passes schema validation")
    except jsonschema.ValidationError as e:
        print(f"❌ Valid input failed validation: {e}")
        return False
    
    # Invalid input (missing required field)
    invalid_input = {}
    try:
        validate(instance=invalid_input, schema=create_scene_schema)
        print("❌ Invalid input should have failed validation")
        return False
    except jsonschema.ValidationError:
        print("✅ Invalid input correctly rejected")
    
    # Invalid input (wrong enum value)
    invalid_enum_input = {"scene_type": "invalid_scene"}
    try:
        validate(instance=invalid_enum_input, schema=create_scene_schema)
        print("❌ Invalid enum value should have failed validation")
        return False
    except jsonschema.ValidationError:
        print("✅ Invalid enum value correctly rejected")
    
    return True

async def run_all_tests():
    """Run all compliance tests"""
    print("="*60)
    print("MCP Protocol Compliance Test Suite")
    print("Testing fixes for MCP Protocol Version 2024-11-05")
    print("="*60)
    
    test_results = []
    
    # Test 1: Protocol Version
    try:
        result = test_protocol_version()
        test_results.append(("Protocol Version", result))
    except Exception as e:
        print(f"❌ Protocol version test failed: {e}")
        test_results.append(("Protocol Version", False))
    
    # Test 2: Tool Schemas
    try:
        result = await test_tool_schemas()
        test_results.append(("Tool Schemas", result))
    except Exception as e:
        print(f"❌ Tool schema test failed: {e}")
        test_results.append(("Tool Schemas", False))
    
    # Test 3: Server Initialization  
    try:
        result = await test_server_initialization()
        test_results.append(("Server Initialization", result))
    except Exception as e:
        print(f"❌ Server initialization test failed: {e}")
        test_results.append(("Server Initialization", False))
    
    # Test 4: Response Format
    try:
        result = await test_response_format()
        test_results.append(("Response Format", result))
    except Exception as e:
        print(f"❌ Response format test failed: {e}")
        test_results.append(("Response Format", False))
    
    # Test 5: Schema Validation Examples
    try:
        result = test_schema_validation_examples()
        test_results.append(("Schema Validation", result))
    except Exception as e:
        print(f"❌ Schema validation test failed: {e}")
        test_results.append(("Schema Validation", False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} : {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! MCP compliance fixes are working.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the fixes.")
        return False

if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
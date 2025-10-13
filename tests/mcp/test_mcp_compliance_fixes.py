#!/usr/bin/env python3
"""
Test suite covering the key MCP protocol compliance guarantees for MuJoCo MCP.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import jsonschema
import pytest
from jsonschema import validate

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.mujoco_mcp.mcp_server import (
    MCP_PROTOCOL_VERSION,
    handle_call_tool,
    handle_list_resources,
    handle_list_tools,
    handle_read_resource,
    main,
    server,
)


def _check_protocol_version() -> bool:
    """Ensure the advertised protocol version matches the 2024-11-05 spec."""

    print("Testing Protocol Version...")
    assert MCP_PROTOCOL_VERSION == "2024-11-05", (
        f"Expected '2024-11-05', got '{MCP_PROTOCOL_VERSION}'"
    )
    print("✅ Protocol version is correctly set to 2024-11-05")
    return True


async def _check_tool_schemas() -> bool:
    """Validate tool schemas comply with JSON Schema Draft-07."""

    print("Testing Tool Schema Compliance...")
    tools = await handle_list_tools()
    for tool in tools:
        schema = tool.inputSchema
        assert schema.get("$schema") == "http://json-schema.org/draft-07/schema#", (
            f"Tool '{tool.name}' has incorrect $schema"
        )
        assert schema.get("additionalProperties") is False, (
            f"Tool '{tool.name}' must set additionalProperties to False"
        )
        jsonschema.Draft7Validator.check_schema(schema)
        print(f"✅ Tool '{tool.name}' has valid JSON Schema Draft 7")
    return True


async def _check_server_initialization() -> bool:
    """Verify the server advertises tool capabilities during initialization."""

    print("Testing Server Initialization...")
    from mcp.server import NotificationOptions

    capabilities = server.get_capabilities(
        notification_options=NotificationOptions(),
        experimental_capabilities={},
    )
    assert capabilities is not None
    assert getattr(capabilities, "tools", None) is not None
    print("✅ Server initialization appears correct")
    return True


async def _check_response_format() -> bool:
    """Ensure tool responses follow the structured JSON envelope."""

    print("Testing Response Format...")
    response = await handle_call_tool("get_server_info", {})
    assert response and response[0].type == "text"
    payload = json.loads(response[0].text)
    assert payload.get("status") == "ok"
    details = payload.get("data", {})
    required = {"name", "version", "description", "protocol_version"}
    missing = required - details.keys()
    assert not missing, f"Missing keys: {', '.join(sorted(missing))}"
    assert details["protocol_version"] == MCP_PROTOCOL_VERSION
    print("✅ Response format is consistent and valid")
    return True


def _check_schema_validation_examples() -> bool:
    """Exercise example payloads against the create_scene schema."""

    print("Testing Schema Validation with Examples...")
    create_scene_schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "scene_type": {
                "type": "string",
                "description": "Type of scene to create",
                "enum": ["pendulum", "double_pendulum", "cart_pole", "arm"],
            }
        },
        "required": ["scene_type"],
        "additionalProperties": False,
    }

    validate(instance={"scene_type": "pendulum"}, schema=create_scene_schema)
    print("✅ Valid input passes schema validation")

    with pytest.raises(jsonschema.ValidationError):
        validate(instance={}, schema=create_scene_schema)
    print("✅ Invalid input correctly rejected")

    with pytest.raises(jsonschema.ValidationError):
        validate(instance={"scene_type": "invalid"}, schema=create_scene_schema)
    print("✅ Invalid enum value correctly rejected")
    return True


async def _check_resource_listing() -> bool:
    """Confirm required resources are advertised."""

    print("Testing Resource Listing...")
    resources = await handle_list_resources()
    uris = {str(resource.uri) for resource in resources}
    assert "simulation://state" in uris
    assert "simulation://config" in uris
    print("✅ Resource listing advertises expected URIs")
    return True


async def _check_resource_read() -> bool:
    """Validate resource reads return JSON payloads."""

    print("Testing Resource Read...")
    state_resource = await handle_read_resource("simulation://state")
    state_payload = json.loads(state_resource[0].content)
    assert "status" in state_payload

    config_resource = await handle_read_resource("simulation://config")
    config_payload = json.loads(config_resource[0].content)
    assert config_payload.get("status") == "ok"
    assert config_payload.get("data", {}).get("protocol_version") == MCP_PROTOCOL_VERSION
    print("✅ Resource read delivers JSON payloads")
    return True


async def run_all_tests() -> List[tuple[str, bool]]:
    """Run the compliance checks sequentially for CLI consumption."""

    print("=" * 60)
    print("MCP Protocol Compliance Test Suite")
    print("Testing fixes for MCP Protocol Version 2024-11-05")
    print("=" * 60)

    results: List[tuple[str, bool]] = []

    try:
        results.append(("Protocol Version", _check_protocol_version()))
    except Exception as exc:
        print(f"❌ Protocol version test failed: {exc}")
        results.append(("Protocol Version", False))

    for label, coroutine in [
        ("Tool Schemas", _check_tool_schemas()),
        ("Server Initialization", _check_server_initialization()),
        ("Response Format", _check_response_format()),
        ("Resource Listing", _check_resource_listing()),
        ("Resource Read", _check_resource_read()),
    ]:
        try:
            success = await coroutine
            results.append((label, success))
        except Exception as exc:
            print(f"❌ {label} test failed: {exc}")
            results.append((label, False))

    try:
        results.append(("Schema Validation", _check_schema_validation_examples()))
    except Exception as exc:
        print(f"❌ Schema validation test failed: {exc}")
        results.append(("Schema Validation", False))

    return results


def main() -> None:
    try:
        results = asyncio.run(run_all_tests())
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(130)

    passed = sum(success for _, success in results)
    total = len(results)

    print("\n" + "=" * 60)
    print("Compliance Summary")
    print("=" * 60)
    for name, success in results:
        marker = "✅" if success else "❌"
        print(f"{marker} {name}")

    print(f"\nPassed {passed}/{total} checks")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Pytest wrappers to ensure compliance checks execute under pytest as well
# ---------------------------------------------------------------------------


def test_protocol_version():
    assert _check_protocol_version()


@pytest.mark.asyncio
async def test_tool_schemas():
    assert await _check_tool_schemas()


@pytest.mark.asyncio
async def test_server_initialization():
    assert await _check_server_initialization()


@pytest.mark.asyncio
async def test_response_format():
    assert await _check_response_format()


def test_schema_validation_examples():
    assert _check_schema_validation_examples()


@pytest.mark.asyncio
async def test_resource_listing():
    assert await _check_resource_listing()


@pytest.mark.asyncio
async def test_resource_read():
    assert await _check_resource_read()

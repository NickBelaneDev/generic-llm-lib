import pytest
from generic_llm_lib.llm_impl.gemini import schema_sanitizer


def test_strip_additional_properties():
    """Tests that 'additionalProperties' is recursively removed."""
    schema = {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "additionalProperties": False,
            },
            "items": {
                "type": "array",
                "items": [
                    {
                        "type": "object",
                        "properties": {"price": {"type": "number"}},
                        "additionalProperties": False,
                    }
                ],
            },
        },
        "additionalProperties": False,
    }

    sanitized = schema_sanitizer.sanitize(schema)

    assert "additionalProperties" not in sanitized
    assert "additionalProperties" not in sanitized["properties"]["user"]
    assert "additionalProperties" not in sanitized["properties"]["items"]["items"][0]
    assert "price" in sanitized["properties"]["items"]["items"][0]["properties"]


def test_ensure_required_params_removes_undefined():
    """Tests that required properties not in 'properties' are removed."""
    schema = {
        "type": "object",
        "properties": {"defined_prop": {"type": "string"}},
        "required": ["defined_prop", "undefined_prop"],
    }

    sanitized = schema_sanitizer.sanitize(schema)

    assert "required" in sanitized
    assert sanitized["required"] == ["defined_prop"]


def test_ensure_required_params_removes_key_if_empty():
    """Tests that the 'required' key is removed if no properties are valid."""
    schema = {
        "type": "object",
        "properties": {"defined_prop": {"type": "string"}},
        "required": ["undefined_prop_1", "undefined_prop_2"],
    }

    sanitized = schema_sanitizer.sanitize(schema)

    assert "required" not in sanitized


def test_sanitize_handles_clean_schema():
    """Tests that a schema that is already clean remains unchanged."""
    schema = {
        "type": "object",
        "properties": {"prop1": {"type": "string"}},
        "required": ["prop1"],
    }
    # Make a copy to ensure it's not modified in place
    original_schema = schema.copy()

    sanitized = schema_sanitizer.sanitize(schema)

    assert sanitized == original_schema


def test_sanitize_integration_e2e():
    """
    Tests both sanitization steps working together on a complex schema.
    """
    schema = {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "additionalProperties": False,
                "required": ["name", "age"],  # 'age' is not defined
            }
        },
        "additionalProperties": False,
        "required": ["user", "order_id"],  # 'order_id' is not defined
    }

    sanitized = schema_sanitizer.sanitize(schema)

    # Top-level checks
    assert "additionalProperties" not in sanitized
    assert sanitized["required"] == ["user"]

    # Nested checks
    user_prop = sanitized["properties"]["user"]
    assert "additionalProperties" not in user_prop
    assert user_prop["required"] == ["name"]

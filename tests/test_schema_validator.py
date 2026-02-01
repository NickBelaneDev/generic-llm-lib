import pytest
from llm_core.tools.schema_validator import SchemaValidator
from llm_core.exceptions.exceptions import ToolValidationError

def test_assert_no_recursive_refs_no_recursion():
    schema = {
        "type": "object",
        "properties": {
            "prop1": {"type": "string"},
            "prop2": {
                "type": "object",
                "properties": {
                    "subprop": {"type": "integer"}
                }
            }
        }
    }
    # Should not raise
    SchemaValidator.assert_no_recursive_refs(schema)

def test_assert_no_recursive_refs_with_recursion():
    schema = {
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {
                    "child": {"$ref": "#/$defs/Node"}
                }
            }
        },
        "properties": {
            "root": {"$ref": "#/$defs/Node"}
        }
    }
    with pytest.raises(ToolValidationError, match="Recursive structure detected"):
        SchemaValidator.assert_no_recursive_refs(schema)

def test_sanitize_schema_removes_metadata():
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "http://example.com/schema",
        "title": "MySchema",
        "type": "object",
        "properties": {
            "field": {"type": "string", "title": "FieldTitle"}
        },
        "definitions": {"SomeDef": {}}
    }
    sanitized = SchemaValidator.sanitize_schema(schema)
    
    assert "$schema" not in sanitized
    assert "$id" not in sanitized
    assert "title" not in sanitized
    assert "definitions" not in sanitized
    assert "title" not in sanitized["properties"]["field"]

def test_sanitize_schema_simplifies_optional():
    # Simulating Optional[int] -> anyOf: [type: integer, type: null]
    schema = {
        "type": "object",
        "properties": {
            "optional_field": {
                "anyOf": [
                    {"type": "integer", "description": "An integer"},
                    {"type": "null"}
                ],
                "description": "Parent description"
            }
        }
    }
    sanitized = SchemaValidator.sanitize_schema(schema)
    field = sanitized["properties"]["optional_field"]
    
    # Should be simplified to just the type
    assert "anyOf" not in field
    assert field["type"] == "integer"
    # Description should be preserved (parent takes precedence in current logic if present)
    assert field["description"] == "Parent description"

def test_sanitize_schema_enforces_additional_properties():
    schema = {
        "type": "object",
        "properties": {
            "field": {"type": "string"}
        }
    }
    sanitized = SchemaValidator.sanitize_schema(schema)
    assert sanitized["additionalProperties"] is False

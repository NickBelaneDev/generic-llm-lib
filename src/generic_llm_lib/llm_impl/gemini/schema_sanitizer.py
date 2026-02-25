"""
A module for sanitizing tool schemas for the Google Gemini API.

This module centralizes the logic for cleaning and adapting a generic tool
schema to meet the specific validation requirements of the Gemini API.
"""

from typing import Dict, Any, Set, cast
from functools import singledispatch


def sanitize(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Performs all necessary, recursive sanitization steps on a tool schema.

    Args:
        schema: The tool parameter schema to sanitize.

    Returns:
        A sanitized schema dictionary ready for the Gemini API.
    """
    # We can safely cast the result because we know the top-level input is a dict,
    # and our dispatch function for dicts returns a dict.
    return cast(Dict[str, Any], _recursive_sanitize(schema, set()))


@singledispatch
def _recursive_sanitize(schema: Any, seen: Set[int]) -> Any:
    """
    Recursively sanitizes a schema object. The implementation is chosen
    based on the object's type (dict, list, or other).
    """
    # Base case for non-dict and non-list types.
    return schema


@_recursive_sanitize.register(dict)
def _(schema: dict, seen: Set[int]) -> dict:
    """
    Sanitizes a dictionary. It first checks for circular references,
    then sanitizes the current level, and finally recurses on its values.
    """
    obj_id = id(schema)
    if obj_id in seen:
        return schema  # Circular reference detected
    seen.add(obj_id)

    # 1. Sanitize the current dictionary level.
    sanitized_at_level = _ensure_required_params(schema)

    # 2. Recursively sanitize all values and filter out 'additionalProperties'.
    result = {
        key: _recursive_sanitize(value, seen)
        for key, value in sanitized_at_level.items()
        if key != "additionalProperties"
    }

    seen.remove(obj_id)
    return result


@_recursive_sanitize.register(list)
def _(schema: list, seen: Set[int]) -> list:
    """Sanitizes a list by recursively sanitizing all of its items."""
    obj_id = id(schema)
    if obj_id in seen:
        return schema  # Circular reference detected
    seen.add(obj_id)

    result = [_recursive_sanitize(item, seen) for item in schema]

    seen.remove(obj_id)
    return result


def _ensure_required_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensures all required parameters in a dictionary are defined in its
    'properties'. This operates on a single dictionary level.
    """
    if "required" not in params or "properties" not in params:
        return params

    _params = params.copy()
    defined_properties = set(_params["properties"].keys())
    required_properties = set(_params["required"])
    valid_required = list(required_properties.intersection(defined_properties))

    if valid_required:
        _params["required"] = valid_required
    else:
        _params.pop("required", None)

    return _params

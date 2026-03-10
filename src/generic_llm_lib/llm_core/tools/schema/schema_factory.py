"""Factory and helper functions for creating and manipulating JSON schemas."""

from typing import Any


def preserve_ref_siblings(schema: Any) -> Any:
    """
    Recursively transforms a JSON schema to preserve siblings of $ref keywords.

    This function addresses a common issue with JSON schema reference resolvers
    that might discard sibling keywords (like "description") next to a "$ref".
    It wraps the "$ref" and its siblings in an "allOf" structure, which is the
    standard and safe way to extend a referenced schema.

    Example:
        Input:
        {
            "description": "A reference to a user.",
            "$ref": "#/$defs/User"
        }

        Output:
        {
            "allOf": [ { "$ref": "#/$defs/User" } ],
            "description": "A reference to a user."
        }

    Args:
        schema: The JSON schema (or a part of it) to transform.

    Returns:
        The transformed schema.
    """
    if isinstance(schema, dict):
        # Check if the dictionary contains a $ref and has other keys (siblings)
        if "$ref" in schema and len(schema) > 1:
            ref_value = schema.pop("$ref")
            # All other items are siblings. Wrap the $ref in allOf.
            return {"allOf": [{"$ref": ref_value}], **schema}

        # Recurse into the values of the dictionary
        return {key: preserve_ref_siblings(value) for key, value in schema.items()}

    if isinstance(schema, list):
        # Recurse into the items of the list
        return [preserve_ref_siblings(item) for item in schema]

    # Return primitives as is
    return schema


def flatten_single_all_of(schema: Any) -> Any:
    """
    Recursively flattens "allOf" arrays that contain only a single element.

    This function simplifies schemas that use "allOf" for extending a single
    referenced schema, which is a common pattern after resolving references
    where sibling keywords (like "description") were preserved.

    Example:
        Input:
        {
            "description": "A user object.",
            "allOf": [
                {
                    "type": "object",
                    "properties": { "name": { "type": "string" } }
                }
            ]
        }

        Output:
        {
            "description": "A user object.",
            "type": "object",
            "properties": { "name": { "type": "string" } }
        }

    Args:
        schema: The JSON schema (or a part of it) to transform.

    Returns:
        The flattened schema.
    """
    if isinstance(schema, dict):
        # Recurse first to handle nested cases
        processed_schema = {key: flatten_single_all_of(value) for key, value in schema.items()}

        # Check for a single-element "allOf"
        if (
            "allOf" in processed_schema
            and isinstance(processed_schema["allOf"], list)
            and len(processed_schema["allOf"]) == 1
        ):
            all_of_content = processed_schema.pop("allOf")[0]
            # Merge the content of allOf with the parent, parent's keys take precedence
            merged = {**all_of_content, **processed_schema}
            return merged

        return processed_schema

    if isinstance(schema, list):
        return [flatten_single_all_of(item) for item in schema]

    return schema

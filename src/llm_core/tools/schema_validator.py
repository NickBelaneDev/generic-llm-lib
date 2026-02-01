from typing import Any, Dict, Set

from llm_core.exceptions.exceptions import ToolValidationError
from llm_core.logger import get_logger

logger = get_logger(__name__)


class SchemaValidator:
    """
    Helper class for validating and sanitizing JSON schemas for LLM tools.
    """

    @staticmethod
    def assert_no_recursive_refs(schema: Dict[str, Any]) -> None:
        """
        Checks if the schema contains recursive references by traversing the graph.
        Raises ToolValidationError if a cycle is detected.

        Args:
            schema: The JSON schema to check.

        Raises:
            ToolValidationError: If a recursive reference is found.
        """
        defs = schema.get("$defs", {}) or schema.get("definitions", {})

        def check(node: Any, path: Set[str]) -> None:
            if isinstance(node, dict):
                if "$ref" in node:
                    ref = node["$ref"]
                    if ref in path:
                        msg = (
                            f"Recursive structure detected: {ref}. "
                            "Recursive structures are not allowed in tool inputs. "
                            "Use parent_id, lists, or a workflow loop instead."
                        )
                        logger.error(msg)
                        raise ToolValidationError(msg)

                    # If it's a local ref, follow it
                    if ref.startswith("#"):
                        # Extract definition name
                        # e.g. #/$defs/MyModel
                        parts = ref.split("/")
                        if len(parts) >= 3:
                            def_name = parts[-1]
                            if def_name in defs:
                                check(defs[def_name], path | {ref})
                    return

                for v in node.values():
                    check(v, path)
            elif isinstance(node, list):
                for item in node:
                    check(item, path)

        check(schema, set())

    @staticmethod
    def sanitize_schema(schema: Any) -> Any:
        """
        Cleans up the schema for better compatibility with LLM providers.
        Removes $defs, $schema, $id, title.
        Simplifies Optional fields (anyOf with null).
        Enforces additionalProperties: false for objects.

        Args:
            schema: The JSON schema to sanitize.

        Returns:
            The sanitized schema.
        """
        if not isinstance(schema, dict):
            return schema

        new_schema = schema.copy()

        # 1. Remove metadata keys
        for key in ["$defs", "$schema", "$id", "title", "definitions"]:
            new_schema.pop(key, None)

        # 2. Handle anyOf with null (Optional fields)
        if "anyOf" in new_schema:
            any_of = new_schema["anyOf"]
            # Check if it's a simple Optional (one type + null)
            non_null = [x for x in any_of if x.get("type") != "null"]

            if len(non_null) == 1:
                # Simplify to the single type
                simplified = non_null[0]

                if isinstance(simplified, dict):
                    # Merge simplified into new_schema
                    # We prefer the description from the parent (new_schema) if present
                    merged = simplified.copy()
                    if "description" in new_schema:
                        merged["description"] = new_schema["description"]

                    # Recurse on the merged result
                    return SchemaValidator.sanitize_schema(merged)

        # 3. Enforce additionalProperties: false for objects
        if new_schema.get("type") == "object":
            if "additionalProperties" not in new_schema:
                new_schema["additionalProperties"] = False

        # Recurse on children
        for key, value in new_schema.items():
            if isinstance(value, dict):
                new_schema[key] = SchemaValidator.sanitize_schema(value)
            elif isinstance(value, list):
                new_schema[key] = [
                    SchemaValidator.sanitize_schema(item) if isinstance(item, dict) else item
                    for item in value
                ]

        return new_schema

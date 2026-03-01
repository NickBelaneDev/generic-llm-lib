"""Schema validation helpers for LLM tool parameter schemas."""

from typing import Any, Dict, List, Optional, Set

from ...exceptions import ToolValidationError
from ...logger import get_logger

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
        SchemaValidator._check_recursion(schema, set(), defs)

    @staticmethod
    def _check_recursion(node: Any, path: Set[str], defs: Dict[str, Any]) -> None:
        if isinstance(node, dict):
            SchemaValidator._check_dict_recursion(node, path, defs)
        elif isinstance(node, list):
            SchemaValidator._check_list_recursion(node, path, defs)

    @staticmethod
    def _check_dict_recursion(node: Dict[str, Any], path: Set[str], defs: Dict[str, Any]) -> None:
        if "$ref" in node:
            SchemaValidator._validate_ref(node["$ref"], path, defs)
            return

        for v in node.values():
            SchemaValidator._check_recursion(v, path, defs)

    @staticmethod
    def _check_list_recursion(node: List[Any], path: Set[str], defs: Dict[str, Any]) -> None:
        for item in node:
            SchemaValidator._check_recursion(item, path, defs)

    @staticmethod
    def _validate_ref(ref: str, path: Set[str], defs: Dict[str, Any]) -> None:
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
            SchemaValidator._follow_local_ref(ref, path, defs)

    @staticmethod
    def _follow_local_ref(ref: str, path: Set[str], defs: Dict[str, Any]) -> None:
        # Extract definition name
        # e.g. #/$defs/MyModel
        parts = ref.split("/")
        if len(parts) >= 3:
            def_name = parts[-1]
            if def_name in defs:
                SchemaValidator._check_recursion(defs[def_name], path | {ref}, defs)

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

        # 1. Try to simplify structure (e.g. anyOf -> single type)
        simplified = SchemaValidator._simplify_any_of(schema)
        if simplified:
            return SchemaValidator.sanitize_schema(simplified)

        new_schema = schema.copy()

        # 2. Remove metadata keys
        SchemaValidator._remove_metadata(new_schema)

        # 3. Enforce constraints
        SchemaValidator._enforce_object_constraints(new_schema)

        # 4. Recurse on children
        return SchemaValidator._sanitize_children(new_schema)

    @staticmethod
    def _simplify_any_of(schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Simplifies 'anyOf' structures if they represent an Optional field (type + null).
        Returns the simplified schema dict if simplification occurred, else None.
        """
        any_of = schema.get("anyOf")
        if not any_of:
            return None

        non_null = SchemaValidator._get_non_null_types(any_of)

        if len(non_null) == 1:
            return SchemaValidator._merge_schema_description(non_null[0], schema)
        return None

    @staticmethod
    def _get_non_null_types(any_of: List[Any]) -> List[Dict[str, Any]]:
        return [x for x in any_of if isinstance(x, dict) and x.get("type") != "null"]

    @staticmethod
    def _merge_schema_description(target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
        """Merges description from source to target."""
        merged = target.copy()
        if "description" in source:
            merged["description"] = source["description"]
        return merged

    @staticmethod
    def _remove_metadata(schema: Dict[str, Any]) -> None:
        """Removes metadata keys from the schema in-place."""
        for key in ["$defs", "$schema", "$id", "title", "definitions"]:
            schema.pop(key, None)

    @staticmethod
    def _enforce_object_constraints(schema: Dict[str, Any]) -> None:
        """Enforces additionalProperties: false for objects in-place."""
        if schema.get("type") == "object":
            if "additionalProperties" not in schema:
                schema["additionalProperties"] = False

    @staticmethod
    def _sanitize_children(schema: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively sanitizes children of the schema."""
        for key, value in schema.items():
            schema[key] = SchemaValidator._sanitize_value(value)
        return schema

    @staticmethod
    def _sanitize_value(value: Any) -> Any:
        if isinstance(value, dict):
            return SchemaValidator.sanitize_schema(value)
        if isinstance(value, list):
            return SchemaValidator._sanitize_list(value)
        return value

    @staticmethod
    def _sanitize_list(data: List[Any]) -> List[Any]:
        """Sanitizes a list of items."""
        return [
            SchemaValidator.sanitize_schema(item) if isinstance(item, dict) else item
            for item in data
        ]

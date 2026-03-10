"""
Factory for creating ToolDefinition objects from Python callables.

This module provides the `ToolFactory` class, which uses reflection and Pydantic
to automatically generate tool schemas and metadata from functions, ensuring
they are compatible with LLM tool calling interfaces.
"""

from typing import Callable, Dict, Any, Optional, cast

import jsonref  # type: ignore
from pydantic import create_model

from ..models import ToolDefinition
from ..schema import ToolParameterFactory, preserve_ref_siblings, flatten_single_all_of
from ...exceptions import ToolValidationError
from ..schema import SchemaValidator
from ...logger import get_logger
import inspect

logger = get_logger(__name__)

class ToolFactory:
    """Creates tools from Callables, schemas and so on."""
    def generate_tool_definition(
        self, func: Callable, name: Optional[str] = None, description: Optional[str] = None
    ) -> ToolDefinition:
        """Generate a ToolDefinition from a callable function.

        Args:
            func: The function to generate a definition for.
            name: Optional name override for the tool.
            description: Optional description override for the tool.

        Returns:
            A ToolDefinition object containing the tool's metadata and schema.

        Raises:
            ToolValidationError: If the function is missing a docstring or parameter descriptions.
        """

        tool_name = name or func.__name__
        try:
            if description is None:
                description = self._get_docstring_from_func(func, tool_name)
        except ToolValidationError:
            raise

        signature = inspect.signature(func)
        fields = self._build_fields(signature, tool_name)

        # We need to cast fields to Any to satisfy mypy's strictness on kwargs unpacking for create_model
        # create_model expects **field_definitions: Any
        dynamic_params_model = create_model(f"{tool_name}Params", **cast(Dict[str, Any], fields))
        raw_schema = dynamic_params_model.model_json_schema()

        # 1. Preserve $ref siblings before resolving refs
        transformed_schema = preserve_ref_siblings(raw_schema)

        # 2. Check for recursion
        SchemaValidator.assert_no_recursive_refs(transformed_schema)

        # 3. Resolve refs using jsonref
        # proxies=False ensures we get a plain dict back, not JsonRef objects
        parameters_schema = jsonref.replace_refs(transformed_schema, proxies=False)

        # 4. Flatten single-element allOf arrays (restore simple structure)
        parameters_schema = flatten_single_all_of(parameters_schema)

        # 5. Sanitize schema (remove $defs, title, etc.)
        parameters_schema = SchemaValidator.sanitize_schema(parameters_schema)

        return ToolDefinition(
            name=tool_name,
            description=description,
            func=func,
            parameters=parameters_schema,
            args_model=dynamic_params_model,
        )

    @staticmethod
    def _get_docstring_from_func(func: Callable, tool_name: str) -> str:
        doc = inspect.getdoc(func)
        if not doc:
            msg = f"Tool '{tool_name}' missing docstring. LLMs need a description of what the tool does."
            logger.error(msg)
            raise ToolValidationError(msg)
        return doc

    @staticmethod
    def _build_fields(signature: inspect.Signature, tool_name: str) -> Dict[str, Any]:

        fields: Dict[str, Any] = {}
        for param_name, param in signature.parameters.items():
            if param_name == "self":
                continue
            ft = ToolParameterFactory.build_field_tuple(param_name=param_name, param=param, tool_name=tool_name)
            fields[param_name] = (ft.annotation, ft.field)
        return fields

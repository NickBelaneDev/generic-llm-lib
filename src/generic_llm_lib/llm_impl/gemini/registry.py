"""Adapt generic tool definitions into Gemini-compatible function declaration structures."""

from google.genai import types
from generic_llm_lib.llm_core import ToolRegistry, ToolDefinition
from typing import Callable, Dict, Any, Union, Optional


class GeminiToolRegistry(ToolRegistry):
    """
    A specialized ToolRegistry for Google Gemini models.

    This class extends the base ToolRegistry to provide Gemini-specific
    tool object generation, which is required for integrating tools
    with the Google Generative AI client.
    """

    def __init__(self) -> None:
        """Initialize the GeminiToolRegistry."""
        super().__init__()

    def register(
        self,
        name_or_tool: Union[str, ToolDefinition, Callable],
        description: Optional[str] = None,
        func: Optional[Callable] = None,
        parameters: Optional[Any] = None,
    ) -> None:
        """
        Registers a tool with the GeminiToolRegistry.

        This method supports two ways of registration:
        1. Passing a `ToolDefinition` object as the first argument.
        2. Passing individual arguments: `name`, `description`, `func`, and `parameters`.
        3. Passing a Callable as the first argument to automatically generate the definition.

        Args:
            name_or_tool: Either a `ToolDefinition` object, the name of the tool (str), or a Callable.
            description: A description of what the tool does. Required if `name_or_tool` is a string.
            func: The callable function implementing the tool's logic. Required if `name_or_tool` is a string.
            parameters: A dictionary or types.Schema defining the tool's input parameters. Required if `name_or_tool` is a string.
        """
        super().register(name_or_tool, description, func, parameters)

    @property
    def tool_object(self) -> types.Tool | None:
        """
        Generates a `types.Tool` object suitable for the Gemini API
        based on the registered tools.

        Returns:
            A `types.Tool` object containing all registered function declarations,
            or None if no tools are registered.
        """
        if not self.tools:
            return None

        declarations = []
        for tool in self.tools.values():

            if tool.parameters:
                # Gemini does not support 'additionalProperties' in the schema
                clean_params = self._strip_additional_properties(tool.parameters)
                declarations.append(
                    types.FunctionDeclaration(name=tool.name, description=tool.description, parameters=clean_params)
                )
            else:
                declarations.append(types.FunctionDeclaration(name=tool.name, description=tool.description))

        return types.Tool(function_declarations=declarations)

    def _strip_additional_properties(self, schema: Any) -> Any:
        """Recursively removes 'additionalProperties' from the schema."""
        if not isinstance(schema, dict):
            return schema

        new_schema = schema.copy()
        new_schema.pop("additionalProperties", None)

        for key, value in new_schema.items():
            if isinstance(value, dict):
                new_schema[key] = self._strip_additional_properties(value)
            elif isinstance(value, list):
                new_schema[key] = [
                    self._strip_additional_properties(item) if isinstance(item, dict) else item for item in value
                ]
        return new_schema

    @property
    def implementations(self) -> Dict[str, Callable]:
        """
        Returns a dictionary mapping tool names to their callable implementations.

        Returns:
            A dictionary where keys are tool names and values are the corresponding
            callable functions.
        """
        return {name: tool.func for name, tool in self.tools.items()}

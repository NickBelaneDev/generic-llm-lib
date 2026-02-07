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
            # Ensure parameters are compatible with types.FunctionDeclaration
            # If it's a dict, the library usually handles it, but type checkers might complain.
            # If it's already a types.Schema (or similar), it's passed through.

            if tool.parameters:
                declarations.append(
                    types.FunctionDeclaration(name=tool.name, description=tool.description, parameters=tool.parameters)
                )
            else:
                declarations.append(types.FunctionDeclaration(name=tool.name, description=tool.description))

        return types.Tool(function_declarations=declarations)

    @property
    def implementations(self) -> Dict[str, Callable]:
        """
        Returns a dictionary mapping tool names to their callable implementations.

        Returns:
            A dictionary where keys are tool names and values are the corresponding
            callable functions.
        """
        return {name: tool.func for name, tool in self.tools.items()}

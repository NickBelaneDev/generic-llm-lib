from google.genai import types
from llm_core import ToolRegistry
from typing import Callable, Dict, Any, Union

class GeminiToolRegistry(ToolRegistry):
    """
    A specialized ToolRegistry for Google Gemini models.

    This class extends the base ToolRegistry to provide Gemini-specific
    tool object generation, which is required for integrating tools
    with the Google Generative AI client.
    """
    def register(self, name: str, description: str, func: Callable, parameters: Union[Dict[str, Any], types.Schema, Any]):
        """
        Registers a tool with the GeminiToolRegistry.

        Args:
            name: The name of the tool.
            description: A description of what the tool does.
            func: The callable function implementing the tool's logic.
            parameters: A dictionary or types.Schema defining the tool's input parameters.
        """
        super().register(name, description, func, parameters)

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
            declarations.append(types.FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters
            ))

        return types.Tool(function_declarations=declarations)

    @property
    def implementations(self):
        """
        Returns a dictionary mapping tool names to their callable implementations.

        Returns:
            A dictionary where keys are tool names and values are the corresponding
            callable functions.
        """
        return {name: tool.func for name, tool in self.tools.items()}

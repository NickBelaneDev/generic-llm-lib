import inspect
from llm_core import ToolRegistry, ToolDefinition
from llm_core.exceptions import ToolRegistrationError
from typing import Callable, Dict, Any, Union, Optional, List

TYPE_MAPPING = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object"
}

class OpenAIToolRegistry(ToolRegistry):
    """
    A specialized ToolRegistry for OpenAI models.

    This class extends the base ToolRegistry to provide OpenAI-specific
    tool object generation, which is required for integrating tools
    with the OpenAI client.
    """
    def __init__(self):
        self.tools: Dict[str, ToolDefinition] = {}

    def register(self,
                 name_or_tool: Union[str, ToolDefinition, Callable],
                 description: Optional[str] = None,
                 func: Optional[Callable] = None,
                 parameters: Optional[Any] = None):
        """
        Registers a tool with the OpenAIToolRegistry.

        This method supports two ways of registration:
        1. Passing a `ToolDefinition` object as the first argument.
        2. Passing individual arguments: `name`, `description`, `func`, and `parameters`.
        3. Passing a Callable as the first argument to automatically generate the definition.

        Args:
            name_or_tool: Either a `ToolDefinition` object, the name of the tool (str), or a Callable.
            description: A description of what the tool does. Required if `name_or_tool` is a string.
            func: The callable function implementing the tool's logic. Required if `name_or_tool` is a string.
            parameters: A dictionary defining the tool's input parameters (JSON Schema). Required if `name_or_tool` is a string.
        """
        super().register(name_or_tool, description, func, parameters)

    @property
    def tool_object(self) -> List[Dict[str, Any]] | None:
        """
        Generates a list of tool definitions suitable for the OpenAI API
        based on the registered tools.

        Returns:
            A list of tool dictionaries, or None if no tools are registered.
        """
        if not self.tools:
            return None

        tools_list = []
        for tool in self.tools.values():
            function_def = {
                "name": tool.name,
                "description": tool.description,
            }
            
            if tool.parameters:
                function_def["parameters"] = tool.parameters
            else:
                 # OpenAI requires parameters to be present even if empty for some models/versions, 
                 # but usually it's a JSON schema. If no params, we can provide an empty object schema.
                 function_def["parameters"] = {
                     "type": "object",
                     "properties": {},
                 }

            tools_list.append({
                "type": "function",
                "function": function_def
            })

        return tools_list

    @property
    def implementations(self):
        """
        Returns a dictionary mapping tool names to their callable implementations.

        Returns:
            A dictionary where keys are tool names and values are the corresponding
            callable functions.
        """
        return {name: tool.func for name, tool in self.tools.items()}

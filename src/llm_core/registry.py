from abc import abstractmethod, ABC
from typing import Callable, Dict, Any, Union, Optional
from .types import ToolDefinition
import inspect

TYPE_MAPPING = {
    str: "STRING",
    int: "INTEGER",
    float: "NUMBER",
    bool: "BOOLEAN",
    list: "ARRAY",
    dict: "OBJECT"
}

class ToolRegistry(ABC):
    """
    A central registry to manage and access all available LLM tools.

    This class holds the function declarations to be sent to the LLM and
    maps function names to their actual Python implementations.
    """

    def __init__(self):
        self.tools: Dict[str, ToolDefinition] = {}

    def register(self,
                 name_or_tool: Union[str, ToolDefinition],
                 description: Optional[str] = None,
                 func: Optional[Callable] = None,
                 parameters: Optional[Any] = None):
        """
        Register a new tool for the LLM.

        This method allows registering a tool either by providing a `ToolDefinition` object
        directly or by providing the individual components (name, description, function, parameters).

        Args:
            name_or_tool: Either a `ToolDefinition` object or the name of the tool (str).
            description: A brief description of what the tool does. Required if `name_or_tool` is a string.
            func: The callable function implementing the tool's logic. Required if `name_or_tool` is a string.
            parameters: A schema defining the tool's input parameters. Required if `name_or_tool` is a string.

        Raises:
            ValueError: If individual arguments are provided but some are missing.
        """
        if isinstance(name_or_tool, ToolDefinition):
            tool = name_or_tool
        else:
            if description is None or func is None or parameters is None:
                raise ValueError("If passing name as string; description, func, and parameters are required.")
            tool = ToolDefinition(name=name_or_tool, description=description, func=func, parameters=parameters)

        self.tools[tool.name] = tool

    def tool(self, func: Callable) -> Callable:
        """A decorator to turn a function into a Gemini tool"""
        name = func.__name__
        description = inspect.getdoc(func) or "No description provided"
        signature = inspect.signature(func)

        properties = {}
        required = []

        for param_name, param in signature.parameters.items():
            if param_name == "self":
                continue

            python_type = param.annotation

            if python_type == inspect.Parameter.empty:
                schema_type = "STRING"
            else:
                schema_type = TYPE_MAPPING.get(python_type, "STRING")

            properties[param_name] = {"type": schema_type}

            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        parameters = {
            "type": "OBJECT",
            "properties": properties,
        }
        if required:
            parameters["required"] = required

        self.register(
            name_or_tool=name,
            description=description,
            func=func,
            parameters=parameters
        )

        return func

    @property
    @abstractmethod
    def tool_object(self):
        """Constructs the final Tool object specific to the LLM provider."""
        pass

    @property
    def implementations(self) -> Dict[str, Callable]:
        """Returns a dictionary mapping function names to their callables."""
        return {name: tool.func for name, tool in self.tools.items()}
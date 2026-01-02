from abc import abstractmethod, ABC
from typing import Callable, Dict, Any, Union
from .types import ToolDefinition

 # Allow Any for provider-specific types

class ToolRegistry(ABC):
    """
    A central registry to manage and access all available LLM tools.

    This class holds the function declarations to be sent to the LLM and
    maps function names to their actual Python implementations.
    """

    def __init__(self):
        self.tools: Dict[str, ToolDefinition] = {}

    def register(self,
                 name: str,
                 description: str,
                 func: Callable,
                 parameters: Union[Dict[str, Any], Any]
                 ):
        """Register new tools for the llm"""
        tool_def = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            func=func
        )
        self.tools[name] = tool_def

    @property
    @abstractmethod
    def tool_object(self):
        """Constructs the final Tool object specific to the LLM provider."""
        pass

    @property
    def implementations(self) -> Dict[str, Callable]:
        """Returns a dictionary mapping function names to their callables."""
        return {name: tool.func for name, tool in self.tools.items()}

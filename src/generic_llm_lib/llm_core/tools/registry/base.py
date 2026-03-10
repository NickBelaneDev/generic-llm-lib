"""
Tool registry abstraction and helper utilities.

This module defines the base `ToolRegistry` class, which serves as a central
repository for managing tools that can be invoked by an LLM. It handles
registration, unregistration, and automatic generation of tool definitions
from Python callables using reflection and Pydantic.
"""

from abc import abstractmethod, ABC
from typing import Callable, Dict, Any, Union, Optional


from ..models import ToolDefinition
from .tool_definition_factory import ToolFactory
from ...exceptions import ToolNotFoundError, ToolRegistrationError

from ...logger import get_logger

logger = get_logger(__name__)


# class ToolRegistry(ABC, Generic[ProviderResT]):
class ToolRegistry(ABC):
    """
    A central registry to manage and access all available LLM tools.

    This class holds the function declarations to be sent to the LLM and
    maps function names to their actual Python implementations.
    """

    def __init__(self) -> None:
        """Initialize the ToolRegistry."""
        self.tools: Dict[str, ToolDefinition] = {}
        self.tool_factory: ToolFactory = ToolFactory()

    def _register_tool_definition(self, tool: ToolDefinition) -> None:
        """Registers a ToolDefinition object directly."""
        if tool.name in self.tools:
            msg = f"Tool '{tool.name}' is already registered."
            logger.error(msg)
            raise ToolRegistrationError(msg)

        self.tools[tool.name] = tool
        logger.info(f"Successfully registered tool: '{tool.name}'")

    def _register_callable(self, func: Callable, name: Optional[str] = None, description: Optional[str] = None) -> None:
        """Registers a callable function by generating a ToolDefinition."""
        # tool = self._generate_tool_definition(func, name=name, description=description)
        tool = self.tool_factory.generate_tool_definition(func, name=name, description=description)
        self._register_tool_definition(tool)

    def _register_manual(
        self, name: str, func: Callable, parameters: Optional[Any] = None, description: Optional[str] = None
    ) -> None:
        """Registers a tool with manually provided components."""
        if parameters is None:
            # If no parameters provided, try to generate definition from function
            self._register_callable(func, name=name, description=description)
        else:
            if description is None:
                raise ToolRegistrationError("If passing name and parameters, description is required.")
            tool = ToolDefinition(name=name, description=description, func=func, parameters=parameters)
            self._register_tool_definition(tool)

    def register(
        self,
        name_or_tool: Union[str, ToolDefinition, Callable],
        description: Optional[str] = None,
        func: Optional[Callable] = None,
        parameters: Optional[Any] = None,
    ) -> None:
        """
        Register a new tool for the LLM.

        This method allows registering a tool either by providing a `ToolDefinition` object
        directly, by providing the individual components (name, description, function, parameters),
        or by providing a function (Callable) to automatically generate the definition.

        Args:
            name_or_tool: Either a `ToolDefinition` object, the name of the tool (str), or a Callable.
            description: A brief description of what the tool does. Required if `name_or_tool` is a string and parameters are provided.
            func: The callable function implementing the tool's logic. Required if `name_or_tool` is a string.
            parameters: A schema defining the tool's input parameters. If None, it will be inferred from `func`.

        Raises:
            ToolRegistrationError: If individual arguments are provided but some are missing or if the tool already exists.
        """
        if isinstance(name_or_tool, ToolDefinition):
            self._register_tool_definition(name_or_tool)

        elif callable(name_or_tool):
            self._register_callable(name_or_tool, description=description)

        elif isinstance(name_or_tool, str):
            if func is None:
                raise ToolRegistrationError("If passing name as string, func is required.")
            self._register_manual(name_or_tool, func, parameters, description)

        else:
            raise ToolRegistrationError(f"Invalid type for name_or_tool: {type(name_or_tool)}")

    def unregister(self, tool_name: str) -> None:
        """Unregister a tool from the registry.

        Args:
            tool_name: The name of the tool to remove.

        Raises:
            ToolNotFoundError: If the tool does not exist in the registry.
        """
        if tool_name in self.tools:
            del self.tools[tool_name]
            logger.info(f"Successfully unregistered tool: '{tool_name}'")
        else:
            raise ToolNotFoundError(f"Tool '{tool_name}' not found in the registry.")

    def tool(self, func: Callable) -> Callable:
        """A decorator to turn a function into an LLM tool.

        Args:
            func: The function to decorate.

        Returns:
            The original function, after registering it as a tool.
        """
        self.register(func)
        return func

    @property
    @abstractmethod
    def tool_object(self) -> Any:
        """Constructs the final Tool object specific to the LLM provider.

        Returns:
            The provider-specific tool representation.
        """
        pass

    @property
    def implementations(self) -> Dict[str, Callable]:
        """Returns a dictionary mapping function names to their callables.

        Returns:
            A dictionary where keys are tool names and values are the functions.
        """
        return {name: tool.func for name, tool in self.tools.items()}

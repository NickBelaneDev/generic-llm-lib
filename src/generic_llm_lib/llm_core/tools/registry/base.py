"""Tool registry abstraction and helper utilities."""

from abc import abstractmethod, ABC
from typing import Callable, Dict, Any, Union, Optional, cast

import jsonref  # type: ignore
from pydantic import create_model

from ..models import ToolDefinition
from ..schema import ToolParameterFactory
from ...exceptions import ToolNotFoundError, ToolRegistrationError, ToolValidationError
from ..schema import SchemaValidator
from ...logger import get_logger
import inspect

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
            tool = name_or_tool
        elif callable(name_or_tool):
            tool = self._generate_tool_definition(name_or_tool, description=description)
        else:
            # name_or_tool is a string (name)
            if func is None:
                raise ToolRegistrationError("If passing name as string, func is required.")

            if parameters is None:
                tool = self._generate_tool_definition(func, name=name_or_tool, description=description)
            else:
                if description is None:
                    raise ToolRegistrationError("If passing name and parameters, description is required.")
                tool = ToolDefinition(name=name_or_tool, description=description, func=func, parameters=parameters)

        if tool.name in self.tools:
            msg = f"Tool '{tool.name}' is already registered."
            logger.error(msg)
            raise ToolRegistrationError(msg)

        self.tools[tool.name] = tool
        logger.info(f"Successfully registered tool: '{tool.name}'")

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

    def _generate_tool_definition(
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
        # 1. Check for recursion
        SchemaValidator.assert_no_recursive_refs(raw_schema)

        # 2. Resolve refs using jsonref
        # proxies=False ensures we get a plain dict back, not JsonRef objects
        parameters_schema = jsonref.replace_refs(raw_schema, proxies=False)

        # 3. Sanitize schema (remove $defs, title, etc.)
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

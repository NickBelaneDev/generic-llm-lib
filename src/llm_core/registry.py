import copy
from abc import abstractmethod, ABC
from typing import Callable, Dict, Any, Union, Optional, get_origin, Annotated, get_args

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic import create_model # NOTE: For context: it was pydantic.v1 before.

from .types import ToolDefinition
from .exceptions import ToolRegistrationError, ToolValidationError
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
                 name_or_tool: Union[str, ToolDefinition, Callable],
                 description: Optional[str] = None,
                 func: Optional[Callable] = None,
                 parameters: Optional[Any] = None):
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
             raise ToolRegistrationError(f"Tool '{tool.name}' is already registered.")

        self.tools[tool.name] = tool

    def _generate_tool_definition(self, func: Callable, name: Optional[str] = None, description: Optional[str] = None) -> ToolDefinition:
        tool_name = name or func.__name__

        if description is None:
            doc = inspect.getdoc(func)
            if not doc:
                raise ToolValidationError(f"Tool '{tool_name}' missing docstring. LLMs need a description of what the tool does.")
            description = doc

        signature = inspect.signature(func)
        fields = {}

        for param_name, param in signature.parameters.items():
            if param_name == "self":
                continue

            annotation = param.annotation
            default = param.default

            has_description = False

            # Every Tools parameter needs 'Annotated[<class>, Field(description='...')] = ...' as its annotation
            if get_origin(annotation) is Annotated:
                for metadata in get_args(annotation):
                    if isinstance(metadata, FieldInfo) and metadata.description:
                        has_description = True
                        break

            if not has_description:
                raise ToolValidationError(
                    f"Parameter '{param_name}' in tool '{tool_name}' is missing a description.\n"
                    f"Usage: {param_name}: Annotated[Type, Field(description='...')] = ..."
                )

            pydantic_default = default if default != inspect.Parameter.empty else ...
            fields[param_name] = (annotation, pydantic_default)

        DynamicParamsModel: BaseModel = create_model(f"{tool_name}Params", **fields)

        raw_schema = DynamicParamsModel.model_json_schema()
        parameters_schema = self._resolve_schema_refs(raw_schema)

        parameters_schema.pop("title", None)

        return ToolDefinition(
            name=tool_name, 
            description=description, 
            func=func, 
            parameters=parameters_schema,
            args_model=DynamicParamsModel
        )

    def tool(self, func: Callable) -> Callable:
        """A decorator to turn a function into a Gemini tool"""
        self.register(func)
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

    def _resolve_schema_refs(self, schema: dict, defs: dict = None) -> dict:
        schema = copy.deepcopy(schema)

        if defs is None:
            defs = schema.pop("$defs", {})

        if "$ref" in schema:
            ref_key = schema["$ref"].split("/")[-1]
            if ref_key in defs:
                ref_content = defs[ref_key]
                resolved_content = self._resolve_schema_refs(ref_content, defs)

                for k, v in resolved_content.items():
                    if k not in schema or k == "$ref":
                        schema[k] = v

                del schema["$ref"]
                return schema

        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                schema["properties"][prop_name] = self._resolve_schema_refs(prop_schema, defs)

        if "items" in schema:
            schema["items"] = self._resolve_schema_refs(schema["items"], defs)

        for key in ["anyOf", "allOf", "oneOf"]:
            if key in schema:
                schema[key] = [self._resolve_schema_refs(item, defs) for item in schema[key]]

        return schema
from abc import abstractmethod, ABC
from typing import Callable, Dict, Any, Union, Optional, get_origin, Annotated, get_args

import jsonref
from pydantic.fields import FieldInfo
from pydantic import create_model

from llm_core.tools.models import ToolDefinition
from llm_core.exceptions.exceptions import ToolRegistrationError, ToolValidationError
import inspect

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


    def _generate_tool_definition(
            self, func: Callable,
            name: Optional[str] = None,
            description: Optional[str] = None
    ) -> ToolDefinition:

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

        dynamic_params_model = create_model(f"{tool_name}Params", **fields)

        raw_schema = dynamic_params_model.model_json_schema()
        
        # 1. Check for recursion
        self._assert_no_recursive_refs(raw_schema)
        
        # 2. Resolve refs using jsonref
        # proxies=False ensures we get a plain dict back, not JsonRef objects
        parameters_schema = jsonref.replace_refs(raw_schema, proxies=False)
        
        # 3. Sanitize schema (remove $defs, title, etc.)
        parameters_schema = self._sanitize_schema(parameters_schema)

        return ToolDefinition(
            name=tool_name, 
            description=description, 
            func=func, 
            parameters=parameters_schema,
            args_model=dynamic_params_model
        )


    def tool(self, func: Callable) -> Callable:
        """A decorator to turn a function into an LLM tool."""
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


    @staticmethod
    def _assert_no_recursive_refs(schema: dict):
        """
        Checks if the schema contains recursive references by traversing the graph.
        Raises ToolValidationError if a cycle is detected.
        """
        defs = schema.get("$defs", {}) or schema.get("definitions", {})
        
        def check(node, path):
            if isinstance(node, dict):
                if "$ref" in node:
                    ref = node["$ref"]
                    if ref in path:
                        raise ToolValidationError(
                            f"Recursive structure detected: {ref}. "
                            "Recursive structures are not allowed in tool inputs. "
                            "Use parent_id, lists, or a workflow loop instead."
                        )
                    
                    # If it's a local ref, follow it
                    if ref.startswith("#"):
                        # Extract definition name
                        # e.g. #/$defs/MyModel
                        parts = ref.split("/")
                        if len(parts) >= 3:
                            def_name = parts[-1]
                            if def_name in defs:
                                check(defs[def_name], path | {ref})
                    return

                for v in node.values():
                    check(v, path)
            elif isinstance(node, list):
                for item in node:
                    check(item, path)

        check(schema, set())

    def _sanitize_schema(self, schema: dict) -> dict:
        """
        Cleans up the schema for better compatibility with LLM providers.
        Removes $defs, $schema, $id, title.
        Simplifies Optional fields (anyOf with null).
        Enforces additionalProperties: false for objects.
        """
        if not isinstance(schema, dict):
            return schema
            
        new_schema = schema.copy()
        
        # 1. Remove metadata keys
        for key in ["$defs", "$schema", "$id", "title", "definitions"]:
            new_schema.pop(key, None)
            
        # 2. Handle anyOf with null (Optional fields)
        if "anyOf" in new_schema:
            any_of = new_schema["anyOf"]
            # Check if it's a simple Optional (one type + null)
            non_null = [x for x in any_of if x.get("type") != "null"]
            
            if len(non_null) == 1:
                # Simplify to the single type
                simplified = non_null[0]
                
                if isinstance(simplified, dict):
                    # Merge simplified into new_schema
                    # We prefer the description from the parent (new_schema) if present
                    merged = simplified.copy()
                    if "description" in new_schema:
                        merged["description"] = new_schema["description"]
                    
                    # Recurse on the merged result
                    return self._sanitize_schema(merged)
        
        # 3. Enforce additionalProperties: false for objects
        if new_schema.get("type") == "object":
            if "additionalProperties" not in new_schema:
                new_schema["additionalProperties"] = False
                
        # Recurse on children
        for key, value in new_schema.items():
            if isinstance(value, dict):
                new_schema[key] = self._sanitize_schema(value)
            elif isinstance(value, list):
                new_schema[key] = [self._sanitize_schema(item) if isinstance(item, dict) else item for item in value]
                
        return new_schema

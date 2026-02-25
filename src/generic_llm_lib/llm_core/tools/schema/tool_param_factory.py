import inspect
from typing import Any, get_origin, Annotated, get_args
from pydantic import Field, BaseModel, ConfigDict
from pydantic.fields import FieldInfo
from generic_llm_lib.llm_core.exceptions.exceptions import ToolValidationError

from ...logger import get_logger

logger = get_logger(__name__)


class FieldTuple(BaseModel):
    """Ensures, that the dynamic model field definition is correctly typed for Pydantic's create_model."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    annotation: Any
    field: FieldInfo


class ToolParameterFactory:
    """Capsules the extraction and validation of single function parameters for pydantic"""

    @classmethod
    def build_field_tuple(cls, param_name: str, param: inspect.Parameter, tool_name: str) -> FieldTuple:
        """Creates the tuple of (annotation, FieldInfo) for a single function parameter.

        Args:
            param_name: The name of the parameter.
            param: The inspect.Parameter object.
            tool_name: The name of the tool for error reporting.

        Returns:
            A FieldTuple containing the type annotation and Pydantic Field configuration.
        """

        annotation = param.annotation
        description = cls._extract_description(annotation=annotation, param_name=param_name, tool_name=tool_name)

        pydantic_default = param.default if param.default != inspect.Parameter.empty else ...

        return FieldTuple(annotation=annotation, field=Field(default=pydantic_default, description=description))

    @staticmethod
    def _extract_description(annotation: Any, param_name: str, tool_name: str) -> str:
        """Every Tools parameter needs 'Annotated[<class>, Field(description='...')] = ...' as its annotation.

        Args:
            annotation: The type annotation to inspect.
            param_name: The name of the parameter being checked.
            tool_name: The name of the tool for error reporting.

        Raises:
            ToolValidationError: If the parameter is missing a Pydantic Field description.
        """

        if get_origin(annotation) is Annotated:
            for metadata in get_args(annotation):
                if isinstance(metadata, FieldInfo) and metadata.description:
                    return metadata.description

        msg = (
            f"Parameter '{param_name}' in tool '{tool_name}' is missing a description.\n"
            f"Usage: {param_name}: Annotated[Type, Field(description='...')] = ..."
        )
        logger.error(msg)
        raise ToolValidationError(msg)

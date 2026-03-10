"""Tests for the ToolFactory class."""

import pytest
from typing import Optional, Annotated
from pydantic import BaseModel, Field

from generic_llm_lib.llm_core.tools.registry.tool_definition_factory import ToolFactory
from generic_llm_lib.llm_core.exceptions import ToolValidationError
from generic_llm_lib.llm_core.tools.models import ToolDefinition


class TestToolFactory:
    """Tests for the ToolFactory class."""

    def test_generate_tool_definition_simple(self):
        """Test generating a tool definition from a simple function using Annotated."""

        def simple_func(
            x: Annotated[int, Field(description="The first number.")],
            y: Annotated[int, Field(description="The second number.")],
        ) -> int:
            """Add two numbers."""
            return x + y

        factory = ToolFactory()
        tool_def = factory.generate_tool_definition(simple_func)

        assert isinstance(tool_def, ToolDefinition)
        assert tool_def.name == "simple_func"
        assert tool_def.description == "Add two numbers."
        assert tool_def.func == simple_func

        # Check parameters schema
        params = tool_def.parameters
        assert "properties" in params
        assert "x" in params["properties"]
        assert "y" in params["properties"]
        assert params["properties"]["x"]["type"] == "integer"
        assert params["properties"]["y"]["type"] == "integer"
        assert params["properties"]["x"]["description"] == "The first number."
        assert params["properties"]["y"]["description"] == "The second number."
        assert "required" in params
        assert "x" in params["required"]
        assert "y" in params["required"]

    def test_generate_tool_definition_with_defaults(self):
        """Test generating a tool definition with default values."""

        def func_with_defaults(
            name: Annotated[str, Field(description="The person's name.")],
            greeting: Annotated[str, Field(description="The greeting to use.")] = "Hello",
        ) -> str:
            """Greet a person."""
            return f"{greeting}, {name}!"

        factory = ToolFactory()
        tool_def = factory.generate_tool_definition(func_with_defaults)

        params = tool_def.parameters
        assert "properties" in params
        assert "name" in params["properties"]
        assert "greeting" in params["properties"]

        # 'name' is required, 'greeting' is not
        assert "required" in params
        assert "name" in params["required"]
        assert "greeting" not in params["required"]

    def test_generate_tool_definition_missing_docstring(self):
        """Test that ToolValidationError is raised when docstring is missing."""

        def no_docstring(x: Annotated[int, Field(description="Input value")]):
            pass

        factory = ToolFactory()
        with pytest.raises(ToolValidationError, match="missing docstring"):
            factory.generate_tool_definition(no_docstring)

    def test_generate_tool_definition_missing_annotation_description(self):
        """Test that ToolValidationError is raised when parameter description is missing in annotation."""

        def missing_annotation(x: int):
            """Docstring exists."""
            pass

        factory = ToolFactory()
        with pytest.raises(ToolValidationError, match="missing a description"):
            factory.generate_tool_definition(missing_annotation)

    def test_generate_tool_definition_override_name_description(self):
        """Test overriding name and description."""

        def my_func(x: Annotated[int, Field(description="Input value")]):
            """Original docstring."""
            pass

        factory = ToolFactory()
        tool_def = factory.generate_tool_definition(my_func, name="custom_name", description="Custom description")

        assert tool_def.name == "custom_name"
        assert tool_def.description == "Custom description"

    def test_generate_tool_definition_complex_types(self):
        """Test generating tool definition with complex types (Pydantic models)."""

        class User(BaseModel):
            name: str = Field(..., description="User name")
            age: int = Field(..., description="User age")

        # Even for Pydantic models, we need to annotate the parameter itself with a description
        def process_user(
            user: Annotated[User, Field(description="The user object.")],
            active: Annotated[bool, Field(description="Whether the user is active.")] = True,
        ) -> str:
            """Process a user."""
            return "processed"

        factory = ToolFactory()
        tool_def = factory.generate_tool_definition(process_user)

        params = tool_def.parameters
        assert "properties" in params
        assert "user" in params["properties"]
        assert params["properties"]["user"]["description"] == "The user object."

        # Check that the Pydantic model was correctly converted to JSON schema
        user_schema = params["properties"]["user"]

        # After resolving refs and sanitizing, the structure should be clean
        assert user_schema["type"] == "object"
        assert "properties" in user_schema
        assert "name" in user_schema["properties"]
        assert "age" in user_schema["properties"]

        # Check that definitions are resolved (no $ref)
        assert "$defs" not in params
        assert "definitions" not in params

    def test_generate_tool_definition_recursive_schema_error(self):
        """Test that recursive schemas raise an error."""

        class Node(BaseModel):
            value: int
            next: Optional["Node"] = None

        def process_node(node: Annotated[Node, Field(description="The node to process.")]):
            """Process a linked list node."""
            pass

        factory = ToolFactory()

        # Expecting an error due to recursion check
        with pytest.raises(Exception):
            factory.generate_tool_definition(process_node)

    def test_generate_tool_definition_method(self):
        """Test generating a tool definition from a class method (ignoring self)."""

        class MyClass:
            def my_method(self, x: Annotated[int, Field(description="Input value.")]) -> int:
                """Method docstring."""
                return x

        obj = MyClass()
        factory = ToolFactory()
        tool_def = factory.generate_tool_definition(obj.my_method)

        params = tool_def.parameters
        assert "properties" in params
        assert "x" in params["properties"]
        assert "self" not in params["properties"]

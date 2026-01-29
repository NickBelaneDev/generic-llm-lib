import pytest
from typing import Annotated, List, Dict
from pydantic import Field, BaseModel
from llm_core.registry import ToolRegistry
from llm_impl.gemini.registry import GeminiToolRegistry
from llm_impl.open_api.registry import OpenAIToolRegistry
from google.genai import types

# Create a concrete implementation for testing base class functionality
# Renamed to avoid PytestCollectionWarning
class ConcreteTestRegistry(ToolRegistry):
    @property
    def tool_object(self):
        return None

def test_registry_tool_decorator():
    registry = ConcreteTestRegistry()
    
    @registry.tool
    def my_tool(x: Annotated[int, Field(description="An integer")]) -> int:
        """My tool description."""
        return x * 2
        
    assert "my_tool" in registry.tools
    tool_def = registry.tools["my_tool"]
    assert tool_def.description == "My tool description."
    assert tool_def.func(2) == 4

def test_gemini_registry_tool_object():
    registry = GeminiToolRegistry()
    
    @registry.tool
    def my_tool(x: Annotated[int, Field(description="An integer")]) -> int:
        """My tool description."""
        return x * 2
        
    tool_obj = registry.tool_object
    assert isinstance(tool_obj, types.Tool)
    assert len(tool_obj.function_declarations) == 1
    decl = tool_obj.function_declarations[0]
    assert decl.name == "my_tool"
    assert decl.description == "My tool description."

def test_openai_registry_tool_object():
    registry = OpenAIToolRegistry()
    
    @registry.tool
    def my_tool(x: Annotated[int, Field(description="An integer")]) -> int:
        """My tool description."""
        return x * 2
        
    tool_obj = registry.tool_object
    assert isinstance(tool_obj, list)
    assert len(tool_obj) == 1
    
    tool_def = tool_obj[0]
    assert tool_def["type"] == "function"
    assert tool_def["function"]["name"] == "my_tool"
    assert tool_def["function"]["description"] == "My tool description."
    assert "parameters" in tool_def["function"]
    assert tool_def["function"]["parameters"]["type"] == "object"

def test_registry_missing_docstring():
    registry = ConcreteTestRegistry()
    with pytest.raises(ValueError, match="missing docstring"):
        @registry.tool
        def no_doc_tool(x: Annotated[int, Field(description="desc")]):
            pass

def test_registry_missing_param_description():
    registry = ConcreteTestRegistry()
    with pytest.raises(ValueError, match="missing a description"):
        @registry.tool
        def bad_param_tool(x: int):
            """Docstring."""
            pass

def test_nested_pydantic_models_schema_resolution():
    """
    Tests that nested Pydantic models are correctly resolved into a single schema
    without $ref definitions, as some LLM providers (like Gemini) require fully resolved schemas.
    """
    registry = ConcreteTestRegistry()

    class Address(BaseModel):
        street: str = Field(description="Street name")
        city: str = Field(description="City name")

    class User(BaseModel):
        name: str = Field(description="User's full name")
        age: int = Field(description="User's age")
        address: Address = Field(description="User's address")
        tags: List[str] = Field(description="User tags")

    @registry.tool
    def create_user(user: Annotated[User, Field(description="The user object to create")]) -> str:
        """Creates a new user in the system."""
        return f"Created user {user.name}"

    tool_def = registry.tools["create_user"]
    schema = tool_def.parameters

    # Verify schema structure
    assert schema["type"] == "object"
    assert "properties" in schema
    assert "user" in schema["properties"]
    
    user_schema = schema["properties"]["user"]
    assert user_schema["type"] == "object"
    
    # Check that Address is resolved inline (no $ref)
    assert "address" in user_schema["properties"]
    address_schema = user_schema["properties"]["address"]
    assert address_schema["type"] == "object"
    assert "street" in address_schema["properties"]
    assert "city" in address_schema["properties"]
    
    # Ensure no $defs or definitions remain at the top level
    assert "$defs" not in schema
    assert "definitions" not in schema
    
    # Ensure no $ref exists in the resolved schema
    import json
    schema_str = json.dumps(schema)
    assert "$ref" not in schema_str

def test_tool_without_parameters():
    """
    Tests registering a tool that takes no parameters.
    """
    registry = ConcreteTestRegistry()

    @registry.tool
    def get_current_time() -> str:
        """Returns the current server time."""
        return "12:00 PM"

    assert "get_current_time" in registry.tools
    tool_def = registry.tools["get_current_time"]
    
    assert tool_def.description == "Returns the current server time."
    assert tool_def.func() == "12:00 PM"
    
    # Verify schema is empty object or None-like structure for no params
    schema = tool_def.parameters
    assert schema["type"] == "object"
    assert schema["properties"] == {}

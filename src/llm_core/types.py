from typing import Optional, Any, Callable, Type
from pydantic import BaseModel, Field

# Placeholder for future generic types to decouple from provider specific types
# e.g. GenericMessage, GenericRole, etc.

class ToolDefinition(BaseModel):
    """
    Represents the definition of a tool that can be registered with an LLM.

    Attributes:
        name: The unique name of the tool.
        description: A brief description of what the tool does.
        func: The callable Python function that implements the tool's logic.
        parameters: A schema (e.g., JSON schema) defining the input parameters
                    for the tool's function. This can be a dictionary or a provider-specific
                    schema object.
        args_model: Optional Pydantic model used for validating and coercing arguments.
    """
    name: str
    description: str
    func: Callable
    parameters: Optional[Any] = None
    args_model: Optional[Type[BaseModel]] = None


class LLMConfig(BaseModel):
    """
    Configuration parameters for an LLM.

    Attributes:
        temperature: Controls the randomness of the output. Higher values mean more random.
                     Must be between 0 and 2, inclusive.
        max_tokens: The maximum number of tokens to generate in the LLM's response.
                    Must be at least 10.
        system_instruction: An optional system-level instruction or persona for the LLM.
    """
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=1024, ge=10)
    system_instruction: Optional[str] = None
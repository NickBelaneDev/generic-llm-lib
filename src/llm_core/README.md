# llm_core Package

The `llm_core` package provides the fundamental abstractions, interfaces, and types for the Generic LLM Library. It defines the contract that different LLM providers (like OpenAI, Gemini, etc.) must implement, ensuring a consistent API for the end-user.

## Modules

### 1. `base.py`

Defines the core interface for LLM interactions.

#### `GenericLLM` (Abstract Base Class)
The primary abstract base class that all LLM provider implementations must inherit from.

*   **Methods:**
    *   `chat(history: List[Any], user_prompt: str) -> Any`: Conducts a multi-turn chat conversation. It takes a provider-specific history and a user prompt, returning a response object.
    *   `ask(prompt: str, model: str = None) -> Any`: Performs a single-turn question/answer interaction without maintaining history.

### 2. `registry.py`

Handles the registration and management of "tools" (functions) that LLMs can invoke.

#### `ToolRegistry` (Abstract Base Class)
A central registry to manage available tools. It bridges Python functions with LLM tool definitions.

*   **Key Features:**
    *   **Automatic Schema Generation:** Uses `pydantic` and `inspect` to automatically generate JSON schemas from Python function signatures and type hints.
    *   **Validation:** Ensures registered tools have proper docstrings and parameter descriptions (using `Annotated` and `Field`).
    *   **Decorator Support:** Provides a `@tool` decorator for easy registration.

*   **Methods:**
    *   `register(name_or_tool, description, func, parameters)`: Registers a tool. Can accept a `ToolDefinition`, a callable, or explicit components.
    *   `tool(func)`: Decorator to register a function as a tool.
    *   `implementations`: Returns a dictionary mapping tool names to their Python callables.
    *   `tool_object` (Abstract Property): Must be implemented by subclasses to convert the internal tool definitions into the specific format required by the LLM provider (e.g., OpenAI function calling format).

### 3. `types.py`

Defines the data structures used throughout the library.

#### `ToolDefinition`
A Pydantic model representing a registered tool.
*   `name`: Tool name.
*   `description`: Tool description.
*   `func`: The actual Python callable.
*   `parameters`: The schema defining inputs.
*   `args_model`: A dynamically generated Pydantic model for argument validation.

#### `LLMConfig`
Configuration parameters for LLM behavior.
*   `temperature`: Controls randomness (0.0 - 2.0).
*   `max_tokens`: Limit on generated tokens.
*   `system_instruction`: Optional system prompt.

### 4. `exceptions.py`

Defines the exception hierarchy for the library.

*   `LLMToolError`: Base exception for tool-related errors.
*   `ToolRegistrationError`: Issues during tool registration (e.g., missing descriptions, duplicate names).
*   `ToolNotFoundError`: Requested tool not found.
*   `ToolExecutionError`: The tool function failed during execution.
*   `ToolValidationError`: Invalid tool definition or parameters.

## Usage Example

### Defining a Tool

Tools require type hints with `Annotated` and `Field` descriptions to generate precise schemas for the LLM.

```python
from typing import Annotated
from pydantic import Field
from llm_core.registry import ToolRegistry

# Assuming a concrete implementation of ToolRegistry exists
registry = MyProviderRegistry()

@registry.tool
def get_weather(
    location: Annotated[str, Field(description="The city and state, e.g. San Francisco, CA")],
    unit: Annotated[str, Field(description="The unit of temperature, either 'celsius' or 'fahrenheit'")] = "celsius"
):
    """Get the current weather in a given location."""
    return f"Weather in {location} is sunny ({unit})"
```

### Implementing a Provider

To add a new LLM provider, subclass `GenericLLM` and implement the abstract methods.

```python
from llm_core.base import GenericLLM

class MyLLM(GenericLLM):
    async def chat(self, history, user_prompt):
        # Implementation specific to the provider
        pass

    async def ask(self, prompt, model=None):
        # Implementation specific to the provider
        pass
```

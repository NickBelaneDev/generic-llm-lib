# llm_impl.openai_api Package

The `llm_impl.openai_api` package provides the concrete implementation of the Generic LLM Library for OpenAI's models (e.g., GPT-4o, GPT-3.5). It handles the specifics of the OpenAI API, including chat completion, function calling (tools), and response parsing.

## Modules

### 1. `core.py`

Contains the main implementation of the LLM wrapper.

#### `GenericOpenAI`
A subclass of `GenericLLM` that interfaces with the OpenAI API using the `AsyncOpenAI` client.

*   **Initialization:**
    *   Requires an initialized `AsyncOpenAI` client.
    *   Configurable parameters: `model_name`, `sys_instruction`, `registry`, `temp`, `max_tokens`, `max_function_loops`, `tool_timeout`, `tool_manager`.
    *   Uses a `ToolExecutionLoop` to manage automated tool execution cycles.

*   **Methods:**
    *   `chat(history, user_prompt)`: The primary entry point for multi-turn conversations. It automatically handles message formatting, API calls, and tool execution loops.
    *   `ask(prompt)`: A convenience method for single-turn stateless queries.

### 2. `registry.py`

Handles OpenAI-specific tool registration.

#### `OpenAIToolRegistry`
A subclass of `ToolRegistry` tailored for OpenAI's tool definition format.

*   **Key Features:**
    *   **`tool_object` Property:** Automatically generates the JSON structure required by OpenAI's `tools` API (e.g., `{"type": "function", "function": {...}}`).
    *   **Pydantic Integration**: Uses Pydantic models for robust argument validation.

### 3. `adapter.py`

Provides the `OpenAIToolAdapter`, which bridges the OpenAI-specific API with the core library's `ToolExecutionLoop`. It handles the specifics of sending tool results back to OpenAI.

### 4. `history_converter.py`

Contains utilities to convert between the library's internal `BaseMessage` format and OpenAI's native message dictionary format.

## Usage Example

```python
import asyncio
import os
from openai import AsyncOpenAI
from generic_llm_lib.llm_impl.openai_api import GenericOpenAI, OpenAIToolRegistry
from generic_llm_lib.llm_core.messages import HistoryHandler

# 1. Setup Registry and Tools
registry = OpenAIToolRegistry()

@registry.register
def get_current_time(timezone: str = "UTC"):
    """Returns the current time in the specified timezone."""
    from datetime import datetime
    return f"The time is {datetime.now()} in {timezone}"

async def main():
    # 2. Initialize Client and Wrapper
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    llm = GenericOpenAI(
        client=client,
        model_name="gpt-4o",
        sys_instruction="You are a helpful assistant.",
        registry=registry
    )

    # 3. Run Chat
    history = HistoryHandler()
    result = await llm.chat(history, "What time is it in London?")
    print(f"Assistant: {result.content}")

if __name__ == "__main__":
    asyncio.run(main())
```

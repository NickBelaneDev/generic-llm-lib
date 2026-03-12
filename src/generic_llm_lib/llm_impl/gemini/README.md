# llm_impl.gemini Package

The `llm_impl.gemini` package provides the concrete implementation of the Generic LLM Library for Google's Gemini models (using the `google-genai` SDK). It handles the specifics of the Gemini API, including chat sessions, function calling (tools), and response parsing.

## Modules

### 1. `core.py`

Contains the main implementation of the LLM wrapper.

#### `GenericGemini`
A subclass of `GenericLLM` that interfaces with the Google GenAI API using the `AsyncClient`.

*   **Initialization:**
    *   Requires an initialized `google.genai.AsyncClient`.
    *   Configurable parameters: `model_name`, `sys_instruction`, `registry`, `temp`, `max_tokens`, `max_function_loops`, `tool_timeout`, `tool_manager`.
    *   Configures the `types.GenerateContentConfig` with tools and generation settings.

*   **Methods:**
    *   `chat(history, user_prompt)`: Manages the chat session. It uses the Gemini `chats` API to maintain state and handles automated tool execution loops.
    *   `ask(prompt)`: A convenience method for single-turn stateless queries.

### 2. `registry.py`

Handles Gemini-specific tool registration.

#### `GeminiToolRegistry`
A subclass of `ToolRegistry` tailored for Gemini's function declaration format.

*   **Key Features:**
    *   **`tool_object` Property:** Converts the internal `ToolDefinition` objects into a `types.Tool` object containing `types.FunctionDeclaration` entries.
    *   **Direct Integration**: Generates schemas compatible with Gemini's strict requirements.

### 3. `adapter.py`

Provides the `GeminiToolAdapter`, which integrates with Gemini's stateful chat sessions to execute tool calls and feed results back to the model within the library's `ToolExecutionLoop`.

### 4. `history_converter.py`

Contains utilities to convert between the library's internal `BaseMessage` format and Gemini's native `types.Content` objects.

## Usage Example

```python
import asyncio
import os
from google import genai
from generic_llm_lib.llm_impl.gemini import GenericGemini, GeminiToolRegistry
from generic_llm_lib.llm_core.messages import HistoryHandler

# 1. Setup Registry and Tools
registry = GeminiToolRegistry()

@registry.register
def get_current_time(timezone: str = "UTC"):
    """Returns the current time in the specified timezone."""
    from datetime import datetime
    return f"The time is {datetime.now()} in {timezone}"

async def main():
    # 2. Initialize Client and Wrapper
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    llm = GenericGemini(
        aclient=client,
        model_name="gemini-2.0-flash",
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

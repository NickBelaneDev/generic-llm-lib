# llm_impl.open_api Package

The `llm_impl.open_api` package provides the concrete implementation of the Generic LLM Library for OpenAI's models (e.g., GPT-3.5, GPT-4). It handles the specifics of the OpenAI API, including chat completion, function calling (tools), and response parsing.

## Modules

### 1. `core.py`

Contains the main implementation of the LLM wrapper.

#### `GenericOpenAI`
A subclass of `GenericLLM` that interfaces with the OpenAI API.

*   **Initialization:**
    *   Requires an initialized `AsyncOpenAI` client.
    *   Configurable parameters: `model_name`, `sys_instruction`, `registry`, `temp`, `max_tokens`, `max_function_loops`.
    *   Initializes the shared `llm_core.tool_helper.ToolHelper` to manage tool execution.

*   **Methods:**
    *   `chat(history, user_prompt)`: Manages the chat session. It constructs the message payload (including system instructions), calls the OpenAI API, and delegates function calling to `_handle_function_calls`. It also cleans the history of intermediate tool calls to save tokens.
    *   `ask(prompt, model)`: A simplified interface for single-turn interactions. It wraps `chat` internally.
    *   `_handle_function_calls`: Delegates the complex loop of checking for tool calls, executing them, and feeding results back to the model to the shared core tool helper.
    *   `_clean_history`: Removes intermediate tool call messages and outputs from the history to keep the context window manageable, preserving only the final results and conversation flow.

### 2. `registry.py`

Handles OpenAI-specific tool registration.

#### `OpenAIToolRegistry`
A subclass of `ToolRegistry` tailored for OpenAI's function calling format.

*   **Key Features:**
    *   **`tool_object` Property:** Converts the internal `ToolDefinition` objects into the specific JSON structure required by OpenAI's `tools` API parameter (e.g., `{"type": "function", "function": {...}}`).
    *   **Registration:** Inherits the flexible registration logic from the base class but ensures compatibility with OpenAI's requirements.

### 3. `models.py`

Defines Pydantic models for structured OpenAI responses.

*   `OpenAITokens`: Tracks token usage (prompt, completion, total).
*   `OpenAIMessageResponse`: Contains the text content and token usage for a single message.
*   `OpenAIChatResponse`: The top-level response object returned by `chat`, containing the final response and the updated conversation history.

## Usage Example

```python
import asyncio
from openai import AsyncOpenAI
from src.llm_impl.open_api import GenericOpenAI, OpenAIToolRegistry

# 1. Setup Registry and Tools
registry = OpenAIToolRegistry()

@registry.tool
def get_current_time(timezone: str = "UTC"):
    """Returns the current time in the specified timezone."""
    from datetime import datetime
    return f"The time is {datetime.now()} in {timezone}"

# 2. Initialize Client and Wrapper
client = AsyncOpenAI(api_key="your-api-key")
llm = GenericOpenAI(
    client=client,
    model_name="gpt-3.5-turbo",
    sys_instruction="You are a helpful assistant.",
    registry=registry
)

# 3. Run Chat
async def main():
    # Single question
    response = await llm.ask("What time is it in London?")
    print(response.text)

    # Chat session
    history = []
    chat_response = await llm.chat(history, "My name is Alice.")
    print(chat_response.last_response.text)
    
    # History is updated automatically in the response object
    history = chat_response.history
    chat_response = await llm.chat(history, "What is my name?")
    print(chat_response.last_response.text)

if __name__ == "__main__":
    asyncio.run(main())
```

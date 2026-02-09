# llm_impl.gemini Package

The `llm_impl.gemini` package provides the concrete implementation of the Generic LLM Library for Google's Gemini models (using the `google-genai` SDK). It handles the specifics of the Gemini API, including chat sessions, function calling (tools), and response parsing.

## Modules

### 1. `core.py`

Contains the main implementation of the LLM wrapper.

#### `GenericGemini`
A subclass of `GenericLLM` that interfaces with the Google GenAI API.

*   **Initialization:**
    *   Requires an initialized `genai.Client`.
    *   Configurable parameters: `model_name`, `sys_instruction`, `registry`, `temp`, `max_tokens`, `max_function_loops`.
    *   Configures the `types.GenerateContentConfig` with tools and generation settings.

*   **Methods:**
    *   `chat(history, user_prompt)`: Manages the chat session. It creates a `chats` session with the provided history, sends the user message, and handles the response. It also cleans the history of intermediate tool calls to save tokens.
    *   `ask(prompt, model)`: A simplified interface for single-turn interactions. It wraps `chat` internally.
    *   `_handle_function_calls`: Manages the loop of executing function calls returned by the model. It supports parallel function calling (multiple calls in one turn), executes them (sync or async), and sends the results back to the model until completion or the loop limit is reached.
    *   `_clean_history`: Removes intermediate tool call parts and function responses from the history to keep the context window manageable, preserving only the final text responses and conversation flow.

### 2. `registry.py`

Handles Gemini-specific tool registration.

#### `GeminiToolRegistry`
A subclass of `ToolRegistry` tailored for Gemini's function declaration format.

*   **Key Features:**
    *   **`tool_object` Property:** Converts the internal `ToolDefinition` objects into a `types.Tool` object containing a list of `types.FunctionDeclaration`. This is the format required by the Gemini API.
    *   **Registration:** Inherits the flexible registration logic from the base class.

### 3. `models.py`

Defines Pydantic models for structured Gemini responses.

*   `GeminiTokens`: Tracks token usage (prompt, candidate, total, thoughts, tool use).
*   `GeminiMessageResponse`: Contains the text content and token usage for a single message.
*   `GeminiChatResponse`: The top-level response object returned by `chat`, containing the final response and the updated conversation history (as a list of `types.Content` objects).

## Usage Example

```python
import asyncio
from google import genai
from generic_llm_lib.llm_impl.gemini import GenericGemini, GeminiToolRegistry

# 1. Setup Registry and Tools
registry = GeminiToolRegistry()


@registry.tool
def get_current_time(timezone: str = "UTC"):
    """Returns the current time in the specified timezone."""
    from datetime import datetime
    return f"The time is {datetime.now()} in {timezone}"


# 2. Initialize Client and Wrapper
client = genai.Client(api_key="your-api-key")
llm = GenericGemini(
    aclient=client,
    model_name="gemini-2.5-flash",
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

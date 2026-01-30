# Proposal: Centralized Tool and Function Handling in `llm_core`

## Context
Currently, tool execution logic is partially handled within `llm_core/tool_loop.py`, but there is a need to further centralize business logic and standardize how different LLM providers (Gemini, OpenAI) interact with tools. The goal is to move all core tool handling logic into `llm_core` and use adapters for specific implementations.

## Proposed Architecture

### 1. Core Abstractions (`llm_core`)

We will enhance `llm_core` to provide a unified interface for tool execution and response handling.

#### `ToolExecutionLoop` (Enhanced)
The existing `ToolExecutionLoop` in `tool_loop.py` is a good starting point. It already handles:
- Argument normalization
- Validation via Pydantic
- Tool execution
- Error handling

We need to ensure it remains strictly provider-agnostic.

#### `ToolAdapter` Protocol
We will introduce a `ToolAdapter` protocol (or abstract base class) in `llm_core`. This adapter will define the contract that every provider must implement to bridge their specific API formats with the generic `ToolExecutionLoop`.

```python
from typing import Protocol, Any, Sequence, Awaitable

class ToolAdapter(Protocol):
    """
    Protocol for adapting provider-specific tool handling to the generic loop.
    """

    def get_tool_calls(self, response: Any) -> Sequence[ToolCallRequest]:
        """Extracts generic tool calls from a provider-specific response."""
        ...

    def record_assistant_message(self, response: Any) -> None:
        """Records the assistant's message (including tool calls) to history."""
        ...

    def build_tool_response_message(self, result: ToolCallResult) -> Any:
        """Converts a generic tool result into a provider-specific message."""
        ...

    async def send_tool_responses(self, messages: Sequence[Any]) -> Any:
        """Sends the tool response messages back to the provider and awaits the next response."""
        ...
```

### 2. Implementation Layer (`llm_impl`)

Each provider (Gemini, OpenAI) will implement this `ToolAdapter`.

#### `GeminiToolAdapter`
- **`get_tool_calls`**: Iterates over `response.parts`, checks for `function_call`, and converts them to `ToolCallRequest`.
- **`record_assistant_message`**: Appends the model's response to the chat history (Gemini maintains history in the chat session object, so this might be a no-op or a state update).
- **`build_tool_response_message`**: Creates a `Part` with `function_response` containing the result.
- **`send_tool_responses`**: Calls `chat.send_message` with the list of function responses.

#### `OpenAIToolAdapter`
- **`get_tool_calls`**: Inspects `message.tool_calls` and converts them.
- **`record_assistant_message`**: Appends the assistant message to the messages list.
- **`build_tool_response_message`**: Creates a message with `role: "tool"`, `tool_call_id`, and `content`.
- **`send_tool_responses`**: Calls `client.chat.completions.create` with the updated history.

### 3. Workflow Integration

The `GenericLLM` implementations will instantiate the `ToolExecutionLoop` and their specific `ToolAdapter`.

```python
# Pseudo-code inside a concrete LLM implementation (e.g., GeminiLLM)

async def chat(self, user_prompt: str):
    # 1. Send initial user message
    initial_response = await self._send_initial_message(user_prompt)

    # 2. Create Adapter
    adapter = GeminiToolAdapter(self.chat_session)

    # 3. Run Loop
    final_response = await self.tool_loop.run(
        initial_response=initial_response,
        get_tool_calls=adapter.get_tool_calls,
        record_assistant_message=adapter.record_assistant_message,
        build_tool_response_message=adapter.build_tool_response_message,
        send_tool_responses=adapter.send_tool_responses
    )

    return final_response
```

## Benefits

1.  **Single Responsibility**: `llm_core` owns the "how" of executing tools (validation, timeout, error catching). `llm_impl` only owns the "translation" (parsing API responses, formatting API requests).
2.  **Consistency**: Error messages, timeouts, and validation logic are identical across all providers.
3.  **Extensibility**: Adding a new provider (e.g., Anthropic) only requires implementing the `ToolAdapter` methods.
4.  **Testability**: The `ToolExecutionLoop` can be tested in isolation with a mock adapter, ensuring robust core logic.

## Next Steps

1.  Define the `ToolAdapter` protocol in `llm_core/tool_loop.py` (or a new file if preferred).
2.  Refactor `ToolExecutionLoop.run` to accept an instance of `ToolAdapter` instead of individual callables (cleaner signature).
3.  Implement `GeminiToolAdapter` in `llm_impl/gemini`.
4.  Implement `OpenAIToolAdapter` in `llm_impl/openai_api`.
5.  Update the `chat` methods in both implementations to use the new flow.

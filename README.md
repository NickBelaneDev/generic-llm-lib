# Generic LLM Library

A flexible, async-first library designed for building scalable agents and chatbots. It provides a unified interface for interacting with Large Language Models (LLMs) like OpenAI, automating complex tasks like function calling loops while remaining lightweight enough for quick server-side deployments.

## Features

- **Async-First Architecture**: Built from the ground up for high-performance, non-blocking operations, making it ideal for server environments.
- **Flexible Agent System**: Create adaptable agents capable of handling complex interactions and tool usage.
- **Unified Interface**: Abstract away provider differences behind a consistent API.
- **Automated Function Calling**: Seamlessly handles the "Model -> Tool -> Model" execution loop.
- **Tool Registry**: Simple decorator-based registration to turn Python functions into LLM-accessible tools.
- **Chatbot Ready**: Designed to be quickly integrated into web backends for powering chatbots.

## Installation

Install the package via pip:

```bash
pip install generic-llm-lib
```

Or using [uv](https://github.com/astral-sh/uv):

```bash
uv add generic-llm-lib
```

## Usage

### Generic Agent

```python
from llm_impl.openai_api import OpenAIToolRegistry, GenericOpenAI
from openai import AsyncOpenAI
from pydantic import Field
from typing import Annotated
import asyncio
import os

# 1. Create a registry
registry = OpenAIToolRegistry()

# 2. Register tools
@registry.tool # makes the function a tool
def get_weather(location: Annotated[str, Field(description="The city and state, e.g. San Francisco, CA")]):
    """Get the current weather in a given location"""
    return f"Sunny in {location}"

# 3. Initialize
client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
llm = GenericOpenAI(client, "gpt-4o", "You are helpful.", registry)

# 4. Chat
async def main():
    openai_chat = await llm.chat([], "Weather in Berlin?")
    print(openai_chat.last_response.text)
    print(openai_chat.last_response.tokens.total_tokens)
    print(openai_chat.history)
    


if __name__ == "__main__":
    asyncio.run(main())
```

## Architecture & Tool Execution Logic

The library uses a centralized `ToolExecutionLoop` to handle the interaction between the LLM and the registered tools. This ensures consistent behavior and simplifies the implementation.

### How it works

1.  **Initial Request**: The user sends a message to the LLM via the `chat` method.
2.  **Model Response**: The LLM processes the message and may decide to call one or more tools. It returns a response containing "tool calls".
3.  **Tool Execution Loop**:
    *   The `ToolExecutionLoop` inspects the response.
    *   If tool calls are present, it iterates through them.
    *   **Validation**: It validates the tool name against the registry and parses/validates the arguments using the tool's Pydantic model (if available).
    *   **Execution**: It executes the corresponding Python function.
        *   **Async Support**: Async functions are awaited directly.
        *   **Sync Support**: Synchronous functions are run in a separate thread (`asyncio.to_thread`) to prevent blocking the event loop.
    *   **Error Handling**: Errors during validation or execution are caught and formatted as error messages to be sent back to the LLM, allowing the model to self-correct.
4.  **Feedback Loop**: The results (or errors) of the tool executions are sent back to the LLM.
5.  **Final Response**: The LLM processes the tool results and generates a final natural language response, which is returned to the user.

This loop continues until the model stops requesting tool calls or the `max_function_loops` limit is reached.

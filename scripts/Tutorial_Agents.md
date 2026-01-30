# Tutorial: How to use `generic-llm-lib`

This tutorial explains how to use the `generic-llm-lib` library to build LLM-powered applications. It is designed for AI agents to understand the library's usage patterns.

## 1. Core Concepts

The library is split into two main layers:
- **`llm_core`**: Contains the base abstractions (`GenericLLM`, `ToolRegistry`).
- **`llm_impl`**: Contains concrete implementations (e.g., `GenericGemini` for Google Gemini, `GenericOpenAI` for OpenAI).

The main workflow involves:
1.  Defining tools (optional).
2.  Initializing a `ToolRegistry` and registering tools.
3.  Initializing an LLM implementation (e.g., `GenericGemini` or `GenericOpenAI`) with the registry.
4.  Using `.chat()` for conversation or `.ask()` for single queries.

## 2. Defining Tools

Tools are Python functions that the LLM can execute. They must have type hints and docstrings.

```python
from typing import Annotated
from pydantic import Field

def get_weather(
    location: Annotated[str, Field(description="The city and state, e.g. San Francisco, CA")],
    unit: Annotated[str, Field(description="Temperature unit 'celsius' or 'fahrenheit'")] = "celsius"
) -> str:
    """
    Get the current weather in a given location.
    """
    # Implementation logic here...
    return f"Weather in {location} is 20 degrees {unit}."
```

**Key Requirements:**
- **Docstring**: Essential. The LLM uses this to understand *what* the tool does.
- **Type Hints**: Must use `Annotated` with `Field(description="...")` for parameters. The LLM uses this to understand *how* to use the arguments.

## 3. Registering Tools

Use `ToolRegistry` to manage your tools.

```python
from llm_core.registry import ToolRegistry

registry = ToolRegistry()

# Register a function directly
registry.register(get_weather)

# Or use the decorator
@registry.tool
def calculate_sum(
    a: Annotated[int, Field(description="First number")], 
    b: Annotated[int, Field(description="Second number")]
) -> int:
    """Adds two numbers."""
    return a + b
```

## 4. Initializing the LLM

### Option A: Google Gemini

Initialize the `GenericGemini` class. You need a `google.genai.Client`.

```python
import os
from google import genai
from llm_impl.gemini.core import GenericGemini

# 1. Setup Client
api_key = os.environ["GEMINI_API_KEY"]
client = genai.Client(api_key=api_key)

# 2. Initialize Wrapper
agent = GenericGemini(
    client=client,
    model_name="gemini-1.5-flash",
    sys_instruction="You are a helpful assistant.",
    registry=registry,  # Pass the registry here
    temp=0.7
)
```

### Option B: OpenAI

Initialize the `GenericOpenAI` class. You need an `openai.AsyncOpenAI` client.

```python
import os
from openai import AsyncOpenAI
from llm_impl.open_api.core import GenericOpenAI

# 1. Setup Client
api_key = os.environ["OPENAI_API_KEY"]
client = AsyncOpenAI(api_key=api_key)

# 2. Initialize Wrapper
agent = GenericOpenAI(
    client=client,
    model_name="gpt-4o",
    sys_instruction="You are a helpful assistant.",
    registry=registry,  # Pass the registry here
    temp=0.7
)
```

## 5. Chatting with the Agent

### Single Turn (`ask`)
Use `ask` for one-off questions. It returns a provider-specific message response (e.g., `GeminiMessageResponse` or `OpenAIMessageResponse`).

```python
import asyncio

async def main():
    response = await agent.ask("What is the weather in Berlin?")
    print(response.text) 
    # Output: "Weather in Berlin is 20 degrees celsius." (if tool was called)

if __name__ == "__main__":
    asyncio.run(main())
```

### Multi-Turn Chat (`chat`)
Use `chat` to maintain history. It requires a list of history objects (provider-specific) and returns a chat response object.

**Note on History:**
- **Gemini**: Uses `google.genai.types.Content` objects.
- **OpenAI**: Uses `Dict[str, Any]` (standard OpenAI message format: `{"role": "user", "content": "..."}`).

```python
# Example loop (abstracted for both providers)
async def chat_loop():
    history = [] # Initialize empty list
    
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        # The agent handles tool execution loops automatically inside .chat()
        response = await agent.chat(history, user_input)
        
        print(f"Agent: {response.last_response.text}")
        
        # Update history for the next turn
        history = response.history

if __name__ == "__main__":
    asyncio.run(chat_loop())
```

## 6. Advanced: Tool Execution Loop

Both `GenericGemini` and `GenericOpenAI` automatically handle the "function calling loop":
1.  LLM generates a tool call.
2.  Library executes the Python function.
3.  Library sends the result back to the LLM.
4.  LLM generates the final text response.

This is controlled by `max_function_loops` (default 5) in the constructor.

## 7. Summary of Classes

| Class | Path | Description |
| :--- | :--- | :--- |
| `GenericLLM` | `llm_core.base` | Abstract base class for all LLMs. |
| `ToolRegistry` | `llm_core.registry` | Manages tools and generates schemas. |
| `GenericGemini` | `llm_impl.gemini.core` | Gemini implementation. |
| `GenericOpenAI` | `llm_impl.open_api.core` | OpenAI implementation. |

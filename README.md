# Generic LLM Library

A flexible, async-first library designed for building scalable agents and chatbots. It provides a unified interface for interacting with Large Language Models (LLMs) like Google Gemini and OpenAI, automating complex tasks like function calling loops while remaining lightweight enough for quick server-side deployments.

## Features

- **Async-First Architecture**: Built from the ground up for high-performance, non-blocking operations, making it ideal for server environments.
- **Flexible Agent System**: Create adaptable agents capable of handling complex interactions and tool usage.
- **Unified Interface**: Abstract away provider differences (Gemini, OpenAI) behind a consistent API.
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
from llm_impl import OpenAIToolRegistry, GenericOpenAI
from google import genai
from pydantic import Field
from typing import Annotated
import asyncio

# 1. Create a registry
registry = OpenAIToolRegistry() # Replace this with GeminiToolRegistry if you want to use gemini

# 2. Register tools
@registry.tool # makes the function a tool
def get_weather(location: Annotated[str, Field(description="The city and state, e.g. San Francisco, CA")]):
    """Get the current weather in a given location"""
    return f"Sunny in {location}"

# 3. Initialize
client = genai.Client(api_key="YOUR_KEY")
llm = GenericOpenAI(client, "gemini-2.0-flash-exp", "You are helpful.", registry) # Replace with GenericGemini if you want to use gemini

# 4. Chat
async def main():
    gemini_chat = await llm.chat([], "Weather in Berlin?")
    print(gemini_chat.last_response.text)
    print(gemini_chat.last_response.tokens.total_token_count)
    print(gemini_chat.history)
    


if __name__ == "__main__":
    asyncio.run(main())
```

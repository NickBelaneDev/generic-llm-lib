# Generic LLM Library

A flexible, async-first library designed for building scalable agents and chatbots. It provides a unified interface for interacting with Large Language Models (LLMs) like Google Gemini, automating complex tasks like function calling loops while remaining lightweight enough for quick server-side deployments.

## Features

- **Async-First Architecture**: Built from the ground up for high-performance, non-blocking operations, making it ideal for server environments.
- **Flexible Agent System**: Create adaptable agents capable of handling complex interactions and tool usage.
- **Unified Interface**: Abstract away provider differences (starting with Gemini) behind a consistent API.
- **Automated Function Calling**: Seamlessly handles the "Model -> Tool -> Model" execution loop.
- **Tool Registry**: Simple decorator-based registration to turn Python functions into LLM-accessible tools.
- **Chatbot Ready**: Designed to be quickly integrated into web backends for powering chatbots.

## Usage

```python
from src.llm_impl import GenericGemini, GeminiToolRegistry
from google import genai
import asyncio

# 1. Create a registry
registry = GeminiToolRegistry()

# 2. Register tools
@registry.tool # makes the function a tool
def get_weather(location: str):
    return f"Sunny in {location}"

# 3. Initialize
client = genai.Client(api_key="YOUR_KEY")
llm = GenericGemini(client, "gemini-2.0-flash-exp", "You are helpful.", registry)

# 4. Chat
async def main():
    gemini_chat = await llm.chat([], "Weather in Berlin?")
    print(gemini_chat.last_response.text)
    print(gemini_chat.last_response.tokens.total_token_count)
    print(gemini_chat.history)
    


if __name__ == "__main__":
    asyncio.run(main())
```
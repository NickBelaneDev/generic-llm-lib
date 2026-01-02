# Generic LLM Library

A unified interface for interacting with various Large Language Models (LLMs), starting with Google Gemini. This library simplifies chat management and automates the function calling loop.

## Features

- **Unified Interface**: Use `GenericLLM` to interact with different providers.
- **Automated Function Calling**: Handles the "Model -> Tool -> Model" loop automatically.
- **Tool Registry**: Easy registration of Python functions as tools for the LLM.

## Usage

```python
from src.impl import GenericGemini, GeminiToolRegistry
from google import genai
import asyncio

# 1. Define tools
def get_weather(location: str):
    return f"Sunny in {location}"

# 2. Register tools
registry = GeminiToolRegistry()
registry.register("get_weather", "Get weather info", get_weather, {"type": "OBJECT", "properties": {"location": {"type": "STRING"}}})

# 3. Initialize
client = genai.Client(api_key="YOUR_KEY")
llm = GenericGemini(client, "gemini-2.0-flash-exp", "You are helpful.", registry)

# 4. Chat
async def main():
    response, history = await llm.chat([], "Weather in Berlin?")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
```

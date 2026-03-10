<div align="center">

# 🤖 Generic LLM Library

**A modern, async-first Python framework for building scalable AI agents and chatbots.**

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue)](http://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Core Concepts](#-core-concepts) • [CI/CD & Quality](#-cicd--quality)

</div>

## ✨ Features

- 🚀 **Async-First Architecture**: Built for high-performance, non-blocking operations.
- 🤖 **Unified Provider API**: Switch between **OpenAI** and **Google Gemini** with zero logic changes.
- 🛠️ **Automated Tool Loops**: Handles the complex "Model -> Tool -> Model" execution cycle out-of-the-box.
- 🔌 **MCP Support**: Seamlessly integrate with [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) servers.
- 📝 **Smart History Management**: Centralized `HistoryHandler` for effortless conversation state tracking.
- 📦 **Type-Safe Tooling**: Define tools using standard Python type hints and Pydantic models.
- 🛡️ **Production Ready**: Built-in exponential backoff, retry logic, and comprehensive error handling.

## 📦 Installation

Since this library is currently in development, you can install it directly from the source.

### Using [uv](https://github.com/astral-sh/uv) (Recommended)

```bash
# Clone the repository
git clone https://github.com/NickBelaneDev/generic-llm-lib.git
cd generic-llm-lib

# Install dependencies and create a virtual environment
uv sync
```

### Using pip

```bash
git clone https://github.com/NickBelaneDev/generic-llm-lib.git
cd generic-llm-lib
pip install -e .
```

## 🚀 Quick Start

Build a tool-capable chatbot in minutes.

```python
import asyncio
import os
from typing import Annotated
from pydantic import Field
from openai import AsyncOpenAI
from generic_llm_lib.llm_impl import GenericOpenAI, OpenAIToolRegistry
from generic_llm_lib.llm_core.messages import HistoryHandler

# 1. Define your tools
registry = OpenAIToolRegistry()

@registry.register
async def get_weather(
    location: Annotated[str, Field(description="The city and state, e.g. Berlin, DE")]
) -> str:
    """Get the current weather in a given location."""
    return f"The weather in {location} is sunny and 22°C."

async def main():
    # 2. Initialize the LLM
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    llm = GenericOpenAI(
        client=client,
        model_name="gpt-4o",
        sys_instruction="You are a helpful assistant.",
        registry=registry
    )

    # 3. Start a conversation with the HistoryHandler
    history = HistoryHandler(system_instruction="You are a helpful assistant.")

    # 4. Run the chat
    result = await llm.chat(history, "What's the weather like in Berlin?")
    
    print(f"Assistant: {result.content}")
    
    # The history is automatically updated and can be used for the next turn
    # result.history contains the full conversation including tool calls

if __name__ == "__main__":
    asyncio.run(main())
```

## 🔌 Model Context Protocol (MCP) Integration

This library provides a built-in wrapper to easily connect and use tools from any MCP-compliant server.

```python
import asyncio
import os
from openai import AsyncOpenAI
from generic_llm_lib.llm_impl import GenericOpenAI, OpenAIToolRegistry
from generic_llm_lib.mcp_wrapper import MCPClientWrapper
from generic_llm_lib.llm_core.messages import HistoryHandler

async def main():
    registry = OpenAIToolRegistry()
    
    # Connect to an MCP server (e.g., a local SQLite MCP server)
    # Ensure the MCP server is installed and runnable via command line
    async with MCPClientWrapper(
        command="uvx", 
        args=["mcp-server-sqlite", "--db-path", "./test.db"]
    ) as mcp_client:
        
        # Load all tools from the MCP server into your registry
        await mcp_client.load_into(registry)
        
        # Initialize LLM with the registry containing MCP tools
        client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        llm = GenericOpenAI(
            client=client,
            model_name="gpt-4o",
            sys_instruction="You are a helpful assistant with access to a database.",
            registry=registry
        )
        
        history = HistoryHandler()
        
        # The LLM can now use tools provided by the MCP server
        result = await llm.chat(history, "Can you check the database for user 'Alice'?")
        print(result.content)

if __name__ == "__main__":
    asyncio.run(main())
```

## ⚙️ Configuration

The `GenericOpenAI` class offers several configuration options to fine-tune the LLM's behavior.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `client` | `AsyncOpenAI` | **Required** | The initialized AsyncOpenAI client. |
| `model_name` | `str` | **Required** | The model identifier (e.g., `gpt-4o`, `gpt-3.5-turbo`). |
| `sys_instruction` | `str` | **Required** | System prompt defining the agent's persona. |
| `registry` | `OpenAIToolRegistry` | `None` | Registry containing tools the LLM can use. |
| `temp` | `float` | `1.0` | Sampling temperature (0.0 to 2.0). Higher values mean more randomness. |
| `max_tokens` | `int` | `3000` | Maximum number of tokens to generate. |
| `max_function_loops` | `int` | `5` | Maximum consecutive tool calls allowed in a single turn. |
| `tool_timeout` | `float` | `180.0` | Timeout in seconds for tool execution. |

### Environment Variables

Ensure the following environment variables are set depending on your provider:

- **OpenAI**: `OPENAI_API_KEY`
- **Google Gemini**: `GOOGLE_API_KEY` (if using the Gemini implementation)

## 🧠 Core Concepts

### 📝 HistoryHandler
The `HistoryHandler` is your central state manager. It abstracts away the complexity of managing message lists, ensuring that system prompts, user inputs, and tool interactions are correctly ordered and formatted.

- **`add_user_message(content)`**: Manually add user input.
- **`clean_tool_calls()`**: Strip intermediate tool outputs to save tokens.
- **`copy()`**: Create a deep copy of the current conversation state.

### 🛠️ Tool Registry
Turn any Python function into an LLM tool. The library automatically generates the required JSON schema from your function's docstring and type hints.

```python
@registry.register
def calculate(a: int, b: int, op: str = "+") -> int:
    """Perform basic arithmetic operations."""
    return a + b if op == "+" else a - b
```

## 🛠 CI/CD & Quality

We maintain high code quality standards through a rigorous CI/CD pipeline:

- **Linting & Formatting**: [Ruff](https://github.com/astral-sh/ruff) and [Black](https://github.com/psf/black) for consistent style.
- **Static Analysis**: [MyPy](http://mypy-lang.org/) for strict type checking.
- **Security**: [Bandit](https://github.com/PyCQA/bandit) for vulnerability scanning and [pip-audit](https://github.com/pypa/pip-audit) for dependency checks.
- **Complexity & Docs**: [Xenon](https://github.com/rubik/xenon) for code complexity and [Interrogate](https://github.com/econchick/interrogate) for docstring coverage.
- **Testing**: [Pytest](https://docs.pytest.org/) with `pytest-asyncio` and `pytest-recording` (VCR) for deterministic integration tests.

---
<div align="center">
Made with ❤️ for the AI Community
</div>

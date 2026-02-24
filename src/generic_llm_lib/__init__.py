"""Generic LLM Library - A provider-agnostic interface for LLMs and tools."""

from .llm_core import (
    GenericLLM,
    ChatResult,
    UserMessage,
    AssistantMessage,
    SystemMessage,
    ToolMessage,
    ToolRegistry,
    ToolDefinition,
)
from .llm_impl.gemini import GenericGemini, GeminiToolRegistry
from .llm_impl.openai_api import GenericOpenAI, OpenAIToolRegistry

__all__ = [
    "GenericLLM",
    "ChatResult",
    "UserMessage",
    "AssistantMessage",
    "SystemMessage",
    "ToolMessage",
    "ToolRegistry",
    "ToolDefinition",
    "GenericGemini",
    "GeminiToolRegistry",
    "GenericOpenAI",
    "OpenAIToolRegistry",
]

"""Base abstractions for LLM providers."""

from .base import GenericLLM, ChatResult

__all__ = [
    "GenericLLM",
    "ChatResult",
]

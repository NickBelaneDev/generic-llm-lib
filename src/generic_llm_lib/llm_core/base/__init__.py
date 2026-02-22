"""Re-export base LLM interfaces and shared response models used by all providers."""

from .base import GenericLLM, ChatResult

__all__ = [
    "GenericLLM",
    "ChatResult",
]

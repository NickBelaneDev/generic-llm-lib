"""Provider-agnostic message models for chat history."""

from pydantic import BaseModel
from abc import ABC
from typing import Optional, List, Any


class BaseMessage(ABC, BaseModel):
    """Base model for messages exchanged with an LLM.

    Attributes:
        author: Role associated with the message.
        content: Text payload of the message.
    """

    author: str
    content: str


class SystemMessage(BaseMessage):
    """Message authored by the system to steer behavior."""

    author: str = "system"


class UserMessage(BaseMessage):
    """Message authored by an end user."""

    author: str = "user"


class AssistantMessage(BaseMessage):
    """Message authored by the assistant, optionally containing tool calls."""

    author: str = "assistant"
    tool_calls: Optional[List[Any]] = None


class ToolMessage(BaseMessage):
    """Message emitted by a tool invocation."""

    author: str = "tool"
    tool_call_id: str
    name: str

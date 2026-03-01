"""Provider-agnostic message models for chat history."""

from pydantic import BaseModel, Field
from abc import ABC
from typing import Optional, List, Any


class BaseMessage(ABC, BaseModel):
    """Base model for messages exchanged with an LLM.

    Attributes:
        role: Role associated with the message (e.g., 'user', 'assistant', 'system').
        content: Text payload of the message.
    """

    role: str
    content: str


class SystemMessage(BaseMessage):
    """Message authored by the system to steer behavior."""

    role: str = "system"


class UserMessage(BaseMessage):
    """Message authored by an end user."""

    role: str = "user"


class AssistantMessage(BaseMessage):
    """Message authored by the assistant, optionally containing tool calls."""

    role: str = "assistant"
    tool_calls: Optional[List[Any]] = None


class ToolMessage(BaseMessage):
    """Message emitted by a tool invocation."""

    role: str = "tool"
    tool_call_id: str
    name: str

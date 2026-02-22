"""Implement normalized conversation message models used to bridge provider payload formats."""

from pydantic import BaseModel
from abc import ABC
from typing import Optional, List, Any


class BaseMessage(ABC, BaseModel):
    """Base structure for all provider-agnostic chat messages."""

    author: str
    content: str


class SystemMessage(BaseMessage):
    """System instruction message authored by the assistant framework."""

    author: str = "system"


class UserMessage(BaseMessage):
    """End-user message in the conversation history."""

    author: str = "user"


class AssistantMessage(BaseMessage):
    """Assistant response message, optionally including tool call metadata."""

    author: str = "assistant"
    tool_calls: Optional[List[Any]] = None


class ToolMessage(BaseMessage):
    """Message that captures the output emitted by a tool invocation."""

    author: str = "tool"
    tool_call_id: str
    name: str

from pydantic import BaseModel
from abc import ABC
from typing import Optional, List, Any


class BaseMessage(ABC, BaseModel):
    author: str
    content: str


class SystemMessage(BaseMessage):
    author: str = "system"


class UserMessage(BaseMessage):
    author: str = "user"


class AssistantMessage(BaseMessage):
    author: str = "assistant"
    tool_calls: Optional[List[Any]] = None


class ToolMessage(BaseMessage):
    author: str = "tool"
    tool_call_id: str
    name: str

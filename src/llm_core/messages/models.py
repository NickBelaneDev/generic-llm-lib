from pydantic import BaseModel
from abc import ABC


class BaseMessage(ABC, BaseModel):
    author: str
    content: dict[str, str]


class SystemMessage(BaseMessage):
    author: str = "system"


class UserMessage(BaseMessage):
    author: str = "user"


class AssistantMessage(BaseMessage):
    author: str = "assistant"


class ToolMessage(BaseMessage):
    author: str = "tool"
    tool_call_id: str
    name: str

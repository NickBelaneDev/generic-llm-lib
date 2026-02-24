"""Data models for tool execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ToolCallRequest:
    """Represents a normalized tool call request from an LLM response."""

    name: str
    arguments: Any
    call_id: Optional[str] = None


@dataclass(frozen=True)
class ToolCallResult:
    """Represents the outcome of executing a tool call."""

    name: str
    response: Dict[str, Any]
    call_id: Optional[str] = None

"""Compatibility wrapper for the core tool helper."""

from .tool_helper import ToolHelper as ToolExecutionLoop
from .tool_helper import ToolCallRequest, ToolCallResult

__all__ = ["ToolExecutionLoop", "ToolCallRequest", "ToolCallResult"]

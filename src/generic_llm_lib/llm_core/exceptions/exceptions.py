# Please write a docstring for this file
"""
Custom exception classes for the LLM tool system.

This module defines a hierarchy of exceptions used to handle errors during
tool discovery, registration, validation, and execution within the LLM core.
"""

class LLMToolError(Exception):
    """Base exception for all tool-related errors."""

    pass


class ToolLoadError(LLMToolError):
    """Raised when there is an error loading a tool."""

    pass


class ToolRegistrationError(LLMToolError):
    """Raised when there is an error registering a tool."""

    pass


class ToolNotFoundError(LLMToolError):
    """Raised when a requested tool is not found in the registry."""

    pass


class ToolExecutionError(LLMToolError):
    """Raised when a tool fails during execution."""

    pass


class ToolValidationError(LLMToolError):
    """Raised when tool parameters or definition are invalid."""

    pass

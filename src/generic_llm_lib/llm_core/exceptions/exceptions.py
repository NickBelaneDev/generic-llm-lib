class LLMToolError(Exception):
    """Base exception for all tool-related errors."""

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

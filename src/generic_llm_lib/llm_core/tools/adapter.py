"""Protocol for adapting provider-specific tool handling."""

from __future__ import annotations

from typing import Any, Protocol, Sequence

from .call_protocol import ToolCallRequest, ToolCallResult


class ToolAdapter(Protocol):
    """
    Protocol for adapting provider-specific tool handling to the generic loop.
    """

    def get_tool_calls(self, response: Any) -> Sequence[ToolCallRequest]:
        """Extracts generic tool calls from a provider-specific response."""
        ...

    def record_assistant_message(self, response: Any) -> None:
        """Records the assistant's message (including tool calls) to history."""
        ...

    def build_tool_response_message(self, result: ToolCallResult) -> Any:
        """Converts a generic tool result into a provider-specific message."""
        ...

    async def send_tool_responses(self, messages: Sequence[Any]) -> Any:
        """Sends the tool response messages back to the provider and awaits the next response."""
        ...

"""History handler for managing conversation state."""

from typing import List, Optional, Any, Iterator
import copy
from .models import BaseMessage, SystemMessage, UserMessage, AssistantMessage, ToolMessage


class HistoryHandler:
    """
    Manages the conversation history with an LLM.

    This class provides a structured way to handle message history, including
    adding different types of messages and managing the system instruction.
    """

    def __init__(self, system_instruction: Optional[str] = None, messages: Optional[List[BaseMessage]] = None):
        """
        Initialize the history handler.

        Args:
            system_instruction: Optional initial system instruction.
            messages: Optional list of existing messages.
        """
        self._messages: List[BaseMessage] = messages or []
        if system_instruction:
            # Ensure system message is at the beginning if not present
            if not self._messages or not isinstance(self._messages[0], SystemMessage):
                self._messages.insert(0, SystemMessage(content=system_instruction))

    @property
    def messages(self) -> List[BaseMessage]:
        """Return the current list of messages."""
        return self._messages

    def add_message(self, message: BaseMessage) -> None:
        """Add a generic message to the history."""
        self._messages.append(message)

    def add_system_message(self, content: str) -> None:
        """Add a system message."""
        self._messages.append(SystemMessage(content=content))

    def add_user_message(self, content: str) -> None:
        """Add a user message."""
        self._messages.append(UserMessage(content=content))

    def add_assistant_message(self, content: str, tool_calls: Optional[List[Any]] = None) -> None:
        """Add an assistant message."""
        self._messages.append(AssistantMessage(content=content, tool_calls=tool_calls))

    def add_tool_message(self, content: str, tool_call_id: str, name: str) -> None:
        """Add a tool output message."""
        self._messages.append(ToolMessage(content=content, tool_call_id=tool_call_id, name=name))

    def clear(self) -> None:
        """Clear the history."""
        self._messages.clear()

    def clean_tool_calls(self) -> None:
        """
        Removes intermediate tool calls and responses from the history.

        Keeps user messages, system messages, and assistant messages that have content.
        This is useful for saving context window space.
        """
        cleaned: List[BaseMessage] = []
        for msg in self._messages:
            if isinstance(msg, ToolMessage):
                continue

            if isinstance(msg, AssistantMessage):
                if msg.tool_calls:
                    if msg.content:
                        # Keep content, remove tool_calls
                        cleaned.append(AssistantMessage(content=msg.content, tool_calls=None))
                    # If no content, skip
                else:
                    cleaned.append(msg)
            else:
                cleaned.append(msg)

        self._messages = cleaned

    def copy(self) -> "HistoryHandler":
        """Return a deep copy of the history handler."""
        return HistoryHandler(messages=copy.deepcopy(self._messages))

    def __iter__(self) -> Iterator[BaseMessage]:
        """Iterate over messages."""
        return iter(self._messages)

    def __len__(self) -> int:
        """Return number of messages."""
        return len(self._messages)

    def __getitem__(self, index: int) -> BaseMessage:
        """Get message by index."""
        return self._messages[index]

    def __repr__(self) -> str:
        """Return string representation."""
        return f"HistoryHandler(messages={len(self._messages)})"

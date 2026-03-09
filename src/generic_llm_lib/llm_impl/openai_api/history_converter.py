"""
Handles the conversion between the generic `BaseMessage` format and the
OpenAI-specific dictionary format for chat messages.

This module isolates data transformation logic, simplifying the `GenericOpenAI`
class and ensuring that all conversion functions have low cyclomatic complexity
(Rank A) for high maintainability.
"""

from typing import List, Dict, Any, Optional

from generic_llm_lib.llm_core.messages import (
    BaseMessage,
    UserMessage,
    AssistantMessage,
    SystemMessage,
    ToolMessage,
)

# --- Conversion TO OpenAI format ---


def _convert_user_message(msg: UserMessage) -> Dict[str, Any]:
    """Converts a generic UserMessage to the OpenAI dictionary format."""
    return {"role": "user", "content": msg.content}


def _convert_assistant_message(msg: AssistantMessage) -> Dict[str, Any]:
    """Converts a generic AssistantMessage to the OpenAI dictionary format."""
    openai_msg: Dict[str, Any] = {"role": "assistant", "content": msg.content}
    if msg.tool_calls:
        openai_msg["tool_calls"] = msg.tool_calls
    return openai_msg


def _convert_system_message(msg: SystemMessage) -> Dict[str, Any]:
    """Converts a generic SystemMessage to the OpenAI dictionary format."""
    return {"role": "system", "content": msg.content}


def _convert_tool_message(msg: ToolMessage) -> Dict[str, Any]:
    """Converts a generic ToolMessage to the OpenAI dictionary format."""
    # The 'name' field is added here to be retrieved during the reverse conversion.
    # It is not a standard field for the OpenAI Chat Completions API's tool message.
    return {
        "role": "tool",
        "content": msg.content,
        "tool_call_id": msg.tool_call_id,
        "name": msg.name,
    }


def _convert_single_message_to_openai(msg: BaseMessage) -> Optional[Dict[str, Any]]:
    """Dispatches a single generic message to the appropriate OpenAI conversion function."""
    if isinstance(msg, UserMessage):
        return _convert_user_message(msg)
    if isinstance(msg, AssistantMessage):
        return _convert_assistant_message(msg)
    if isinstance(msg, SystemMessage):
        return _convert_system_message(msg)
    if isinstance(msg, ToolMessage):
        return _convert_tool_message(msg)
    return None


def convert_to_openai_history(history: List[BaseMessage]) -> List[Dict[str, Any]]:
    """Converts a list of generic BaseMessage objects to an OpenAI-specific history.

    Args:
        history: A list of BaseMessage objects.

    Returns:
        A list of message dictionaries compatible with the OpenAI API.
    """
    openai_history = []
    for msg in history:
        converted_msg = _convert_single_message_to_openai(msg)
        if converted_msg:
            openai_history.append(converted_msg)
    return openai_history


# --- Conversion FROM OpenAI format ---


def _convert_openai_user_role(msg: Dict[str, Any]) -> Optional[UserMessage]:
    """Converts an OpenAI message dictionary with 'user' role to a UserMessage."""
    content = msg.get("content")
    if isinstance(content, str) and content:
        return UserMessage(content=content)
    return None


def _convert_openai_assistant_role(msg: Dict[str, Any]) -> Optional[AssistantMessage]:
    """Converts an OpenAI message dictionary with 'assistant' role to an AssistantMessage."""
    content = msg.get("content")
    tool_calls = msg.get("tool_calls")
    # An assistant message is valid if it has text content or tool calls.
    if (isinstance(content, str) and content) or tool_calls:
        return AssistantMessage(content=content or "", tool_calls=tool_calls)
    return None


def _convert_openai_system_role(msg: Dict[str, Any]) -> Optional[SystemMessage]:
    """Converts an OpenAI message dictionary with 'system' role to a SystemMessage."""
    content = msg.get("content")
    if isinstance(content, str) and content:
        return SystemMessage(content=content)
    return None


def _convert_openai_tool_role(msg: Dict[str, Any]) -> Optional[ToolMessage]:
    """Converts an OpenAI message dictionary with 'tool' role to a ToolMessage."""
    tool_call_id = msg.get("tool_call_id")
    name = msg.get("name")  # Relies on the non-standard 'name' field from the forward conversion.
    content = msg.get("content", "")

    if tool_call_id and name:
        return ToolMessage(content=str(content), tool_call_id=tool_call_id, name=name)
    return None


def _convert_single_message_from_openai(msg: Dict[str, Any]) -> Optional[BaseMessage]:
    """Dispatches a single OpenAI message dictionary to the appropriate generic conversion function."""
    role = msg.get("role")
    if role == "user":
        return _convert_openai_user_role(msg)
    if role == "assistant":
        return _convert_openai_assistant_role(msg)
    if role == "system":
        return _convert_openai_system_role(msg)
    if role == "tool":
        return _convert_openai_tool_role(msg)
    return None


def convert_from_openai_history(history: List[Dict[str, Any]]) -> List[BaseMessage]:
    """Converts an OpenAI-specific message history back to the generic format.

    Args:
        history: A list of message dictionaries from the OpenAI API.

    Returns:
        A list of generic BaseMessage objects.
    """
    generic_history: List[BaseMessage] = []
    for msg in history:
        converted_msg = _convert_single_message_from_openai(msg)
        if converted_msg:
            generic_history.append(converted_msg)
    return generic_history

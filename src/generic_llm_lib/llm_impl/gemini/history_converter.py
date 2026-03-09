"""
Handles the conversion between the generic `BaseMessage` format and the
Gemini-specific `types.Content` format.

This module provides a clear separation of concerns, isolating the logic
for data transformation from the core LLM interaction logic. It breaks down
the conversion process into small, single-responsibility functions to ensure
low cyclomatic complexity (Rank A) and high maintainability.
"""

from typing import List, Sequence, Optional, Union, Any
from google.genai import types

from generic_llm_lib.llm_core.messages import (
    BaseMessage,
    UserMessage,
    AssistantMessage,
    SystemMessage,
    ToolMessage,
)

# --- Conversion TO Gemini format ---


def _create_tool_call_part(tool_call: Union[types.FunctionCall, dict]) -> types.Part:
    """Creates a Gemini Part from a tool call (FunctionCall object or dict)."""
    if isinstance(tool_call, types.FunctionCall):
        return types.Part(function_call=tool_call)
    # Handle cases where tool_calls might be dicts
    return types.Part(function_call=types.FunctionCall(**tool_call))


def _convert_assistant_message(msg: AssistantMessage) -> types.Content:
    """Converts a generic AssistantMessage to a Gemini Content object."""
    parts = []
    if msg.content:
        parts.append(types.Part(text=msg.content))
    if msg.tool_calls:
        parts.extend([_create_tool_call_part(tc) for tc in msg.tool_calls])
    return types.Content(role="model", parts=parts)


def _convert_tool_message(msg: ToolMessage) -> types.Content:
    """Converts a generic ToolMessage to a Gemini Content object."""
    part = types.Part(
        function_response=types.FunctionResponse(
            name=msg.name,
            response={"result": msg.content},
        )
    )
    return types.Content(role="user", parts=[part])


def _convert_single_message_to_gemini(msg: BaseMessage) -> Optional[types.Content]:
    """Dispatches a single message to the appropriate conversion function."""
    if isinstance(msg, UserMessage):
        return types.Content(role="user", parts=[types.Part(text=msg.content)])
    if isinstance(msg, AssistantMessage):
        return _convert_assistant_message(msg)
    if isinstance(msg, ToolMessage):
        return _convert_tool_message(msg)
    # SystemMessage is ignored in history
    return None


def convert_to_gemini_history(history: List[BaseMessage]) -> List[types.Content]:
    """Converts a list of generic BaseMessage objects to a Gemini-specific history.

    Args:
        history: A list of BaseMessage objects.

    Returns:
        A list of Gemini Content objects.
    """
    gemini_history = []
    for msg in history:
        converted = _convert_single_message_to_gemini(msg)
        if converted:
            gemini_history.append(converted)
    return gemini_history


# --- Conversion FROM Gemini format ---


def _create_tool_message_from_response(tr: types.FunctionResponse) -> ToolMessage:
    """Creates a generic ToolMessage from a Gemini FunctionResponse."""
    response_content = tr.response
    # Extract 'result' if the response is wrapped in a dict (our convention)
    if isinstance(response_content, dict) and "result" in response_content:
        response_content = response_content["result"]

    return ToolMessage(
        content=str(response_content),
        tool_call_id="",  # Gemini does not use tool_call_id
        name=tr.name,
    )


def _extract_tool_responses(parts: List[types.Part]) -> List[ToolMessage]:
    """Extracts tool responses from a list of Gemini Parts."""
    tool_responses = [p.function_response for p in parts if p.function_response]
    return [_create_tool_message_from_response(tr) for tr in tool_responses]


def _extract_user_text_message(parts: List[types.Part]) -> List[UserMessage]:
    """Extracts a user text message from a list of Gemini Parts."""
    text_parts = [p.text for p in parts if p.text]
    if text_parts:
        return [UserMessage(content="".join(text_parts))]
    return []


def _convert_gemini_user_role(content: types.Content) -> List[BaseMessage]:
    """Converts a Gemini 'user' role Content object to generic messages.

    Handles both standard user text messages and tool responses (which share the 'user' role).
    """
    if not content.parts:
        return []

    # Case 1: Tool Responses
    tool_messages = _extract_tool_responses(content.parts)
    if tool_messages:
        return tool_messages  # type: ignore[return-value]

    # Case 2: User Text
    return _extract_user_text_message(content.parts)  # type: ignore[return-value]


def _extract_model_text(parts: List[types.Part]) -> str:
    """Extracts text content from model parts."""
    return "".join([p.text for p in parts if p.text])


def _extract_model_tool_calls(parts: List[types.Part]) -> Optional[List[types.FunctionCall]]:
    """Extracts tool calls from model parts."""
    tool_calls = [p.function_call for p in parts if p.function_call]
    return tool_calls if tool_calls else None


def _convert_gemini_model_role(content: types.Content) -> Optional[AssistantMessage]:
    """Converts a Gemini 'model' role Content object to an AssistantMessage."""
    if not content.parts:
        return None

    text_content = _extract_model_text(content.parts)
    tool_calls = _extract_model_tool_calls(content.parts)

    if text_content or tool_calls:
        return AssistantMessage(
            content=text_content,
            tool_calls=tool_calls,  # type: ignore[arg-type]
        )
    return None


def convert_from_gemini_history(history: Sequence[types.Content]) -> List[BaseMessage]:
    """Converts a Gemini-specific message history back to the generic format.

    Args:
        history: A sequence of Gemini Content objects.

    Returns:
        A list of generic BaseMessage objects.
    """
    generic_history: List[BaseMessage] = []
    for content in history:
        if content.role == "user":
            generic_history.extend(_convert_gemini_user_role(content))
        elif content.role == "model":
            msg = _convert_gemini_model_role(content)
            if msg:
                generic_history.append(msg)
    return generic_history

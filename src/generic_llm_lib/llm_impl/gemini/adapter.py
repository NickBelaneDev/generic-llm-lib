"""Translate Gemini tool-calling payloads into the library's provider-agnostic tool protocol."""

from google.genai import types
from typing import Any, Sequence, cast
from google.genai.types import GenerateContentResponse
from generic_llm_lib.llm_core import ToolCallRequest, ToolCallResult, ToolAdapter
from generic_llm_lib.llm_core import get_logger

logger = get_logger(__name__)


class GeminiToolAdapter(ToolAdapter):
    """Adapter for Gemini tool handling."""

    def __init__(self, chat_session: Any):
        """Initialize the Gemini tool adapter.

        Args:
            chat_session: The Gemini chat session instance.
        """
        self.chat_session = chat_session

    def get_tool_calls(self, response: GenerateContentResponse) -> Sequence[ToolCallRequest]:
        """Extract tool calls from a Gemini content response.

        Args:
            response: The content response from Gemini.

        Returns:
            A sequence of tool call requests extracted from the response.
        """
        function_calls = [p.function_call for p in (response.parts or []) if p.function_call]
        return [
            ToolCallRequest(
                name=cast(str, function_call.name),
                arguments=getattr(function_call, "args", None),
            )
            for function_call in function_calls
        ]

    def record_assistant_message(self, response: GenerateContentResponse) -> None:
        """Records the assistant's message.

        Gemini handles history internally in the chat session, so this is a no-op.

        Args:
            response: The content response from Gemini.
        """
        pass

    def build_tool_response_message(self, result: ToolCallResult) -> types.Part:
        """Build a tool response part for the Gemini API.

        Args:
            result: The result of a tool call.

        Returns:
            A Gemini Part object containing the function response.
        """
        return types.Part(
            function_response=types.FunctionResponse(
                name=result.name,
                response=result.response,
            )
        )

    async def send_tool_responses(self, messages: Sequence[types.Part]) -> GenerateContentResponse:
        """Send tool response parts back to the Gemini API and get a new response.

        Args:
            messages: A sequence of Gemini Part objects containing tool responses.

        Returns:
            The next content response from Gemini.
        """
        response = await self.chat_session.send_message(list(messages))
        return cast(GenerateContentResponse, response)

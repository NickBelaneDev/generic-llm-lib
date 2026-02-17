from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionToolParam
from typing import List, Optional, Any, Dict, Sequence, Iterable, cast
import json
from generic_llm_lib.llm_core.tools.call_protocol import ToolCallRequest, ToolCallResult
from generic_llm_lib.llm_core.tools.adapter import ToolAdapter


class OpenAIToolAdapter(ToolAdapter):
    """Adapter for OpenAI tool handling."""

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[Iterable[ChatCompletionToolParam]],
        temperature: float,
        max_tokens: int,
    ):
        """Initialize the OpenAI tool adapter.

        Args:
            client: The OpenAI client instance.
            model: The name of the model to use.
            messages: The conversation history.
            tools: Optional list of tool definitions.
            temperature: Sampling temperature.
            max_tokens: Maximum number of tokens to generate.
        """
        self.client = client
        self.model = model
        self.messages = messages
        self.tools = tools
        self.temperature = temperature
        self.max_tokens = max_tokens

    def get_tool_calls(self, response: ChatCompletion) -> Sequence[ToolCallRequest]:
        """Extract tool calls from an OpenAI chat completion response.

        Args:
            response: The chat completion response from OpenAI.

        Returns:
            A sequence of tool call requests extracted from the response.
        """
        if not response.choices:
            return []

        tool_calls = response.choices[0].message.tool_calls
        if not tool_calls:
            return []

        requests = []
        for tool_call in tool_calls:
            # Check if it's a function tool call (has 'function' attribute)
            if tool_call.type == "function":
                requests.append(
                    ToolCallRequest(
                        name=tool_call.function.name,
                        arguments=tool_call.function.arguments,
                        call_id=tool_call.id,
                    )
                )
        return requests

    def record_assistant_message(self, response: ChatCompletion) -> None:
        """Record the assistant's message from the response into the message history.

        Args:
            response: The chat completion response containing the assistant's message.
        """
        self.messages.append(response.choices[0].message.model_dump())

    def build_tool_response_message(self, result: ToolCallResult) -> Dict[str, Any]:
        """Build a tool response message for the OpenAI API.

        Args:
            result: The result of a tool call.

        Returns:
            A dictionary representing the tool response message.
        """
        return {
            "role": "tool",
            "tool_call_id": result.call_id,
            "name": result.name,
            "content": json.dumps(result.response),
        }

    async def send_tool_responses(self, tool_messages: Sequence[Dict[str, Any]]) -> ChatCompletion:
        """Send tool responses back to the OpenAI API and get a new completion.

        Args:
            tool_messages: A sequence of tool response messages.

        Returns:
            The next chat completion response from OpenAI.
        """
        self.messages.extend(tool_messages)

        # We need to cast messages to Iterable[Any] because the library expects a specific union of message types
        # but we are using List[Dict[str, Any]] which is structurally compatible.
        return await self.client.chat.completions.create(
            model=self.model,
            messages=cast(Iterable[Any], self.messages),
            tools=self.tools,  # type: ignore[arg-type]
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

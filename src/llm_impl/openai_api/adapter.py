from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from typing import List, Optional, Any, Dict, Sequence
import json
from llm_core.tools import ToolCallRequest, ToolCallResult, ToolAdapter



class OpenAIToolAdapter(ToolAdapter):
    """Adapter for OpenAI tool handling."""

    def __init__(self,
                 client: AsyncOpenAI,
                 model: str,
                 messages: List[Dict[str, Any]],
                 tools: Optional[List[Dict[str, Any]]],
                 temperature: float,
                 max_tokens: int):
        self.client = client
        self.model = model
        self.messages = messages
        self.tools = tools
        self.temperature = temperature
        self.max_tokens = max_tokens

    def get_tool_calls(self, response: ChatCompletion) -> Sequence[ToolCallRequest]:
        if not response.choices:
            return []

        tool_calls = response.choices[0].message.tool_calls
        if not tool_calls:
            return []

        return [
            ToolCallRequest(
                name=tool_call.function.name,
                arguments=tool_call.function.arguments,
                call_id=tool_call.id,
            )
            for tool_call in tool_calls
        ]

    def record_assistant_message(self, response: ChatCompletion) -> None:
        self.messages.append(response.choices[0].message.model_dump())

    def build_tool_response_message(self, result: ToolCallResult) -> Dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": result.call_id,
            "name": result.name,
            "content": json.dumps(result.response),
        }

    async def send_tool_responses(self, tool_messages: Sequence[Dict[str, Any]]) -> ChatCompletion:
        self.messages.extend(tool_messages)
        return await self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.tools,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )


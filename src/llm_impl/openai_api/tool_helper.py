from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from typing import List, Tuple, Optional, Any, Dict, Sequence
import json
from llm_core import ToolRegistry
from llm_core.tool_loop import ToolExecutionLoop, ToolCallRequest, ToolCallResult


class ToolHelper:
    def __init__(self,
                 client: AsyncOpenAI,
                 model: str,
                 registry: Optional[ToolRegistry],
                 temperature: float,
                 max_tokens: int,
                 max_function_loops: int,
                 tool_timeout: float = 60.0):
        self.client = client
        self.model = model
        self.registry = registry
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_function_loops = max_function_loops
        self.tool_timeout = tool_timeout
        self._tool_loop = ToolExecutionLoop(
            registry=registry,
            max_function_loops=max_function_loops,
            tool_timeout=tool_timeout,
            argument_error_formatter=self._format_argument_error,
        )

    async def handle_function_calls(self,
                                    messages: List[Dict[str, Any]],
                                    initial_response: ChatCompletion) -> Tuple[List[Dict[str, Any]], ChatCompletion]:
        tools = self.registry.tool_object if self.registry else None

        if not initial_response.choices:
            return messages, initial_response

        final_response = await self._tool_loop.run(
            initial_response=initial_response,
            get_tool_calls=self._get_tool_calls,
            record_assistant_message=lambda response: messages.append(
                response.choices[0].message.model_dump()
            ),
            build_tool_response_message=self._build_tool_response_message,
            send_tool_responses=lambda tool_messages: self._send_tool_messages(
                messages,
                tool_messages,
                tools,
            ),
        )

        return messages, final_response

    def _get_tool_calls(self, response: ChatCompletion) -> Sequence[ToolCallRequest]:
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

    @staticmethod
    def _build_tool_response_message(tool_result: ToolCallResult) -> Dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": tool_result.call_id,
            "name": tool_result.name,
            "content": json.dumps(tool_result.response),
        }

    async def _send_tool_messages(
        self,
        messages: List[Dict[str, Any]],
        tool_messages: Sequence[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
    ) -> ChatCompletion:
        messages.extend(tool_messages)
        return await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    @staticmethod
    def _format_argument_error(tool_name: str, error: Exception) -> str:
        return f"Failed to decode function arguments: {error}"

import pytest
from unittest.mock import AsyncMock
from pydantic import BaseModel
from typing import Sequence, Any

from llm_core.tools import ToolExecutionLoop, ToolCallRequest, ToolCallResult, ToolAdapter
from llm_impl.openai_api.registry import OpenAIToolRegistry
from llm_core.tools.models import ToolDefinition


class SampleArgs(BaseModel):
    required_value: int


class MockAdapter(ToolAdapter):
    def __init__(self, get_tool_calls_func, record_func, build_func, send_func):
        self.get_tool_calls_func = get_tool_calls_func
        self.record_func = record_func
        self.build_func = build_func
        self.send_func = send_func

    def get_tool_calls(self, response: Any) -> Sequence[ToolCallRequest]:
        return self.get_tool_calls_func(response)

    def record_assistant_message(self, response: Any) -> None:
        self.record_func(response)

    def build_tool_response_message(self, result: ToolCallResult) -> Any:
        return self.build_func(result)

    async def send_tool_responses(self, messages: Sequence[Any]) -> Any:
        return await self.send_func(messages)


@pytest.mark.asyncio
async def test_tool_execution_loop_runs_tools():
    registry = OpenAIToolRegistry()
    tool_func = AsyncMock(return_value="ok")
    registry.tools["sample"] = ToolDefinition(
        name="sample",
        description="sample tool",
        func=tool_func,
    )

    loop = ToolExecutionLoop(
        registry=registry,
        max_function_loops=2,
    )

    initial_response = {"tool_calls": [ToolCallRequest(name="sample", arguments={"a": 1})]}
    final_response = {"tool_calls": []}
    recorded = []
    captured_tool_results = []

    def get_tool_calls(response):
        return response["tool_calls"]

    def record_assistant_message(response):
        recorded.append(response)

    def build_tool_response_message(tool_result: ToolCallResult):
        captured_tool_results.append(tool_result)
        return tool_result

    async def send_tool_responses(tool_messages):
        return final_response

    adapter = MockAdapter(
        get_tool_calls,
        record_assistant_message,
        build_tool_response_message,
        send_tool_responses
    )

    response = await loop.run(
        initial_response=initial_response,
        adapter=adapter
    )

    tool_func.assert_called_once_with(a=1)
    assert response == final_response
    assert recorded == [initial_response, final_response]
    assert captured_tool_results[0].response == {"result": "ok"}


@pytest.mark.asyncio
async def test_tool_execution_loop_handles_invalid_arguments():
    registry = OpenAIToolRegistry()

    async def tool_func(required_value: int) -> int:
        return required_value

    registry.tools["validated"] = ToolDefinition(
        name="validated",
        description="validated tool",
        func=tool_func,
        args_model=SampleArgs,
    )

    loop = ToolExecutionLoop(
        registry=registry,
        max_function_loops=1,
    )

    initial_response = {"tool_calls": [ToolCallRequest(name="validated", arguments={"required_value": "bad"})]}
    recorded = []
    tool_results = []

    def get_tool_calls(response):
        return response["tool_calls"]

    def record_assistant_message(response):
        recorded.append(response)

    def build_tool_response_message(tool_result: ToolCallResult):
        tool_results.append(tool_result)
        return tool_result

    async def send_tool_responses(tool_messages):
        return {"tool_calls": []}

    adapter = MockAdapter(
        get_tool_calls,
        record_assistant_message,
        build_tool_response_message,
        send_tool_responses
    )

    await loop.run(
        initial_response=initial_response,
        adapter=adapter
    )

    assert tool_results
    error_message = tool_results[0].response["error"]
    assert "Argument validation failed:" in error_message
    assert "required_value" in error_message

import pytest
from unittest.mock import AsyncMock
from pydantic import BaseModel

from llm_core.tool_loop import ToolExecutionLoop, ToolCallRequest, ToolCallResult
from llm_impl.open_api.registry import OpenAIToolRegistry
from llm_core.types import ToolDefinition


class SampleArgs(BaseModel):
    required_value: int


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

    response = await loop.run(
        initial_response=initial_response,
        get_tool_calls=get_tool_calls,
        record_assistant_message=record_assistant_message,
        build_tool_response_message=build_tool_response_message,
        send_tool_responses=send_tool_responses,
    )

    tool_func.assert_called_once_with(a=1)
    assert response == final_response
    assert recorded == [initial_response]
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

    await loop.run(
        initial_response=initial_response,
        get_tool_calls=get_tool_calls,
        record_assistant_message=record_assistant_message,
        build_tool_response_message=build_tool_response_message,
        send_tool_responses=send_tool_responses,
    )

    assert tool_results
    error_message = tool_results[0].response["error"]
    assert "Argument validation failed:" in error_message
    assert "required_value" in error_message

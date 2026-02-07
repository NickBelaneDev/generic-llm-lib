import pytest
from unittest.mock import MagicMock, AsyncMock
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage
from llm_impl.openai_api.core import GenericOpenAI
from llm_impl.openai_api.models import OpenAIChatResponse, OpenAIMessageResponse
from llm_impl.openai_api.registry import OpenAIToolRegistry
import json
from typing import Any


@pytest.fixture
def mock_openai_client() -> Any:
    client = MagicMock(spec=AsyncOpenAI)
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock()
    return client


@pytest.mark.asyncio
async def test_generic_openai_initialization(mock_openai_client: Any) -> None:
    openai_llm = GenericOpenAI(client=mock_openai_client, model_name="gpt-4", sys_instruction="You are a helper.")
    assert openai_llm.model == "gpt-4"
    assert openai_llm.client == mock_openai_client
    assert openai_llm.sys_instruction == "You are a helper."


@pytest.mark.asyncio
async def test_ask_method(mock_openai_client: Any) -> None:
    # Mock response
    mock_message = MagicMock(spec=ChatCompletionMessage)
    mock_message.content = "Hello world"
    mock_message.tool_calls = None
    mock_message.model_dump.return_value = {"role": "assistant", "content": "Hello world"}

    mock_choice = MagicMock(spec=Choice)
    mock_choice.message = mock_message
    mock_choice.finish_reason = "stop"

    mock_usage = MagicMock(spec=CompletionUsage)
    mock_usage.prompt_tokens = 5
    mock_usage.completion_tokens = 10
    mock_usage.total_tokens = 15

    mock_response = MagicMock(spec=ChatCompletion)
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage

    mock_openai_client.chat.completions.create.return_value = mock_response

    openai_llm = GenericOpenAI(client=mock_openai_client, model_name="gpt-4", sys_instruction="You are a helper.")

    # Execute
    response = await openai_llm.ask("Hello")

    # Verify
    assert isinstance(response, OpenAIMessageResponse)
    assert response.text == "Hello world"
    assert response.tokens.total_tokens == 15

    # Verify call arguments
    # Note: The messages list is modified in-place by the chat method to include the response.
    # So we expect the final state of the list.
    call_args = mock_openai_client.chat.completions.create.call_args
    assert call_args.kwargs["messages"] == [
        {"role": "system", "content": "You are a helper."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hello world"},
    ]


@pytest.mark.asyncio
async def test_chat_method(mock_openai_client: Any) -> None:
    # Mock response
    mock_message = MagicMock(spec=ChatCompletionMessage)
    mock_message.content = "Chat response"
    mock_message.tool_calls = None
    mock_message.model_dump.return_value = {"role": "assistant", "content": "Chat response"}

    mock_choice = MagicMock(spec=Choice)
    mock_choice.message = mock_message
    mock_choice.finish_reason = "stop"

    mock_response = MagicMock(spec=ChatCompletion)
    mock_response.choices = [mock_choice]
    mock_response.usage = None

    mock_openai_client.chat.completions.create.return_value = mock_response

    openai_llm = GenericOpenAI(client=mock_openai_client, model_name="gpt-4", sys_instruction="You are a helper.")

    # Execute
    response = await openai_llm.chat([], "Hi")

    # Verify
    assert isinstance(response, OpenAIChatResponse)
    assert response.last_response.text == "Chat response"
    # History should contain system (if added internally), user prompt, and assistant response
    # The implementation adds system prompt if history is empty
    assert len(response.history) == 3
    assert response.history[0]["role"] == "system"
    assert response.history[1]["role"] == "user"
    assert response.history[2]["role"] == "assistant"


@pytest.mark.asyncio
async def test_function_calling(mock_openai_client: Any) -> None:
    # Setup registry with a mock tool
    registry = OpenAIToolRegistry()
    mock_tool = AsyncMock(return_value="Tool Result")
    mock_tool.__name__ = "test_tool"

    # Manually register to avoid pydantic complexity in test
    mock_tool_def = MagicMock()
    mock_tool_def.name = "test_tool"
    mock_tool_def.func = mock_tool
    mock_tool_def.parameters = {}
    mock_tool_def.description = "Test tool"
    mock_tool_def.args_model = None

    registry.tools["test_tool"] = mock_tool_def

    # First response triggers function call
    # We use a simple MagicMock without spec to avoid attribute issues with nested properties
    tool_call = MagicMock()
    tool_call.id = "call_123"
    tool_call.type = "function"
    tool_call.function.name = "test_tool"
    tool_call.function.arguments = '{"arg": "val"}'

    msg1 = MagicMock(spec=ChatCompletionMessage)
    msg1.content = None
    msg1.tool_calls = [tool_call]
    msg1.model_dump.return_value = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "call_123", "function": {"name": "test_tool", "arguments": '{"arg": "val"}'}, "type": "function"}
        ],
    }

    choice1 = MagicMock(spec=Choice)
    choice1.message = msg1
    choice1.finish_reason = "tool_calls"

    response1 = MagicMock(spec=ChatCompletion)
    response1.choices = [choice1]
    # Add usage to response1 to avoid AttributeError
    response1.usage = None

    # Second response is final answer
    msg2 = MagicMock(spec=ChatCompletionMessage)
    msg2.content = "Final answer"
    msg2.tool_calls = None
    msg2.model_dump.return_value = {"role": "assistant", "content": "Final answer"}

    choice2 = MagicMock(spec=Choice)
    choice2.message = msg2
    choice2.finish_reason = "stop"

    response2 = MagicMock(spec=ChatCompletion)
    response2.choices = [choice2]
    response2.usage = None

    # Configure mock to return sequence of responses
    mock_openai_client.chat.completions.create.side_effect = [response1, response2]

    openai_llm = GenericOpenAI(
        client=mock_openai_client, model_name="gpt-4", sys_instruction="System", registry=registry
    )

    # Execute
    await openai_llm.chat([], "Call tool")

    # Verify tool was called
    mock_tool.assert_called_once_with(arg="val")

    # Verify chat sent function result back
    assert mock_openai_client.chat.completions.create.call_count == 2

    # Check the messages sent in the second call (which includes the tool result)
    second_call_args = mock_openai_client.chat.completions.create.call_args_list[1]
    messages_sent = second_call_args.kwargs["messages"]

    # Expected: System, User, Assistant (Tool Call), Tool (Result)
    # BUT: The list is modified in place after the call to add the final response.
    # So we see 5 messages.
    assert len(messages_sent) == 5
    assert messages_sent[2]["role"] == "assistant"
    assert messages_sent[3]["role"] == "tool"
    assert messages_sent[3]["tool_call_id"] == "call_123"
    assert json.loads(messages_sent[3]["content"]) == {"result": "Tool Result"}
    # The final answer added after the second call
    assert messages_sent[4]["role"] == "assistant"
    assert messages_sent[4]["content"] == "Final answer"

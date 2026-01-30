import pytest
from unittest.mock import MagicMock, AsyncMock
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from llm_impl.open_api.tool_helper import ToolHelper
from llm_core import ToolRegistry
import json

@pytest.fixture
def mock_openai_client():
    client = MagicMock(spec=AsyncOpenAI)
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock()
    return client

@pytest.fixture
def mock_registry():
    registry = MagicMock(spec=ToolRegistry)
    registry.tool_object = [{"type": "function", "function": {"name": "test_tool"}}]
    registry.tools = {}
    return registry

@pytest.mark.asyncio
async def test_tool_helper_initialization(mock_openai_client, mock_registry):
    helper = ToolHelper(
        client=mock_openai_client,
        model="gpt-4",
        registry=mock_registry,
        temperature=0.7,
        max_tokens=100,
        max_function_loops=5
    )
    assert helper.client == mock_openai_client
    assert helper.model == "gpt-4"
    assert helper.registry == mock_registry
    assert helper.temperature == 0.7
    assert helper.max_tokens == 100
    assert helper.max_function_loops == 5

@pytest.mark.asyncio
async def test_handle_function_calls_no_calls(mock_openai_client, mock_registry):
    helper = ToolHelper(
        client=mock_openai_client,
        model="gpt-4",
        registry=mock_registry,
        temperature=0.7,
        max_tokens=100,
        max_function_loops=5
    )

    # Mock response with no tool calls
    mock_message = MagicMock(spec=ChatCompletionMessage)
    mock_message.content = "Hello"
    mock_message.tool_calls = None
    mock_message.model_dump.return_value = {"role": "assistant", "content": "Hello"}

    mock_choice = MagicMock(spec=Choice)
    mock_choice.message = mock_message

    mock_response = MagicMock(spec=ChatCompletion)
    mock_response.choices = [mock_choice]

    messages = [{"role": "user", "content": "Hi"}]
    
    updated_messages, final_response = await helper.handle_function_calls(messages, mock_response)

    assert len(updated_messages) == 2
    assert updated_messages[-1]["content"] == "Hello"
    assert final_response == mock_response
    mock_openai_client.chat.completions.create.assert_not_called()

@pytest.mark.asyncio
async def test_handle_function_calls_execution(mock_openai_client, mock_registry):
    # Setup mock tool
    mock_tool_func = AsyncMock(return_value="Success")
    mock_tool_def = MagicMock()
    mock_tool_def.func = mock_tool_func
    mock_tool_def.args_model = None
    mock_registry.tools = {"test_tool": mock_tool_def}

    helper = ToolHelper(
        client=mock_openai_client,
        model="gpt-4",
        registry=mock_registry,
        temperature=0.7,
        max_tokens=100,
        max_function_loops=5
    )

    # First response triggers tool call
    tool_call = MagicMock()
    tool_call.id = "call_1"
    tool_call.function.name = "test_tool"
    tool_call.function.arguments = '{"arg": "value"}'

    msg1 = MagicMock(spec=ChatCompletionMessage)
    msg1.tool_calls = [tool_call]
    msg1.model_dump.return_value = {
        "role": "assistant",
        "tool_calls": [{"id": "call_1", "function": {"name": "test_tool", "arguments": '{"arg": "value"}'}}]
    }

    choice1 = MagicMock(spec=Choice)
    choice1.message = msg1
    
    response1 = MagicMock(spec=ChatCompletion)
    response1.choices = [choice1]

    # Second response (final)
    msg2 = MagicMock(spec=ChatCompletionMessage)
    msg2.content = "Done"
    msg2.tool_calls = None
    msg2.model_dump.return_value = {"role": "assistant", "content": "Done"}

    choice2 = MagicMock(spec=Choice)
    choice2.message = msg2

    response2 = MagicMock(spec=ChatCompletion)
    response2.choices = [choice2]

    mock_openai_client.chat.completions.create.return_value = response2

    messages = [{"role": "user", "content": "Run tool"}]
    
    updated_messages, final_response = await helper.handle_function_calls(messages, response1)

    # Verify tool execution
    mock_tool_func.assert_called_once_with(arg="value")
    
    # Verify messages structure
    # 1. User
    # 2. Assistant (Tool Call)
    # 3. Tool (Result)
    # 4. Assistant (Final)
    assert len(updated_messages) == 4
    assert updated_messages[1]["role"] == "assistant"
    assert updated_messages[2]["role"] == "tool"
    assert updated_messages[2]["tool_call_id"] == "call_1"
    assert json.loads(updated_messages[2]["content"]) == {"result": "Success"}
    assert updated_messages[3]["role"] == "assistant"
    assert updated_messages[3]["content"] == "Done"

@pytest.mark.asyncio
async def test_handle_function_calls_error(mock_openai_client, mock_registry):
    # Setup mock tool that raises exception
    mock_tool_func = AsyncMock(side_effect=Exception("Tool failed"))
    mock_tool_def = MagicMock()
    mock_tool_def.func = mock_tool_func
    mock_tool_def.args_model = None
    mock_registry.tools = {"test_tool": mock_tool_def}

    helper = ToolHelper(
        client=mock_openai_client,
        model="gpt-4",
        registry=mock_registry,
        temperature=0.7,
        max_tokens=100,
        max_function_loops=5
    )

    # Response triggers tool call
    tool_call = MagicMock()
    tool_call.id = "call_1"
    tool_call.function.name = "test_tool"
    tool_call.function.arguments = '{}'

    msg1 = MagicMock(spec=ChatCompletionMessage)
    msg1.tool_calls = [tool_call]
    msg1.model_dump.return_value = {
        "role": "assistant",
        "tool_calls": [{"id": "call_1", "function": {"name": "test_tool", "arguments": '{}'}}]
    }

    choice1 = MagicMock(spec=Choice)
    choice1.message = msg1
    
    response1 = MagicMock(spec=ChatCompletion)
    response1.choices = [choice1]

    # Final response
    msg2 = MagicMock(spec=ChatCompletionMessage)
    msg2.content = "Error handled"
    msg2.tool_calls = None
    msg2.model_dump.return_value = {"role": "assistant", "content": "Error handled"}

    choice2 = MagicMock(spec=Choice)
    choice2.message = msg2

    response2 = MagicMock(spec=ChatCompletion)
    response2.choices = [choice2]

    mock_openai_client.chat.completions.create.return_value = response2

    messages = []
    updated_messages, _ = await helper.handle_function_calls(messages, response1)

    # Verify error message in tool output
    # Index 0: Assistant (Tool Call)
    # Index 1: Tool (Error)
    # Index 2: Assistant (Final)
    assert updated_messages[1]["role"] == "tool"
    content = json.loads(updated_messages[1]["content"])
    assert "error" in content
    assert content["error"] == "Tool failed"

@pytest.mark.asyncio
async def test_handle_function_calls_empty_arguments(mock_openai_client, mock_registry):
    mock_tool_func = AsyncMock(return_value="Success")
    mock_tool_def = MagicMock()
    mock_tool_def.func = mock_tool_func
    mock_tool_def.args_model = None
    mock_registry.tools = {"test_tool": mock_tool_def}

    helper = ToolHelper(
        client=mock_openai_client,
        model="gpt-4",
        registry=mock_registry,
        temperature=0.7,
        max_tokens=100,
        max_function_loops=5
    )

    tool_call = MagicMock()
    tool_call.id = "call_empty"
    tool_call.function.name = "test_tool"
    tool_call.function.arguments = ""

    msg1 = MagicMock(spec=ChatCompletionMessage)
    msg1.tool_calls = [tool_call]
    msg1.model_dump.return_value = {
        "role": "assistant",
        "tool_calls": [{"id": "call_empty", "function": {"name": "test_tool", "arguments": ""}}]
    }

    choice1 = MagicMock(spec=Choice)
    choice1.message = msg1

    response1 = MagicMock(spec=ChatCompletion)
    response1.choices = [choice1]

    msg2 = MagicMock(spec=ChatCompletionMessage)
    msg2.content = "Done"
    msg2.tool_calls = None
    msg2.model_dump.return_value = {"role": "assistant", "content": "Done"}

    choice2 = MagicMock(spec=Choice)
    choice2.message = msg2

    response2 = MagicMock(spec=ChatCompletion)
    response2.choices = [choice2]

    mock_openai_client.chat.completions.create.return_value = response2

    messages = [{"role": "user", "content": "Run tool"}]
    updated_messages, _ = await helper.handle_function_calls(messages, response1)

    mock_tool_func.assert_called_once_with()
    assert updated_messages[2]["role"] == "tool"
    assert json.loads(updated_messages[2]["content"]) == {"result": "Success"}

@pytest.mark.asyncio
async def test_handle_function_calls_invalid_arguments(mock_openai_client, mock_registry):
    mock_tool_func = AsyncMock(return_value="Success")
    mock_tool_def = MagicMock()
    mock_tool_def.func = mock_tool_func
    mock_tool_def.args_model = None
    mock_registry.tools = {"test_tool": mock_tool_def}

    helper = ToolHelper(
        client=mock_openai_client,
        model="gpt-4",
        registry=mock_registry,
        temperature=0.7,
        max_tokens=100,
        max_function_loops=5
    )

    tool_call = MagicMock()
    tool_call.id = "call_invalid"
    tool_call.function.name = "test_tool"
    tool_call.function.arguments = '["not", "object"]'

    msg1 = MagicMock(spec=ChatCompletionMessage)
    msg1.tool_calls = [tool_call]
    msg1.model_dump.return_value = {
        "role": "assistant",
        "tool_calls": [{"id": "call_invalid", "function": {"name": "test_tool", "arguments": '["not", "object"]'}}]
    }

    choice1 = MagicMock(spec=Choice)
    choice1.message = msg1

    response1 = MagicMock(spec=ChatCompletion)
    response1.choices = [choice1]

    msg2 = MagicMock(spec=ChatCompletionMessage)
    msg2.content = "Done"
    msg2.tool_calls = None
    msg2.model_dump.return_value = {"role": "assistant", "content": "Done"}

    choice2 = MagicMock(spec=Choice)
    choice2.message = msg2

    response2 = MagicMock(spec=ChatCompletion)
    response2.choices = [choice2]

    mock_openai_client.chat.completions.create.return_value = response2

    messages = [{"role": "user", "content": "Run tool"}]
    updated_messages, _ = await helper.handle_function_calls(messages, response1)

    mock_tool_func.assert_not_called()
    assert updated_messages[2]["role"] == "tool"
    content = json.loads(updated_messages[2]["content"])
    assert "error" in content

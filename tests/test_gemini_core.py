import pytest
from unittest.mock import MagicMock, AsyncMock
from llm_impl.gemini.core import GenericGemini
from llm_impl.gemini.models import GeminiChatResponse, GeminiMessageResponse
from llm_impl.gemini.registry import GeminiToolRegistry
from typing import Any


@pytest.mark.asyncio
async def test_generic_gemini_initialization(mock_genai_client: Any) -> None:
    gemini = GenericGemini(client=mock_genai_client, model_name="gemini-pro", sys_instruction="You are a helper.")
    assert gemini.model == "gemini-pro"
    assert gemini.client == mock_genai_client
    assert gemini.config.system_instruction == "You are a helper."


@pytest.mark.asyncio
async def test_ask_method(mock_genai_client: Any, mock_chat_session: Any) -> None:
    # Setup
    mock_genai_client.aio.chats.create.return_value = mock_chat_session

    # Mock response
    mock_response = MagicMock()
    mock_response.parts = [MagicMock(text="Hello world", function_call=None)]
    mock_response.usage_metadata = MagicMock(
        candidates_token_count=10,
        prompt_token_count=5,
        total_token_count=15,
        thoughts_token_count=0,
        tool_use_prompt_token_count=0,
    )
    mock_chat_session.send_message = AsyncMock(return_value=mock_response)

    gemini = GenericGemini(client=mock_genai_client, model_name="gemini-pro", sys_instruction="You are a helper.")

    # Execute
    response = await gemini.ask("Hello")

    # Verify
    assert isinstance(response, GeminiMessageResponse)
    assert response.text == "Hello world"
    assert response.tokens.total_token_count == 15
    mock_chat_session.send_message.assert_called_once_with("Hello")


@pytest.mark.asyncio
async def test_chat_method(mock_genai_client: Any, mock_chat_session: Any) -> None:
    # Setup
    mock_genai_client.aio.chats.create.return_value = mock_chat_session

    mock_response = MagicMock()
    mock_response.parts = [MagicMock(text="Chat response", function_call=None)]
    mock_response.usage_metadata = None
    mock_chat_session.send_message = AsyncMock(return_value=mock_response)

    # Mock history content to satisfy Pydantic validation
    mock_content = MagicMock()
    mock_content.role = "user"
    mock_content.parts = []
    mock_chat_session.get_history.return_value = [mock_content]

    gemini = GenericGemini(client=mock_genai_client, model_name="gemini-pro", sys_instruction="You are a helper.")

    # Execute
    response = await gemini.chat([], "Hi")

    # Verify
    assert isinstance(response, GeminiChatResponse)
    assert response.last_response.text == "Chat response"
    assert len(response.history) == 1


@pytest.mark.asyncio
async def test_function_calling(mock_genai_client: Any, mock_chat_session: Any) -> None:
    # Setup registry with a mock tool
    registry = GeminiToolRegistry()
    mock_tool = AsyncMock(return_value="Tool Result")
    mock_tool.__name__ = "test_tool"

    # Manually register to avoid pydantic complexity in test
    # We need to mock the tool definition because we are bypassing the decorator
    mock_tool_def = MagicMock()
    mock_tool_def.name = "test_tool"
    mock_tool_def.func = mock_tool
    mock_tool_def.parameters = {}  # Mock parameters
    mock_tool_def.description = "Test tool"
    mock_tool_def.args_model = None  # Mock args_model

    registry.tools["test_tool"] = mock_tool_def

    mock_genai_client.aio.chats.create.return_value = mock_chat_session

    # First response triggers function call
    response1 = MagicMock()
    func_call = MagicMock()
    func_call.name = "test_tool"
    func_call.args = {"arg": "val"}
    response1.parts = [MagicMock(function_call=func_call, text=None)]

    # Second response is final answer
    response2 = MagicMock()
    response2.parts = [MagicMock(text="Final answer", function_call=None)]
    response2.usage_metadata = None

    # Configure mock to return sequence of responses
    mock_chat_session.send_message = AsyncMock(side_effect=[response1, response2])

    # Mock history for the chat response construction
    mock_content = MagicMock()
    mock_content.role = "model"
    mock_content.parts = []
    mock_chat_session.get_history.return_value = [mock_content]

    gemini = GenericGemini(
        client=mock_genai_client, model_name="gemini-pro", sys_instruction="System", registry=registry
    )

    # Execute
    await gemini.chat([], "Call tool")

    # Verify tool was called
    mock_tool.assert_called_once_with(arg="val")
    # Verify chat sent function result back
    assert mock_chat_session.send_message.call_count == 2


@pytest.mark.asyncio
async def test_function_calling_with_empty_args(mock_genai_client: Any, mock_chat_session: Any) -> None:
    registry = GeminiToolRegistry()
    mock_tool = AsyncMock(return_value="Tool Result")
    mock_tool.__name__ = "empty_args_tool"

    mock_tool_def = MagicMock()
    mock_tool_def.name = "empty_args_tool"
    mock_tool_def.func = mock_tool
    mock_tool_def.parameters = {}
    mock_tool_def.description = "Tool with empty args"
    mock_tool_def.args_model = None

    registry.tools["empty_args_tool"] = mock_tool_def

    mock_genai_client.aio.chats.create.return_value = mock_chat_session

    response1 = MagicMock()
    func_call = MagicMock()
    func_call.name = "empty_args_tool"
    func_call.args = None
    response1.parts = [MagicMock(function_call=func_call, text=None)]

    response2 = MagicMock()
    response2.parts = [MagicMock(text="Final answer", function_call=None)]
    response2.usage_metadata = None

    mock_chat_session.send_message = AsyncMock(side_effect=[response1, response2])

    mock_content = MagicMock()
    mock_content.role = "model"
    mock_content.parts = []
    mock_chat_session.get_history.return_value = [mock_content]

    gemini = GenericGemini(
        client=mock_genai_client, model_name="gemini-pro", sys_instruction="System", registry=registry
    )

    await gemini.chat([], "Call tool without args")

    mock_tool.assert_called_once_with()
    assert mock_chat_session.send_message.call_count == 2

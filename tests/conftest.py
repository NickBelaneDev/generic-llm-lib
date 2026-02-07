import pytest
from unittest.mock import MagicMock, AsyncMock
from typing import Any


@pytest.fixture
def mock_genai_client() -> Any:
    client = MagicMock()
    client.chats.create.return_value = MagicMock()
    client.aio.chats.create.return_value = MagicMock()
    return client


@pytest.fixture
def mock_chat_session() -> Any:
    chat = MagicMock()
    chat.send_message.return_value = MagicMock()
    chat.get_history.return_value = []
    return chat


@pytest.fixture
def mock_openai_client() -> Any:
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock()
    return client

import pytest
from unittest.mock import MagicMock, AsyncMock
from google.genai import types

@pytest.fixture
def mock_genai_client():
    client = MagicMock()
    client.chats.create.return_value = MagicMock()
    return client

@pytest.fixture
def mock_chat_session():
    chat = MagicMock()
    chat.send_message.return_value = MagicMock()
    chat.get_history.return_value = []
    return chat

@pytest.fixture
def mock_openai_client():
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock()
    return client
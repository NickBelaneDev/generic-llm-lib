import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from typing import List

from generic_llm_lib.llm_core.base import GenericLLM, ChatResult
from generic_llm_lib.llm_core.messages import BaseMessage
from generic_llm_lib.llm_impl.gemini.core import GenericGemini
from generic_llm_lib.llm_impl.openai_api.core import GenericOpenAI


# Mock implementation for testing GenericLLM base logic
class MockLLM(GenericLLM[str]):
    def __init__(self, max_retries: int = 3, base_retry_delay: float = 0.1):
        super().__init__(max_retries=max_retries, base_retry_delay=base_retry_delay)
        self.chat_impl_mock = AsyncMock()

    async def _chat_impl(self, history: List[BaseMessage], user_prompt: str) -> ChatResult[str]:
        return await self.chat_impl_mock(history, user_prompt)


@pytest.mark.asyncio
async def test_initialization():
    """Test initialization of GenericLLM."""
    llm = MockLLM(max_retries=5, base_retry_delay=2.0)
    assert llm.max_retries == 5
    assert llm.base_retry_delay == 2.0


@pytest.mark.asyncio
async def test_generic_llm_chat_happy_path():
    """Test that chat works correctly on the first attempt."""
    llm = MockLLM()
    expected_result = ChatResult(content="Success", history=[], raw="raw")
    llm.chat_impl_mock.return_value = expected_result

    result = await llm.chat([], "hello")
    assert result == expected_result
    assert llm.chat_impl_mock.call_count == 1


@pytest.mark.asyncio
async def test_generic_llm_chat_retry_success():
    """Test that chat retries and eventually succeeds."""
    llm = MockLLM(max_retries=3, base_retry_delay=0.01)
    expected_result = ChatResult(content="Success", history=[], raw="raw")

    # Fail twice, then succeed
    llm.chat_impl_mock.side_effect = [Exception("Fail 1"), Exception("Fail 2"), expected_result]

    result = await llm.chat([], "hello")
    assert result == expected_result
    assert llm.chat_impl_mock.call_count == 3


@pytest.mark.asyncio
async def test_generic_llm_chat_failure_capture():
    """Test that chat raises the last exception after max retries."""
    llm = MockLLM(max_retries=2, base_retry_delay=0.01)

    # Always fail
    llm.chat_impl_mock.side_effect = Exception("Persistent Failure")

    with pytest.raises(Exception) as excinfo:
        await llm.chat([], "hello")

    assert "Persistent Failure" in str(excinfo.value)
    # Initial call + 2 retries = 3 calls
    assert llm.chat_impl_mock.call_count == 3


@pytest.mark.asyncio
async def test_generic_llm_ask_happy_path():
    """Test that ask works correctly on the first attempt."""
    llm = MockLLM()
    expected_result = ChatResult(content="Success", history=[], raw="raw")
    llm.chat_impl_mock.return_value = expected_result

    result = await llm.ask("hello")
    assert result == expected_result
    assert llm.chat_impl_mock.call_count == 1


@pytest.mark.asyncio
async def test_generic_llm_ask_retry_success():
    """Test that ask retries and eventually succeeds."""
    llm = MockLLM(max_retries=3, base_retry_delay=0.01)
    expected_result = ChatResult(content="Success", history=[], raw="raw")

    # Fail once, then succeed
    llm.chat_impl_mock.side_effect = [Exception("Fail 1"), expected_result]

    result = await llm.ask("hello")
    assert result == expected_result
    assert llm.chat_impl_mock.call_count == 2


@pytest.mark.asyncio
async def test_generic_llm_ask_failure_capture():
    """Test that ask raises the last exception after max retries."""
    llm = MockLLM(max_retries=2, base_retry_delay=0.01)

    llm.chat_impl_mock.side_effect = Exception("Ask Failure")

    with pytest.raises(Exception) as excinfo:
        await llm.ask("hello")

    assert "Ask Failure" in str(excinfo.value)
    assert llm.chat_impl_mock.call_count == 3


@pytest.mark.asyncio
async def test_generic_llm_zero_retries():
    """Edge Case: Test behavior when max_retries is 0."""
    llm = MockLLM(max_retries=0, base_retry_delay=0.01)

    llm.chat_impl_mock.side_effect = Exception("Fail immediately")

    with pytest.raises(Exception) as excinfo:
        await llm.chat([], "hello")

    assert "Fail immediately" in str(excinfo.value)
    assert llm.chat_impl_mock.call_count == 1


@pytest.mark.asyncio
async def test_gemini_retry_integration():
    """Test that GenericGemini uses the retry logic."""
    client_mock = AsyncMock()

    gemini = GenericGemini(aclient=client_mock, model_name="test-model", sys_instruction="sys", tool_timeout=0.1)
    gemini.max_retries = 2
    gemini.base_retry_delay = 0.01

    # We mock _chat_impl to simulate failures
    with patch.object(
        gemini,
        "_chat_impl",
        side_effect=[Exception("Gemini Fail"), ChatResult(content="Recovered", history=[], raw=MagicMock())],
    ) as mock_impl:
        result = await gemini.chat([], "test")

        assert result.content == "Recovered"
        assert mock_impl.call_count == 2


@pytest.mark.asyncio
async def test_openai_retry_integration():
    """Test that GenericOpenAI uses the retry logic."""
    client_mock = AsyncMock()
    openai_llm = GenericOpenAI(client=client_mock, model_name="test-model", sys_instruction="sys")
    openai_llm.max_retries = 2
    openai_llm.base_retry_delay = 0.01

    with patch.object(
        openai_llm,
        "_chat_impl",
        side_effect=[Exception("OpenAI Fail"), ChatResult(content="Recovered", history=[], raw=MagicMock())],
    ) as mock_impl:
        result = await openai_llm.chat([], "test")

        assert result.content == "Recovered"
        assert mock_impl.call_count == 2


@pytest.mark.asyncio
async def test_gemini_ask_delegates_to_chat_with_retry():
    """
    Edge Case: GenericGemini.ask calls self.chat.
    Since ask delegates to chat, it should inherit chat's retry logic.
    """
    client_mock = AsyncMock()
    gemini = GenericGemini(aclient=client_mock, model_name="test-model", sys_instruction="sys", tool_timeout=0.1)
    gemini.max_retries = 2
    gemini.base_retry_delay = 0.01

    expected_result = ChatResult(content="Recovered", history=[], raw=MagicMock())

    # We mock _chat_impl to fail twice then succeed.
    # ask -> chat -> _execute_with_retry -> _chat_impl
    # So _chat_impl should be called 3 times (1 initial + 2 retries)

    with patch.object(
        gemini, "_chat_impl", side_effect=[Exception("Fail 1"), Exception("Fail 2"), expected_result]
    ) as mock_impl:
        result = await gemini.ask("test")

        assert result.content == "Recovered"
        assert mock_impl.call_count == 3


@pytest.mark.asyncio
async def test_openai_ask_delegates_to_chat_with_retry():
    """
    Edge Case: GenericOpenAI.ask calls self.chat.
    Since ask delegates to chat, it should inherit chat's retry logic.
    """
    client_mock = AsyncMock()
    openai_llm = GenericOpenAI(client=client_mock, model_name="test-model", sys_instruction="sys")
    openai_llm.max_retries = 2
    openai_llm.base_retry_delay = 0.01

    expected_result = ChatResult(content="Recovered", history=[], raw=MagicMock())

    # Fail twice, succeed on 3rd attempt
    with patch.object(
        openai_llm, "_chat_impl", side_effect=[Exception("Fail 1"), Exception("Fail 2"), expected_result]
    ) as mock_impl:
        result = await openai_llm.ask("test")

        assert result.content == "Recovered"
        assert mock_impl.call_count == 3

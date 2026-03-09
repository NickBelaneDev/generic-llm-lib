import pytest
from google.genai.client import AsyncClient
from generic_llm_lib.llm_impl import GenericGemini
from generic_llm_lib.llm_core.messages import HistoryHandler, UserMessage, AssistantMessage


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_gemini_with_history_handler(genai_client: AsyncClient) -> None:
    """Test GenericGemini using HistoryHandler for conversation state."""
    gemini = GenericGemini(
        aclient=genai_client, model_name="gemini-2.0-flash", sys_instruction="You are a helpful assistant."
    )

    # Initialize history with a system instruction (though Gemini uses config,
    # HistoryHandler can still hold it for consistency)
    history = HistoryHandler(system_instruction="You are a test helper.")

    # First turn
    response = await gemini.chat(history, "My name is Alice.")
    assert "Alice" in response.content

    # Update history from response
    history = HistoryHandler(messages=response.history)
    assert len(history) == 2
    assert isinstance(history[0], UserMessage)
    assert isinstance(history[1], AssistantMessage)

    # Second turn - should remember the name
    response2 = await gemini.chat(history, "What is my name?")
    assert "Alice" in response2.content

    # Verify history grew
    history2 = HistoryHandler(messages=response2.history)
    assert len(history2) == 4  # User, Assistant, User, Assistant
    assert history2[0].content == "My name is Alice."
    assert "Alice" in history2[1].content
    assert history2[2].content == "What is my name?"
    assert "Alice" in history2[3].content

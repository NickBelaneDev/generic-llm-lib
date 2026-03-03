import pytest
from google.genai.client import AsyncClient
from generic_llm_lib.llm_impl import GenericGemini
from generic_llm_lib.llm_core.messages import HistoryHandler

@pytest.mark.vcr
@pytest.mark.asyncio
async def test_gemini_history_auto_update(genai_client: AsyncClient) -> None:
    """Test that GenericGemini updates HistoryHandler in-place."""
    gemini = GenericGemini(
        aclient=genai_client, 
        model_name="gemini-2.0-flash", 
        sys_instruction="You are a helpful assistant."
    )
    
    history = HistoryHandler()
    
    # First turn
    await gemini.chat(history, "My name is Bob.")
    
    # Check if history was updated
    # If not updated, this assertion will fail
    assert len(history) >= 2, "HistoryHandler was not updated after chat turn"
    
    # Second turn
    response = await gemini.chat(history, "What is my name?")
    
    assert "Bob" in response.content
    assert len(history) >= 4

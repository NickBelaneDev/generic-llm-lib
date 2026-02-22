import os
from pathlib import Path

import pytest
from google.genai import Client

from generic_llm_lib import ChatResult, GenericGemini

CASSETTE_NAME = "gemini_chat.yaml"


def _cassette_path() -> Path:
    return Path(__file__).parent / "cassettes" / CASSETTE_NAME


def _gemini_api_key() -> str | None:
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GENAI_API_KEY")


def _gemini_model() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def _skip_without_key_and_cassette() -> None:
    if _gemini_api_key() is None and not _cassette_path().is_file():
        pytest.skip("Set GEMINI_API_KEY to record or add tests/cassettes/gemini_chat.yaml for playback.")


@pytest.mark.asyncio
@pytest.mark.vcr(cassette_name=CASSETTE_NAME)
async def test_gemini_chat_roundtrip() -> None:
    _skip_without_key_and_cassette()

    client = Client(api_key=_gemini_api_key() or "test")
    llm = GenericGemini(
        aclient=client.aio,
        model_name=_gemini_model(),
        sys_instruction="Respond with a short answer.",
        temp=0,
        max_tokens=32,
    )

    response = await llm.ask("Reply with the word recorded.")

    assert isinstance(response, ChatResult)
    assert response.content
    assert response.history
    assert response.history[-1].content == response.content

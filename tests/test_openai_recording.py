import os
from pathlib import Path

import pytest
from openai import AsyncOpenAI

from generic_llm_lib import ChatResult, GenericOpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
CASSETTE_NAME = "openai_chat.yaml"


def _cassette_path() -> Path:
    return Path(__file__).parent / "cassettes" / CASSETTE_NAME


def _openai_api_key() -> str | None:
    return os.getenv("OPENAI_API_KEY")


def _openai_model() -> str:
    return os.getenv("OPENAI_MODEL", "openai/gpt-oss-20b")


def _openai_base_url() -> str:
    return os.getenv("OPENAI_BASE_URL")


def _skip_without_key_and_cassette() -> None:
    if _openai_api_key() is None and not _cassette_path().is_file():
        pytest.skip("Set OPENAI_API_KEY to record or add tests/cassettes/openai_chat.yaml for playback.")


@pytest.mark.asyncio
@pytest.mark.vcr(cassette_name=CASSETTE_NAME)
async def test_openai_chat_roundtrip() -> None:
    _skip_without_key_and_cassette()

    client = AsyncOpenAI(api_key=_openai_api_key() or "test", base_url=_openai_base_url())
    llm = GenericOpenAI(
        client=client,
        model_name=_openai_model(),
        sys_instruction="Respond with a short answer.",
        temp=0,
        max_tokens=32,
    )

    response = await llm.ask("Reply with the word recorded.")

    assert isinstance(response, ChatResult)
    assert response.content
    assert response.history
    assert response.history[-1].content == response.content

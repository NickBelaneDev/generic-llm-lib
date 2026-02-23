import pytest
from openai import AsyncOpenAI
from generic_llm_lib import GenericOpenAI, ChatResult, OpenAIToolRegistry, UserMessage, AssistantMessage, SystemMessage
from typing import Annotated
from pydantic import Field
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())


def _get_openai_model() -> str:
    return os.getenv("OPENAI_MODEL", "openai/gpt-oss-20b")


@pytest.mark.asyncio
async def test_generic_openai_initialization(openai_client: AsyncOpenAI) -> None:
    openai_llm = GenericOpenAI(
        client=openai_client, model_name=_get_openai_model(), sys_instruction="You are a helper."
    )
    assert openai_llm.model == _get_openai_model()
    assert openai_llm.client == openai_client
    assert openai_llm.sys_instruction == "You are a helper."


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_openai_ask_method(openai_client: AsyncOpenAI) -> None:
    openai_llm = GenericOpenAI(
        client=openai_client, model_name=_get_openai_model(), sys_instruction="You are a helper."
    )

    # Execute
    response = await openai_llm.ask("Hello")

    # Verify
    assert isinstance(response, ChatResult)
    assert len(response.content) > 0
    assert response.raw.usage.total_tokens > 0

    # Verify history in response
    assert len(response.history) == 3
    assert isinstance(response.history[0], SystemMessage)
    assert isinstance(response.history[1], UserMessage)
    assert isinstance(response.history[2], AssistantMessage)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_openai_chat_method(openai_client: AsyncOpenAI) -> None:
    openai_llm = GenericOpenAI(
        client=openai_client, model_name=_get_openai_model(), sys_instruction="You are a helper."
    )

    # Execute
    response = await openai_llm.chat([], "Hi")

    # Verify
    assert isinstance(response, ChatResult)
    assert len(response.content) > 0
    assert len(response.history) == 3
    assert isinstance(response.history[0], SystemMessage)
    assert isinstance(response.history[1], UserMessage)
    assert isinstance(response.history[2], AssistantMessage)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_openai_function_calling(openai_client: AsyncOpenAI) -> None:
    # Setup registry with a mock tool
    registry = OpenAIToolRegistry()

    async def get_weather(location: Annotated[str, Field(description="The location to get the weather for.")]) -> str:
        """Get the weather for a location."""
        return f"The weather in {location} is sunny."

    registry.register(get_weather)

    openai_llm = GenericOpenAI(
        client=openai_client,
        model_name=_get_openai_model(),
        sys_instruction="You are a helpful assistant.",
        registry=registry,
    )

    # Execute
    response = await openai_llm.chat([], "What is the weather in Berlin?")

    # Verify content
    assert "sunny" in response.content.lower()
    assert len(response.history) > 2
    assert response.history[-1].content == response.content

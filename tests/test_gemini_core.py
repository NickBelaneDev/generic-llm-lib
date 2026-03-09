import pytest

from dotenv import load_dotenv, find_dotenv

from generic_llm_lib.llm_impl import GenericGemini, GeminiToolRegistry
from generic_llm_lib.llm_core import ChatResult, UserMessage, AssistantMessage, HistoryHandler

from typing import Annotated
from pydantic import Field
from google.genai.client import AsyncClient

load_dotenv(find_dotenv())


@pytest.mark.asyncio
async def test_generic_gemini_initialization(genai_client: AsyncClient) -> None:
    gemini = GenericGemini(aclient=genai_client, model_name="gemini-2.5-flash", sys_instruction="You are a helper.")
    assert gemini.model == "gemini-2.5-flash"
    assert gemini.client == genai_client
    assert gemini.config.system_instruction == "You are a helper."


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_gemini_ask_method(genai_client: AsyncClient) -> None:
    gemini = GenericGemini(aclient=genai_client, model_name="gemini-2.5-flash", sys_instruction="You are a helper.")

    # Execute
    response = await gemini.ask("Hello")

    # Verify
    assert isinstance(response, ChatResult)
    assert len(response.content) > 0
    assert response.raw.usage_metadata.total_token_count > 0

    # Verify history in response
    assert len(response.history) == 2
    assert isinstance(response.history[0], UserMessage)
    assert response.history[0].content == "Hello"
    assert isinstance(response.history[1], AssistantMessage)
    assert len(response.history[1].content) > 0


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_gemini_chat_method(genai_client: AsyncClient) -> None:
    gemini = GenericGemini(aclient=genai_client, model_name="gemini-2.5-flash", sys_instruction="You are a helper.")

    # Execute
    response = await gemini.chat([], "Hi")

    # Verify
    assert isinstance(response, ChatResult)
    assert len(response.content) > 0
    assert len(response.history) == 2
    assert isinstance(response.history[0], UserMessage)
    assert isinstance(response.history[1], AssistantMessage)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_gemini_function_calling(genai_client: AsyncClient) -> None:
    # Setup registry with a mock tool
    registry = GeminiToolRegistry()

    async def get_weather(location: Annotated[str, Field(description="The location to get the weather for.")]) -> str:
        """Get the weather for a location."""
        return f"The weather in {location} is sunny."

    registry.register(get_weather)

    gemini = GenericGemini(
        aclient=genai_client,
        model_name="gemini-2.5-flash",
        sys_instruction="You are a helpful assistant.",
        registry=registry,
    )

    # Execute
    response = await gemini.chat([], "What is the weather in Berlin?")

    # Verify content
    assert "sunny" in response.content.lower()
    # History should contain: User -> Assistant (Tool Call) -> Tool (Response) -> Assistant (Final Answer)
    assert len(response.history) == 4
    assert response.history[-1].content == response.content


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_gemini_function_calling_with_empty_args(genai_client: AsyncClient) -> None:
    registry = GeminiToolRegistry()

    async def get_current_time() -> str:
        """Get the current time."""
        return "It is 12:00 PM."

    registry.register(get_current_time)

    gemini = GenericGemini(
        aclient=genai_client,
        model_name="gemini-2.5-flash",
        sys_instruction="You are a helpful assistant.",
        registry=registry,
    )

    response = await gemini.chat([], "What time is it?")

    # Verify content contains the time returned by the tool
    assert "12:00" in response.content or "noon" in response.content.lower()


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_gemini_history_handler_integration(genai_client: AsyncClient) -> None:
    """Test that HistoryHandler correctly maintains context across multiple turns."""
    gemini = GenericGemini(
        aclient=genai_client, model_name="gemini-2.5-flash", sys_instruction="You are a helpful assistant."
    )

    # Initialize history
    history = HistoryHandler(system_instruction="You are a helpful assistant.")

    # First turn: User introduces themselves
    response1 = await gemini.chat(history, "My name is Alice.")

    # Update history with the response
    # Note: response.history contains the FULL history of that session (User + Assistant)
    # We need to be careful not to duplicate if we were appending manually,
    # but here we can just re-initialize or append the new messages.
    # The GenericLLM.chat method returns the full history of the interaction.
    history = HistoryHandler(messages=response1.history)

    # Second turn: User asks for their name
    response2 = await gemini.chat(history, "What is my name?")

    # Verify the assistant remembers the name
    assert "Alice" in response2.content

    # Verify the history length grew
    # 1. User: My name is Alice
    # 2. Assistant: Hello Alice...
    # 3. User: What is my name?
    # 4. Assistant: Your name is Alice.
    assert len(response2.history) >= 4

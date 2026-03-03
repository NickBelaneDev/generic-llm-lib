"""Core abstractions for LLM provider implementations."""

import asyncio
from typing import List, TypeVar, Generic, Callable, Coroutine, Any, Union
from abc import ABC, abstractmethod
from pydantic import BaseModel
from ..messages import BaseMessage, HistoryHandler
from ..logger import get_logger

logger = get_logger(__name__)


ProviderResT = TypeVar("ProviderResT")


class ChatResult(BaseModel, Generic[ProviderResT]):
    """Normalized chat output returned by provider implementations.

    Attributes:
        content: Text content returned by the provider.
        history: Updated conversation history in provider-agnostic format.
        raw: Provider-specific response payload for advanced use cases.
    """

    content: str
    history: List[BaseMessage]
    raw: ProviderResT


class GenericLLM(ABC, Generic[ProviderResT]):
    """Abstract base class for LLM implementations.

    Implementations should return provider-specific data via ``ChatResult.raw`` while
    keeping ``ChatResult.content`` and ``ChatResult.history`` consistent for consumers.
    """

    def __init__(self, max_retries: int = 3, base_retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_retry_delay = base_retry_delay

    async def _execute_with_retry(
        self, func: Callable[..., Coroutine[Any, Any, ChatResult[ProviderResT]]], *args: Any, **kwargs: dict[str, Any]
    ) -> ChatResult[ProviderResT]:
        """
        Executes a function with retry logic.

        Args:
            func: The asynchronous function to execute.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            The result of the function call.

        Raises:
            Exception: The last encountered exception if all retries fail.
            TimeoutError: If the maximum number of retries is exceeded - which actually should not happen...
        """

        delay = self.base_retry_delay
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Here we could check, what kind of error it is.

                if attempt == self.max_retries:
                    raise e

                logger.warning(f"API Error (Retry: {attempt + 1}/{self.max_retries}): {e}. Waiting {delay}s...")
                await asyncio.sleep(delay)
                delay *= 2  # Exponential backoff

        msg = f"Failed to get response after {self.max_retries} retries."
        logger.error(msg)
        raise TimeoutError(msg)

    async def chat(
        self, history: Union[List[BaseMessage], HistoryHandler], user_prompt: str
    ) -> ChatResult[ProviderResT]:
        """
        Conducts a chat turn with the LLM.

        Args:
            history: The conversation history as a list of messages or a HistoryHandler.
            user_prompt: The user's input message.

        Returns:
            A provider-specific response object containing the LLM's response
            and updated conversation history.
        """
        # Normalize history to List[BaseMessage]
        if isinstance(history, HistoryHandler):
            history_list = history.messages
        else:
            history_list = history

        result = await self._execute_with_retry(self._chat_impl, history_list, user_prompt)

        # Update HistoryHandler if provided
        if isinstance(history, HistoryHandler):
            history.update(result.history)

        return result

    async def ask(self, prompt: str) -> ChatResult[ProviderResT]:
        """
        Single-turn question without maintaining history.

        Args:
            prompt: The question to ask.

        Returns:
            A provider-specific response object containing the LLM's response.
        """
        return await self.chat([], prompt)

    @abstractmethod
    async def _chat_impl(self, history: List[BaseMessage], user_prompt: str) -> ChatResult[ProviderResT]:
        pass

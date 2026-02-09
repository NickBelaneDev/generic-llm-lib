from typing import List, Optional, TypeVar, Generic
from abc import ABC, abstractmethod
from generic_llm_lib.llm_core.messages.models import BaseMessage
from pydantic import BaseModel

ProviderResT = TypeVar("ProviderResT")


class ChatResult(BaseModel, Generic[ProviderResT]):
    content: str
    history: List[BaseMessage]
    raw: ProviderResT


class GenericLLM(ABC, Generic[ProviderResT]):
    """
    Abstract Base Class for Generic LLM implementations.
    Defines the standard interface for chatting and asking questions.
    """

    @abstractmethod
    async def chat(self, history: List[BaseMessage], user_prompt: str) -> ChatResult[ProviderResT]:
        """
        Conducts a chat turn with the LLM.

        Args:
            history: The conversation history (provider-agnostic format).
            user_prompt: The user's input message.

        Returns:
            A provider-specific response object containing the LLM's response
            and updated conversation history.
        """
        pass

    @abstractmethod
    async def ask(self, prompt: str, model: Optional[str] = None) -> ChatResult[ProviderResT]:
        """
        Single-turn question without maintaining history.

        Args:
            prompt: The question to ask.
            model: Optional model override.

        Returns:
            A provider-specific response object containing the LLM's response.
        """
        pass

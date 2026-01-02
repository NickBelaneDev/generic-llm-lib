from typing import List, Any, Tuple
from abc import ABC, abstractmethod


class GenericLLM(ABC):
    """
    Abstract Base Class for Generic LLM implementations.
    Defines the standard interface for chatting and asking questions.
    """
    @abstractmethod
    async def chat(self, history: List[Any], user_prompt: str) -> Tuple[str, List[Any]]:
        """
        Conducts a chat turn with the LLM.

        Args:
            history: The conversation history (provider-specific format).
            user_prompt: The user's input message.

        Returns:
            A tuple containing:
            - The text response from the LLM.
            - The updated conversation history.
        """
        pass

    @abstractmethod
    async def ask(self, prompt: str, model: str = None) -> str:
        """
        Single-turn question without maintaining history.

        Args:
            prompt: The question to ask.
            model: Optional model override.

        Returns:
            The text response.
        """
        pass
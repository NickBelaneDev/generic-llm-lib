from pydantic import BaseModel, Field
from typing import List, Optional
from generic_llm_lib.llm_core.messages.models import BaseMessage


class OpenAITokens(BaseModel):
    """
    Represents the token counts for an OpenAI model response.

    Attributes:
        prompt_tokens: The number of tokens in the prompt.
        completion_tokens: The number of tokens in the completion response.
        total_tokens: The total number of tokens used.
    """

    prompt_tokens: Optional[int] = Field(default=None)
    completion_tokens: Optional[int] = Field(default=None)
    total_tokens: Optional[int] = Field(default=None)


class OpenAIMessageResponse(BaseModel):
    """
    Represents the response for a single message in the OpenAI model.

    Attributes:
        text: The text content of the response.
        tokens: The token counts for the response.
    """

    text: str
    tokens: OpenAITokens


class OpenAIChatResponse(BaseModel):
    """
    Represents the response for a chat session with the OpenAI model.

    Attributes:
        last_response: The last message response in the chat session.
        history: The chat history, represented as a list of BaseMessage objects.
    """

    last_response: OpenAIMessageResponse
    history: List[BaseMessage] = Field(default_factory=list)

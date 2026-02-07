from pydantic import BaseModel, Field
from typing import List, Optional
from google.genai import types


class GeminiTokens(BaseModel):
    """
    Represents the token counts for a Gemini model response.

    Attributes:
        prompt_token_count: The number of tokens in the prompt.
        candidate_token_count: The number of tokens in the candidate response.
        total_token_count: The total number of tokens in the prompt and response.
        thoughts_token_count: The number of tokens in the thoughts.
        tool_use_prompt_token_count: The number of tokens in the tool use prompt.
    """

    prompt_token_count: Optional[int] = Field(default=None)
    candidate_token_count: Optional[int] = Field(default=None)
    total_token_count: Optional[int] = Field(default=None)
    thoughts_token_count: Optional[int] = Field(default=None)
    tool_use_prompt_token_count: Optional[int] = Field(default=None)


class GeminiMessageResponse(BaseModel):
    """
    Represents the response for a single message in the Gemini model.

    Attributes:
        text: The text content of the response.
        tokens: The token counts for the response.
    """

    text: str
    tokens: GeminiTokens


class GeminiChatResponse(BaseModel):
    """
    Represents the response for a chat session with the Gemini model.

    Attributes:
        last_response: The last message response in the chat session.
        history: The chat history, represented as a list of Content objects.
    """

    last_response: GeminiMessageResponse
    history: List[types.Content] = Field(default_factory=list)

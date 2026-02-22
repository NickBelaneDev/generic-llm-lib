from google.genai import types
from typing import List, Tuple, Optional, Any, Sequence

from google.genai.client import AsyncClient
from google.genai.types import GenerateContentResponse
from generic_llm_lib.llm_core import GenericLLM, ToolExecutionLoop, get_logger
from generic_llm_lib.llm_core.messages.models import BaseMessage, UserMessage, AssistantMessage, SystemMessage
from generic_llm_lib.llm_core.base.base import ChatResult
from .adapter import GeminiToolAdapter
from .registry import GeminiToolRegistry
from ...llm_core.tools.tool_manager import ToolManager

logger = get_logger(__name__)


class GenericGemini(GenericLLM[GenerateContentResponse]):
    """
    Implementation of GenericLLM for Google's Gemini models.
    Handles chat sessions and automatic function calling loops.
    """

    def __init__(
        self,
        aclient: AsyncClient,
        model_name: str,
        sys_instruction: str,
        registry: Optional[GeminiToolRegistry] = None,
        temp: float = 1.0,
        max_tokens: int = 3000,
        max_function_loops: int = 5,
        tool_timeout: float = 180.0,
        tool_manager: Optional[ToolManager[GeminiToolRegistry]] = None,
    ):
        """
        Initializes the GenericGemini LLM wrapper.

        Args:
            aclient: The initialized Google GenAI client.
            model_name: The identifier for the Gemini model to use (e.g., 'gemini-pro', 'gemini-flash-latest').
            sys_instruction: A system-level instruction or persona for the LLM.
            registry: An optional GeminiToolRegistry instance containing tools the LLM can use.
            temp: The temperature for text generation, controlling randomness.
            max_tokens: The maximum number of tokens to generate in the response.
            max_function_loops: The maximum number of consecutive function calls the LLM can make.
            tool_timeout: The maximum time in seconds to wait for a tool execution.
        """
        self.model: str = model_name

        self.max_function_loops = max_function_loops
        self.tool_timeout = tool_timeout

        self.tool_manager = tool_manager

        if registry:
            self.registry = registry
        elif self.tool_manager:

            self.registry = self.tool_manager.registry
        else:
            self.registry = GeminiToolRegistry()

        self._tool_loop = ToolExecutionLoop(
            registry=registry,
            max_function_loops=max_function_loops,
            tool_timeout=tool_timeout,
        )

        # Only include tools if there are any registered
        tools_config: Optional[List[Any]] = None
        if self.registry:
            tool_obj = self.registry.tool_object
            if tool_obj:
                tools_config = [tool_obj]
                logger.info(f"Registered {len(self.registry.tools)} tools for Gemini model '{model_name}'.")

        self.config: types.GenerateContentConfig = types.GenerateContentConfig(
            system_instruction=sys_instruction, temperature=temp, max_output_tokens=max_tokens, tools=tools_config
        )

        self.client: AsyncClient = aclient
        logger.info(f"Initialized GenericGemini with model='{model_name}', temp={temp}, max_tokens={max_tokens}")

    async def ask(self, prompt: str, model: Optional[str] = None) -> ChatResult[GenerateContentResponse]:
        """
        Generates a text response from the LLM based on a single prompt.
        This method handles potential function calls internally by initiating a temporary chat session.

        Args:
            prompt: The user's input prompt.
            model: Optional. Overrides the default model for this specific request.

        Returns:
            GeminiMessageResponse: The generated text response from the LLM.
        """
        if not model:
            model = self.model

        logger.debug(f"Asking Gemini (model={model}): {prompt[:50]}...")

        # We use a temporary chat session to handle the tool execution loop (Model -> Tool -> Model)
        # We start with an empty history.
        response: ChatResult[GenerateContentResponse] = await self.chat([], prompt)

        return response

    async def chat(
        self, history: List[BaseMessage], user_prompt: str, clean_history: bool = False
    ) -> ChatResult[GenerateContentResponse]:
        """
        Processes a single turn of a chat conversation, including handling user input,
        generating LLM responses, and executing any requested function calls.

        The method supports a multi-turn interaction where the LLM can call functions
        and receive their results within the same turn, up to `max_function_loops` times.

        Args:
            history: A list of `BaseMessage` objects representing the conversation history.
            user_prompt: The current message from the user.
            clean_history: Decide if the history may contain complete function_call parts or not.

        Returns:
            GeminiChatResponse: An object containing:
            - The final text response from the LLM after all function calls (if any) are resolved.
            - The updated conversation history, including the user's prompt, LLM's responses,
              and any tool calls/responses.
        """
        logger.debug(f"Starting chat turn. History length: {len(history)}. Prompt: {user_prompt[:50]}...")

        # Convert generic history to Gemini specific history
        gemini_history = self._convert_history(history)

        # Create a chat session with the provided history
        chat = self.client.chats.create(
            model=self.model,
            config=self.config,
            history=gemini_history,  # type: ignore[arg-type]
        )

        # Send the user message
        try:
            _response = await chat.send_message(user_prompt)
        except Exception as e:
            logger.error(f"Error sending message to Gemini: {e}", exc_info=True)
            raise

        response, chat = await self._handle_function_calls(_response, chat)

        # Get full history
        full_history = chat.get_history()

        # Clean history to remove intermediate tool calls and outputs to save tokens
        if clean_history:
            logger.debug("Cleaning chat history (removing intermediate tool calls).")
            cleaned_history = self._clean_history(full_history)
            return self._build_response(response, cleaned_history)

        else:
            return self._build_response(response, full_history)

    @staticmethod
    def _convert_history(history: List[BaseMessage]) -> List[types.Content]:
        """
        Converts generic BaseMessage history to Gemini specific Content history.

        Args:
            history: List of BaseMessage objects.

        Returns:
            List of Gemini Content objects.
        """
        gemini_history = []
        for msg in history:
            if isinstance(msg, UserMessage):
                gemini_history.append(types.Content(role="user", parts=[types.Part(text=msg.content)]))
            elif isinstance(msg, AssistantMessage):
                gemini_history.append(types.Content(role="model", parts=[types.Part(text=msg.content)]))
            elif isinstance(msg, SystemMessage):
                # System messages are typically handled via system_instruction in config,
                # but if they appear in history, we might treat them as user or model depending on context,
                # or skip if already set in config. For now, let's treat as user message or skip.
                # Gemini doesn't have a 'system' role in chat history in the same way as OpenAI.
                # It's usually set at initialization.
                pass
            # Tool messages handling would be more complex as it requires mapping back to function calls/responses
            # For now, we focus on text history.
        return gemini_history

    async def _handle_function_calls(
        self, response: GenerateContentResponse, chat: Any
    ) -> Tuple[GenerateContentResponse, Any]:
        """
        Handles the function calling loop.
        Iterates through the response to check for function calls, executes them,
        and sends the results back to the model until no more function calls are made
        or the limit is reached.

        Args:
            response: The initial response from the model.
            chat: The current chat session object.

        Returns:
            Tuple[GenerateContentResponse, Any]: The final response from the model and the updated chat object.
        """

        if not response:
            return response, chat

        adapter = GeminiToolAdapter(chat)
        result = await self._tool_loop.run(initial_response=response, adapter=adapter)

        if not isinstance(result, GenerateContentResponse):
            raise TypeError(f"Expected GenerateContentResponse, got {type(result)}")

        return result, chat

    @staticmethod
    def _clean_history(history: Sequence[types.Content]) -> List[types.Content]:
        """Removes intermediate tool calls and outputs from the history to save tokens.

        Keeps user messages, system instructions, and assistant messages that have content.

        Args:
            history: The conversation history to clean.

        Returns:
            A list of cleaned content objects.
        """
        cleaned_history = []
        for content in history:
            # Skip if no parts
            if not content.parts:
                cleaned_history.append(content)
                continue

            # Check if it is a function response
            has_function_response = any(part.function_response for part in content.parts)
            if has_function_response:
                continue

            # Check for function calls
            has_function_call = any(part.function_call for part in content.parts)

            if has_function_call:
                # Filter out function_call parts, keep text parts
                new_parts = [part for part in content.parts if not part.function_call]

                if new_parts:
                    # Create new Content with remaining parts (e.g. text)
                    cleaned_history.append(types.Content(role=content.role, parts=new_parts))
                # If no parts left (only function calls), we skip this message
            else:
                # No function calls or responses, keep as is
                cleaned_history.append(content)

        return cleaned_history

    @staticmethod
    def _build_response(
        response: GenerateContentResponse, history: Sequence[types.Content]
    ) -> ChatResult[GenerateContentResponse]:
        """
        Constructs the final GeminiChatResponse object from the raw API response and chat history.

        Args:
            response: The final GenerateContentResponse from the model.
            history: The chat history list.

        Returns:
            GeminiChatResponse: The structured response containing text, tokens, and history.
        """
        text_response = "".join([p.text for p in response.parts if p.text]) if response.parts else ""
        # Convert Gemini history back to generic BaseMessage history
        generic_history = GenericGemini._convert_to_generic_history(history)

        return ChatResult(
            content=text_response,
            history=generic_history,
            raw=response,
        )

    @staticmethod
    def _convert_to_generic_history(history: Sequence[types.Content]) -> List[BaseMessage]:
        """
        Converts Gemini specific Content history to generic BaseMessage history.

        Args:
            history: List of Gemini Content objects.

        Returns:
            List of BaseMessage objects.
        """
        generic_history: List[BaseMessage] = []
        for content in history:
            role = content.role
            # Check if parts is None before iterating
            if content.parts is None:
                continue

            text_parts = [part.text for part in content.parts if part.text]
            text_content = "".join(text_parts)

            if not text_content:
                continue

            if role == "user":
                generic_history.append(UserMessage(content=text_content))
            elif role == "model":
                generic_history.append(AssistantMessage(content=text_content))
            # Handle other roles if necessary
        return generic_history

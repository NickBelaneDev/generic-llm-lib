from google.genai import types
from typing import List, Tuple, Optional, Any, Sequence

from google.genai.client import AsyncClient
from google.genai.types import GenerateContentResponse
from generic_llm_lib.llm_core import GenericLLM, get_logger
from generic_llm_lib.llm_core.tools import ToolExecutionLoop
from generic_llm_lib.llm_core.messages import BaseMessage
from generic_llm_lib.llm_core.base import ChatResult
from .adapter import GeminiToolAdapter
from .registry import GeminiToolRegistry
from generic_llm_lib.llm_core.tools import ToolManager
from .history_converter import convert_to_gemini_history, convert_from_gemini_history

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
        super().__init__()
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
            registry=self.registry,
            max_function_loops=self.max_function_loops,
            tool_timeout=self.tool_timeout,
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

    async def _chat_impl(self, history: List[BaseMessage], user_prompt: str) -> ChatResult[GenerateContentResponse]:
        """
        Processes a single turn of a chat conversation, including handling user input,
        generating LLM responses, and executing any requested function calls.

        Args:
            history: A list of `BaseMessage` objects representing the conversation history.
            user_prompt: The current message from the user.

        Returns:
            ChatResult[GenerateContentResponse]: An object containing the response and updated history.
        """
        logger.debug(f"Starting chat turn. History length: {len(history)}. Prompt: {user_prompt[:50]}...")

        # Convert generic history to Gemini specific history
        gemini_history = convert_to_gemini_history(history)

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

        # Get full history from the chat session
        full_history = chat.get_history()

        return self._build_response(response, full_history)

    async def _handle_function_calls(
        self, response: GenerateContentResponse, chat: Any
    ) -> Tuple[GenerateContentResponse, Any]:
        """
        Handles the function calling loop.

        Args:
            response: The initial response from the model.
            chat: The current chat session object.

        Returns:
            Tuple[GenerateContentResponse, Any]: The final response and updated chat object.
        """
        if not response:
            return response, chat

        adapter = GeminiToolAdapter(chat)
        result = await self._tool_loop.run(initial_response=response, adapter=adapter)

        if not isinstance(result, GenerateContentResponse):
            raise TypeError(f"Expected GenerateContentResponse, got {type(result)}")

        return result, chat

    @staticmethod
    def _build_response(
        response: GenerateContentResponse, history: Sequence[types.Content]
    ) -> ChatResult[GenerateContentResponse]:
        """
        Constructs the final ChatResult object.

        Args:
            response: The final GenerateContentResponse from the model.
            history: The chat history list.

        Returns:
            ChatResult[GenerateContentResponse]: The structured response.
        """
        text_response = "".join([p.text for p in response.parts if p.text]) if response.parts else ""
        generic_history = convert_from_gemini_history(history)

        return ChatResult(
            content=text_response,
            history=generic_history,
            raw=response,
        )

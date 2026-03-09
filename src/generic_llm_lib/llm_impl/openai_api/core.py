"""
OpenAI LLM implementation core module.

This module provides the `GenericOpenAI` class, which implements the
`GenericLLM` interface for OpenAI's chat completion models. It handles
the interaction with the OpenAI API, including chat history management
and automatic tool execution loops.
"""

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionToolParam
from typing import List, Tuple, Optional, Any, Dict, Iterable, cast
import logging
from generic_llm_lib.llm_core import GenericLLM
from generic_llm_lib.llm_core.tools import ToolExecutionLoop, ToolManager
from generic_llm_lib.llm_core.messages import BaseMessage
from generic_llm_lib.llm_core.base import ChatResult
from .adapter import OpenAIToolAdapter
from .registry import OpenAIToolRegistry
from .history_converter import convert_to_openai_history, convert_from_openai_history

logger = logging.getLogger(__name__)


class GenericOpenAI(GenericLLM[ChatCompletion]):
    """
    Implementation of GenericLLM for OpenAI's models.
    Handles chat sessions and automatic function calling loops.
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        model_name: str,
        sys_instruction: str,
        registry: Optional[OpenAIToolRegistry] = None,
        temp: float = 1.0,
        max_tokens: int = 3000,
        max_function_loops: int = 5,
        tool_timeout: float = 180.0,
        tool_manager: Optional[ToolManager[OpenAIToolRegistry]] = None,
    ):
        """
        Initializes the GenericOpenAI LLM wrapper.

        Args:
            client: The initialized AsyncOpenAI client.
            model_name: The identifier for the OpenAI model to use (e.g., 'gpt-4', 'gpt-3.5-turbo').
            sys_instruction: A system-level instruction or persona for the LLM.
            registry: An optional ToolRegistry instance containing tools the LLM can use.
            temp: The temperature for text generation, controlling randomness.
            max_tokens: The maximum number of tokens to generate in the response.
            max_function_loops: The maximum number of consecutive function calls the LLM can make.
            tool_timeout: The maximum time in seconds to wait for a tool execution.
        """
        super().__init__()
        self.model: str = model_name
        self.registry: Optional[OpenAIToolRegistry] | None = registry
        self.max_function_loops = max_function_loops
        self.sys_instruction = sys_instruction
        self.temperature = temp
        self.max_tokens = max_tokens
        self.tool_timeout = tool_timeout
        self.client: AsyncOpenAI = client

        self.tool_manager = tool_manager

        if registry:
            self.registry = registry
        elif self.tool_manager:
            # When injecting a ToolManager we need to use his registry
            self.registry = self.tool_manager.registry
        else:
            self.registry = OpenAIToolRegistry()

        self._tool_loop = ToolExecutionLoop(
            registry=self.registry,
            max_function_loops=self.max_function_loops,
            tool_timeout=self.tool_timeout,
            argument_error_formatter=self._format_argument_error,
        )

    def _prepare_messages(self, history: List[BaseMessage], user_prompt: str) -> List[Dict[str, Any]]:
        """Prepares the initial list of messages for the OpenAI API call."""
        openai_history = convert_to_openai_history(history)
        messages = list(openai_history)

        if not messages and self.sys_instruction:
            messages.append({"role": "system", "content": self.sys_instruction})

        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _get_tools_config(self) -> Optional[Iterable[ChatCompletionToolParam]]:
        """Retrieves the tool configuration from the registry."""
        if not self.registry:
            return None

        tools = self.registry.tool_object
        if tools:
            logger.debug(f"Tools registered: {[t.get('function', {}).get('name') for t in tools]}")
        return tools

    @staticmethod
    def _log_initial_response_status(response: ChatCompletion) -> None:
        """Logs the status of the initial OpenAI response."""
        if not response.choices:
            logger.debug(f"Initial response has no choices. Response dump: {response.model_dump()}")
        else:
            logger.debug(f"Initial response received. Finish reason: {response.choices[0].finish_reason}")

    @staticmethod
    def _append_final_response_to_messages(messages: List[Dict[str, Any]], final_response: ChatCompletion) -> None:
        """Appends the final LLM response to the messages list if it contains content."""
        if final_response.choices and final_response.choices[0].message.content:
            last_msg = final_response.choices[0].message.model_dump()
            # Avoid duplicates if the message is already present (e.g., from tool loop)
            if not messages or messages[-1] != last_msg:
                messages.append(last_msg)

    async def _chat_impl(self, history: List[BaseMessage], user_prompt: str) -> ChatResult[ChatCompletion]:
        """
        Processes a single turn of a chat conversation, including handling user input,
        generating LLM responses, and executing any requested function calls.

        The method supports a multi-turn interaction where the LLM can call functions
        and receive their results within the same turn, up to `max_function_loops` times.

        Args:
            history: A list of `BaseMessage` objects representing the conversation history.
            user_prompt: The current message from the user.

        Returns:
            ChatResult[ChatCompletion]: An object containing:
            - The final text response from the LLM after all function calls (if any) are resolved.
            - The updated conversation history, including the user's prompt, LLM's responses,
              and any tool calls/responses.
            - The raw ChatCompletion object.
        """
        logger.debug(f"chat() called. History length: {len(history)}, User prompt: {user_prompt}")

        messages = self._prepare_messages(history, user_prompt)
        tools = self._get_tools_config()

        # Initial call to OpenAI
        logger.debug(f"Sending initial request to OpenAI model: {self.model}")
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=cast(Iterable[Any], messages),
            tools=tools,  # type: ignore
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        self._log_initial_response_status(response)

        messages, final_response = await self._handle_function_calls(messages, response, tools)

        self._append_final_response_to_messages(messages, final_response)

        return self._build_response(final_response, messages)

    async def _handle_function_calls(
        self,
        messages: List[Dict[str, Any]],
        initial_response: ChatCompletion,
        tools: Optional[Iterable[ChatCompletionToolParam]],
    ) -> Tuple[List[Dict[str, Any]], ChatCompletion]:
        """
        Handles the function calling loop.
        Delegates to ToolExecutionLoop via OpenAIToolAdapter.

        Args:
            messages: The current list of messages in the conversation.
            initial_response: The initial response object from the model.
            tools: The list of available tools.

        Returns:
            Tuple[List[Dict[str, Any]], ChatCompletion]: The updated messages list and the final response object from the model.
        """
        if not initial_response.choices:
            return messages, initial_response

        adapter = OpenAIToolAdapter(
            client=self.client,
            model=self.model,
            messages=messages,
            registry=self.registry,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        final_response: ChatCompletion = await self._tool_loop.run(initial_response=initial_response, adapter=adapter)

        return messages, final_response

    @staticmethod
    def _build_response(response: ChatCompletion, history: List[Dict[str, Any]]) -> ChatResult[ChatCompletion]:
        """
        Constructs the final ChatResult object from the raw API response and chat history.

        Args:
            response: The final ChatCompletion from the model.
            history: The chat history list.

        Returns:
            ChatResult[ChatCompletion]: The structured response containing text, history, and raw response.
        """
        if response.choices:
            message_content = response.choices[0].message.content or ""
        else:
            message_content = ""

        # Convert OpenAI history back to generic BaseMessage history
        generic_history = convert_from_openai_history(history)

        return ChatResult(content=message_content, history=generic_history, raw=response)

    @staticmethod
    def _format_argument_error(tool_name: str, error: Exception) -> str:
        """Format an error message when tool argument decoding fails.

        Args:
            tool_name: The name of the tool that failed.
            error: The exception that occurred.

        Returns:
            A formatted error string.
        """
        return f"Failed to decode function arguments: {error}"

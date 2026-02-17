from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionToolParam
from typing import List, Tuple, Optional, Any, Dict, Iterable, cast
import logging
from generic_llm_lib.llm_core import GenericLLM
from generic_llm_lib.llm_core import ToolExecutionLoop
from generic_llm_lib.llm_core.messages.models import (
    BaseMessage,
    UserMessage,
    AssistantMessage,
    SystemMessage,
    ToolMessage,
)
from generic_llm_lib.llm_core.base.base import ChatResult
from .adapter import OpenAIToolAdapter
from .registry import OpenAIToolRegistry

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
        self.model: str = model_name
        self.registry: Optional[OpenAIToolRegistry] | None = registry
        self.max_function_loops = max_function_loops
        self.sys_instruction = sys_instruction
        self.temperature = temp
        self.max_tokens = max_tokens
        self.tool_timeout = tool_timeout

        self.client: AsyncOpenAI = client

        if self.registry is None:
            self.registry = OpenAIToolRegistry()

        self._tool_loop = ToolExecutionLoop(
            registry=self.registry,
            max_function_loops=self.max_function_loops,
            tool_timeout=self.tool_timeout,
            argument_error_formatter=self._format_argument_error,
        )

    async def ask(self, prompt: str, model: Optional[str] = None) -> ChatResult[ChatCompletion]:
        """
        Generates a text response from the LLM based on a single prompt.
        This method handles potential function calls internally by initiating a temporary chat session.

        Args:
            prompt: The user's input prompt.
            model: Optional. Overrides the default model for this specific request.

        Returns:
            ChatResult[ChatCompletion]: The generated text response from the LLM.
        """
        # We use a temporary chat session to handle the tool execution loop (Model -> Tool -> Model)
        # We start with an empty history.
        # logger.debug(f"ask() called with prompt: {prompt}")
        return await self.chat([], prompt)

    async def chat(
        self, history: List[BaseMessage], user_prompt: str, clean_history: bool = False
    ) -> ChatResult[ChatCompletion]:
        """
        Processes a single turn of a chat conversation, including handling user input,
        generating LLM responses, and executing any requested function calls.

        The method supports a multi-turn interaction where the LLM can call functions
        and receive their results within the same turn, up to `max_function_loops` times.

        Args:
            history: A list of `BaseMessage` objects representing the conversation history.
            user_prompt: The current message from the user.
            clean_history: Removes function calls from the chat history.

        Returns:
            ChatResult[ChatCompletion]: An object containing:
            - The final text response from the LLM after all function calls (if any) are resolved.
            - The updated conversation history, including the user's prompt, LLM's responses,
              and any tool calls/responses.
            - The raw ChatCompletion object.
        """
        # logger.debug(f"chat() called. History length: {len(history)}, User prompt: {user_prompt}")

        # Convert generic history to OpenAI specific history
        openai_history = self._convert_history(history)

        # Prepare the messages list. If history is empty, add system instruction first.
        messages = list(openai_history)
        if not messages and self.sys_instruction:
            messages.append({"role": "system", "content": self.sys_instruction})

        # Add user message
        messages.append({"role": "user", "content": user_prompt})

        # Get tools configuration
        tools: Iterable[ChatCompletionToolParam] | None = None
        if self.registry:
            tools = self.registry.tool_object

            # logger.debug(f"Tools registered: {[t.get('function', {}).get('name') for t in tools]}")

        # Initial call to OpenAI
        # logger.debug(f"Sending initial request to OpenAI model: {self.model}")
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=cast(Iterable[Any], messages),
            tools=tools,  # type: ignore
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        if not response.choices:
            # logger.debug(f"Initial response has no choices. Response dump: {response.model_dump()}")
            pass
        else:
            # logger.debug(f"Initial response received. Finish reason: {response.choices[0].finish_reason}")
            pass

        messages, final_response = await self._handle_function_calls(messages, response, tools)

        # Ensure the final response is added to the history if it has content
        if final_response.choices and final_response.choices[0].message.content:
            # Check if the last message in history is already this response (to avoid duplicates if logic changes)
            # The tool loop does NOT record the final response, so we must add it here.
            last_msg = final_response.choices[0].message.model_dump()
            # Only add if not already present (simple check)
            if not messages or messages[-1] != last_msg:
                messages.append(last_msg)

        # Clean history to remove intermediate tool calls and outputs to save tokens
        if clean_history:
            logger.debug("Cleaning chat history (removing intermediate tool calls).")
            messages = self._clean_history(messages)

        return self._build_response(final_response, messages)

    @staticmethod
    def _convert_history(history: List[BaseMessage]) -> List[Dict[str, Any]]:
        """
        Converts generic BaseMessage history to OpenAI specific dictionary history.

        Args:
            history: List of BaseMessage objects.

        Returns:
            List of OpenAI message dictionaries.
        """
        openai_history = []
        for msg in history:
            if isinstance(msg, UserMessage):
                openai_history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AssistantMessage):
                openai_msg = {"role": "assistant", "content": msg.content}
                if msg.tool_calls:
                    openai_msg["tool_calls"] = msg.tool_calls  # type: ignore
                openai_history.append(openai_msg)
            elif isinstance(msg, SystemMessage):
                openai_history.append({"role": "system", "content": msg.content})
            elif isinstance(msg, ToolMessage):
                openai_history.append(
                    {"role": "tool", "content": msg.content, "tool_call_id": msg.tool_call_id, "name": msg.name}
                )
        return openai_history

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
            tools=tools,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        final_response: ChatCompletion = await self._tool_loop.run(initial_response=initial_response, adapter=adapter)

        return messages, final_response

    @staticmethod
    def _clean_history(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Removes intermediate tool calls and outputs from the history to save tokens.

        Keeps user messages, system instructions, and assistant messages that have content.

        Args:
            messages: The list of messages to clean.

        Returns:
            A list of cleaned messages.
        """
        # logger.debug(f"Cleaning history. Original size: {len(messages)}")
        cleaned_messages = []
        for msg in messages:
            role = msg.get("role")
            if role == "tool":
                continue

            if role == "assistant":
                # If the message has tool_calls, we only keep it if it has content.
                # We strip the tool_calls field to avoid sending it back to the model in future turns.
                if msg.get("tool_calls"):
                    if msg.get("content"):
                        # Create a clean copy without tool_calls
                        clean_msg = {k: v for k, v in msg.items() if k != "tool_calls"}
                        cleaned_messages.append(clean_msg)
                    # If no content, we skip this message entirely
                else:
                    # No tool calls, keep the message
                    cleaned_messages.append(msg)
            else:
                # Keep system and user messages
                cleaned_messages.append(msg)

        # logger.debug(f"History cleaned. New size: {len(cleaned_messages)}")
        return cleaned_messages

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
        generic_history = GenericOpenAI._convert_to_generic_history(history)

        return ChatResult(content=message_content, history=generic_history, raw=response)

    @staticmethod
    def _convert_to_generic_history(history: List[Dict[str, Any]]) -> List[BaseMessage]:
        """
        Converts OpenAI specific dictionary history to generic BaseMessage history.

        Args:
            history: List of OpenAI message dictionaries.

        Returns:
            List of BaseMessage objects.
        """
        generic_history: List[BaseMessage] = []
        for msg in history:
            role = msg.get("role")
            content = msg.get("content")
            tool_calls = msg.get("tool_calls")

            # Handle None content
            if content is None:
                if role == "assistant" and tool_calls:
                    content = ""
                else:
                    # Skip messages with None content (except assistant with tools)
                    continue

            # Handle empty content
            if content == "":
                # Keep if it's an assistant with tool calls
                if role == "assistant" and tool_calls:
                    pass
                # Keep if it's a tool response (must not be skipped to maintain conversation flow)
                elif role == "tool":
                    pass
                else:
                    # Skip empty user or system messages
                    continue

            if role == "user":
                generic_history.append(UserMessage(content=content))
            elif role == "assistant":
                generic_history.append(AssistantMessage(content=content, tool_calls=tool_calls))
            elif role == "system":
                generic_history.append(SystemMessage(content=content))
            elif role == "tool":
                tool_call_id = msg.get("tool_call_id")
                name = msg.get("name", "unknown_tool")
                if tool_call_id:
                    generic_history.append(ToolMessage(content=content, tool_call_id=tool_call_id, name=name))
            # Handle other roles if necessary
        return generic_history

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

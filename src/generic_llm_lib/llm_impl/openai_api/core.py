from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionToolParam
from typing import List, Tuple, Optional, Any, Dict, Iterable, cast
import logging
from generic_llm_lib.llm_core import GenericLLM, ToolRegistry
from generic_llm_lib.llm_core import ToolExecutionLoop
from generic_llm_lib.llm_core.messages.models import BaseMessage, UserMessage, AssistantMessage, SystemMessage
from .models import OpenAIMessageResponse, OpenAIChatResponse, OpenAITokens
from .adapter import OpenAIToolAdapter

logger = logging.getLogger(__name__)


class GenericOpenAI(GenericLLM):
    """
    Implementation of GenericLLM for OpenAI's models.
    Handles chat sessions and automatic function calling loops.
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        model_name: str,
        sys_instruction: str,
        registry: Optional[ToolRegistry] = None,
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
        self.registry: Optional[ToolRegistry] = registry
        self.max_function_loops = max_function_loops
        self.sys_instruction = sys_instruction
        self.temperature = temp
        self.max_tokens = max_tokens
        self.tool_timeout = tool_timeout

        self.client: AsyncOpenAI = client

        self._tool_loop = ToolExecutionLoop(
            registry=registry,
            max_function_loops=max_function_loops,
            tool_timeout=tool_timeout,
            argument_error_formatter=self._format_argument_error,
        )

    async def ask(self, prompt: str, model: Optional[str] = None) -> OpenAIMessageResponse:
        """
        Generates a text response from the LLM based on a single prompt.
        This method handles potential function calls internally by initiating a temporary chat session.

        Args:
            prompt: The user's input prompt.
            model: Optional. Overrides the default model for this specific request.

        Returns:
            OpenAIMessageResponse: The generated text response from the LLM.
        """
        # We use a temporary chat session to handle the tool execution loop (Model -> Tool -> Model)
        # We start with an empty history.
        # logger.debug(f"ask() called with prompt: {prompt}")
        response: OpenAIChatResponse = await self.chat([], prompt)

        return response.last_response

    async def chat(
        self, history: List[BaseMessage], user_prompt: str, clean_history: bool = False
    ) -> OpenAIChatResponse:
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
            OpenAIChatResponse: An object containing:
            - The final text response from the LLM after all function calls (if any) are resolved.
            - The updated conversation history, including the user's prompt, LLM's responses,
              and any tool calls/responses.
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
            # We cast the tool_object to the expected type for OpenAI
            # The registry returns List[Dict[str, Any]], which is compatible with ChatCompletionToolParam
            # if structured correctly.
            tools = cast(Iterable[ChatCompletionToolParam], self.registry.tool_object)

            # logger.debug(f"Tools registered: {[t.get('function', {}).get('name') for t in tools]}")

        # Initial call to OpenAI
        # logger.debug(f"Sending initial request to OpenAI model: {self.model}")
        # We need to cast messages to the expected type for OpenAI
        # The history is List[Dict[str, Any]], but OpenAI expects Iterable[ChatCompletionMessageParam]
        # Since Dict[str, Any] is compatible with the structure, we can cast it.
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

    def _convert_history(self, history: List[BaseMessage]) -> List[Dict[str, Any]]:
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
                openai_history.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                openai_history.append({"role": "system", "content": msg.content})
            # Tool messages handling would be more complex
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

        # Convert Iterable[ChatCompletionToolParam] back to List[Dict[str, Any]] for the adapter if needed,
        # or update adapter to accept Iterable. For now, we assume it's a list or cast it.
        # We cast tools to Iterable[Dict[str, Any]] to satisfy the list constructor if needed, or just use list()
        # The error was: Argument 1 to "list" has incompatible type "Iterable[ChatCompletionFunctionToolParam]"; expected "Iterable[dict[str, Any]]"
        # ChatCompletionFunctionToolParam is a TypedDict, which is compatible with dict at runtime but mypy is strict.
        tools_list: Optional[List[Dict[str, Any]]] = list(cast(Iterable[Dict[str, Any]], tools)) if tools else None

        adapter = OpenAIToolAdapter(
            client=self.client,
            model=self.model,
            messages=messages,
            tools=tools_list,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        final_response = await self._tool_loop.run(initial_response=initial_response, adapter=adapter)

        return messages, cast(ChatCompletion, final_response)

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
    def _build_response(response: ChatCompletion, history: List[Dict[str, Any]]) -> OpenAIChatResponse:
        """
        Constructs the final OpenAIChatResponse object from the raw API response and chat history.

        Args:
            response: The final ChatCompletion from the model.
            history: The chat history list.

        Returns:
            OpenAIChatResponse: The structured response containing text, tokens, and history.
        """
        if response.choices:
            message_content = response.choices[0].message.content or ""
        else:
            message_content = ""

        usage = response.usage
        if usage:
            response_tokens = OpenAITokens(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
            )
        else:
            response_tokens = OpenAITokens()

        response_message = OpenAIMessageResponse(text=message_content, tokens=response_tokens)

        # Convert OpenAI history back to generic BaseMessage history
        generic_history = GenericOpenAI._convert_to_generic_history(history)

        openai_response = OpenAIChatResponse(last_response=response_message, history=generic_history)

        return openai_response

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

            if not content:
                continue

            if role == "user":
                generic_history.append(UserMessage(content=content))
            elif role == "assistant":
                generic_history.append(AssistantMessage(content=content))
            elif role == "system":
                generic_history.append(SystemMessage(content=content))
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

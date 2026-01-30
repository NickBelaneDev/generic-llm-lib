from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from typing import List, Tuple, Optional, Any, Dict
import inspect
import json
import asyncio
import logging
from llm_core import GenericLLM, ToolRegistry
from .models import OpenAIMessageResponse, OpenAIChatResponse, OpenAITokens
from .tool_helper import ToolHelper

import pprint

logger = logging.getLogger(__name__)

class GenericOpenAI(GenericLLM):
    """
    Implementation of GenericLLM for OpenAI's models.
    Handles chat sessions and automatic function calling loops.
    """

    def __init__(self,
                 client: AsyncOpenAI,
                 model_name: str,
                 sys_instruction: str,
                 registry: Optional[ToolRegistry] = None,
                 temp: float = 1.0,
                 max_tokens: int = 100,
                 max_function_loops: int = 5,
                 tool_timeout: float = 60.0
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
        
        self.tool_helper = ToolHelper(
            client=self.client,
            model=self.model,
            registry=self.registry,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_function_loops=self.max_function_loops,
            tool_timeout=self.tool_timeout
        )

    async def ask(self, prompt: str, model: str = None) -> OpenAIMessageResponse:
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

    async def chat(self,
                   history: List[Dict[str, Any]],
                   user_prompt: str) -> OpenAIChatResponse:
        """
        Processes a single turn of a chat conversation, including handling user input,
        generating LLM responses, and executing any requested function calls.

        The method supports a multi-turn interaction where the LLM can call functions
        and receive their results within the same turn, up to `max_function_loops` times.

        Args:
            history: A list of message dictionaries representing the conversation history.
            user_prompt: The current message from the user.

        Returns:
            OpenAIChatResponse: An object containing:
            - The final text response from the LLM after all function calls (if any) are resolved.
            - The updated conversation history, including the user's prompt, LLM's responses,
              and any tool calls/responses.
        """
        # logger.debug(f"chat() called. History length: {len(history)}, User prompt: {user_prompt}")

        # Prepare the messages list. If history is empty, add system instruction first.
        messages = list(history)
        if not messages and self.sys_instruction:
            messages.append({"role": "system", "content": self.sys_instruction})
        
        # Add user message
        messages.append({"role": "user", "content": user_prompt})

        # Get tools configuration
        tools = None
        if self.registry:
            tools = self.registry.tool_object
            # logger.debug(f"Tools registered: {[t.get('function', {}).get('name') for t in tools]}")

        # Initial call to OpenAI
        # logger.debug(f"Sending initial request to OpenAI model: {self.model}")
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        if not response.choices:
            # logger.debug(f"Initial response has no choices. Response dump: {response.model_dump()}")
            pass
        else:
            # logger.debug(f"Initial response received. Finish reason: {response.choices[0].finish_reason}")
            pass

        # Handle function calls loop
        messages, final_response = await self._handle_function_calls(messages, response, tools)
        
        # Clean history to remove intermediate tool calls and outputs to save tokens
        clean_history = self._clean_history(messages)
        
        return self._build_response(final_response, clean_history)

    async def _handle_function_calls(self, 
                                     messages: List[Dict[str, Any]], 
                                     initial_response: ChatCompletion,
                                     tools: Optional[List[Dict[str, Any]]]) -> Tuple[List[Dict[str, Any]], ChatCompletion]:
        """
        Handles the function calling loop.
        Delegates to ToolHelper.

        Args:
            messages: The current list of messages in the conversation.
            initial_response: The initial response object from the model.
            tools: The list of available tools (unused here as ToolHelper has access to registry).

        Returns:
            Tuple[List[Dict[str, Any]], ChatCompletion]: The updated messages list and the final response object from the model.
        """
        return await self.tool_helper.handle_function_calls(messages, initial_response)

    @staticmethod
    def _clean_history(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Removes intermediate tool calls and outputs from the history to save tokens.
        Keeps user messages, system instructions, and assistant messages that have content.
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
                total_tokens=usage.total_tokens
            )
        else:
            response_tokens = OpenAITokens()

        response_message = OpenAIMessageResponse(
            text=message_content,
            tokens=response_tokens
        )

        openai_response = OpenAIChatResponse(
            last_response=response_message,
            history=history
        )
    
        return openai_response
from google import genai
from google.genai import types
from typing import List, Tuple, Optional, Any
import inspect
import asyncio
from google.genai.types import GenerateContentResponse
from llm_core import GenericLLM, ToolRegistry
from llm_core.exceptions import ToolExecutionError, ToolNotFoundError
from .models import GeminiMessageResponse, GeminiChatResponse, GeminiTokens


class GenericGemini(GenericLLM):
    """
    Implementation of GenericLLM for Google's Gemini models.
    Handles chat sessions and automatic function calling loops.
    """

    def __init__(self,
                 client: genai.Client,
                 model_name: str,
                 sys_instruction: str,
                 registry: Optional[ToolRegistry] = None,
                 temp: float = 1.0,
                 max_tokens: int = 100,
                 max_function_loops: int = 5
                 ):
        """
        Initializes the GenericGemini LLM wrapper.

        Args:
            client: The initialized Google GenAI client.
            model_name: The identifier for the Gemini model to use (e.g., 'gemini-pro', 'gemini-flash-latest').
            sys_instruction: A system-level instruction or persona for the LLM.
            registry: An optional ToolRegistry instance containing tools the LLM can use.
            temp: The temperature for text generation, controlling randomness.
            max_tokens: The maximum number of tokens to generate in the response.
            max_function_loops: The maximum number of consecutive function calls the LLM can make.
        """
        self.model: str = model_name
        self.registry: Optional[ToolRegistry] = registry
        self.max_function_loops = max_function_loops

        # Only include tools if there are any registered
        tools_config = None
        if self.registry:
            tool_obj = self.registry.tool_object
            if tool_obj:
                tools_config = [tool_obj]

        self.config: types.GenerateContentConfig = types.GenerateContentConfig(
            system_instruction=sys_instruction,
            temperature=temp,
            max_output_tokens=max_tokens,

            tools=tools_config
        )

        self.client: genai.Client = client

    async def ask(self, prompt: str, model: str = None) -> GeminiMessageResponse:
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

        # We use a temporary chat session to handle the tool execution loop (Model -> Tool -> Model)
        # We start with an empty history.
        response: GeminiChatResponse = await self.chat([], prompt)

        return response.last_response

    async def chat(self,
                   history: List[types.Content],
                   user_prompt: str) -> GeminiChatResponse:
        """
        Processes a single turn of a chat conversation, including handling user input,
        generating LLM responses, and executing any requested function calls.

        The method supports a multi-turn interaction where the LLM can call functions
        and receive their results within the same turn, up to `max_function_loops` times.

        Args:
            history: A list of `types.Content` objects representing the conversation history.
            user_prompt: The current message from the user.

        Returns:
            GeminiChatResponse: An object containing:
            - The final text response from the LLM after all function calls (if any) are resolved.
            - The updated conversation history, including the user's prompt, LLM's responses,
              and any tool calls/responses.
        """
        # Create a chat session with the provided history
        chat = self.client.chats.create(
            model=self.model,
            config=self.config,
            history=history
        )
        
        # Send the user message
        _response = chat.send_message(user_prompt)
        response, chat = await self._handle_function_calls(_response, chat)
        
        # Get full history
        full_history = chat.get_history()
        
        # Clean history to remove intermediate tool calls and outputs to save tokens
        cleaned_history = self._clean_history(full_history)
        
        gemini_response = self._build_response(response, cleaned_history)
        
        return gemini_response
    
    async def _handle_function_calls(self, response: GenerateContentResponse, chat) -> Tuple[GenerateContentResponse, Any]:
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
        
        for _ in range(self.max_function_loops):
            # Collect all function calls from the response parts
            # Gemini can return multiple function calls in a single turn (parallel function calling)
            function_calls = [p.function_call for p in (response.parts or []) if p.function_call]
            
            if not function_calls:
                # If there are no function calls, we have our final text response.
                break

            parts_to_send = []

            for function_call in function_calls:
                function_name = function_call.name

                if not self.registry or function_name not in self.registry.implementations:
                    # If tool is not found, report error to LLM
                    parts_to_send.append(
                        types.Part(function_response=types.FunctionResponse(
                            name=function_name,
                            response={"error": f"Tool '{function_name}' not found in registry."},
                        ))
                    )
                    continue

                try:
                    function_args = self._normalize_function_args(function_call)
                    tool_function = self.registry.implementations[function_name]
                    # Execute the tool, either async oder sync
                    if inspect.iscoroutinefunction(tool_function):
                        function_result = await tool_function(**function_args)
                    else:
                        # Run synchronous tools in a thread to avoid blocking the event loop
                        function_result = await asyncio.to_thread(tool_function, **function_args)
                    
                    # Create the response part
                    parts_to_send.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=function_name,
                                response={"result": function_result},
                            )
                        )
                    )

                except Exception as e:
                    # Send execution error back to the LLM
                    parts_to_send.append(
                        types.Part(function_response=types.FunctionResponse(
                            name=function_name,
                            response={"error": str(e)},
                        ))
                    )
            
            # Send all function results back to the model in a single message
            if parts_to_send:
                response = chat.send_message(parts_to_send)
                
        return response, chat

    @staticmethod
    def _normalize_function_args(function_call: Any) -> dict:
        raw_args = getattr(function_call, "args", None)
        if raw_args is None:
            return {}
        if isinstance(raw_args, dict):
            return raw_args
        try:
            return dict(raw_args)
        except (TypeError, ValueError) as exc:
            raise ToolExecutionError(
                f"Failed to parse arguments for tool '{function_call.name}': {exc}"
            ) from exc

    def _clean_history(self, history: List[types.Content]) -> List[types.Content]:
        """
        Removes intermediate tool calls and outputs from the history to save tokens.
        Keeps user messages, system instructions, and assistant messages that have content.
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
    def _build_response(response: GenerateContentResponse, history: List[types.Content]) -> GeminiChatResponse:
        """
        Constructs the final GeminiChatResponse object from the raw API response and chat history.

        Args:
            response: The final GenerateContentResponse from the model.
            history: The chat history list.

        Returns:
            GeminiChatResponse: The structured response containing text, tokens, and history.
        """
        text_response = "".join([p.text for p in response.parts if p.text]) if response.parts else ""

        # Handle cases where usage_metadata might be missing
        if response.usage_metadata:
            response_tokens = GeminiTokens(
                candidate_token_count=response.usage_metadata.candidates_token_count,
                prompt_token_count=response.usage_metadata.prompt_token_count,
                total_token_count=response.usage_metadata.total_token_count,
                thoughts_token_count=response.usage_metadata.thoughts_token_count,
                tool_use_prompt_token_count=response.usage_metadata.tool_use_prompt_token_count
            )
        else:
            response_tokens = GeminiTokens()

        response_message = GeminiMessageResponse(
            text=text_response,
            tokens=response_tokens
        )

        gemini_response = GeminiChatResponse(
            last_response=response_message,
            history=history
        )
    
        return gemini_response

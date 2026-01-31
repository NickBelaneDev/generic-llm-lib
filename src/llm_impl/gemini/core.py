from google import genai
from google.genai import types
from typing import List, Tuple, Optional, Any, Sequence
from google.genai.types import GenerateContentResponse
from llm_core import GenericLLM, ToolRegistry
from llm_core.tools import ToolExecutionLoop, ToolCallRequest, ToolCallResult, ToolAdapter
from .models import GeminiMessageResponse, GeminiChatResponse, GeminiTokens

class GeminiToolAdapter(ToolAdapter):
    """Adapter for Gemini tool handling."""

    def __init__(self, chat_session: Any):
        self.chat_session = chat_session

    def get_tool_calls(self, response: GenerateContentResponse) -> Sequence[ToolCallRequest]:
        function_calls = [p.function_call for p in (response.parts or []) if p.function_call]
        return [
            ToolCallRequest(
                name=function_call.name,
                arguments=getattr(function_call, "args", None),
            )
            for function_call in function_calls
        ]

    def record_assistant_message(self, response: GenerateContentResponse) -> None:
        # Gemini handles history internally in the chat session, so we don't need to manually append.
        pass

    def build_tool_response_message(self, result: ToolCallResult) -> types.Part:
        return types.Part(
            function_response=types.FunctionResponse(
                name=result.name,
                response=result.response,
            )
        )

    async def send_tool_responses(self, messages: Sequence[types.Part]) -> GenerateContentResponse:
        return self.chat_session.send_message(list(messages))


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
                 max_function_loops: int = 5,
                 tool_timeout: float = 180.0
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
            tool_timeout: The maximum time in seconds to wait for a tool execution.
        """
        self.model: str = model_name
        self.registry: Optional[ToolRegistry] = registry
        self.max_function_loops = max_function_loops
        self.tool_timeout = tool_timeout
        self._tool_loop = ToolExecutionLoop(
            registry=registry,
            max_function_loops=max_function_loops,
            tool_timeout=tool_timeout,
        )

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
        
        if not response:
            return response, chat

        adapter = GeminiToolAdapter(chat)
        final_response = await self._tool_loop.run(
            initial_response=response,
            adapter=adapter
        )

        return final_response, chat

    @staticmethod
    def _clean_history(history: List[types.Content]) -> List[types.Content]:
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

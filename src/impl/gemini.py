from google import genai
from google.genai import types
from typing import List, Dict, Any, Tuple, Optional
from src.llm_core import GenericLLM, ToolRegistry


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
        Initialize the Gemini wrapper.

        Args:
            client: The initialized Google GenAI client.
            model_name: The model identifier (e.g., 'gemini-2.0-flash-exp').
            sys_instruction: System prompt for the agent.
            registry: Optional ToolRegistry containing available tools.
            temp: Temperature for generation.
            max_tokens: Max output tokens.
            max_function_loops: Maximum number of sequential function calls allowed.
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

    async def ask(self, prompt: str, model: str = None) -> str:
        """Generates content from a given prompt. Handles function calls via a temporary chat."""
        if not model:
            model = self.model

        # We use a temporary chat session to handle the tool execution loop (Model -> Tool -> Model)
        # We start with an empty history.
        response, _ = await self.chat([], prompt)
        return response

    async def chat(self,
                   history: List[types.Content],
                   user_prompt: str) -> Tuple[str, List[types.Content]]:
        """
        Processes a single turn of a chat, handling user input and any subsequent
        function calls requested by the model.

        Args:
            history: The active chat session object (list of types.Content).
            user_prompt: The user's message.

        Returns:
            The final text response from the LLM and the updated history.
        """
        # Create a chat session with the provided history
        chat = self.client.chats.create(
            model=self.model,
            config=self.config,
            history=history
        )
        
        # Send the user message
        response = chat.send_message(user_prompt)

        # This loop continues as long as the model requests function calls.
        for _ in range(self.max_function_loops):
            # Search for a function call in any part of the response
            target_part = next((p for p in (response.parts or []) if p.function_call), None)
            if not target_part:
                # If there's no function call, we have our final text response.
                break

            # --- Execute the function call ---
            function_call = target_part.function_call
            function_name = function_call.name

            if not self.registry or function_name not in self.registry.implementations:
                 # If tool is not found, report error to LLM
                 response = chat.send_message(
                    types.Part(function_response=types.FunctionResponse(
                        name=function_name,
                        response={"error": f"Tool '{function_name}' not found in registry."},
                    ))
                )
                 continue

            try:
                # 1. Look up the implementation and call it with the provided arguments
                tool_function = self.registry.implementations[function_name]
                function_result = tool_function(**dict(function_call.args))

                # 2. Send the function's result back to the model.
                response = chat.send_message(
                    types.Part(
                        function_response=types.FunctionResponse(
                        name=function_name,
                        response={"result": function_result},
                    )
                    )
                )

            except Exception as e:
                # Send execution error back to the LLM
                response = chat.send_message(
                    types.Part(function_response=types.FunctionResponse(
                        name=function_name,
                        response={"error": str(e)},
                    ))
                )

        # After the loop, return the final text response from the LLM.
        text_response = "".join([p.text for p in response.parts if p.text]) if response.parts else ""
        
        # Return text and the curated history (which includes the tool calls/responses)
        return text_response, chat.get_history(curated=True)

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from typing import List, Tuple, Optional, Any, Dict
import inspect
import json
import asyncio
from llm_core import ToolRegistry
from llm_core.exceptions import ToolExecutionError, ToolNotFoundError

class ToolHelper:
    def __init__(self,
                 client: AsyncOpenAI,
                 model: str,
                 registry: Optional[ToolRegistry],
                 temperature: float,
                 max_tokens: int,
                 max_function_loops: int):
        self.client = client
        self.model = model
        self.registry = registry
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_function_loops = max_function_loops

    async def handle_function_calls(self,
                                    messages: List[Dict[str, Any]],
                                    initial_response: ChatCompletion) -> Tuple[List[Dict[str, Any]], ChatCompletion]:
        
        current_response = initial_response
        tools = self.registry.tool_object if self.registry else None

        if not current_response.choices:
            return messages, current_response

        current_message = current_response.choices[0].message
        
        for loop_index in range(self.max_function_loops):
            tool_calls = current_message.tool_calls
            
            if not tool_calls:
                messages.append(current_message.model_dump())
                return messages, current_response

            messages.append(current_message.model_dump())

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                tool_call_id = tool_call.id
                
                try:
                    function_args = self._parse_tool_arguments(tool_call.function.arguments)
                except ValueError as e:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": function_name,
                        "content": json.dumps({"error": f"Failed to decode function arguments: {e}"})
                    })
                    continue

                if not self.registry or function_name not in self.registry.tools:
                    error_msg = f"Tool '{function_name}' not found in registry."
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": function_name,
                        "content": json.dumps({"error": error_msg})
                    })
                    continue
                
                tool_def = self.registry.tools[function_name]

                # Validate and coerce arguments using the Pydantic model if available
                if tool_def.args_model:
                    try:
                        validated_args = tool_def.args_model(**function_args)
                        function_args = validated_args.model_dump()
                    except Exception as validation_error:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": function_name,
                            "content": json.dumps({"error": f"Argument validation failed: {validation_error}"})
                        })
                        continue

                messages, function_result = await self._execute_tool(
                    tool_def.func,
                    function_args,
                    messages,
                    tool_call_id,
                    function_name
                )
            
            current_response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            if not current_response.choices:
                return messages, current_response

            current_message = current_response.choices[0].message
            
        messages.append(current_message.model_dump())
        
        return messages, current_response

    @staticmethod
    def _parse_tool_arguments(arguments: Optional[str]) -> Dict[str, Any]:
        if not arguments:
            return {}
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(str(exc)) from exc
        if parsed is None:
            return {}
        if not isinstance(parsed, dict):
            raise ValueError("Function arguments must decode to a JSON object.")
        return parsed

    async def _execute_tool(self,
                            tool_function: Any,
                            function_args: Dict[str, Any],
                            messages: List[Dict[str, Any]],
                            tool_call_id: str,
                            function_name: str) -> Tuple[List[Dict[str, Any]], Any]:
        
        try:
            if inspect.iscoroutinefunction(tool_function):
                function_result = await tool_function(**function_args)
            else:
                function_result = await asyncio.to_thread(tool_function, **function_args)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": function_name,
                "content": json.dumps({"result": function_result})
            })
            return messages, function_result
        
        except Exception as e:
            # Wrap the original exception in a ToolExecutionError for better context if needed,
            # but here we primarily want to report the error back to the LLM.
            # We can log the specific exception type if we had a logger.
            error_message = str(e)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": function_name,
                "content": json.dumps({"error": error_message})
            })
            return messages, None
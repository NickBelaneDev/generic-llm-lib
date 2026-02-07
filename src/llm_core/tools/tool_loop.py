"""Shared tool execution loop utilities for LLM implementations."""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any, Callable, Dict, Optional

from llm_core.exceptions.exceptions import ToolExecutionError
from llm_core.logger import get_logger
from .adapter import ToolAdapter
from .call_protocol import ToolCallRequest, ToolCallResult
from .registry import ToolRegistry

logger = get_logger(__name__)


class ToolExecutionLoop:
    """Centralized tool execution loop for LLM providers.

    This loop handles argument normalization, validation, tool execution,
    and error handling in a provider-agnostic way.
    """

    # Exceptions that are considered recoverable and should be returned to the LLM.
    # System errors (like ConnectionError, MemoryError) are NOT included and will
    # propagate, stopping the loop.
    RECOVERABLE_ERRORS = (
        ToolExecutionError,
        FileNotFoundError,
        FileExistsError,
        PermissionError,
        IsADirectoryError,
        NotADirectoryError,
        ValueError,
        TypeError,
    )

    def __init__(
        self,
        *,
        registry: Optional[ToolRegistry],
        max_function_loops: int,
        tool_timeout: float = 180.0,
        argument_error_formatter: Optional[Callable[[str, Exception], str]] = None,
    ) -> None:
        """Initialize the tool execution loop.

        Args:
            registry: Tool registry used to resolve tool definitions.
            max_function_loops: Maximum number of tool-call iterations allowed.
            tool_timeout: Timeout in seconds for tool execution. Default is 180 seconds.
            argument_error_formatter: Optional formatter for argument parsing errors.
        """
        self._registry = registry
        self._max_function_loops = max_function_loops
        self._tool_timeout = tool_timeout
        self._argument_error_formatter = argument_error_formatter or self._default_argument_error

    async def run(
        self,
        *,
        initial_response: Any,
        adapter: ToolAdapter,
    ) -> Any:
        """Run the tool execution loop.

        Args:
            initial_response: Initial provider response to inspect.
            adapter: The provider-specific adapter for tool handling.

        Returns:
            The final provider response after tool execution completes.
        """
        current_response = initial_response

        for loop_index in range(self._max_function_loops):
            tool_calls = list(adapter.get_tool_calls(current_response))

            if not tool_calls:
                logger.debug("No tool calls found in response. Loop finished.")
                adapter.record_assistant_message(current_response)
                return current_response

            logger.info(f"Loop {loop_index + 1}/{self._max_function_loops}: Processing {len(tool_calls)} tool call(s).")
            adapter.record_assistant_message(current_response)

            response_messages = []
            tasks = [self._handle_tool_call(tc) for tc in tool_calls]
            results = await asyncio.gather(*tasks)
            for result in results:
                response_messages.append(adapter.build_tool_response_message(result))

            if response_messages:
                current_response = await adapter.send_tool_responses(response_messages)
            else:
                logger.debug("No tool responses generated. Loop finished.")
                return current_response

        logger.warning(f"Max tool loops ({self._max_function_loops}) reached. Stopping execution.")
        return current_response

    async def _handle_tool_call(self, tool_call: ToolCallRequest) -> ToolCallResult:
        """Handle a single tool call request.

        Validates the tool existence, normalizes arguments, validates against
        schema (if present), and executes the tool.

        Args:
            tool_call: The tool call request containing name, ID, and arguments.

        Returns:
            The result of the tool execution, including any errors.
        """
        logger.debug(f"Handling tool call: {tool_call.name} (ID: {tool_call.call_id})")

        if not self._registry or tool_call.name not in self._registry.tools:
            msg = f"Tool '{tool_call.name}' not found in registry."
            logger.warning(msg)
            return ToolCallResult(
                name=tool_call.name,
                response={"error": msg},
                call_id=tool_call.call_id,
            )

        tool_def = self._registry.tools[tool_call.name]

        try:
            function_args = self._normalize_function_args(tool_call.name, tool_call.arguments)
        except ToolExecutionError as exc:
            msg = str(exc)
            logger.warning(f"Argument normalization failed for '{tool_call.name}': {msg}")
            return ToolCallResult(
                name=tool_call.name,
                response={"error": msg},
                call_id=tool_call.call_id,
            )

        if tool_def.args_model:
            try:
                validated_args = tool_def.args_model(**function_args)
                function_args = validated_args.model_dump()
            except Exception as validation_error:
                msg = f"Argument validation failed: {validation_error}"
                logger.warning(f"Validation error for '{tool_call.name}': {msg}")
                return ToolCallResult(
                    name=tool_call.name,
                    response={"error": msg},
                    call_id=tool_call.call_id,
                )

        # Execute the tool function
        try:
            logger.info(f"Executing tool '{tool_call.name}'...")
            function_result = await self._execute_tool(tool_def.func, function_args)
            logger.info(f"Tool '{tool_call.name}' executed successfully.")
        except self.RECOVERABLE_ERRORS as exc:
            msg = str(exc)
            logger.warning(f"Recoverable error in '{tool_call.name}': {msg} ({type(exc).__name__})")
            return ToolCallResult(
                name=tool_call.name,
                response={"error": msg},
                call_id=tool_call.call_id,
            )

        return ToolCallResult(
            name=tool_call.name,
            response={"result": function_result},
            call_id=tool_call.call_id,
        )

    def _normalize_function_args(self, tool_name: str, raw_args: Any) -> Dict[str, Any]:
        """Normalize tool arguments into a dictionary.

        Handles JSON strings, dictionaries, or None values.

        Args:
            tool_name: Name of the tool (for error reporting).
            raw_args: The raw arguments (dict, string, or None).

        Returns:
            A dictionary of normalized arguments.

        Raises:
            ToolExecutionError: If arguments cannot be parsed or are invalid.
        """
        if raw_args is None or raw_args == "":
            return {}

        if isinstance(raw_args, dict):
            return raw_args

        if isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args)

            except json.JSONDecodeError as exc:
                raise ToolExecutionError(self._argument_error_formatter(tool_name, exc)) from exc

            if parsed is None:
                return {}

            if not isinstance(parsed, dict):
                msg = ValueError("Function arguments must decode to a JSON object.")
                raise ToolExecutionError(self._argument_error_formatter(tool_name, msg))

            return parsed

        try:
            return dict(raw_args)
        except (TypeError, ValueError) as exc:
            raise ToolExecutionError(self._argument_error_formatter(tool_name, exc)) from exc

    async def _execute_tool(self, tool_function: Any, function_args: Dict[str, Any]) -> Any:
        """Execute the tool function, handling async/sync and timeouts.

        Args:
            tool_function: The callable to execute.
            function_args: The arguments to pass to the function.

        Returns:
            The result of the function execution.

        Raises:
            ToolExecutionError: If execution times out.
        """
        try:
            if inspect.iscoroutinefunction(tool_function):
                return await asyncio.wait_for(
                    tool_function(**function_args),
                    timeout=self._tool_timeout,
                )

            return await asyncio.wait_for(
                asyncio.to_thread(tool_function, **function_args),
                timeout=self._tool_timeout,
            )

        except asyncio.TimeoutError as exc:
            msg = f"Tool execution timed out after {self._tool_timeout} seconds."
            raise ToolExecutionError(msg) from exc

    @staticmethod
    def _default_argument_error(tool_name: str, error: Exception) -> str:
        """Format a default error message for argument parsing failures.

        Args:
            tool_name: Name of the tool.
            error: The exception that occurred.

        Returns:
            A formatted error message string.
        """
        return f"Failed to parse arguments for tool '{tool_name}': {error}"

"""Shared tool execution helper utilities for LLM implementations."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional, Sequence

from .exceptions import ToolExecutionError
from .registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolCallRequest:
    """Represents a normalized tool call request from an LLM response."""

    name: str
    arguments: Any
    call_id: Optional[str] = None


@dataclass(frozen=True)
class ToolCallResult:
    """Represents the outcome of executing a tool call."""

    name: str
    response: Dict[str, Any]
    call_id: Optional[str] = None


class ToolHelper:
    """Centralized tool execution helper for LLM providers.

    This helper handles argument normalization, validation, tool execution,
    and error handling in a provider-agnostic way.
    """

    def __init__(
        self,
        *,
        registry: Optional[ToolRegistry],
        max_function_loops: int,
        tool_timeout: float = 60.0,
        argument_error_formatter: Optional[Callable[[str, Exception], str]] = None,
    ) -> None:
        """Initialize the tool execution helper.

        Args:
            registry: Tool registry used to resolve tool definitions.
            max_function_loops: Maximum number of tool-call iterations allowed.
            tool_timeout: Timeout in seconds for tool execution.
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
        get_tool_calls: Callable[[Any], Sequence[ToolCallRequest]],
        record_assistant_message: Callable[[Any], None],
        build_tool_response_message: Callable[[ToolCallResult], Any],
        send_tool_responses: Callable[[Sequence[Any]], Awaitable[Any]],
    ) -> Any:
        """Run the tool execution loop.

        Args:
            initial_response: Initial provider response to inspect.
            get_tool_calls: Extracts tool calls from a provider response.
            record_assistant_message: Persists the assistant response in provider history.
            build_tool_response_message: Builds a provider message from tool results.
            send_tool_responses: Sends tool results back to the provider to get a new response.

        Returns:
            The final provider response after tool execution completes.
        """

        current_response = initial_response

        for _ in range(self._max_function_loops):
            tool_calls = list(get_tool_calls(current_response))

            if not tool_calls:
                record_assistant_message(current_response)
                return current_response

            record_assistant_message(current_response)

            response_messages = []
            for tool_call in tool_calls:
                result = await self._handle_tool_call(tool_call)
                response_messages.append(build_tool_response_message(result))

            if response_messages:
                current_response = await send_tool_responses(response_messages)
            else:
                return current_response

        return current_response

    async def _handle_tool_call(self, tool_call: ToolCallRequest) -> ToolCallResult:
        if not self._registry or tool_call.name not in self._registry.tools:
            msg = f"Tool '{tool_call.name}' not found in registry."
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
            logger.warning("ToolExecutionError in '%s': %s", tool_call.name, msg)
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
                logger.warning("ToolExecutionError in '%s': %s", tool_call.name, msg)
                return ToolCallResult(
                    name=tool_call.name,
                    response={"error": msg},
                    call_id=tool_call.call_id,
                )

        try:
            function_result = await self._execute_tool(tool_def.func, function_args)
        except ToolExecutionError as exc:
            msg = str(exc)
            logger.warning("ToolExecutionError in '%s': %s", tool_call.name, msg)
            return ToolCallResult(
                name=tool_call.name,
                response={"error": msg},
                call_id=tool_call.call_id,
            )
        except Exception as exc:
            logger.error(
                "Unexpected error executing tool '%s': %s",
                tool_call.name,
                exc,
                exc_info=True,
            )
            msg = "An internal error occurred during tool execution."
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
        return f"Failed to parse arguments for tool '{tool_name}': {error}"

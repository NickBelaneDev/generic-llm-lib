"""Bridge MCP server tools into ToolRegistry entries through async stdio client sessions."""

import logging
from contextlib import AsyncExitStack
from types import TracebackType
from typing import Any, Optional, Type, cast

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.types import Tool as MCPTool, TextContent, ImageContent, EmbeddedResource

from generic_llm_lib.llm_core import ToolRegistry
from generic_llm_lib.llm_core import SchemaValidator

logger = logging.getLogger(__name__)

__all__ = ["MCPClientWrapper"]


class MCPClientWrapper:
    """Wrapper for the Model Context Protocol (MCP) client to integrate with ToolRegistry."""

    def __init__(self, command: str, args: list[str], env: Optional[dict[str, str]] = None):
        """Initializes the wrapper with parameters for the MCP server process.

        Args:
            command: The command to run the server.
            args: List of arguments for the command.
            env: Optional dictionary of environment variables.
        """
        self._server_params = StdioServerParameters(command=command, args=args, env=env)
        self._session: Optional[ClientSession] = None
        self._exit_stack = AsyncExitStack()

    async def __aenter__(self) -> "MCPClientWrapper":
        """Opens the connection (transport) and initializes the session.

        Returns:
            The initialized MCPClientWrapper instance.
        """
        logger.debug("Initializing MCP client session...")
        read, write = await self._exit_stack.enter_async_context(stdio_client(self._server_params))

        self._session = await self._exit_stack.enter_async_context(ClientSession(read, write))

        await self._session.initialize()
        logger.info("MCP client session initialized successfully.")
        return self

    async def __aexit__(
        self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]
    ) -> None:
        """Cleanly closes all connections."""
        logger.debug("Closing MCP client session...")
        await self._exit_stack.aclose()
        self._session = None
        logger.info("MCP client session closed.")

    async def load_into(self, registry: ToolRegistry) -> None:
        """Loads all tools from the MCP server and registers them in the given registry.

        Args:
            registry: The ToolRegistry to register the tools into.

        Raises:
            RuntimeError: If the MCP Client is not connected.
        """
        if not self._session:
            raise RuntimeError("MCP Client is not connected. Use 'async with'.")

        logger.debug("Fetching tools from MCP server...")
        result = await self._session.list_tools()
        logger.info("Found %d tools from MCP server.", len(result.tools))

        for tool in result.tools:
            self._register_single_tool(registry, tool)

    def _register_single_tool(self, registry: ToolRegistry, tool: MCPTool) -> None:
        """Creates the proxy function and registers it.

        Args:
            registry: The ToolRegistry to register the tool into.
            tool: The MCPTool object containing tool metadata.
        """
        tool_name = tool.name
        # Ensure description is present, as ToolRegistry might require it if parameters are present.
        tool_description = tool.description or f"Tool {tool_name} provided by MCP server."

        async def mcp_proxy(**kwargs: Any) -> Any:
            """Proxy MCP tool calls through the active client session.

            Args:
                **kwargs: Keyword arguments forwarded to the remote MCP tool.

            Returns:
                A normalized string representation of MCP content blocks.
            """
            if not self._session:
                raise RuntimeError(f"Cannot call tool '{tool_name}': MCP session is not active.")

            logger.info("Delegating tool '%s' to MCP Server...", tool_name)
            logger.debug("Tool arguments: %s", kwargs)

            mcp_result = await self._session.call_tool(tool_name, arguments=kwargs)

            output = []
            if not mcp_result.content:
                return "Success"

            for c in mcp_result.content:
                if c.type == "text":
                    text_content = cast(TextContent, c)
                    output.append(text_content.text)
                elif c.type == "image":
                    image_content = cast(ImageContent, c)
                    output.append(f"[Image: {image_content.mimeType}]")
                elif c.type == "resource":
                    resource_content = cast(EmbeddedResource, c)
                    # EmbeddedResource has 'resource' attribute which contains 'uri'
                    output.append(f"[Resource: {resource_content.resource.uri}]")
                else:
                    output.append(f"[Unknown content type: {c.type}]")

            result_text = "\n".join(output)
            logger.debug(
                "Tool '%s' result: %s", tool_name, result_text[:200] + "..." if len(result_text) > 200 else result_text
            )
            return result_text

        mcp_proxy.__name__ = tool_name
        mcp_proxy.__doc__ = tool_description

        # Sanitize schema to ensure compatibility with LLMs
        parameters = tool.inputSchema
        parameters = SchemaValidator.sanitize_schema(parameters)

        try:
            registry.register(
                name_or_tool=tool_name, description=tool_description, func=mcp_proxy, parameters=parameters
            )
            logger.info("MCP Tool '%s' successfully registered.", tool_name)
        except Exception as e:
            logger.error("Error registering MCP Tool '%s': %s", tool_name, e)

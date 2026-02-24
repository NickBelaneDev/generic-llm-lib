import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from mcp.types import Tool as MCPTool, TextContent, CallToolResult, ListToolsResult
from generic_llm_lib.mcp_wrapper import MCPClientWrapper
from generic_llm_lib.llm_core.tools import ToolRegistry
from typing import Any


@pytest.fixture
def mock_registry() -> Any:
    return MagicMock(spec=ToolRegistry)


@pytest.fixture
def mock_session() -> Any:
    session = AsyncMock()
    session.initialize = AsyncMock()
    # Ensure context manager returns the session itself
    session.__aenter__.return_value = session
    return session


@pytest.mark.asyncio
async def test_mcp_wrapper_lifecycle(mock_session: Any) -> None:
    """Test that the MCP wrapper correctly initializes and closes the session."""
    with patch("generic_llm_lib.mcp_wrapper.wrapper.stdio_client", new_callable=MagicMock) as mock_stdio:
        mock_stdio.return_value = AsyncMock()
        mock_stdio.return_value.__aenter__.return_value = (AsyncMock(), AsyncMock())

        with patch("generic_llm_lib.mcp_wrapper.wrapper.ClientSession", return_value=mock_session):
            async with MCPClientWrapper("cmd", ["arg"]) as wrapper:
                assert wrapper._session is not None
                # Check if initialize was called (using await count or similar if needed, but called is standard mock)
                # Mypy complains about .called on Coroutine, but AsyncMock handles it.
                # We can cast or just ignore for test
                assert mock_session.initialize.call_count > 0

            # Session should be None or closed (implementation detail, but wrapper sets it to None)
            assert wrapper._session is None


@pytest.mark.asyncio
async def test_load_into_registers_tools(mock_registry: Any, mock_session: Any) -> None:
    """Test that tools from MCP are registered into the ToolRegistry."""

    # Mock MCP tools
    tools_result = ListToolsResult(
        tools=[
            MCPTool(name="tool1", description="desc1", inputSchema={"type": "object"}),
            MCPTool(name="tool2", description="desc2", inputSchema={"type": "object"}),
        ]
    )
    # Explicitly set the return value for the async method
    mock_session.list_tools = AsyncMock(return_value=tools_result)

    with patch("generic_llm_lib.mcp_wrapper.wrapper.stdio_client", new_callable=MagicMock) as mock_stdio:
        mock_stdio.return_value = AsyncMock()
        mock_stdio.return_value.__aenter__.return_value = (AsyncMock(), AsyncMock())

        with patch("generic_llm_lib.mcp_wrapper.wrapper.ClientSession", return_value=mock_session):
            async with MCPClientWrapper("cmd", ["arg"]) as wrapper:
                await wrapper.load_into(mock_registry)

                # Verify list_tools was called
                assert mock_session.list_tools.call_count > 0

                # Check if register was called twice
                assert mock_registry.register.call_count == 2

                # Verify calls
                calls = mock_registry.register.call_args_list
                assert calls[0].kwargs["name_or_tool"] == "tool1"
                assert calls[0].kwargs["description"] == "desc1"
                assert calls[1].kwargs["name_or_tool"] == "tool2"


@pytest.mark.asyncio
async def test_mcp_proxy_execution(mock_registry: Any, mock_session: Any) -> None:
    """Test that the registered proxy function calls the MCP session."""

    # Mock MCP tool
    tools_result = ListToolsResult(tools=[MCPTool(name="test_tool", description="desc", inputSchema={})])
    mock_session.list_tools = AsyncMock(return_value=tools_result)

    # Mock execution result
    mock_session.call_tool = AsyncMock(
        return_value=CallToolResult(content=[TextContent(type="text", text="Result from MCP")])
    )

    with patch("generic_llm_lib.mcp_wrapper.wrapper.stdio_client", new_callable=MagicMock) as mock_stdio:
        mock_stdio.return_value = AsyncMock()
        mock_stdio.return_value.__aenter__.return_value = (AsyncMock(), AsyncMock())

        with patch("generic_llm_lib.mcp_wrapper.wrapper.ClientSession", return_value=mock_session):
            async with MCPClientWrapper("cmd", ["arg"]) as wrapper:
                await wrapper.load_into(mock_registry)

                assert mock_registry.register.call_count > 0

                # Get the proxy function passed to register
                proxy_func = mock_registry.register.call_args.kwargs["func"]

                # Execute proxy
                result = await proxy_func(param="value")

                # Verify MCP call
                mock_session.call_tool.assert_called_with("test_tool", arguments={"param": "value"})
                assert result == "Result from MCP"

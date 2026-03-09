from generic_llm_lib.llm_core.messages import UserMessage, AssistantMessage, SystemMessage, ToolMessage
from generic_llm_lib.llm_impl.openai_api.history_converter import (
    convert_to_openai_history,
    convert_from_openai_history,
    _convert_user_message,
    _convert_assistant_message,
    _convert_system_message,
    _convert_tool_message,
    _convert_openai_user_role,
    _convert_openai_assistant_role,
    _convert_openai_system_role,
    _convert_openai_tool_role,
    _convert_single_message_to_openai,
    _convert_single_message_from_openai,
)


class TestMessageConversion:
    """Tests for message conversion between generic and OpenAI formats."""

    def test_convert_history_to_openai(self):
        """Test converting generic messages to OpenAI format."""
        history = [
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content="Hello"),
            AssistantMessage(content="Hi there!", tool_calls=[{"id": "call_123", "function": {"name": "test_tool"}}]),
            ToolMessage(content="Tool output", tool_call_id="call_123", name="test_tool"),
        ]

        openai_history = convert_to_openai_history(history)

        assert len(openai_history) == 4
        assert openai_history[0] == {"role": "system", "content": "You are a helpful assistant."}
        assert openai_history[1] == {"role": "user", "content": "Hello"}
        assert openai_history[2] == {
            "role": "assistant",
            "content": "Hi there!",
            "tool_calls": [{"id": "call_123", "function": {"name": "test_tool"}}],
        }
        assert openai_history[3] == {
            "role": "tool",
            "content": "Tool output",
            "tool_call_id": "call_123",
            "name": "test_tool",
        }

    def test_convert_history_from_openai(self):
        """Test converting OpenAI messages back to generic format."""
        openai_history = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User query"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_abc", "function": {"name": "my_func"}}],
            },
            {
                "role": "tool",
                "content": "Function result",
                "tool_call_id": "call_abc",
                "name": "my_func",
            },
        ]

        generic_history = convert_from_openai_history(openai_history)

        assert len(generic_history) == 4
        assert isinstance(generic_history[0], SystemMessage)
        assert generic_history[0].content == "System prompt"

        assert isinstance(generic_history[1], UserMessage)
        assert generic_history[1].content == "User query"

        assert isinstance(generic_history[2], AssistantMessage)
        assert generic_history[2].content == ""  # None content becomes empty string if tool calls exist
        assert generic_history[2].tool_calls == [{"id": "call_abc", "function": {"name": "my_func"}}]

        assert isinstance(generic_history[3], ToolMessage)
        assert generic_history[3].content == "Function result"
        assert generic_history[3].tool_call_id == "call_abc"
        assert generic_history[3].name == "my_func"

    def test_convert_history_assistant_no_tool_calls(self):
        """Test converting assistant message without tool calls."""
        history = [AssistantMessage(content="Just text")]
        openai_history = convert_to_openai_history(history)

        assert len(openai_history) == 1
        assert openai_history[0] == {"role": "assistant", "content": "Just text"}
        assert "tool_calls" not in openai_history[0]

    def test_convert_from_openai_skip_empty_content_no_tools(self):
        """Test that messages with empty content and no tool calls are skipped."""
        openai_history = [
            {"role": "user", "content": ""},  # Should be skipped
            {"role": "assistant", "content": None},  # Should be skipped
            {"role": "user", "content": "Valid"},
        ]

        generic_history = convert_from_openai_history(openai_history)

        assert len(generic_history) == 1
        assert generic_history[0].content == "Valid"

    # --- Tests for individual conversion functions (TO OpenAI) ---

    def test_convert_user_message_func(self):
        """Test _convert_user_message function."""
        msg = UserMessage(content="Hello from user")
        expected = {"role": "user", "content": "Hello from user"}
        assert _convert_user_message(msg) == expected

    def test_convert_assistant_message_func_with_content(self):
        """Test _convert_assistant_message function with content."""
        msg = AssistantMessage(content="Hi there!")
        expected = {"role": "assistant", "content": "Hi there!"}
        assert _convert_assistant_message(msg) == expected

    def test_convert_assistant_message_func_with_tool_calls(self):
        """Test _convert_assistant_message function with tool calls."""
        tool_calls = [{"id": "call_123", "function": {"name": "test_tool"}}]
        msg = AssistantMessage(content="", tool_calls=tool_calls)
        expected = {"role": "assistant", "content": "", "tool_calls": tool_calls}
        assert _convert_assistant_message(msg) == expected

    def test_convert_assistant_message_func_with_content_and_tool_calls(self):
        """Test _convert_assistant_message function with content and tool calls."""
        tool_calls = [{"id": "call_123", "function": {"name": "test_tool"}}]
        msg = AssistantMessage(content="Thinking...", tool_calls=tool_calls)
        expected = {"role": "assistant", "content": "Thinking...", "tool_calls": tool_calls}
        assert _convert_assistant_message(msg) == expected

    def test_convert_system_message_func(self):
        """Test _convert_system_message function."""
        msg = SystemMessage(content="You are a bot.")
        expected = {"role": "system", "content": "You are a bot."}
        assert _convert_system_message(msg) == expected

    def test_convert_tool_message_func(self):
        """Test _convert_tool_message function."""
        msg = ToolMessage(content="Tool result data", tool_call_id="call_abc", name="my_tool")
        expected = {
            "role": "tool",
            "content": "Tool result data",
            "tool_call_id": "call_abc",
            "name": "my_tool",
        }
        assert _convert_tool_message(msg) == expected

    def test_convert_single_message_to_openai_func(self):
        """Test _convert_single_message_to_openai function for various message types."""
        assert _convert_single_message_to_openai(UserMessage(content="Hi")) == {"role": "user", "content": "Hi"}
        assert _convert_single_message_to_openai(SystemMessage(content="Sys")) == {"role": "system", "content": "Sys"}
        assert _convert_single_message_to_openai(AssistantMessage(content="Asst")) == {
            "role": "assistant",
            "content": "Asst",
        }
        assert _convert_single_message_to_openai(ToolMessage(content="Tool", tool_call_id="1", name="t")) == {
            "role": "tool",
            "content": "Tool",
            "tool_call_id": "1",
            "name": "t",
        }
        # SystemMessage should return None if not explicitly handled as part of history
        assert _convert_single_message_to_openai(SystemMessage(content="")) == {"role": "system", "content": ""}

    # --- Tests for individual conversion functions (FROM OpenAI) ---

    def test_convert_openai_user_role_func(self):
        """Test _convert_openai_user_role function."""
        openai_msg = {"role": "user", "content": "Hello"}
        result = _convert_openai_user_role(openai_msg)
        assert isinstance(result, UserMessage)
        assert result.content == "Hello"

        openai_msg_empty = {"role": "user", "content": ""}
        assert _convert_openai_user_role(openai_msg_empty) is None

    def test_convert_openai_assistant_role_func_with_content(self):
        """Test _convert_openai_assistant_role function with content."""
        openai_msg = {"role": "assistant", "content": "Response"}
        result = _convert_openai_assistant_role(openai_msg)
        assert isinstance(result, AssistantMessage)
        assert result.content == "Response"
        assert result.tool_calls is None

    def test_convert_openai_assistant_role_func_with_tool_calls(self):
        """Test _convert_openai_assistant_role function with tool calls."""
        tool_calls = [{"id": "call_abc", "function": {"name": "my_func"}}]
        openai_msg = {"role": "assistant", "content": None, "tool_calls": tool_calls}
        result = _convert_openai_assistant_role(openai_msg)
        assert isinstance(result, AssistantMessage)
        assert result.content == ""
        assert result.tool_calls == tool_calls

    def test_convert_openai_assistant_role_func_empty(self):
        """Test _convert_openai_assistant_role function with empty message."""
        openai_msg = {"role": "assistant", "content": None, "tool_calls": None}
        assert _convert_openai_assistant_role(openai_msg) is None

    def test_convert_openai_system_role_func(self):
        """Test _convert_openai_system_role function."""
        openai_msg = {"role": "system", "content": "System instruction"}
        result = _convert_openai_system_role(openai_msg)
        assert isinstance(result, SystemMessage)
        assert result.content == "System instruction"

        openai_msg_empty = {"role": "system", "content": ""}
        assert _convert_openai_system_role(openai_msg_empty) is None

    def test_convert_openai_tool_role_func(self):
        """Test _convert_openai_tool_role function."""
        openai_msg = {
            "role": "tool",
            "content": "Tool output",
            "tool_call_id": "call_123",
            "name": "test_tool",
        }
        result = _convert_openai_tool_role(openai_msg)
        assert isinstance(result, ToolMessage)
        assert result.content == "Tool output"
        assert result.tool_call_id == "call_123"
        assert result.name == "test_tool"

        openai_msg_missing_id = {"role": "tool", "content": "Output", "name": "test_tool"}
        assert _convert_openai_tool_role(openai_msg_missing_id) is None

    def test_convert_single_message_from_openai_func(self):
        """Test _convert_single_message_from_openai function for various message types."""
        assert isinstance(_convert_single_message_from_openai({"role": "user", "content": "Hi"}), UserMessage)
        assert isinstance(_convert_single_message_from_openai({"role": "system", "content": "Sys"}), SystemMessage)
        assert isinstance(
            _convert_single_message_from_openai({"role": "assistant", "content": "Asst"}), AssistantMessage
        )
        assert isinstance(
            _convert_single_message_from_openai({"role": "tool", "content": "Tool", "tool_call_id": "1", "name": "t"}),
            ToolMessage,
        )
        assert _convert_single_message_from_openai({"role": "unknown", "content": "Unknown"}) is None

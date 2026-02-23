from generic_llm_lib import UserMessage, AssistantMessage, SystemMessage, ToolMessage, GenericOpenAI


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

        openai_history = GenericOpenAI._convert_history(history)

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

        generic_history = GenericOpenAI._convert_to_generic_history(openai_history)

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
        openai_history = GenericOpenAI._convert_history(history)

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

        generic_history = GenericOpenAI._convert_to_generic_history(openai_history)

        assert len(generic_history) == 1
        assert generic_history[0].content == "Valid"

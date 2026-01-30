import pytest
import asyncio
import logging
from unittest.mock import MagicMock, AsyncMock
from llm_core.registry import ToolRegistry
from llm_impl.gemini.core import GenericGemini
from llm_impl.gemini.registry import GeminiToolRegistry
from llm_impl.open_api.core import GenericOpenAI
from llm_core.types import LLMConfig

# --- Test 1: Information Leakage Prevention ---
@pytest.mark.asyncio
async def test_gemini_tool_error_sanitization(mock_genai_client, caplog):
    """
    Verifies that exceptions in Gemini tools are caught, logged, and sanitized
    before being sent back to the LLM.
    """
    mock_chat = MagicMock()
    mock_genai_client.chats.create.return_value = mock_chat

    # Setup: LLM calls a tool that raises a sensitive exception
    func_call = MagicMock()
    func_call.name = "sensitive_tool"
    func_call.args = {}
    
    # First response: Call the tool
    resp_call = MagicMock(parts=[MagicMock(function_call=func_call, text=None)])
    
    # Second response: Final answer (after tool error)
    resp_final = MagicMock(parts=[MagicMock(text="I encountered an error.", function_call=None)], usage_metadata=None)
    
    # Mock send_message sequence
    # 1. User prompt -> returns tool call
    # 2. Tool result (error) sent back -> returns final answer
    mock_chat.send_message = MagicMock(side_effect=[resp_call, resp_final])
    
    # Mock history
    mock_content = MagicMock()
    mock_content.role = "model"
    mock_content.parts = []
    mock_chat.get_history.return_value = [mock_content]

    registry = GeminiToolRegistry()
    
    @registry.tool
    def sensitive_tool() -> str:
        """A tool that fails."""
        raise ValueError("DB_PASSWORD=secret123") # Sensitive info!

    gemini = GenericGemini(
        client=mock_genai_client, 
        model_name="gemini-pro", 
        sys_instruction="sys", 
        registry=registry
    )

    # Capture logs to verify we logged the real error
    with caplog.at_level(logging.ERROR):
        await gemini.chat([], "Trigger error")

    # Verification 1: The real error should be in the server logs
    assert "DB_PASSWORD=secret123" in caplog.text
    assert "Unexpected error executing tool 'sensitive_tool'" in caplog.text

    # Verification 2: The message sent to the LLM should be sanitized
    # We check the arguments passed to the second send_message call
    call_args = mock_chat.send_message.call_args_list[1]
    sent_parts = call_args[0][0] # The argument passed to send_message
    
    # The response sent to LLM
    error_response = sent_parts[0].function_response.response["error"]
    
    assert "DB_PASSWORD" not in error_response
    assert "An internal error occurred" in error_response

# --- Test 2: Tool Execution Timeout ---
@pytest.mark.asyncio
async def test_gemini_tool_timeout(mock_genai_client):
    """
    Verifies that long-running tools are terminated after the timeout period.
    """
    mock_chat = MagicMock()
    mock_genai_client.chats.create.return_value = mock_chat

    func_call = MagicMock()
    func_call.name = "slow_tool"
    func_call.args = {}
    
    resp_call = MagicMock(parts=[MagicMock(function_call=func_call, text=None)])
    resp_final = MagicMock(parts=[MagicMock(text="Timeout happened.", function_call=None)], usage_metadata=None)
    
    mock_chat.send_message = MagicMock(side_effect=[resp_call, resp_final])
    mock_chat.get_history.return_value = [MagicMock(role="model", parts=[])]

    registry = GeminiToolRegistry()
    
    @registry.tool
    async def slow_tool() -> str:
        """Sleeps for 2 seconds."""
        await asyncio.sleep(2)
        return "finished"

    # Set a short timeout (0.1s)
    gemini = GenericGemini(
        client=mock_genai_client, 
        model_name="gemini-pro", 
        sys_instruction="sys", 
        registry=registry,
        tool_timeout=0.1
    )

    await gemini.chat([], "Run slow tool")

    # Verify that the tool result sent back indicates a timeout
    call_args = mock_chat.send_message.call_args_list[1]
    sent_parts = call_args[0][0]
    
    response_payload = sent_parts[0].function_response.response
    
    assert "error" in response_payload
    assert "timed out" in response_payload["error"]

# --- Test 3: Schema Recursion Limit ---
def test_schema_recursion_limit():
    """
    Verifies that the registry detects and stops infinite recursion in schemas.
    """
    class ConcreteRegistry(ToolRegistry):
        @property
        def tool_object(self): return None

    registry = ConcreteRegistry()

    # Create a deeply nested schema manually to simulate recursion/depth
    # A simple recursive dict structure
    recursive_schema = {"type": "object", "properties": {}}
    current = recursive_schema
    for _ in range(25): # Depth > 20
        current["properties"]["next"] = {"type": "object", "properties": {}}
        current = current["properties"]["next"]

    # We can't easily inject this into Pydantic generation without complex setup,
    # so we test the _resolve_schema_refs method directly if possible,
    # or use a recursive reference that expands indefinitely.
    
    # Let's use the method directly as it's the one modified.
    
    # Mock a schema with a self-reference loop
    schema = {
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {
                    "child": {"$ref": "#/$defs/Node"}
                }
            }
        },
        "$ref": "#/$defs/Node"
    }

    try:
        registry._resolve_schema_refs(schema, max_depth=5)
        pytest.fail("Should have raised RecursionError")
    except RecursionError as e:
        assert "Max recursion depth" in str(e)

# --- Test 4: Default Configuration Values ---
def test_default_config_values():
    """
    Verifies that the default configuration values have been updated.
    """
    config = LLMConfig()
    assert config.max_tokens == 1024
    assert config.temperature == 0.7

# --- Test 5: OpenAI Tool Error Sanitization ---
@pytest.mark.asyncio
async def test_openai_tool_error_sanitization(mock_openai_client, caplog):
    """
    Verifies that exceptions in OpenAI tools are caught, logged, and sanitized.
    """
    # Setup mocks
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    
    # 1. Initial response: Tool call
    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_123"
    mock_tool_call.function.name = "sensitive_tool"
    mock_tool_call.function.arguments = "{}"
    
    mock_message.tool_calls = [mock_tool_call]
    mock_message.content = None
    mock_message.role = "assistant"
    
    # Mock model_dump for the tool call message
    mock_message.model_dump.return_value = {
        "role": "assistant", 
        "content": None,
        "tool_calls": [{
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "sensitive_tool",
                "arguments": "{}"
            }
        }]
    }
    
    mock_choice.message = mock_message
    mock_completion.choices = [mock_choice]
    
    # 2. Second response: Final answer
    mock_final_choice = MagicMock()
    mock_final_message = MagicMock()
    mock_final_message.tool_calls = None
    mock_final_message.content = "Error handled."
    
    # Mock model_dump for the final message
    mock_final_message.model_dump.return_value = {
        "role": "assistant",
        "content": "Error handled."
    }

    mock_final_choice.message = mock_final_message
    
    mock_final_completion = MagicMock()
    mock_final_completion.choices = [mock_final_choice]
    mock_final_completion.usage = MagicMock(prompt_tokens=10, completion_tokens=10, total_tokens=20)

    # Sequence of returns from client.chat.completions.create
    mock_openai_client.chat.completions.create.side_effect = [
        mock_completion,       # First call returns tool request
        mock_final_completion  # Second call returns final answer
    ]

    registry = GeminiToolRegistry() # Registry is generic enough
    @registry.tool
    def sensitive_tool() -> str:
        """A sensitive tool."""
        raise ValueError("API_KEY=12345")

    llm = GenericOpenAI(
        client=mock_openai_client,
        model_name="gpt-4",
        sys_instruction="sys",
        registry=registry
    )

    with caplog.at_level(logging.ERROR):
        await llm.chat([], "Run tool")

    # Verify logs
    assert "API_KEY=12345" in caplog.text
    
    # Verify what was sent back to OpenAI
    # The second call to create() contains the tool output in 'messages'
    call_args = mock_openai_client.chat.completions.create.call_args_list[1]
    messages = call_args[1]['messages'] # kwargs['messages']
    
    # The last message should be the tool response
    tool_msg = messages[-1]
    assert tool_msg['role'] == 'tool'
    assert "API_KEY" not in tool_msg['content']
    assert "An internal error occurred" in tool_msg['content']
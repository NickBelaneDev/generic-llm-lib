import pytest
from unittest.mock import MagicMock, AsyncMock
from typing import Annotated, List, Optional
from pydantic import BaseModel, Field
from llm_core.registry import ToolRegistry
from llm_impl.gemini.core import GenericGemini
from llm_impl.gemini.registry import GeminiToolRegistry

# --- Weakness 1: Duplicate Tool Registration ---
def test_duplicate_tool_registration_overwrite():
    """
    Weakness: The registry silently overwrites tools with the same name.
    This can lead to hard-to-debug issues if a user accidentally reuses a name.
    """
    class ConcreteRegistry(ToolRegistry):
        @property
        def tool_object(self): return None

    registry = ConcreteRegistry()

    @registry.tool
    def my_tool() -> str:
        """First implementation."""
        return "first"

    @registry.tool
    def my_tool() -> str: # Same name
        """Second implementation."""
        return "second"

    # Check which one is active
    assert registry.tools["my_tool"].description == "Second implementation."
    assert registry.tools["my_tool"].func() == "second"
    # No warning or error was raised.

# --- Weakness 2: Circular Pydantic Models ---
def test_circular_pydantic_models_recursion_error():
    """
    Weakness: _resolve_schema_refs does not handle circular references,
    leading to a RecursionError.
    """
    class ConcreteRegistry(ToolRegistry):
        @property
        def tool_object(self): return None

    registry = ConcreteRegistry()

    class NodeB(BaseModel):
        pass # Forward declaration placeholder

    class NodeA(BaseModel):
        child: Optional['NodeB'] = Field(default=None, description="Child node")

    class NodeB(BaseModel):
        parent: Optional[NodeA] = Field(default=None, description="Parent node")
    
    # Update forward refs
    NodeA.model_rebuild()
    NodeB.model_rebuild()

    try:
        @registry.tool
        def process_graph(node: Annotated[NodeA, Field(description="Start node")]) -> str:
            """Process a graph node."""
            return "processed"
    except RecursionError:
        pytest.fail("RecursionError caught! The registry cannot handle circular models.")
    except Exception as e:
        # Depending on implementation, it might fail differently
        pytest.fail(f"Failed with {type(e).__name__}: {e}")

# --- Weakness 3: Max Function Loops Exceeded ---
@pytest.mark.asyncio
async def test_max_function_loops_exceeded(mock_genai_client):
    """
    Weakness: When max_function_loops is exceeded, the loop terminates,
    but the return value might be the raw function call request from the LLM,
    not a final text response.
    """
    mock_chat = MagicMock()
    mock_genai_client.chats.create.return_value = mock_chat

    # Response that always requests a function call
    func_call_response = MagicMock()
    func_call = MagicMock()
    func_call.name = "recursive_tool"
    func_call.args = {}
    func_call_response.parts = [MagicMock(function_call=func_call, text=None)]
    func_call_response.usage_metadata = None

    # The chat session always returns a function call request
    mock_chat.send_message.return_value = func_call_response

    # Registry with the tool
    registry = GeminiToolRegistry()
    @registry.tool
    def recursive_tool() -> str:
        """A tool."""
        return "loop"

    gemini = GenericGemini(
        client=mock_genai_client,
        model_name="gemini-pro",
        sys_instruction="sys",
        registry=registry,
        max_function_loops=2 # Low limit
    )

    # Execute
    response = await gemini.chat([], "Start loop")

    # Verification
    # The loop should run 2 times.
    # The final response returned to the user will be 'func_call_response' (the last one received).
    # This response has NO text, only a function call.
    assert response.last_response.text == "" 
    # This is a weakness: the user gets an empty string and doesn't know the model got stuck.

# --- Weakness 4: Non-Serializable Tool Return (Crash) ---
@pytest.mark.asyncio
async def test_non_serializable_tool_return_crash(mock_genai_client):
    """
    Weakness: If a tool returns a non-serializable object, the chat.send_message call
    (which sends the tool result back to LLM) will crash outside the try/except block.
    """
    mock_chat = MagicMock()
    mock_genai_client.chats.create.return_value = mock_chat

    # 1. LLM calls the tool
    llm_request = MagicMock()
    fc = MagicMock()
    fc.name = "bad_tool"
    fc.args = {}
    llm_request.parts = [MagicMock(function_call=fc, text=None)]
    
    # 2. We need to mock send_message to simulate the crash when sending the result
    # The first call is the user prompt -> returns llm_request
    # The second call is sending the tool result -> SHOULD CRASH if not serializable
    
    def send_message_side_effect(content):
        # Check if we are sending a function response
        if isinstance(content, list) and hasattr(content[0], 'function_response'):
            # Simulate Google Library crashing on serialization
            # The content contains the result from the tool
            resp = content[0].function_response.response
            if "result" in resp and not isinstance(resp["result"], (str, int, float, bool, list, dict, type(None))):
                 raise ValueError("Serialization Error: Object is not JSON serializable")
        
        return llm_request # Return same request to keep loop going (if it didn't crash)

    mock_chat.send_message.side_effect = send_message_side_effect

    # Registry
    registry = GeminiToolRegistry()
    
    class Unserializable:
        pass

    @registry.tool
    def bad_tool() -> Unserializable:
        """Returns a bad object."""
        return Unserializable()

    gemini = GenericGemini(
        client=mock_genai_client,
        model_name="gemini-pro",
        sys_instruction="sys",
        registry=registry
    )

    # Execute
    try:
        await gemini.chat([], "Crash me")
        pytest.fail("Should have raised ValueError due to serialization failure")
    except ValueError as e:
        assert "Serialization Error" in str(e)
    # If caught, it confirms the weakness: the library crashes instead of handling the error gracefully.

# --- Weakness 5: Empty Prompt Handling ---
@pytest.mark.asyncio
async def test_empty_prompt_handling(mock_genai_client):
    """
    Weakness: The library does not validate empty prompts. 
    If the underlying API rejects empty strings, the user gets a raw API error.
    """
    mock_chat = MagicMock()
    mock_genai_client.chats.create.return_value = mock_chat
    
    # Simulate API error for empty prompt
    def side_effect(prompt):
        if not prompt:
            raise Exception("400 Invalid Argument: Prompt cannot be empty")
        return MagicMock(parts=[MagicMock(text="OK")])
        
    mock_chat.send_message.side_effect = side_effect

    gemini = GenericGemini(client=mock_genai_client, model_name="gemini-pro", sys_instruction="sys")

    try:
        await gemini.ask("")
        pytest.fail("Should have raised Exception")
    except Exception as e:
        assert "Invalid Argument" in str(e)
    # Confirms weakness: No pre-validation in the library.

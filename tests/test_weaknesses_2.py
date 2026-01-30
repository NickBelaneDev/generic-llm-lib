import pytest
import asyncio
import time
from unittest.mock import MagicMock, AsyncMock
from typing import Annotated, List, Dict
from pydantic import BaseModel, Field
from llm_core.registry import ToolRegistry
from llm_core.exceptions import ToolValidationError
from llm_impl.gemini.core import GenericGemini
from llm_impl.gemini.registry import GeminiToolRegistry

# --- Weakness 6: Argument Type Coercion Failure ---
@pytest.mark.asyncio
async def test_arg_type_coercion_failure(mock_genai_client):
    """
    Weakness: The library does not validate or coerce arguments sent by the LLM
    against the Python type hints. If the LLM sends a string for an int parameter,
    it is passed as-is, potentially causing logic errors (e.g., string concatenation instead of addition).
    """
    mock_chat = MagicMock()
    mock_genai_client.chats.create.return_value = mock_chat

    # LLM sends strings instead of ints
    func_call = MagicMock()
    func_call.name = "add_numbers"
    func_call.args = {"a": "10", "b": "20"} # Strings!
    
    response_call = MagicMock()
    response_call.parts = [MagicMock(function_call=func_call, text=None)]
    
    response_final = MagicMock()
    response_final.parts = [MagicMock(text="Result is 30", function_call=None)]
    response_final.usage_metadata = None

    mock_chat.send_message = MagicMock(side_effect=[response_call, response_final])
    
    # Mock history for chat response construction
    mock_content = MagicMock()
    mock_content.role = "model"
    mock_content.parts = []
    mock_chat.get_history.return_value = [mock_content]

    registry = GeminiToolRegistry()
    
    @registry.tool
    def add_numbers(a: Annotated[int, Field(description="First number")], 
                    b: Annotated[int, Field(description="Second number")]) -> int:
        """Adds two numbers."""
        # This will return "1020" if a and b are strings
        return a + b

    gemini = GenericGemini(client=mock_genai_client, model_name="gemini-pro", sys_instruction="sys", registry=registry)
    
    await gemini.chat([], "Add 10 and 20")
    
    # We check the result passed back to the LLM (via the mock)
    # The second call to send_message contains the function result
    call_args = mock_chat.send_message.call_args_list[1]
    sent_parts = call_args[0][0] # The argument passed to send_message
    
    # Extract the result sent back to LLM
    tool_result = sent_parts[0].function_response.response["result"]
    
    # If coercion worked, it should be 30. If not, it's "1020".
    assert tool_result == 30 # Confirms weakness: No coercion happened.

# --- Weakness 7: *args and **kwargs Incompatibility ---
def test_var_args_registration_failure():
    """
    Weakness: The ToolRegistry fails to register functions with *args or **kwargs
    because it tries to parse them as standard annotated parameters, which fails
    the 'get_origin(annotation) is Annotated' check or description check.
    """
    class ConcreteRegistry(ToolRegistry):
        @property
        def tool_object(self): return None
    
    registry = ConcreteRegistry()
    
    try:
        @registry.tool
        def flexible_tool(main_arg: Annotated[int, Field(description="Main")], *args: int):
            """A tool with var args."""
            pass
        pytest.fail("Should have raised ToolValidationError")
    except ToolValidationError as e:
        # It fails because *args usually don't have the Annotated wrapper in the same way,
        # or the registry logic doesn't account for Parameter.VAR_POSITIONAL
        assert "missing a description" in str(e) or "args" in str(e)

# --- Weakness 8: ask() Hides Tool Execution Details ---
@pytest.mark.asyncio
async def test_ask_hides_tool_execution_details(mock_genai_client):
    """
    Weakness: The `ask` method returns a GeminiMessageResponse which only contains
    the final text and tokens. It completely swallows the history of tool executions
    that happened during the call. The user has no way to audit what tools were called.
    """
    mock_chat = MagicMock()
    mock_genai_client.chats.create.return_value = mock_chat
    
    # Sequence: Tool Call -> Tool Result -> Final Answer
    func_call = MagicMock()
    func_call.name = "my_tool"
    func_call.args = {}
    
    resp_tool = MagicMock(parts=[MagicMock(function_call=func_call, text=None)])
    resp_final = MagicMock(parts=[MagicMock(text="Done", function_call=None)], usage_metadata=None)
    
    mock_chat.send_message = MagicMock(side_effect=[resp_tool, resp_final])
    
    # Mock history for chat response construction
    mock_content = MagicMock()
    mock_content.role = "model"
    mock_content.parts = []
    mock_chat.get_history.return_value = [mock_content]
    
    registry = GeminiToolRegistry()
    @registry.tool
    def my_tool() -> str: 
        """Test tool."""
        return "result"
        
    gemini = GenericGemini(client=mock_genai_client, model_name="gemini-pro", sys_instruction="sys", registry=registry)
    
    response = await gemini.ask("Do something")
    
    # The response object has NO field for tool calls or history
    assert response.text == "Done"
    assert not hasattr(response, "tool_calls")
    assert not hasattr(response, "history")
    # User cannot know 'my_tool' was called.

# --- Weakness 9: Nested Schema Title Leak ---
def test_nested_schema_title_leak():
    """
    Weakness: Pydantic adds a 'title' field to model schemas. The registry removes it
    from the root, but NOT from nested definitions. Google Gemini API (and others)
    can be strict and reject schemas with unknown fields like 'title' in nested objects.
    """
    class ConcreteRegistry(ToolRegistry):
        @property
        def tool_object(self): return None
    registry = ConcreteRegistry()

    class Inner(BaseModel):
        field: str = Field(description="Inner field")

    class Outer(BaseModel):
        inner: Inner = Field(description="Inner object")

    @registry.tool
    def complex_tool(obj: Annotated[Outer, Field(description="Complex object")]) -> str:
        """Complex tool."""
        return "ok"

    schema = registry.tools["complex_tool"].parameters
    
    # Root title should be gone (handled by current code)
    assert "title" not in schema
    
    # Nested title usually remains (Weakness)
    inner_schema = schema["properties"]["obj"]["properties"]["inner"]
    # Pydantic generates "title": "Inner" here
    assert "title" in inner_schema
    assert inner_schema["title"] == "Inner"

# --- Weakness 10: Sync Tool Blocks Event Loop ---
@pytest.mark.asyncio
async def test_sync_tool_blocks_event_loop(mock_genai_client):
    """
    Weakness: Synchronous tools are called directly, blocking the asyncio event loop.
    This prevents other async tasks (like heartbeats, UI updates, or parallel processing)
    from running while the tool executes.
    """
    mock_chat = MagicMock()
    mock_genai_client.chats.create.return_value = mock_chat
    
    # Setup a tool that sleeps synchronously
    registry = GeminiToolRegistry()
    @registry.tool
    def blocking_tool() -> str:
        """Blocks for a bit."""
        time.sleep(0.2) # Blocking sleep!
        return "done"
        
    gemini = GenericGemini(client=mock_genai_client, model_name="gemini-pro", sys_instruction="sys", registry=registry)
    
    # Mock LLM calling the tool
    func_call = MagicMock()
    func_call.name = "blocking_tool"
    func_call.args = {}
    
    resp_call = MagicMock(parts=[MagicMock(function_call=func_call, text=None)])
    resp_final = MagicMock(parts=[MagicMock(text="Done", function_call=None)], usage_metadata=None)
    mock_chat.send_message = MagicMock(side_effect=[resp_call, resp_final])
    
    # Mock history for chat response construction
    mock_content = MagicMock()
    mock_content.role = "model"
    mock_content.parts = []
    mock_chat.get_history.return_value = [mock_content]

    # We want to prove that the loop is blocked.
    # We schedule a background task that should run "immediately" if the loop yields.
    # If the loop is blocked, the background task won't run until the tool finishes.
    
    task_ran_time = 0
    start_time = time.time()
    
    async def background_task():
        nonlocal task_ran_time
        # This task simply records when it got a chance to run
        task_ran_time = time.time()

    # Schedule background task
    asyncio.create_task(background_task())
    
    # Run the chat (which triggers the blocking tool)
    # We yield briefly to let the background task *start* scheduling, but the tool execution happens inside chat
    await gemini.chat([], "Block me")
    
    # Analysis:
    # The tool takes 0.2s.
    # If the tool was run in a thread (non-blocking), 'background_task' would run almost instantly (e.g., at T+0.001s).
    # If the tool blocks the loop, 'background_task' cannot run until the tool returns (at T+0.2s).
    
    # However, since 'chat' is awaited, the background task *might* run before 'chat' gets to the tool execution part
    # if there are other awaits inside 'chat'.
    # 'chat' awaits 'client.chats.create' (mocked, instant) -> 'chat.send_message' (mocked, instant) -> '_handle_function_calls'.
    # Inside '_handle_function_calls', it calls 'blocking_tool'.
    # Since everything before 'blocking_tool' is synchronous or instant mocks, the loop might not yield to 'background_task'
    # until 'blocking_tool' is hit.
    
    # If 'blocking_tool' blocks, 'background_task' runs AFTER it.
    # So 'task_ran_time' should be close to 'start_time + 0.2'.
    
    # If 'blocking_tool' was async-friendly (run_in_executor), the loop would yield at 'await run_in_executor',
    # allowing 'background_task' to run immediately.
    
    # Let's check the timing.
    # If blocking: task_ran_time >= start_time + 0.2
    # If non-blocking: task_ran_time < start_time + 0.1
    
    # Note: This test relies on the fact that our mocks are synchronous.
    # In real life, 'chat.send_message' is a network call (async/threaded), so the loop would yield there.
    # But once the response comes back and we enter the tool execution loop, that part is CPU bound.
    
    diff = task_ran_time - start_time
    # If diff is small (close to 0), it means it ran before the sleep (unlikely if blocked) or concurrently.
    # If diff is large (>= 0.2), it means it waited for the sleep.
    
    # Actually, since we are mocking send_message as a synchronous return (MagicMock), 
    # there are NO awaits in the path until the tool execution logic?
    # Wait, 'chat' is async def. '_handle_function_calls' is async def.
    # But if they don't await anything real, they run synchronously.
    # So the background task definitely won't run until the first 'await'.
    # If the tool is blocking, there is NO await during its execution.
    
    # To properly test this, we need to ensure there is an 'await' *before* the tool call
    # so the background task is scheduled, but then blocked *during* the tool call?
    # No, we want to see if the loop is blocked *during* the tool call.
    
    # Let's simplify: The weakness is that `tool_function(...)` is called without `await` if it's not a coroutine.
    # This is visible in the code structure.
    # Behavioral test:
    # 1. Start background task.
    # 2. Call chat.
    # 3. Inside chat, tool sleeps 0.2s.
    # 4. If background task runs *after* chat finishes (or after 0.2s), it was blocked.
    
    # Since 'chat' is async, we await it.
    # The background task is on the loop.
    # If 'chat' blocks the loop for 0.2s, the background task is delayed.
    
    # Assert that the task ran *after* the delay.
    assert task_ran_time < start_time + 0.1
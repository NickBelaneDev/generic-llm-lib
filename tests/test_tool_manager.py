import pytest
from unittest.mock import MagicMock
from generic_llm_lib.llm_core.tools.tool_manager import ToolManager
from generic_llm_lib.llm_core.tools.registry import ToolRegistry
from generic_llm_lib.llm_core.exceptions.exceptions import ToolRegistrationError

# --- Fixtures ---


@pytest.fixture
def mock_registry():
    """Creates a mock ToolRegistry."""
    return MagicMock(spec=ToolRegistry)


@pytest.fixture
def temp_tools_dir(tmp_path):
    """Creates a temporary directory structure for tools."""
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()

    # Create a subdirectory
    sub_dir = tools_dir / "subdir"
    sub_dir.mkdir()

    # Create a dummy tool file in root
    (tools_dir / "dummy_tool.py").write_text("def my_tool():\n    '''A dummy tool.'''\n    pass\n")

    # Create a dummy tool file in subdir
    (sub_dir / "sub_tool.py").write_text("def sub_func():\n    '''A sub tool.'''\n    pass\n")

    # Create a file without docstring
    (tools_dir / "no_doc_tool.py").write_text("def no_doc():\n    pass\n")

    return tools_dir


@pytest.fixture
def tool_manager(mock_registry, temp_tools_dir):
    """Creates a ToolManager instance with the temp directory."""
    return ToolManager(registry=mock_registry, tools_dir=temp_tools_dir)


# --- Tests ---


def test_browse_plugins_root(tool_manager):
    """Test browsing the root directory."""
    result = tool_manager.browse_plugins()
    assert "[DIR] subdir/" in result
    assert "[MOD] dummy_tool" in result
    assert "[MOD] no_doc_tool" in result


def test_browse_plugins_subdir(tool_manager):
    """Test browsing a subdirectory."""
    result = tool_manager.browse_plugins("subdir")
    assert "[MOD] sub_tool" in result


def test_browse_plugins_invalid_path(tool_manager):
    """Test browsing a non-existent path."""
    result = tool_manager.browse_plugins("non_existent")
    assert "Error: Path not found." in result


def test_inspect_plugin_valid(tool_manager):
    """Test inspecting a valid plugin with tools."""
    # We need to mock importlib to avoid actually importing files if we want pure unit tests,
    # but here we are using tmp_path so we can actually import the generated files.
    # However, since the temp dir is not in sys.path, we rely on ToolManager's implementation
    # which uses spec_from_file_location.

    # Note: The ToolManager implementation uses `dynamic_tools.{plugin_path}` as module name.
    # This should work fine with spec_from_file_location.

    result = tool_manager.inspect_plugin("dummy_tool")
    assert "Available tools in 'dummy_tool':" in result
    assert "- my_tool: A dummy tool." in result


def test_inspect_plugin_no_docstring(tool_manager):
    """Test inspecting a plugin where functions have no docstrings."""
    result = tool_manager.inspect_plugin("no_doc_tool")
    # Should not list the tool because it has no docstring
    assert "No tools found in 'no_doc_tool'." in result


def test_inspect_plugin_not_found(tool_manager):
    """Test inspecting a non-existent plugin."""
    result = tool_manager.inspect_plugin("non_existent")
    assert "Error inspecting plugin" in result


def test_load_specific_tool_success(tool_manager, mock_registry):
    """Test successfully loading a specific tool."""
    result = tool_manager.load_specific_tool("dummy_tool", "my_tool")
    assert "Success: Tool 'my_tool' from 'dummy_tool' is now active." in result
    mock_registry.register.assert_called_once()

    # Verify the function passed to register is correct
    args, _ = mock_registry.register.call_args
    assert args[0].__name__ == "my_tool"


def test_load_specific_tool_function_not_found(tool_manager, mock_registry):
    """Test loading a non-existent function from a valid module."""
    result = tool_manager.load_specific_tool("dummy_tool", "non_existent_func")

    assert "Error: Function 'non_existent_func' not found" in result
    mock_registry.register.assert_not_called()


def test_load_specific_tool_module_not_found(tool_manager, mock_registry):
    """Test loading from a non-existent module."""
    result = tool_manager.load_specific_tool("non_existent_module", "my_tool")

    assert "Critical error loading tool" in result
    mock_registry.register.assert_not_called()


def test_load_specific_tool_registration_error(tool_manager, mock_registry):
    """Test handling of registration errors from the registry."""
    # Simulate registry raising an error (e.g. tool already registered)
    mock_registry.register.side_effect = ToolRegistrationError("Tool already registered")

    result = tool_manager.load_specific_tool("dummy_tool", "my_tool")

    assert "Critical error loading tool: Tool already registered" in result

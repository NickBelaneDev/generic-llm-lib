import pytest
from unittest.mock import MagicMock

from generic_llm_lib.llm_core.exceptions.exceptions import ToolLoadError
from generic_llm_lib.llm_core import ScopedTool

# Angenommen, ScopedTool ist in deiner Codebase importierbar
# from dynamic_tools.scoped_tool import ScopedTool


@pytest.fixture
def mock_tool_manager():
    """Baut eine atomare Mock-Umgebung für den ToolManager und die Registry auf."""
    manager = MagicMock()
    manager.registry = MagicMock()
    return manager


def test_scoped_tool_successful_lifecycle(mock_tool_manager):
    """
    Evaluiert den Idealfall: Tool wird geladen und beim Verlassen des
    Scopes garantiert entfernt.
    """
    # Setup: Simuliere erfolgreiches Laden
    mock_tool_manager.load_specific_tool.return_value = "Success: Tool loaded"

    with ScopedTool(mock_tool_manager, "system.docker", "restart_container") as scoped:
        # Verifiziere Zustand INNERHALB des Blocks
        assert scoped.successfully_loaded is True
        mock_tool_manager.load_specific_tool.assert_called_once_with("system.docker", "restart_container")

    # Verifiziere Zustand NACH dem Block
    # unregister MUSS exakt einmal mit dem richtigen Funktionsnamen aufgerufen worden sein
    mock_tool_manager.registry.unregister.assert_called_once_with("restart_container")


def test_scoped_tool_exception_safety(mock_tool_manager):
    """
    Kritischer Pfad: Evaluiert, ob das Tool entladen wird, wenn die
    Agenten-Logik innerhalb des Context Managers crasht.
    """
    mock_tool_manager.load_specific_tool.return_value = "Success: Tool loaded"

    # Wir provozieren einen Absturz innerhalb des Blocks
    with pytest.raises(RuntimeError, match="LLM API Timeout"):
        with ScopedTool(mock_tool_manager, "system.docker", "restart_container"):
            raise RuntimeError("LLM API Timeout")

    # Das ist der wichtigste Assert: Trotz Absturz MUSS unregister aufgerufen worden sein
    mock_tool_manager.registry.unregister.assert_called_once_with("restart_container")


def test_scoped_tool_load_failure_with_chaining(mock_tool_manager):
    # Setup: Simuliere, dass der Manager eine Exception wirft, statt einen String zu liefern
    original_error = ValueError("Plugin file is corrupted")
    mock_tool_manager.load_specific_tool.side_effect = original_error

    with pytest.raises(ToolLoadError) as excinfo:
        with ScopedTool(mock_tool_manager, "system.docker", "invalid_function"):
            pass

    assert isinstance(excinfo.value.__cause__, ValueError)
    assert "Plugin file is corrupted" in str(excinfo.value.__cause__)

    # Sicherstellen, dass unregister nicht gerufen wurde
    mock_tool_manager.registry.unregister.assert_not_called()

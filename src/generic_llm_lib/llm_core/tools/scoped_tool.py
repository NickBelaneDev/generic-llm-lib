"""To offer dynamic tool handling, we use a context manager, that loads the tool for the call. ScopedTool does exactly that.
The LLM has its own tool_manager.execute_dynamic_tool() which uses the ScopedTool instance to safely load the tool."""

from __future__ import annotations

from types import TracebackType
from typing import TYPE_CHECKING

from generic_llm_lib.llm_core.logger import get_logger
from ..exceptions.exceptions import ToolLoadError

if TYPE_CHECKING:
    from .tool_manager import ToolManager

logger = get_logger(__name__)


class ScopedTool:
    """
    Encapsulates the lifecycle of a dynamically loaded tool.
    Guarantees unloading when leaving the scope.
    """

    def __init__(self, tool_manager: ToolManager, plugin_path: str, function_name: str) -> None:
        """Initialize a scoped wrapper for a dynamically loaded tool.

        Args:
            tool_manager: Manager responsible for loading and unregistering tools.
            plugin_path: Filesystem path to the plugin module.
            function_name: Tool function name to load from the plugin.
        """
        self.tool_manager = tool_manager
        self.plugin_path = plugin_path
        self.function_name = function_name
        self.successfully_loaded = False

    def __enter__(self) -> "ScopedTool":
        """Load the target tool when entering the context manager.

        Returns:
            The current scoped tool instance after successful loading.

        Raises:
            ToolLoadError: If the tool cannot be loaded successfully.
        """
        logger.debug("Entering ScopedTool for '%s' from '%s'.", self.function_name, self.plugin_path)
        try:
            result_msg = self.tool_manager.load_specific_tool(self.plugin_path, self.function_name)
        except Exception as e:
            # Kritische Korrektur: msg muss definiert sein, BEVOR der Fehler geworfen wird.
            self.successfully_loaded = False
            msg = f"ScopedTool failed to initialize '{self.function_name}' due to an exception."
            logger.error(msg)
            raise ToolLoadError(msg) from e

        if result_msg.startswith("Success"):
            self.successfully_loaded = True
            logger.info("ScopedTool loaded '%s' from '%s'.", self.function_name, self.plugin_path)
            return self  # Alles gut, Block betreten.

        # Defensiver Pfad: Methode lief ohne Exception durch, gab aber keinen Success-String zurück.
        self.successfully_loaded = False
        msg = f"ScopedTool failed to initialize: {result_msg}"
        logger.error(msg)
        raise ToolLoadError(msg)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Unload the tool when leaving the context manager.

        Args:
            exc_type: Exception type raised in the context body, if any.
            exc_val: Exception instance raised in the context body, if any.
            exc_tb: Traceback associated with the exception, if any.
        """
        if exc_type is not None:
            logger.debug("ScopedTool exiting with exception: %s.", exc_type)
        if self.successfully_loaded:
            logger.debug("Unregistering tool '%s'.", self.function_name)
            self.tool_manager.registry.unregister(self.function_name)
            logger.info("ScopedTool unloaded '%s'.", self.function_name)

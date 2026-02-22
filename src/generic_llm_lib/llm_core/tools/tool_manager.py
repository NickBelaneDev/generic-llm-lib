import importlib.util
import inspect
import os
from pathlib import Path
from typing import Annotated, Any, Dict

from pydantic import Field

from generic_llm_lib.llm_core.tools.registry import ToolRegistry
from generic_llm_lib.llm_core import get_logger

logger = get_logger(__name__)


class ToolManager:
    """
    Manages dynamic loading and inspection of tools from a specified directory.
    This allows the LLM to discover and load tools at runtime.
    """

    def __init__(self, registry: ToolRegistry, tools_dir: str | Path):
        """
        Initialize the ToolManager.

        Args:
            registry: The ToolRegistry instance to register loaded tools into.
            tools_dir: The root directory containing tool modules.
        """
        self.registry = registry
        self.tools_dir = Path(tools_dir)
        self._module_cache: Dict[str, Any] = {}

    def _get_module(self, plugin_path: str) -> Any:
        """
        Loads a module (or retrieves it from cache) without registering tools.

        Args:
            plugin_path: Dotted path to the plugin (e.g., 'database.mysql').

        Returns:
            The loaded module object.

        Raises:
            FileNotFoundError: If the module file does not exist.
            ImportError: If the module cannot be loaded.
        """
        if plugin_path in self._module_cache:
            return self._module_cache[plugin_path]

        # Convert dotted path to file path
        relative_path = plugin_path.replace(".", os.sep) + ".py"
        file_path = self.tools_dir / relative_path

        if not file_path.exists():
            raise FileNotFoundError(f"Module {plugin_path} not found at {file_path}")

        module_name = f"dynamic_tools.{plugin_path}"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load spec for module {plugin_path}")

        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            raise ImportError(f"Failed to execute module {plugin_path}: {e}") from e

        self._module_cache[plugin_path] = module
        return module

    def browse_plugins(
        self,
        path: Annotated[
            str, Field(description="Sub-directory to browse (e.g., 'system/monitoring'). Defaults to root.")
        ] = "",
    ) -> str:
        """
        Lists categories (directories) and available modules (.py files) in the tools directory.

        Args:
            path: The sub-directory to browse.

        Returns:
            A string listing directories and modules.
        """
        target = self.tools_dir / path

        # Security check to prevent traversing up
        try:
            target.resolve().relative_to(self.tools_dir.resolve())
        except ValueError:
            return "Error: Access denied. Cannot browse outside tools directory."

        if not target.exists():
            return "Error: Path not found."

        entries = []
        try:
            for item in target.iterdir():
                if item.is_dir() and not item.name.startswith("_"):
                    entries.append(f"[DIR] {item.name}/")
                elif item.suffix == ".py" and not item.name.startswith("_"):
                    entries.append(f"[MOD] {item.stem}")
        except Exception as e:
            return f"Error browsing directory: {e}"

        if not entries:
            return "Directory is empty."

        return "\n".join(sorted(entries))

    def inspect_plugin(
        self, plugin_path: Annotated[str, Field(description="Dotted path to the module (e.g., 'database.mysql')")]
    ) -> str:
        """
        Scans a module for functions that can be loaded as tools.

        Args:
            plugin_path: The dotted path to the module.

        Returns:
            A list of available tools in the module with their descriptions.
        """
        try:
            module = self._get_module(plugin_path)
            tools = []
            for name, obj in inspect.getmembers(module, inspect.isfunction):
                # We look for functions with docstrings (ToolRegistry standard)
                if not name.startswith("_") and obj.__doc__:
                    # Get first line of docstring
                    doc_first_line = obj.__doc__.strip().splitlines()[0]
                    tools.append(f"- {name}: {doc_first_line}")

            if not tools:
                return f"No tools found in '{plugin_path}'."

            return f"Available tools in '{plugin_path}':\n" + "\n".join(tools)
        except Exception as e:
            return f"Error inspecting plugin: {str(e)}"

    def load_specific_tool(
        self,
        plugin_path: Annotated[str, Field(description="Dotted path to the module")],
        function_name: Annotated[str, Field(description="Name of the function in the module")],
    ) -> str:
        """
        Loads a specific function from a module into the registry.

        Args:
            plugin_path: The dotted path to the module.
            function_name: The name of the function to load.

        Returns:
            Success or error message.
        """
        try:
            module = self._get_module(plugin_path)
            func = getattr(module, function_name, None)

            if not func or not callable(func):
                return f"Error: Function '{function_name}' not found in module '{plugin_path}'."

            # Register the tool
            # Note: This might raise ToolRegistrationError or ToolValidationError
            self.registry.register(func)
            return f"Success: Tool '{function_name}' from '{plugin_path}' is now active."

        except Exception as e:
            logger.error(f"Failed to load tool {function_name} from {plugin_path}: {e}", exc_info=True)
            return f"Critical error loading tool: {str(e)}"

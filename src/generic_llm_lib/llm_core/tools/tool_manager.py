"""Load, inspect, and execute plugin tools dynamically from a configurable tools directory."""

import importlib.util
import inspect
import os
import json
from pathlib import Path
from typing import Annotated, Any, Dict, TypeVar, Generic

from pydantic import Field

from generic_llm_lib.llm_core.exceptions.exceptions import ToolLoadError
from generic_llm_lib.llm_core.tools.registry import ToolRegistry
from generic_llm_lib.llm_core.logger import get_logger

logger = get_logger(__name__)
R = TypeVar("R", bound=ToolRegistry)


class ToolManager(Generic[R]):
    """
    Manages dynamic loading and inspection of tools from a specified directory.
    This allows the LLM to discover and load tools at runtime.
    """

    def __init__(self, registry: R, tools_dir: str | Path):
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
            logger.debug("Tool module cache hit for '%s'.", plugin_path)
            return self._module_cache[plugin_path]

        # Convert dotted path to file path
        logger.debug("Loading tool module '%s' from '%s'.", plugin_path, self.tools_dir)
        relative_path = plugin_path.replace(".", os.sep) + ".py"
        file_path = self.tools_dir / relative_path

        if not file_path.exists():
            msg = f"Module {plugin_path} not found at {file_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        module_name = f"dynamic_tools.{plugin_path}"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            msg = f"Could not load spec for module {plugin_path}"
            logger.error(msg)
            raise ImportError(msg)

        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            msg = f"Failed to execute module {plugin_path}: {e}"
            logger.error(msg, exc_info=True)
            raise ImportError(msg) from e

        self._module_cache[plugin_path] = module
        logger.debug("Loaded tool module '%s'.", plugin_path)
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
        logger.debug("Browsing tools directory '%s' with path '%s'.", self.tools_dir, path)

        # Security check to prevent traversing up
        try:
            target.resolve().relative_to(self.tools_dir.resolve())
        except ValueError:
            msg = "Error: Access denied. Cannot browse outside tools directory."
            logger.warning("Browse denied for path '%s'.", path)
            return msg

        if not target.exists():
            msg = "Error: Path not found."
            logger.warning("Browse path not found: '%s'.", target)
            return msg

        entries = []
        try:
            for item in target.iterdir():
                if item.is_dir() and not item.name.startswith("_"):
                    entries.append(f"[DIR] {item.name}/")
                elif item.suffix == ".py" and not item.name.startswith("_"):
                    entries.append(f"[MOD] {item.stem}")
        except Exception as e:
            msg = f"Error browsing directory: {e}"
            logger.error(msg, exc_info=True)
            return msg

        if not entries:
            logger.info("Tools directory is empty at '%s'.", target)
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
            logger.debug("Inspecting plugin '%s'.", plugin_path)
            module = self._get_module(plugin_path)
            tools = []
            for name, obj in inspect.getmembers(module, inspect.isfunction):
                # We look for functions with docstrings (ToolRegistry standard)
                if not name.startswith("_") and obj.__doc__:
                    # Get first line of docstring
                    doc_first_line = obj.__doc__.strip().splitlines()[0]
                    tools.append(f"- {name}: {doc_first_line}")

            if not tools:
                logger.info("No tools found in '%s'.", plugin_path)
                return f"No tools found in '{plugin_path}'."

            return f"Available tools in '{plugin_path}':\n" + "\n".join(tools)
        except Exception as e:
            msg = f"Error inspecting plugin: {str(e)}"
            logger.error(msg, exc_info=True)
            return msg

    def execute_dynamic_plugin(
        self,
        plugin_path: Annotated[str, Field(description="Dotted path to the module")],
        function_name: Annotated[str, Field(description="Name of the function to execute")],
        kwargs_json: Annotated[str, Field(description="JSON string of the function's arguments")],
    ) -> str:
        """Load, execute, and unload a plugin function using JSON arguments.

        Args:
            plugin_path: Dotted module path that contains the function.
            function_name: Function name to execute from the plugin module.
            kwargs_json: JSON string containing keyword arguments for the function.

        Returns:
            The function result converted to a string, or a formatted error message.
        """
        from generic_llm_lib.llm_core.tools.scoped_tool import ScopedTool

        logger.info("Executing dynamic tool '%s' from '%s'.", function_name, plugin_path)
        try:
            kwargs = json.loads(kwargs_json)
        except json.JSONDecodeError as e:
            msg = f"Error parsing the arguments: {e}"
            logger.error(msg)
            return msg

        with ScopedTool(self, plugin_path, function_name) as scoped:
            if not scoped.successfully_loaded:
                return f"Critical Error: Could not load '{function_name}' from '{plugin_path}'"

            try:
                func = self.registry.implementations.get(function_name)
                if func is None:
                    msg = f"Error: Tool '{function_name}' is not available in the registry."
                    logger.error(msg)
                    return msg
                result = func(**kwargs)
                logger.debug("Tool '%s' executed successfully.", function_name)
                return str(result)

            except Exception as e:
                msg = f"Error executing '{function_name}': {str(e)}"
                logger.error(msg, exc_info=True)
                return msg

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
            logger.info("Loading tool '%s' from '%s'.", function_name, plugin_path)
            module = self._get_module(plugin_path)
            func = getattr(module, function_name, None)

            if not func or not callable(func):
                msg = f"Error: Function '{function_name}' not found in module '{plugin_path}'."
                logger.error(msg)
                raise ValueError(msg)

            # Register the tool
            # Note: This might raise ToolRegistrationError or ToolValidationError
            self.registry.register(func)
            msg = f"Success: Tool '{function_name}' from '{plugin_path}' is now active."
            logger.info(msg)
            return msg

        except Exception as e:
            msg = f"Failed to load tool {function_name} from {plugin_path}: {e}"
            logger.error(msg, exc_info=True)
            raise ToolLoadError(msg)

import importlib.util
import inspect
import os
import json
import warnings
from pathlib import Path
from typing import Annotated, Any, Dict, TypeVar, Generic, Literal

from pydantic import Field

from ...exceptions import ToolLoadError
from ..registry import ToolRegistry
from ...logger import get_logger

logger = get_logger(__name__)
R = TypeVar("R", bound=ToolRegistry)


class ToolManager(Generic[R]):
    """
    Manages dynamic loading and inspection of tools from a specified directory.
    This allows the LLM to discover and load tools at runtime.
    """

    def __init__(self, registry: R, tools_dir: str | Path, mode: Literal["proxy", "hot_swap"] = "proxy"):
        """
        Initialize the ToolManager.

        Args:
            registry: The ToolRegistry instance to register loaded tools into.
            tools_dir: The root directory containing tool modules.
        """
        self.registry = registry
        self.tools_dir = Path(tools_dir)
        self._module_cache: Dict[str, Any] = {}
        self.mode = mode
        self._register_default_tools()

        warnings.warn(
            "DynamicToolManager is an experimental feature. It may lead to hallucination loops "
            "and JSON parsing errors in models with < 70B parameters. Use with caution.",
            UserWarning,
            stacklevel=2,
        )

    def _register_default_tools(self) -> None:
        """
        Registers the default management tools into the registry.

        This method enables the LLM to browse, inspect, and execute or load
        dynamic tools based on the configured operational mode.
        """

        self.registry.register(self.browse_plugins)
        self.registry.register(self.inspect_plugin)
        if self.mode == "proxy":
            self.registry.register(self.execute_dynamic_plugin)
        elif self.mode == "hot_swap":
            self.registry.register(self.load_specific_tool)
        else:
            raise ValueError(f"Unexpected ToolManager Mode: {self.mode}")

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

        Returns:
            A string listing directories and modules.
        """
        target = self.tools_dir / path
        logger.debug(f"Browsing tools directory '{self.tools_dir}' with path '{path}'.", self.tools_dir, path)

        # Security check to prevent traversing up
        try:
            target.resolve().relative_to(self.tools_dir.resolve())
        except ValueError:
            msg = f"Error: Access denied. Cannot browse outside tools directory. Use the following path: '{self.tools_dir}' to browse your tools."
            logger.warning("Browse denied for path '%s'.", path)
            return msg

        if not target.exists():
            msg = f"Error: Path not found. Use the following path: '{self.tools_dir}' to browse your tools."
            logger.warning(f"Browse path not found: {target}")
            return msg

        entries = []
        try:
            for item in target.iterdir():
                if item.is_dir() and not item.name.startswith("_"):
                    entries.append(f"[DIR] {item.name}/")
                elif item.suffix == ".py" and not item.name.startswith("_"):
                    entries.append(f"[MOD] {item.stem}")
        except Exception as e:
            msg = f"Error browsing directory: {e}. Use the following path: '{self.tools_dir}' to browse your tools."
            logger.error(msg, exc_info=True)
            return msg

        if not entries:
            logger.info(
                f"Tools directory is empty at {target}. Make sure you are using the correct path: {self.tools_dir} to browse your tools"
            )
            return "Directory is empty."

        return "\n".join(sorted(entries))

    def inspect_plugin(
        self, plugin_path: Annotated[str, Field(description="Dotted path to the module (e.g., 'database.mysql')")]
    ) -> str:
        """
        Scans a module for functions that can be loaded as tools.

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
                    try:
                        tool_def = self.registry._generate_tool_definition(obj)
                        schema_str = json.dumps(tool_def.parameters)
                    except Exception:
                        schema_str = "Konnte Schema nicht extrahieren"

                    tools.append(f"- {name}: {obj.__doc__}\n  Expected kwargs_json Format: {schema_str}")
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
        kwargs_json: Annotated[dict[str, Any], Field(description="JSON string of the function's arguments")],
    ) -> str:
        """Executes a function from a dynamic plugin module.

        Returns:
            The result of the function execution as a string, or an error message.
        """

        from .scoped_tool import ScopedTool

        logger.info("Executing dynamic tool '%s' from '%s'.", function_name, plugin_path)
        try:
            kwargs = json.loads(kwargs_json)  # type: ignore
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

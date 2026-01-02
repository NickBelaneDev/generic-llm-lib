from google.genai import types
from src.llm_core import ToolRegistry
from typing import Callable, Dict, Any, Union

class GeminiToolRegistry(ToolRegistry):
    def register(self, name: str, description: str, func: Callable, parameters: Union[Dict[str, Any], types.Schema, Any]):
        super().register(name, description, func, parameters)

    @property
    def tool_object(self) -> types.Tool | None:
        if not self.tools:
            return None

        declarations = []
        for tool in self.tools.values():
            # Ensure parameters are compatible with types.FunctionDeclaration
            # If it's a dict, the library usually handles it, but type checkers might complain.
            # If it's already a types.Schema (or similar), it's passed through.
            declarations.append(types.FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters
            ))

        return types.Tool(function_declarations=declarations)

    @property
    def implementations(self):
        return {name: tool.func for name, tool in self.tools.items()}

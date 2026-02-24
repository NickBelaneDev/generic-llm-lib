from typing import cast
from unittest.mock import MagicMock

from google.genai.client import AsyncClient
from openai import AsyncOpenAI

from generic_llm_lib import (
    GenericGemini,
    GenericOpenAI,
    GeminiToolRegistry,
    OpenAIToolRegistry,
)
from generic_llm_lib.llm_core.tools import ToolManager


def _tool_manager_with_registry(registry: object) -> ToolManager:
    manager = MagicMock(spec=ToolManager)
    manager.registry = registry
    return cast(ToolManager, manager)


def test_openai_prefers_explicit_registry_over_tool_manager(openai_client: AsyncOpenAI) -> None:
    explicit_registry = OpenAIToolRegistry()
    manager_registry = OpenAIToolRegistry()
    tool_manager = _tool_manager_with_registry(manager_registry)

    llm = GenericOpenAI(
        client=openai_client,
        model_name="openai/gpt-oss-20b",
        sys_instruction="You are a helper.",
        registry=explicit_registry,
        tool_manager=tool_manager,
    )

    assert llm.registry is explicit_registry
    assert llm._tool_loop._registry is explicit_registry


def test_openai_uses_tool_manager_registry_when_no_registry(openai_client: AsyncOpenAI) -> None:
    manager_registry = OpenAIToolRegistry()
    tool_manager = _tool_manager_with_registry(manager_registry)

    llm = GenericOpenAI(
        client=openai_client,
        model_name="openai/gpt-oss-20b",
        sys_instruction="You are a helper.",
        tool_manager=tool_manager,
    )

    assert llm.registry is manager_registry
    assert llm._tool_loop._registry is manager_registry


def test_gemini_prefers_explicit_registry_over_tool_manager(genai_client: AsyncClient) -> None:
    explicit_registry = GeminiToolRegistry()
    manager_registry = GeminiToolRegistry()
    tool_manager = _tool_manager_with_registry(manager_registry)

    llm = GenericGemini(
        aclient=genai_client,
        model_name="gemini-2.5-flash",
        sys_instruction="You are a helper.",
        registry=explicit_registry,
        tool_manager=tool_manager,
    )

    assert llm.registry is explicit_registry
    assert llm._tool_loop._registry is explicit_registry


def test_gemini_uses_tool_manager_registry_when_no_registry(genai_client: AsyncClient) -> None:
    manager_registry = GeminiToolRegistry()
    tool_manager = _tool_manager_with_registry(manager_registry)

    llm = GenericGemini(
        aclient=genai_client,
        model_name="gemini-2.5-flash",
        sys_instruction="You are a helper.",
        tool_manager=tool_manager,
    )

    assert llm.registry is manager_registry
    assert llm._tool_loop._registry is manager_registry

import os
from unittest.mock import MagicMock

import pytest
from typing import Any, cast
from dotenv import load_dotenv, find_dotenv
from google.genai.client import Client, AsyncClient
from openai import AsyncOpenAI
from generic_llm_lib import ToolManager, OpenAIToolRegistry

# Load environment variables from .env file
# We explicitly look for .env in the project root if find_dotenv() fails or returns empty
env_file = find_dotenv()
if not env_file:
    # Fallback: try to find .env in the current working directory or parent directories manually
    # This is helpful if the test runner is started from a subdirectory
    current_dir = os.getcwd()
    potential_env = os.path.join(current_dir, ".env")
    if os.path.exists(potential_env):
        env_file = potential_env
    else:
        # Try one level up
        potential_env = os.path.join(os.path.dirname(current_dir), ".env")
        if os.path.exists(potential_env):
            env_file = potential_env

if env_file:
    print(f"Loading .env from: {env_file}")
    load_dotenv(env_file)
else:
    print("Warning: No .env file found.")


@pytest.fixture
def genai_client() -> AsyncClient:
    # Try different common names for the API key
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    # If no key is provided, we use a dummy key.
    # This allows tests to run with VCR cassettes even without a real key.
    # If recording is needed, a real key must be in .env
    if not api_key:
        print("Warning: No GOOGLE_API_KEY or GEMINI_API_KEY found in environment. Using dummy key.")
        api_key = "dummy_key"
    return Client(api_key=api_key).aio


@pytest.fixture
def openai_client() -> AsyncOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = "dummy_key"
    return AsyncOpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL"))


@pytest.fixture
def openai_mock_tool_manager() -> ToolManager[OpenAIToolRegistry]:
    """
    Baut eine atomare Mock-Umgebung für den generischen ToolManager auf.
    """
    # 1. Erstelle die Mocks mit der korrekten Spezifikation (spec)
    manager_mock = MagicMock(spec=ToolManager)
    registry_mock = MagicMock(spec=OpenAIToolRegistry)

    # 2. Verknüpfe die Mocks intern
    manager_mock.registry = registry_mock

    # 3. Zwinge den Typ-Checker, den Mock als instanziierten Generic zu akzeptieren
    return cast(ToolManager[OpenAIToolRegistry], manager_mock)


@pytest.fixture(scope="session")
def vcr_config() -> dict[str, Any]:
    return {
        "cassette_library_dir": "tests/cassettes",
        "record_mode": os.getenv("VCR_RECORD_MODE", "once"),
        "match_on": ["method", "path", "query"],
        "filter_headers": [
            "authorization",
            "openai-organization",
            "x-goog-api-key",
            "x-api-key",
            "api-key",
        ],
        "filter_query_parameters": ["key", "api_key", "access_token"],
        "decode_compressed_response": True,
    }

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

### Added
- **ToolManager**: Allows LLMs to load tools dynamically.

## [0.4.0] - 2026-02-01

### Added
- **Unified Imports**: Simplified library usage by exposing core classes (`GenericLLM`, `ChatResult`, `GenericGemini`, `GenericOpenAI`, etc.) directly from the top-level `generic_llm_lib` package.
- **Project Structure**: Adopted a clean `src`-layout for better packaging and testing isolation.
- **Type Safety**: Enforced strict type checking (`mypy`) across the entire codebase, including `disallow_untyped_defs = true`.
- **Testing**: Added `pytest-recording` (VCR.py) for deterministic and fast integration tests using cassettes.

### Changed
- **Refactoring**: Massive cleanup of internal package structure (`llm_core`, `llm_impl`) to follow the Dependency Inversion Principle more strictly.
- **Dependencies**: Switched to `uv` for modern, fast dependency management.
- **Documentation**: Updated `README.md` and `CONTRIBUTING.md` (implied) to reflect the new structure and usage patterns.

### Fixed
- **Imports**: Resolved circular import issues and exposed missing classes in `__init__.py` files.
- **DX**: Improved Developer Experience by reducing import depth and clarifying public APIs.

## [0.3.1] - 2026-01-31

### Changed
- **Tool Execution**: Refactored and centralized the tool execution loop logic to reduce duplication and improve maintainability.
- **Project Structure**: Reorganized directory structure and improved dependency and registry consistency.

### Fixed
- **General**: Addressed various bugs and improved overall stability.

## [0.2.3] - 2026-01-30

### Added
- **OpenAI Support**: Added `GenericOpenAI` implementation to support OpenAI's chat models with tool calling capabilities.

### Fixed
- **Blocking Event Loop**: Fixed a critical issue where synchronous tools were executed directly in the main thread, blocking the asyncio event loop. Synchronous tools are now executed in a separate thread using `asyncio.to_thread`.
- **JSON Parsing**: Added robust error handling for JSON parsing of function arguments in `GenericOpenAI`. Invalid JSON from the LLM now returns an error message to the model instead of crashing the application.
- **Test Suite**: Fixed various test failures in `test_openai_core.py` related to missing mock attributes and incorrect assertions.

## [0.2.2] - 2026-01-13

### Added
- **@tool decorator**: Now supports a simple decorator `@registry.tool` to declare a function as a tool and add it to the llm's `registry`.

## [0.2.1] - 2026-01-10

### Added
- **async Function calls**: The GenericGemini can now handle async functions as well.

## [0.2.0] - 2026-01-06

### Added
- **Response Models**: Introduced structured Pydantic models (`GeminiMessageResponse`, `GeminiChatResponse`, `GeminiTokens`) to standardize the output format for Gemini interactions.
- **Parallel Function Calling**: `GenericGemini` now supports executing multiple function calls returned by the model in a single turn.
- **Exports**: Added `GeminiMessageResponse`, `GeminiChatResponse`, and `GeminiTokens` to top-level `llm_impl` exports for easier typing.

### Changed
- **Refactoring**: `GenericGemini.chat` method logic split into `_handle_function_calls` and `_build_response` for better readability and maintainability.
- **Error Handling**: Tool execution errors are now caught and returned to the LLM instead of raising an exception in the client.
- **Models**: `GeminiTokens` fields now default to `None` using Pydantic's `Field(default=None)`, simplifying instantiation.
- **Robustness**: Added checks for missing `usage_metadata` in Gemini responses.
- **Documentation**: Updated docstrings for `GenericGemini` methods.

### Fixed
- Fixed access to chat history using `chat.history` instead of `chat.get_history()`.

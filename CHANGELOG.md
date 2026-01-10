# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
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

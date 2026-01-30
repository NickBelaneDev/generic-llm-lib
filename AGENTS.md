# Global development guidelines for generic-llm-lib

This document provides context to understand the generic-llm-lib Python project and assist with development.

## Project architecture and context

### Project structure

This is a Python project using `uv` for dependency management.

```txt
generic-llm-lib/
├── src/
│   ├── llm_core/         # Base abstractions, interfaces, and protocols
│   ├── llm_impl/         # Concrete implementations of LLM providers
│   │   ├── gemini/       # Google Gemini integration
│   │   ├── open_api/     # OpenAI integration
│   │   └── ...
├── tests/                # Unit and integration tests
├── pyproject.toml        # Project configuration and dependencies
└── uv.lock               # Locked dependencies
```

- **Core layer** (`src/llm_core`): Base abstractions, interfaces, and protocols. Users should not need to know about this layer directly if they only use the implementations.
- **Implementation layer** (`src/llm_impl`): Concrete implementations for specific providers (Gemini, OpenAI).

### Development tools & commands

- `uv` – Fast Python package installer and resolver
- `pytest` – Testing framework

This project uses `uv` for dependency management.

```bash
# Run tests
uv run pytest

# Run specific test file
uv run pytest tests/test_specific.py
```

#### Key config files

- pyproject.toml: Main project configuration
- uv.lock: Locked dependencies for reproducible builds

#### Commit standards

Suggest PR titles that follow Conventional Commits format. Note that all commit/PR titles should be in lowercase with the exception of proper nouns/named entities. All PR titles should include a scope with no exceptions. For example:

```txt
feat(core): add new base agent class
fix(gemini): resolve connection timeout
chore(deps): update pydantic
```

#### Pull request guidelines

- Always add a disclaimer to the PR description mentioning how AI agents are involved with the contribution.
- Describe the "why" of the changes, why the proposed solution is the right one. Limit prose.
- Highlight areas of the proposed changes that require careful review.

## Core development principles

### Maintain stable public interfaces

CRITICAL: Always attempt to preserve function signatures, argument positions, and names for exported/public methods. Do not make breaking changes.
You should warn the developer for any function signature changes, regardless of whether they look breaking or not.

**Before making ANY changes to public APIs:**

- Check if the function/class is exported in `__init__.py`
- Look for existing usage patterns in tests and examples
- Use keyword-only arguments for new parameters: `*, new_param: str = "default"`
- Mark experimental features clearly with docstring warnings

Ask: "Would this change break someone's code if they used it last week?"

### Code quality standards

All Python code MUST include type hints and return types.

```python title="Example"
def filter_data(data: list[str], criteria: set[str]) -> list[str]:
    """Single line description of the function.

    Any additional context about the function can go here.

    Args:
        data: List of items to filter.
        criteria: Set of criteria to apply.

    Returns:
        Filtered list of items.
    """
```

- Use descriptive, self-explanatory variable names.
- Follow existing patterns in the codebase you're modifying
- Attempt to break up complex functions (>20 lines) into smaller, focused functions where it makes sense

### Testing requirements

Every new feature or bugfix MUST be covered by unit tests.

- Tests are located in `tests/`.
- We use `pytest` as the testing framework.

**Checklist:**

- [ ] Tests fail when your new logic is broken
- [ ] Happy path is covered
- [ ] Edge cases and error conditions are tested
- [ ] Use fixtures/mocks for external dependencies
- [ ] Tests are deterministic (no flaky tests)
- [ ] Does the test suite fail if your new logic is broken?

### Security and risk assessment

- No `eval()`, `exec()`, or `pickle` on user-controlled input
- Proper exception handling (no bare `except:`) and use a `msg` variable for error messages
- Remove unreachable/commented code before committing
- Race conditions or resource leaks (file handles, sockets, threads).
- Ensure proper resource cleanup (file handles, connections)

### Documentation standards

Use Google-style docstrings with Args section for all public functions.

```python title="Example"
def send_request(url: str, *, timeout: int = 30) -> bool:
    """Send a request to the specified URL.

    Any additional context about the function can go here.

    Args:
        url: The target URL.
        timeout: Timeout in seconds.

    Returns:
        `True` if the request was successful, `False` otherwise.

    Raises:
        ConnectionError: If unable to connect to the server.
    """
```

- Types go in function signatures, NOT in docstrings
  - If a default is present, DO NOT repeat it in the docstring unless there is post-processing or it is set conditionally.
- Focus on "why" rather than "what" in descriptions
- Document all parameters, return values, and exceptions
- Keep descriptions concise but clear
- Ensure American English spelling (e.g., "behavior", not "behaviour")

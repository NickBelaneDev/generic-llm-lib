# Security Audit & Improvement Report

## Overview
This report outlines the findings from a security and code quality review of the `generic-llm-lib`. The library is generally well-structured and uses modern practices (AsyncIO, Pydantic), but there are areas where security and robustness can be improved.

## Security Findings

### 1. Information Leakage in Error Handling
**Severity:** Medium
**Location:** 
- `src/llm_impl/gemini/core.py` (Line 158)
- `src/llm_impl/open_api/tool_helper.py` (Line 115)

**Description:**
When a tool execution fails, the library catches the exception and sends `str(e)` back to the LLM.
```python
response={"error": str(e)}
```
If a tool fails due to a database error, file system error, or internal logic error, the exception message might contain sensitive information (e.g., file paths, SQL fragments, API keys in connection strings).

**Recommendation:**
Sanitize error messages sent to the LLM. Log the full exception on the server side using a secure logger, and return a generic error message to the LLM, or a sanitized version of the error.

### 2. Lack of Tool Execution Timeouts
**Severity:** Medium
**Location:** 
- `src/llm_impl/gemini/core.py`
- `src/llm_impl/open_api/tool_helper.py`

**Description:**
Tools are executed using `await tool_function(...)` or `asyncio.to_thread(...)` without a timeout. If a user-defined tool hangs (e.g., an infinite loop or a hanging network request), the entire chat request will hang indefinitely, potentially leading to a Denial of Service (DoS) if many such requests consume server resources.

**Recommendation:**
Implement a timeout mechanism for tool execution.
```python
try:
    function_result = await asyncio.wait_for(tool_function(**function_args), timeout=self.tool_timeout)
except asyncio.TimeoutError:
    # Handle timeout
```

### 3. Unbounded Conversation History
**Severity:** Low/Medium
**Location:** `src/llm_impl/gemini/core.py`, `src/llm_impl/open_api/core.py`

**Description:**
While the library cleans intermediate tool calls from the history, it does not appear to enforce a maximum limit on the number of turns in the `history` list passed by the user. An indefinitely growing history will eventually exceed the model's context window or consume excessive tokens, leading to crashes or high costs.

**Recommendation:**
Implement a strategy to truncate or summarize older history when it exceeds a certain token count or message limit.

## Code Quality & Improvements

### 1. Logging
**Location:** `src/llm_impl/open_api/core.py`
**Observation:** There are commented-out `print` statements used for debugging.
**Recommendation:** Replace `print` statements with the standard Python `logging` module. This allows for proper log levels (DEBUG, INFO, ERROR) and better integration with production monitoring systems.

### 2. Schema Resolution Recursion
**Location:** `src/llm_core/registry.py` (`_resolve_schema_refs`)
**Observation:** The method uses recursion to resolve JSON schema references. While unlikely with standard tool definitions, a maliciously crafted or extremely deep schema could cause a `RecursionError`.
**Recommendation:** Ensure that tool definitions are trusted or implement a recursion depth limit.

### 3. Hardcoded Configuration
**Location:** `src/llm_core/types.py`
**Observation:** Default values for `temperature` (1.0) and `max_tokens` (100) are hardcoded in the `LLMConfig` model (though they can be overridden).
**Recommendation:** Ensure these defaults are sensible for the intended use cases. 100 tokens might be too short for some "helpful" responses.

## Summary of Recommendations

1.  **Sanitize Tool Errors:** Do not send raw exception strings to the LLM.
2.  **Add Timeouts:** Wrap tool execution in `asyncio.wait_for`.
3.  **Use Logging:** Replace commented-out prints with `logging`.
4.  **Limit History:** Advise users or implement helpers to manage history size.

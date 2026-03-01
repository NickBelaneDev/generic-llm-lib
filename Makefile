# ==============================================================================
# Makefile for generic-llm-lib
#
# This Makefile provides a set of commands to help with development, testing,
# and release management.
#
# Default target: help
# ------------------------------------------------------------------------------

.DEFAULT_GOAL := help

# Phony targets prevent conflicts with files of the same name
.PHONY: help dev-install quality security doc complexity fix test release

# ------------------------------------------------------------------------------
# HELP
# ------------------------------------------------------------------------------

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@echo "  help           Show this help message."
	@echo "  dev-install    Install all dependencies, including development tools."
	@echo ""
	@echo "  -- Code Quality & Analysis --"
	@echo "  quality        Run all code quality checks (ruff, black, mypy)."
	@echo "  security       Run security scans (bandit, pip-audit)."
	@echo "  doc            Check for docstring coverage with interrogate."
	@echo "  complexity     Analyze code complexity with xenon."
	@echo ""
	@echo "  -- Development & Testing --"
	@echo "  fix            Automatically fix linting issues and format code."
	@echo "  test           Run the test suite using pytest."
	@echo ""
	@echo "  -- Release Management --"
	@echo "  release        Create a new release. Requires VERSION (e.g., make release VERSION=v0.1.0)."
	@echo ""

# ------------------------------------------------------------------------------
# DEPENDENCY MANAGEMENT
# ------------------------------------------------------------------------------

dev-install:
	@echo "--> Installing all dependencies..."
	@uv sync --all-extras --dev

# ------------------------------------------------------------------------------
# CODE QUALITY & ANALYSIS
# ------------------------------------------------------------------------------

quality:
	@echo "--> Running code quality checks..."
	@echo "    - Running ruff linter..."
	@uv run ruff check .
	@echo "    - Checking formatting with black..."
	@uv run black --check .
	@echo "    - Running mypy for type checking..."
	@uv run mypy src

security:
	@echo "--> Running security scans..."
	@echo "    - Running bandit for security vulnerabilities..."
	@uv run bandit -r src/ -ll -iii
	@echo "    - Auditing dependencies with pip-audit..."
	@uv run pip-audit

doc:
	@echo "--> Checking docstring coverage..."
	@uv run interrogate -vv --fail-under 90 src/

complexity:
	@echo "--> Analyzing code complexity..."
	@uv run xenon --max-absolute A --max-modules A --max-average A src/

# ------------------------------------------------------------------------------
# DEVELOPMENT & TESTING
# ------------------------------------------------------------------------------

fix:
	@echo "--> Automatically fixing code style and formatting..."
	@uv run ruff check . --fix
	@uv run black .

test:
	@echo "--> Running tests..."
	@uv run pytest

# ------------------------------------------------------------------------------
# RELEASE MANAGEMENT
# ------------------------------------------------------------------------------

release:
ifndef VERSION
	$(error VERSION is not set. Usage: make release VERSION=v0.0.0)
endif
	@echo "--> Creating release $(VERSION)..."
	@echo "    - Checking version consistency..."
	@uv run python scripts/check_version.py $(VERSION)

	@echo "    - Tagging release and creating branch..."
	@git tag -a $(VERSION) -m "Release $(VERSION)"
	@git checkout -b version/$(VERSION)
	@git push origin $(VERSION)
	@git push origin version/$(VERSION)
	@echo "--> Release $(VERSION) created successfully."

.PHONY: quality security doc complexity fix test release

quality:
	uv sync --all-extras --dev
	uv run ruff check .
	uv run black --check .
	uv run mypy src

security:
	uv run bandit -r src/
	uv run pip-audit

doc:
	uv run interrogate -vv --fail-under 90 src/

complexity:
	uv run xenon --max-absolute A --max-modules A --max-average A src/

fix:
	uv run ruff check . --fix
	uv run black .

test:
	uv run pytest

release:
ifndef VERSION
	$(error VERSION is not set. Usage: make release VERSION=v0.0.0)
endif
	@echo "Checking version consistency..."
	@uv run python scripts/check_version.py $(VERSION)

	git tag -a $(VERSION) -m "Release $(VERSION)"
	git checkout -b version/$(VERSION)
	git push origin $(VERSION)
	git push origin version/$(VERSION)

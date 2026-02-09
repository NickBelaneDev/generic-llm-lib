.PHONY: quality fix test release

quality:
	uv sync --all-extras --dev
	uv run ruff check .
	uv run black --check .
	uv run mypy src

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

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
	@uv run python -c "import sys; v = sys.argv[1].lstrip('v'); p = open('pyproject.toml').read(); c = open('CHANGELOG.md').read(); ver_line = f'version = \"{v}\"'; log_line = f'[{v}]'; errs = []; ver_line in p or errs.append(f'Error: Version {v} does not match pyproject.toml'); log_line in c or errs.append(f'Error: Version {v} not found in CHANGELOG.md'); errs and (print('\n'.join(errs)) or sys.exit(1))" $(VERSION)

	git tag -a $(VERSION) -m "Release $(VERSION)"
	git checkout -b version/$(VERSION)
	git push origin $(VERSION)
	git push origin version/$(VERSION)

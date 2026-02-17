import sys
from pathlib import Path


def check_version(version_tag: str) -> None:
    # Strip 'v' prefix if present (e.g., v0.4.0 -> 0.4.0)
    version = version_tag.lstrip("v")

    # Define file paths
    root = Path(__file__).parent.parent
    pyproject_path = root / "pyproject.toml"
    changelog_path = root / "CHANGELOG.md"

    errors = []

    # Check pyproject.toml
    if not pyproject_path.exists():
        errors.append(f"Error: {pyproject_path} not found.")
    else:
        content = pyproject_path.read_text(encoding="utf-8")
        # Look for version = "0.4.0"
        # We use a simple string check or regex. String check is safer for exact match if formatted standardly.
        expected_line = f'version = "{version}"'
        if expected_line not in content:
            errors.append(
                f"Error: Version {version} does not match version in pyproject.toml (expected '{expected_line}')"
            )

    # Check CHANGELOG.md
    if not changelog_path.exists():
        errors.append(f"Error: {changelog_path} not found.")
    else:
        content = changelog_path.read_text(encoding="utf-8")
        # Look for [0.4.0]
        expected_header = f"[{version}]"
        if expected_header not in content:
            errors.append(f"Error: Version {version} not found in CHANGELOG.md (expected header '{expected_header}')")

    if errors:
        print("\n".join(errors))
        sys.exit(1)

    print(f"Version {version} consistency check passed.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_version.py <version>")
        sys.exit(1)

    check_version(sys.argv[1])

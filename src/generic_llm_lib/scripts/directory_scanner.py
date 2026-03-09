"""Scans a directory to create a filtered tree structure of its contents."""

from pathlib import Path
from typing import Any, Dict, Set, Union

# Define the project root relative to this script's location.
# This script is in: /src/generic_llm_lib/scripts/
# Project root is 4 levels up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class DirectoryScanner:
    """
    A scanner to inspect a directory and build a filtered map of its structure.

    This class is designed to walk through a directory tree and create a
    dictionary that represents its structure, but only including files with
    specific extensions and excluding common temporary or virtual environment
    directories.
    """

    def __init__(self) -> None:
        """Initializes the DirectoryScanner with filtering rules."""
        self.max_dirs_to_scan = 100
        self.scanned_dirs_count = 0
        self.allowed_extensions: Set[str] = {".py", ".toml", ".env", ".json"}
        self.excluded_dirs: Set[str] = {
            "__pycache__",
            ".git",
            ".github",
            ".idea",
            ".vscode",
            ".venv",
            "venv",
            "env",
            "node_modules",
            "dist",
            "build",
            ".eggs",
            "lib",
            "lib64",
            "bin",
            "include",
            "share",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            "libraries",  # Explicitly requested exclusion
            "generic_llm_lib.egg-info",
        }

    def read_directory_tree(self, path: Path = PROJECT_ROOT) -> Dict[str, Union[str, Dict[str, Any]]]:
        """
        Walks a directory to create a dictionary from its filtered structure.

        This method recursively scans a directory, including only files with
        allowed extensions and skipping specified directories. This is useful
        for generating a clean overview of a project's source code.

        Args:
            path: The root path from which to start scanning.

        Returns:
            A dictionary representing the filtered directory tree. Files are
            represented as strings with their size, and directories are
            represented as nested dictionaries. Returns an empty dictionary
            if the path is not a directory.
        """
        if not path.is_dir():
            return {}

        data: Dict[str, Union[str, Dict[str, Any]]] = {}

        for item in sorted(path.iterdir()):  # Sort for consistent output
            if item.name in self.excluded_dirs:
                continue

            if item.is_dir():
                if self.scanned_dirs_count >= self.max_dirs_to_scan:
                    continue
                self.scanned_dirs_count += 1
                subtree = self.read_directory_tree(item)
                if subtree:  # Only include non-empty directories
                    data[item.name] = subtree
            elif item.is_file() and item.suffix in self.allowed_extensions:
                try:
                    size_kb = item.stat().st_size / 1024
                    data[item.name] = f"{size_kb:.2f}kb"
                except FileNotFoundError:
                    # This can happen with broken symlinks, so we skip them.
                    continue
        return data

    def json_tree_to_string(self, data: Dict[str, Any], indent: int = 0) -> str:
        """
        Converts a dictionary tree to a formatted string.

        Args:
            data: The dictionary tree to convert.
            indent: The current indentation level for pretty-printing.

        Returns:
            A string representation of the directory tree.
        """
        content_string = ""
        for key, value in data.items():
            prefix = "   " * indent
            if isinstance(value, dict):
                content_string += f"{prefix}{key}/\n"
                content_string += self.json_tree_to_string(value, indent + 1)
            else:
                content_string += f"{prefix}{key}: {value}\n"
        return content_string

    def build_directory_tree(self, data: Dict[str, Any], path: Path = PROJECT_ROOT) -> None:
        """
        Creates a directory structure from a dictionary.

        Note: This is a destructive operation and should be used with caution.

        Args:
            data: The dictionary representing the directory structure.
            path: The root path where the structure will be created.
        """
        for key, value in data.items():
            next_path = path / key
            try:
                if isinstance(value, dict):
                    next_path.mkdir(parents=True, exist_ok=True)
                    print(f"Directory created at {next_path}")
                    self.build_directory_tree(value, next_path)
                else:
                    next_path.touch()
                    print(f"File created {next_path}")
            except FileExistsError:
                print(f"File or directory already exists at {next_path}")


if __name__ == "__main__":
    scanner = DirectoryScanner()
    project_tree = scanner.read_directory_tree()
    print(scanner.json_tree_to_string(project_tree))

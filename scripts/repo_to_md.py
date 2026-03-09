import os
from pathlib import Path

def generate_repo_md(root_dir: str, output_file: str):
    """Generates a single Markdown file containing the repository structure and file contents.

    Args:
        root_dir: The root directory of the repository.
        output_file: The path to the output Markdown file.
    """
    root = Path(root_dir)
    ignore_dirs = {".git", "__pycache__", ".venv", ".pytest_cache", "dist", "build", ".idea", "libraries"}
    ignore_files = {"uv.lock", "package-lock.json", ".DS_Store", output_file}
    extensions = {".py", ".toml", ".env", ".json"}

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# Repository Summary: {root.name}\n\n")
        
        f.write("## Directory Structure\n")
        f.write("```text\n")
        for path in sorted(root.rglob("*")):
            if any(part in ignore_dirs for part in path.parts):
                continue
            depth = len(path.relative_to(root).parts) - 1
            indent = "  " * depth
            if path.is_dir():
                f.write(f"{indent}📁 {path.name}/\n")
            elif path.suffix in extensions and path.name not in ignore_files:
                f.write(f"{indent}📄 {path.name}\n")
        f.write("```\n\n")

        f.write("## File Contents\n\n")
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in extensions:
                if any(part in ignore_dirs for part in path.parts) or path.name in ignore_files:
                    continue
                
                relative_path = path.relative_to(root)
                f.write(f"### File: {relative_path}\n")
                f.write(f"```python\n" if path.suffix == ".py" else "```text\n")
                try:
                    f.write(path.read_text(encoding="utf-8"))
                except Exception as e:
                    f.write(f"Error reading file: {e}")
                f.write("\n```\n\n")

if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(repo_root, "repo_context.md")
    generate_repo_md(repo_root, output_path)
    print(f"Repository context generated at: {output_path}")

from pathlib import Path

def write_agent_md(
        path: Path
):
    """Writes a Tutorial_Agent.md File for LLMs to understand the library."""
    this_file_dir = Path(__file__).parent
    agent_md_path = this_file_dir / "Tutorial_Agents.md"
    path.write_text(
        data=agent_md_path.read_text(),
        encoding="utf-8",
        newline="\n"
    )


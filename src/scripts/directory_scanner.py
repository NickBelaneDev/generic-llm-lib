from pathlib import Path
from typing import Dict, Union, Any

LIBRARY_ROOT = Path(__file__).parent


class DirectoryScanner:
    def __init__(self) -> None:
        self.max_reads = 100
        self.current_read = 0

    def read_directory_tree(self, path: Path = LIBRARY_ROOT) -> Dict[str, Union[str, Dict[str, Any]]]:
        """Walks from a certain directory and creates a dict from its structure."""
        data: Dict[str, Union[str, Dict[str, Any]]] = {}

        for item in path.iterdir():

            if self.current_read >= self.max_reads:
                break

            if item.is_file():
                size_kb = item.stat().st_size / 1024
                data[item.name] = f"{size_kb:.2f}kb"
                continue

            data[item.name] = self.read_directory_tree(item)
            self.current_read += 1
        return data

    def json_tree_to_string(self, data: Dict[str, Any], indent: int = 0) -> str:
        content_string = ""
        for k, v in data.items():
            if isinstance(v, dict):
                content_string += ("   " * indent) + f"{k}/" + "\n"
                content_string += self.json_tree_to_string(v, indent + 1)
            else:
                content_string += "   " * indent + f"{k}: {v}" + "\n"
        return content_string

    def build_directory_tree(self, data: Dict[str, Any], path: Path = LIBRARY_ROOT) -> None:
        """Creates a Directory from a Dictionary."""
        for key in data.keys():
            _next_path = path / key

            try:
                value = data.get(key)
                if isinstance(value, dict):
                    if not Path(_next_path).exists():
                        Path.mkdir(_next_path, parents=True, exist_ok=True)
                        print(f"Directory created at {_next_path}")

                    self.build_directory_tree(value, _next_path)

                else:
                    if not Path(_next_path).exists():
                        Path(_next_path).touch()
                        print(f"File created {_next_path}")

            except FileExistsError:
                print(f"File already exists at {_next_path}")


if __name__ == "__main__":
    ds = DirectoryScanner()
    dict_tree = ds.read_directory_tree()

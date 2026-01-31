import json
from pathlib import Path
import pprint
CS_LAB_ROOT = Path("C:/Users/Anwender/PycharmProjects/cs-lab")
COMPUTER_SCIENCE_FOLDER = Path("/")

with open('C:/Users/Anwender/PycharmProjects/cs-lab/tree.json', 'r') as f:
    tree_tmplt: dict = json.load(f)



class DirectoryScanner:
    def __init__(self):
        self.max_reads = 100
        self.current_read = 0

    def read_directory_tree(
            self,
            path: Path = COMPUTER_SCIENCE_FOLDER
    ) \
            -> dict:
        """Walks from a certain directory and creates a dict from its structure."""
        data = {}

        for item in path.iterdir():

            if self.current_read >= self.max_reads: break

            if item.is_file():
                size_kb = item.stat().st_size / 1024
                data[item.name] = f"{size_kb:.2f}kb"
                continue

            data[item.name] = self.read_directory_tree(item)
            self.current_read += 1
        return data

    def json_tree_to_string(
            self,
            data: dict,
            indent: int = 0
    ) -> str:
        content_string = ""
        for k, v in data.items():
            if isinstance(v, dict):
                content_string += (f"   " * indent) + f"{k}/" + f"\n"
                content_string += self.json_tree_to_string(v, indent + 1)
            else:
                content_string += (f"   " * indent + f"{k}: {v}" + f"\n")
        return content_string


    def build_directory_tree(
            self,
            data: dict,
            path: Path = COMPUTER_SCIENCE_FOLDER
    )\
            -> None:
        """Creates a Directory from a Dictionary."""
        for key in data.keys():
            _next_path = path / key

            try:

                if data.get(key):
                    if not Path(_next_path).exists():
                        Path.mkdir(_next_path, parents=True, exist_ok=True)
                        print(f"Directory created at {_next_path}")

                    self.build_directory_tree(data.get(key), _next_path)

                else:
                    if not Path(_next_path).exists():
                        Path(_next_path).touch()
                        print(f"File created {_next_path}")

            except FileExistsError:
                print(f"File already exists at {_next_path}")

    #build_directory_tree(tree_tmplt)
ds = DirectoryScanner()
dict_tree = ds.read_directory_tree()

str_tree = ds.json_tree_to_string(dict_tree)
pprint.pprint(dict_tree)
print(str_tree)
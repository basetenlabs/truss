from pathlib import Path


def write_str_to_file(filepath: Path, content: str):
    with filepath.open("w") as file:
        file.write(content)

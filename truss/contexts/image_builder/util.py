from pathlib import Path


def file_is_empty(path: Path, ignore_hash_style_comments: bool = True) -> bool:
    if not path.exists():
        return True

    with path.open() as file:
        for line in file.readlines():
            if ignore_hash_style_comments and _is_hash_style_comment(line):
                continue
            if line.strip() != "":
                return False

    return True


def file_is_not_empty(path: Path, ignore_hash_style_comments: bool = True) -> bool:
    return not file_is_empty(path, ignore_hash_style_comments)


def _is_hash_style_comment(line: str) -> bool:
    return line.lstrip().startswith("#")

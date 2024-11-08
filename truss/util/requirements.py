from typing import Optional


def parse_requirement_string(req_str: str) -> Optional[str]:
    """
    Collects requirements from a list of requirement lines.
    """
    stripped_line = req_str.strip()
    if stripped_line and not stripped_line.startswith("#"):
        return stripped_line
    return None

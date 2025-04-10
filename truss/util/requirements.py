from typing import Optional


def parse_requirement_string(req_str: str) -> Optional[str]:
    """
    Collects requirements from a list of requirement lines.
    """
    stripped_line = req_str.strip()
    if stripped_line and not stripped_line.startswith("#"):
        return stripped_line
    return None


def raise_insufficent_revision(repo_id_huggingface: str, revision: str):
    """
    Raises an exception if the revision is insufficient.
    """
    raise ValueError(
        f"Revision '{revision}' is insufficient for repo '{repo_id_huggingface}'. "
        "Please a suitable commit sha under this "
        f"`[link](https://huggingface.co/{repo_id_huggingface}/commits/main)`"
    )

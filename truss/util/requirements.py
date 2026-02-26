import pathlib
from typing import Optional

import tomlkit
from packaging.requirements import InvalidRequirement, Requirement


def parse_requirement_string(req_str: str) -> Optional[str]:
    """
    Collects requirements from a list of requirement lines.
    """
    stripped_line = req_str.strip()
    if not stripped_line or stripped_line.startswith("#"):
        return None

    # NB(nikhil): We intentionally don't delegate to `_is_valid_requirement` here, since `pip` technically supports
    # non PEP 508 compliant requirement strings (e.g. `git+` URLs). We want to be as permissive as possible here, and let `pip`
    # handle the validation of the requirement string.
    return stripped_line


def parse_requirements_from_pyproject(pyproject_path: pathlib.Path) -> list[str]:
    with open(pyproject_path) as f:
        data = tomlkit.load(f)

    raw_deps = data.get("project", {}).get("dependencies", [])
    return [dep for dep in raw_deps if _is_valid_requirement(dep)]


def _is_valid_requirement(req: str) -> bool:
    try:
        Requirement(req)
        return True
    except InvalidRequirement:
        return False


def raise_insufficent_revision(repo_id_huggingface: str, revision: str):
    """
    Raises an exception if the revision is insufficient.
    """
    raise ValueError(
        f"Revision '{revision}' is insufficient for repo '{repo_id_huggingface}'. "
        "Please a suitable commit sha under this "
        f"`[link](https://huggingface.co/{repo_id_huggingface}/commits/main)`"
    )

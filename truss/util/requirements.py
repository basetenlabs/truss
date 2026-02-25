import pathlib
from typing import Optional

import tomlkit


def parse_requirement_string(req_str: str) -> Optional[str]:
    """
    Collects requirements from a list of requirement lines.
    """
    stripped_line = req_str.strip()
    if stripped_line and not stripped_line.startswith("#"):
        return stripped_line
    return None


def parse_requirements_from_pyproject(pyproject_path: pathlib.Path) -> list[str]:
    """Parse [project.dependencies] from a pyproject.toml file.

    Returns a list of PEP 508 requirement strings, filtering out any
    that are not simple PyPI requirements (e.g. local paths, direct URLs).
    """
    with open(pyproject_path) as f:
        data = tomlkit.load(f)

    raw_deps = data.get("project", {}).get("dependencies", [])
    return [req for req in raw_deps if _is_pypi_requirement(req)]


def _is_pypi_requirement(req: str) -> bool:
    """Check if a requirement string looks like a standard PyPI requirement.

    Filters out direct URL references (PEP 440 @ syntax) and local path
    dependencies. Standard version specifiers (>=, ==, ~=, etc.) are allowed.
    """
    stripped = req.strip()
    if not stripped or stripped.startswith("#"):
        return False
    # Direct URL reference: "package @ https://..." or "package @ file://..."
    if " @ " in stripped:
        return False
    # Local path (starts with . or / or contains path separators without version spec)
    if stripped.startswith((".")) or stripped.startswith("/"):
        return False
    return True


def raise_insufficent_revision(repo_id_huggingface: str, revision: str):
    """
    Raises an exception if the revision is insufficient.
    """
    raise ValueError(
        f"Revision '{revision}' is insufficient for repo '{repo_id_huggingface}'. "
        "Please a suitable commit sha under this "
        f"`[link](https://huggingface.co/{repo_id_huggingface}/commits/main)`"
    )

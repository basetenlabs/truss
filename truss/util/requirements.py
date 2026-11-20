import logging
import pathlib
from typing import Optional

import tomlkit
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

logger = logging.getLogger(__name__)

_invalid_dependency_warned = False


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


def add_or_update_requirement(requirements: list[str], requirement: str) -> list[str]:
    """Replace the same named requirement, keeping other markers and extras independent.

    pip also accepts opaque inputs such as bare Git URLs and local paths. Preserve
    those verbatim instead of guessing a package name or rejecting the whole list.
    """
    key = _requirement_key(requirement)
    return [
        existing
        for existing in requirements
        if (
            existing != requirement
            if key is None
            else _requirement_key(existing) != key
        )
    ] + [requirement]


def _requirement_key(
    requirement: str,
) -> Optional[tuple[str, frozenset[str], Optional[str]]]:
    try:
        parsed = Requirement(requirement)
    except InvalidRequirement:
        return None
    return (
        canonicalize_name(parsed.name),
        frozenset(canonicalize_name(extra) for extra in parsed.extras),
        str(parsed.marker) if parsed.marker is not None else None,
    )


def parse_requirements_from_pyproject(
    pyproject_path: pathlib.Path, warn_on_invalid: bool = False
) -> list[str]:
    global _invalid_dependency_warned
    with open(pyproject_path) as f:
        data = tomlkit.load(f)

    raw_deps = data.get("project", {}).get("dependencies", [])
    valid_deps = []
    for dep in raw_deps:
        if _is_valid_requirement(dep):
            valid_deps.append(dep)
        elif (
            warn_on_invalid
            and not _is_local_path(dep)
            and not _invalid_dependency_warned
        ):
            logger.warning(
                f"Ignoring invalid dependency `{dep}` in pyproject.toml: could not be "
                "parsed as a requirement."
            )
            _invalid_dependency_warned = True
    return valid_deps


def _is_valid_requirement(req: str) -> bool:
    try:
        Requirement(req)
        return True
    except InvalidRequirement:
        return False


def _is_local_path(req: str) -> bool:
    return req.strip().startswith((".", "/"))


def raise_insufficent_revision(repo_id_huggingface: str, revision: str):
    """
    Raises an exception if the revision is insufficient.
    """
    raise ValueError(
        f"Revision '{revision}' is insufficient for repo '{repo_id_huggingface}'. "
        "Please a suitable commit sha under this "
        f"`[link](https://huggingface.co/{repo_id_huggingface}/commits/main)`"
    )

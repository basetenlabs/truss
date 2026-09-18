"""User-Agent header for outbound calls to Baseten services."""

import platform
from collections.abc import Mapping

import truss

_client_name = "truss-sdk"


def set_client_name(name: str) -> None:
    """Set the client name used in the User-Agent header (e.g. ``truss-cli``)."""
    global _client_name
    _client_name = name


def user_agent_header() -> str:
    """Build a User-Agent value like ``truss-cli/0.17.2 (Python/3.13.2; Linux)``."""
    return (
        f"{_client_name}/{truss.__version__} "
        f"(Python/{platform.python_version()}; {platform.system()})"
    )


def with_user_agent(headers: Mapping[str, str]) -> dict[str, str]:
    """Return a copy of ``headers`` with our User-Agent set, unless one is already present."""
    if any(key.lower() == "user-agent" for key in headers):
        return dict(headers)
    return {**headers, "User-Agent": user_agent_header()}

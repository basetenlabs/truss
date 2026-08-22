from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import rich_click as click

from truss.cli.cannery.config import CanneryConfig


def resolve_token_file(config: CanneryConfig) -> Optional[Path]:
    configured = os.environ.get("TRUSS_CANNERY_AUTH_TOKEN_FILE")
    if configured:
        token_file = Path(configured).expanduser()
        if not token_file.is_file():
            raise click.UsageError(
                "TRUSS_CANNERY_AUTH_TOKEN_FILE must point to an existing token "
                f"file; got {configured!r}."
            )
        return token_file.resolve()

    if config.is_loopback:
        return None

    raise click.UsageError(
        "A Cannery token is required for non-loopback endpoints. "
        "Set TRUSS_CANNERY_AUTH_TOKEN_FILE to an existing token file. "
        "Automatic Baseten-to-Cannery token exchange is pending RUN-869."
    )


def child_environment(token_file: Optional[Path], org: str) -> Dict[str, str]:
    env = os.environ.copy()
    env.pop("CANNERY_AUTH_TOKEN_FILE", None)
    env["CANNERY_ORG"] = org
    if token_file is not None:
        env["CANNERY_AUTH_TOKEN_FILE"] = str(token_file)
    return env

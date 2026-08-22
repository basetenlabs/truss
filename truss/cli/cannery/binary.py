from __future__ import annotations

import os
import shutil

import rich_click as click


def resolve_cannery_binary() -> str:
    configured = os.environ.get("TRUSS_CANNERY_BIN")
    candidate = configured or "cannery"
    resolved = shutil.which(candidate)
    if resolved is not None:
        return resolved

    if configured:
        raise click.ClickException(
            f"Cannot execute Cannery binary configured by TRUSS_CANNERY_BIN: "
            f"{configured!r}. Set it to an executable Cannery client."
        )
    raise click.ClickException(
        "Cannery client binary was not found on PATH. Install or build `cannery`, "
        "or set TRUSS_CANNERY_BIN to its executable path."
    )

from __future__ import annotations

import rich_click as click


class CanneryProtocolError(click.ClickException):
    """The Cannery subprocess violated its selected machine protocol."""

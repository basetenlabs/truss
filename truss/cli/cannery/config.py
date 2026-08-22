from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass
from urllib.parse import urlparse

import rich_click as click

DEFAULT_LOCAL_API = "http://127.0.0.1:8787"
DEFAULT_LOCAL_ORG = "dev"


@dataclass(frozen=True)
class CanneryConfig:
    api: str
    org: str

    @property
    def is_loopback(self) -> bool:
        return is_loopback_endpoint(self.api)


def is_loopback_endpoint(api: str) -> bool:
    parsed = urlparse(api)
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        raise click.UsageError(
            "TRUSS_CANNERY_API must be an http(s) URL with a hostname."
        )

    hostname = parsed.hostname.rstrip(".").lower()
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def resolve_cannery_config() -> CanneryConfig:
    return CanneryConfig(
        api=os.environ.get("TRUSS_CANNERY_API", DEFAULT_LOCAL_API),
        org=os.environ.get("TRUSS_CANNERY_ORG", DEFAULT_LOCAL_ORG),
    )

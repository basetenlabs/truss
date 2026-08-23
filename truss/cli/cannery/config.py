from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass, field
from typing import Dict, Optional
from urllib.parse import urlparse

from truss.base.constants import DEFAULT_REMOTE_URL
from truss.cli import remote_cli
from truss.cli.cannery.errors import CanneryUsageError
from truss.remote.remote_factory import RemoteFactory
from truss.remote.truss_remote import RemoteConfig

DEFAULT_LOCAL_API = "http://127.0.0.1:8787"
DEFAULT_LOCAL_ORG = "dev"
_ALLOW_PATH_ENV = "TRUSS_CANNERY_ALLOW_PATH"

# Keep the public volume API mapping explicit so an unknown control-plane host
# fails closed instead of guessing its data endpoint.
CANNERY_API_BY_REMOTE_URL: Dict[str, str] = {
    DEFAULT_REMOTE_URL: "https://bdn.baseten.co"
}


@dataclass(frozen=True)
class ActiveRemote:
    name: str
    remote_url: str
    config: RemoteConfig = field(repr=False)


@dataclass(frozen=True)
class CanneryConfig:
    api: str
    org: str
    active_remote: Optional[ActiveRemote] = None
    allow_path_fallback: bool = False

    @property
    def is_loopback(self) -> bool:
        return is_loopback_endpoint(self.api)


def is_loopback_endpoint(api: str) -> bool:
    parsed = urlparse(api)
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        raise CanneryUsageError(
            "TRUSS_CANNERY_API must be an http(s) URL with a hostname."
        )

    hostname = parsed.hostname.rstrip(".").lower()
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _normalize_remote_url(remote_url: str) -> str:
    return remote_url.rstrip("/")


def _resolve_active_remote(remote: Optional[str]) -> Optional[ActiveRemote]:
    available_remotes = RemoteFactory.get_available_config_names()
    if not available_remotes and remote is None:
        return None

    remote_name = remote or remote_cli.inquire_remote_name(allow_create=False)
    try:
        remote_config = RemoteFactory.load_remote_config(remote_name)
    except (FileNotFoundError, ValueError):
        raise CanneryUsageError(
            f"Could not load Truss remote {remote_name!r}. Run `truss auth status "
            f"--remote {remote_name}` to verify its configuration."
        ) from None
    if remote_config.configs.get("remote_provider") != "baseten":
        raise CanneryUsageError(
            "The selected Truss remote is not a Baseten remote. Set "
            "TRUSS_CANNERY_API for local Cannery development."
        )
    remote_url = remote_config.configs.get("remote_url")
    if not isinstance(remote_url, str) or not remote_url:
        raise CanneryUsageError("The selected Truss remote has no valid remote_url.")
    return ActiveRemote(
        name=remote_name,
        remote_url=_normalize_remote_url(remote_url),
        config=remote_config,
    )


def resolve_cannery_config(remote: Optional[str] = None) -> CanneryConfig:
    explicit_api = os.environ.get("TRUSS_CANNERY_API")
    org = os.environ.get("TRUSS_CANNERY_ORG", DEFAULT_LOCAL_ORG)
    if explicit_api:
        if remote is not None:
            raise CanneryUsageError(
                "--remote cannot be combined with TRUSS_CANNERY_API. Unset the "
                "local endpoint override to use a configured Truss remote."
            )
        if not is_loopback_endpoint(explicit_api):
            raise CanneryUsageError(
                "TRUSS_CANNERY_API is restricted to an explicit loopback URL. "
                "Use --remote for authenticated Cannery access."
            )
        allow_path = os.environ.get(_ALLOW_PATH_ENV)
        if allow_path not in {None, "1"}:
            raise CanneryUsageError(f"{_ALLOW_PATH_ENV} must be 1 when enabled.")
        if not os.environ.get("TRUSS_CANNERY_BIN") and allow_path != "1":
            raise CanneryUsageError(
                "Local Cannery development requires TRUSS_CANNERY_BIN. To "
                "explicitly opt in to `cannery` on PATH, set "
                f"{_ALLOW_PATH_ENV}=1."
            )
        return CanneryConfig(
            api=explicit_api, org=org, allow_path_fallback=allow_path == "1"
        )

    active_remote = _resolve_active_remote(remote)
    if active_remote is None:
        raise CanneryUsageError(
            "No Cannery endpoint is configured. Run `truss auth login` and use "
            "--remote, or explicitly configure local development with "
            "TRUSS_CANNERY_API and TRUSS_CANNERY_BIN."
        )

    api = CANNERY_API_BY_REMOTE_URL.get(active_remote.remote_url)
    if api is None:
        raise CanneryUsageError(
            "No public volume API endpoint is configured for the selected Truss "
            "remote. Set TRUSS_CANNERY_API explicitly for local development."
        )
    return CanneryConfig(api=api, org=org, active_remote=active_remote)

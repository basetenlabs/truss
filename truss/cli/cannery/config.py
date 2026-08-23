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

# RUN-867 owns the final discovery contract. Keeping this mapping explicit makes
# an unknown control-plane host fail closed instead of guessing its data endpoint.
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


def _resolve_active_remote() -> Optional[ActiveRemote]:
    if not RemoteFactory.get_available_config_names():
        return None

    remote_name = remote_cli.inquire_remote_name(allow_create=False)
    remote_config = RemoteFactory.load_remote_config(remote_name)
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


def resolve_cannery_config() -> CanneryConfig:
    explicit_api = os.environ.get("TRUSS_CANNERY_API")
    org = os.environ.get("TRUSS_CANNERY_ORG", DEFAULT_LOCAL_ORG)
    if explicit_api:
        config = CanneryConfig(api=explicit_api, org=org)
        is_loopback_endpoint(config.api)
        return config

    active_remote = _resolve_active_remote()
    if active_remote is None:
        return CanneryConfig(api=DEFAULT_LOCAL_API, org=org)

    api = CANNERY_API_BY_REMOTE_URL.get(active_remote.remote_url)
    if api is None:
        raise CanneryUsageError(
            "No Cannery endpoint is configured for the selected Truss remote. "
            "Set TRUSS_CANNERY_API explicitly "
            "for local development."
        )
    return CanneryConfig(api=api, org=org, active_remote=active_remote)

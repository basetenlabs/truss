"""BYOC entry points: ``TrussHandle`` + sibling URL discovery from mounted ``dynamic_chainlet_config``.

:class:`TrussHandle` is the canonical BYOC handle. Its
:meth:`~TrussHandle.http_call_args` / :meth:`~TrussHandle.ws_call_args`
methods return a :class:`CallArgs` ``(url, headers)`` tuple ready to hand
to ``httpx`` or ``websockets``.

:func:`~truss_chains.remote_chainlet.utils.load_dynamic_chainlet_config` is
the shared primitive used by both typed-chainlet wiring and the BYOC entry
points below. :func:`get_service_urls` and :func:`get_baseten_chain_api_key`
are additional BYOC entry points for code without an injected
``DeploymentContext``. Keys are chainlet display names; values parse as
``ServiceDescriptorUrls``. Catch ``MissingDependencyError`` when not deployed
in a chain context.
"""

import functools
from typing import NamedTuple, Optional, Type, Union, overload

from truss.templates.shared import secrets_resolver
from truss_chains import private_types, public_types
from truss_chains.remote_chainlet import utils


class CallArgs(NamedTuple):
    """``(url, headers)`` for a sibling-chainlet call."""

    url: str
    headers: dict[str, str]


@functools.lru_cache(maxsize=1)
def get_baseten_chain_api_key() -> str:
    """Baseten chain API key from the Truss secrets resolver.

    LRU-cached per process. The key is injected by the platform at deploy
    time, so the only failure mode is "secret mount missing entirely" —
    that surfaces as ``MissingDependencyError``.
    """
    secrets = secrets_resolver.Secrets({public_types.CHAIN_API_KEY_SECRET_NAME: ""})
    try:
        return secrets[public_types.CHAIN_API_KEY_SECRET_NAME]
    except secrets_resolver.SecretNotFound:
        raise public_types.MissingDependencyError(
            f"No '{public_types.CHAIN_API_KEY_SECRET_NAME}' secret found at "
            f"`{secrets_resolver.SecretsResolver.SECRETS_MOUNT_DIR}` or via "
            f"env var. Required for sibling-chainlet authorization in BYOC "
            f"code."
        )


@overload
def get_service_urls(target: str) -> public_types.ServiceDescriptorUrls: ...


@overload
def get_service_urls(
    target: Type[private_types.ABCChainlet],
) -> public_types.ServiceDescriptorUrls: ...


def get_service_urls(
    target: Union[str, Type[private_types.ABCChainlet]],
) -> public_types.ServiceDescriptorUrls:
    """Resolve sibling URLs by ``ABCChainlet`` subclass (via ``display_name``) or name string.

    Raises ``TypeError`` or ``MissingDependencyError``.
    """
    if isinstance(target, str):
        name = target
    elif isinstance(target, type):
        name = target.display_name
    else:
        raise TypeError(
            "`get_service_urls` accepts a chainlet class or its display-name "
            f"string, got {type(target).__name__!r}."
        )
    config = utils.load_dynamic_chainlet_config()
    if name not in config:
        raise public_types.MissingDependencyError(
            f"No sibling chainlet named '{name}'. Available: {list(config)}."
        )
    return config[name]


class TrussHandle:
    """Sibling chainlet handle; build once (e.g. in ``__init__``), then call args."""

    urls: public_types.ServiceDescriptorUrls

    def __init__(self, target: Union[str, Type[private_types.ABCChainlet]]) -> None:
        self.urls = get_service_urls(target)

    def http_call_args(
        self,
        *,
        prefer_internal: bool = False,
        sync_path: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> CallArgs:
        """Default ``predict_url`` + ``Authorization``; ``prefer_internal`` uses workload-plane URL + ``Host``.

        ``sync_path`` rewrites the URL to ``/sync/<sync_path>``.
        ``api_key`` overrides :func:`get_baseten_chain_api_key`.
        ``prefer_internal`` uses the internal url if it exists.
        """
        key = api_key if api_key is not None else get_baseten_chain_api_key()
        if (
            prefer_internal or self.urls.predict_url is None
        ) and self.urls.internal_url is not None:
            url = self.urls.internal_url.gateway_run_remote_url
            headers = {
                "Authorization": f"Api-Key {key}",
                "Host": self.urls.internal_url.hostname,
            }
        else:
            assert self.urls.predict_url is not None, (
                "ServiceDescriptorUrls has neither predict_url nor internal_url — "
                "platform invariant violated."
            )
            url = self.urls.predict_url
            headers = {"Authorization": f"Api-Key {key}"}
        if sync_path is not None:
            url = f"{url.removesuffix('/run_remote')}/sync/{sync_path.lstrip('/')}"
        return CallArgs(url=url, headers=headers)

    def ws_call_args(
        self, *, sync_path: Optional[str] = None, api_key: Optional[str] = None
    ) -> CallArgs:
        """Returns a ``wss://`` URL + auth-only headers for a WebSocket sibling call.

        ``websockets.connect`` rejects Host-header overrides (api-gateway
        returns 400), so this has no ``prefer_internal`` kwarg
        """
        key = api_key if api_key is not None else get_baseten_chain_api_key()
        assert self.urls.predict_url is not None, (
            "ServiceDescriptorUrls has no predict_url — WS BYOC requires the chain-host URL."
        )
        base = self.urls.predict_url.removesuffix("/run_remote")
        suffix = (
            f"sync/{sync_path.lstrip('/')}" if sync_path is not None else "websocket"
        )
        url = f"{base}/{suffix}"
        if url.startswith("https://"):
            url = "wss://" + url[len("https://") :]
        elif url.startswith("http://"):
            url = "ws://" + url[len("http://") :]
        return CallArgs(url=url, headers={"Authorization": f"Api-Key {key}"})

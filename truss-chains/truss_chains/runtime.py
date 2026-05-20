"""Discover sibling chainlet URLs from mounted ``dynamic_chainlet_config``.

:class:`ServiceHandle` is the canonical BYOC entry point — also re-exported
at the top level as :class:`truss_chains.ServiceHandle`. Its
:meth:`~ServiceHandle.http_call_args` / :meth:`~ServiceHandle.ws_call_args`
methods return a :class:`CallArgs` ``(url, headers)`` tuple ready to hand
to ``httpx`` or ``websockets``.
:func:`load_dynamic_chainlet_config` is the primitive used by
typed-chainlet wiring; :func:`get_service_urls` and
:func:`get_baseten_chain_api_key` are additional BYOC entry points for code
without an injected ``DeploymentContext``.
Keys are chainlet display names; values parse as ``ServiceDescriptorUrls``.
Catch ``MissingDependencyError`` when not deployed in a chain context.
"""

import functools
import json
from typing import NamedTuple, Type, Union, overload

from truss.templates.shared import dynamic_config_resolver, secrets_resolver
from truss_chains import private_types, public_types


class CallArgs(NamedTuple):
    """``(url, headers)`` for a sibling-chainlet call."""

    url: str
    headers: dict[str, str]


@functools.lru_cache(maxsize=1)
def load_dynamic_chainlet_config() -> dict[str, public_types.ServiceDescriptorUrls]:
    """Load and validate ``dynamic_chainlet_config`` (display name → URLs).

    Result is LRU-cached per process; tests that patch mount paths must call
    ``cache_clear()`` on this function first.

    Raises ``MissingDependencyError`` if unset or empty.
    """
    dynamic_chainlet_config_str = dynamic_config_resolver.get_dynamic_config_value_sync(
        private_types.DYNAMIC_CHAINLET_CONFIG_KEY
    )
    if not dynamic_chainlet_config_str:
        raise public_types.MissingDependencyError(
            f"No '{private_types.DYNAMIC_CHAINLET_CONFIG_KEY}' "
            "found. Cannot override Chainlet configs."
        )
    data = json.loads(dynamic_chainlet_config_str)
    if not isinstance(data, dict):
        # Ignore unexpected root types (e.g. `json.dumps("")` decodes to a string).
        # Historically `in`/`not in` on those values behaved like an empty mapping.
        data = {}
    return {
        name: public_types.ServiceDescriptorUrls.model_validate(entry)
        for name, entry in data.items()
    }


@functools.lru_cache(maxsize=1)
def get_baseten_chain_api_key() -> str:
    """Baseten chain API key from the Truss secrets resolver.

    LRU-cached per process. Uses the same secret name and ``SECRET_DUMMY`` check as
    ``DeploymentContext.get_baseten_api_key``, but reads from env / mount instead of
    an injected ``secrets`` map.

    Raises ``MissingDependencyError`` if missing or placeholder ``SECRET_DUMMY``.
    """
    secrets = secrets_resolver.Secrets({public_types.CHAIN_API_KEY_SECRET_NAME: ""})
    try:
        api_key = secrets[public_types.CHAIN_API_KEY_SECRET_NAME]
    except secrets_resolver.SecretNotFound:
        raise public_types.MissingDependencyError(
            f"No '{public_types.CHAIN_API_KEY_SECRET_NAME}' secret found at "
            f"`{secrets_resolver.SecretsResolver.SECRETS_MOUNT_DIR}` or via "
            f"env var. Required for sibling-chainlet authorization in BYOC "
            f"code."
        )
    if api_key == public_types.SECRET_DUMMY:
        raise public_types.MissingDependencyError(
            f"Chain API key resolved to the placeholder "
            f"`{public_types.SECRET_DUMMY}`. Set the real value on the "
            f"deployed chain."
        )
    return api_key


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
    config = load_dynamic_chainlet_config()
    if name not in config:
        raise public_types.MissingDependencyError(
            f"No sibling chainlet named '{name}'. Available: {list(config)}."
        )
    return config[name]


class ServiceHandle:
    """Sibling chainlet handle; build once (e.g. in ``__init__``), then call args."""

    urls: public_types.ServiceDescriptorUrls

    def __init__(self, target: Union[str, Type[private_types.ABCChainlet]]) -> None:
        self.urls = get_service_urls(target)

    def http_call_args(
        self,
        *,
        prefer_internal: bool = False,
        sync_path: Union[str, None] = None,
        api_key: Union[str, None] = None,
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
        self, *, sync_path: Union[str, None] = None, api_key: Union[str, None] = None
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

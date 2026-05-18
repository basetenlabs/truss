"""Discover sibling chainlet URLs from mounted ``dynamic_chainlet_config``.

:func:`load_dynamic_chainlet_config` is the primitive used by typed-chainlet
wiring; :class:`ServiceHandle`, :func:`get_service_urls`, and
:func:`get_baseten_chain_api_key` are the BYOC entry points for plain Truss
models or any code without an injected ``DeploymentContext``.
Keys are chainlet display names; values parse as ``ServiceDescriptorUrls``.
Catch ``MissingDependencyError`` when not deployed in a chain context.
"""

import functools
import json
from typing import NamedTuple, Type, Union, overload

from truss.templates.shared import dynamic_config_resolver, secrets_resolver
from truss_chains import private_types, public_types


class HttpCallArgs(NamedTuple):
    """``(url, headers)`` for an HTTP POST to a sibling chainlet."""

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
    """Sibling chainlet handle; build once (e.g. in ``__init__``), then ``http_call_args``.

    Same target rules as :func:`get_service_urls`. Raw fields live on ``.urls``.
    """

    urls: public_types.ServiceDescriptorUrls

    def __init__(self, target: Union[str, Type[private_types.ABCChainlet]]) -> None:
        self.urls = get_service_urls(target)

    def http_call_args(
        self, *, prefer_internal: bool = False, api_key: Union[str, None] = None
    ) -> HttpCallArgs:
        """Default ``predict_url`` + ``Authorization``; ``prefer_internal`` uses workload-plane URL + ``Host``.

        Whichever URL is present is used when only one is set, regardless of
        ``prefer_internal``. ``api_key`` overrides :func:`get_baseten_chain_api_key`.
        """
        key = api_key if api_key is not None else get_baseten_chain_api_key()
        if (
            prefer_internal or self.urls.predict_url is None
        ) and self.urls.internal_url is not None:
            return HttpCallArgs(
                url=self.urls.internal_url.gateway_run_remote_url,
                headers={
                    "Authorization": f"Api-Key {key}",
                    "Host": self.urls.internal_url.hostname,
                },
            )
        assert self.urls.predict_url is not None, (
            "ServiceDescriptorUrls has neither predict_url nor internal_url — "
            "platform invariant violated."
        )
        return HttpCallArgs(
            url=self.urls.predict_url, headers={"Authorization": f"Api-Key {key}"}
        )

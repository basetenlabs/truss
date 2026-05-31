"""Public sibling-discovery API for chainlet runtimes.

Reads ``/etc/b10_dynamic_config/dynamic_chainlet_config`` (the ConfigMap mounted
by the Baseten operator into every chainlet pod) and returns typed
``DeployedServiceDescriptor`` instances.

This is a **stable public contract**: any Truss running inside a chain — whether
authored as a typed ``ChainletBase`` or as a raw Truss directory wrapped via
Project 2's ``TrussChainlet`` — can use this module to discover sibling chainlet
URLs without depending on framework internals.

Schema of the dynamic config file (JSON object), with at least one of
``predict_url`` / ``internal_url`` always present per chainlet::

    {
      "<chainlet_display_name>": {
        "predict_url": "https://chain-<id>.api.baseten.co/.../run_remote",
        "internal_url": {
          "gateway_run_remote_url": "https://<wp>.api.baseten.co/.../run_remote",
          "hostname": "chain-<id>.api.baseten.co"
        }
      },
      ...
    }

Note on RPC options: the dynamic config does *not* carry ``RPCOptions`` (retries,
timeout, concurrency_limit). Those are typed-chain-only and live in the static
``chainlet_to_service`` map embedded in ``config.yaml`` by codegen. Descriptors
returned here use default ``RPCOptions``; callers using their own HTTP/WS client
should bring their own retry, timeout, and concurrency policy.
"""

import json
from typing import Mapping

from truss.templates.shared import dynamic_config_resolver
from truss_chains import private_types, public_types


def _load_dynamic_config() -> dict:
    """Read and parse ``/etc/b10_dynamic_config/dynamic_chainlet_config``.

    Raises:
        MissingDependencyError: if the ConfigMap file is absent or empty,
            indicating the Truss is not running inside a chain context.
    """
    raw = dynamic_config_resolver.get_dynamic_config_value_sync(
        private_types.DYNAMIC_CHAINLET_CONFIG_KEY
    )
    if not raw:
        raise public_types.MissingDependencyError(
            f"No '{private_types.DYNAMIC_CHAINLET_CONFIG_KEY}' configmap found at "
            f"`{dynamic_config_resolver.DYNAMIC_CONFIG_MOUNT_DIR}`. "
            "This Truss is not running inside a chain context."
        )
    return json.loads(raw)


def _descriptor_from_raw(
    name: str, raw: dict
) -> public_types.DeployedServiceDescriptor:
    """Build a ``DeployedServiceDescriptor`` from one entry of the dynamic config.

    ``name`` is used for both ``name`` and ``display_name`` in the resulting
    descriptor (the dynamic config keys *are* display names — see module
    docstring). ``options`` defaults to a fresh ``RPCOptions()``.
    """
    kwargs: dict = {}
    if internal_url := raw.get("internal_url"):
        kwargs["internal_url"] = public_types.DeployedServiceDescriptor.InternalURL(
            **internal_url
        )
    if predict_url := raw.get("predict_url"):
        kwargs["predict_url"] = predict_url
    return public_types.DeployedServiceDescriptor(
        name=name, display_name=name, options=public_types.RPCOptions(), **kwargs
    )


def get_service(name: str) -> public_types.DeployedServiceDescriptor:
    """Return the descriptor for a sibling chainlet by name.

    Args:
        name: The chainlet's display name (the key used in the chain
            declaration; matches the keys in the dynamic config map).

    Returns:
        ``DeployedServiceDescriptor`` with ``predict_url``, ``internal_url``
        (when available), and helper methods for building auth headers and
        WebSocket URLs.

    Raises:
        MissingDependencyError: if no chain context is detected (the dynamic
            config file is absent), or if ``name`` is not a registered
            sibling. The error message includes the available names.
    """
    config = _load_dynamic_config()
    if name not in config:
        raise public_types.MissingDependencyError(
            f"No sibling chainlet named '{name}'. Available: {list(config)}."
        )
    return _descriptor_from_raw(name, config[name])


def list_services() -> Mapping[str, public_types.DeployedServiceDescriptor]:
    """Return all sibling chainlet descriptors keyed by name.

    Returns an empty mapping if no chain context is detected (i.e., the Truss
    is running outside a chain). Use this to enumerate available siblings
    without raising on absence — useful for code that wants to detect whether
    it is running inside a chain at all.
    """
    try:
        config = _load_dynamic_config()
    except public_types.MissingDependencyError:
        return {}
    return {n: _descriptor_from_raw(n, r) for n, r in config.items()}

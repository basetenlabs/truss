"""Adversarial: pin the backward-compatible contract of the typed-chain
``populate_chainlet_service_predict_urls`` after the internal refactor."""

import json

import pytest

from truss_chains import private_types, public_types, runtime
from truss_chains.remote_chainlet.utils import populate_chainlet_service_predict_urls


@pytest.fixture
def dynamic_config_mount_dir(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "truss.templates.shared.dynamic_config_resolver.DYNAMIC_CONFIG_MOUNT_DIR",
        str(tmp_path),
    )
    yield


def _write_config(tmp_path, payload):
    with (tmp_path / private_types.DYNAMIC_CHAINLET_CONFIG_KEY).open("w") as f:
        f.write(json.dumps(payload))


def test_typed_path_clears_predict_url_when_internal_url_present(
    tmp_path, dynamic_config_mount_dir
):
    """Historical behavior: in the typed-chain path, when both URLs are in
    the dynamic config, the resulting descriptor has only ``internal_url``
    set. The new public ``runtime.get_service`` carries both — but the typed
    path stays mutually exclusive for backward compatibility."""
    _write_config(
        tmp_path,
        {
            "Hello World!": {
                "predict_url": "https://chain-x.api.baseten.co/.../run_remote",
                "internal_url": {
                    "gateway_run_remote_url": "https://wp.api.baseten.co/.../run_remote",
                    "hostname": "chain-x.api.baseten.co",
                },
            }
        },
    )

    chainlet_to_service = {
        "HelloWorld": private_types.ServiceDescriptor(
            name="HelloWorld",
            display_name="Hello World!",
            options=public_types.RPCOptions(),
        )
    }
    out = populate_chainlet_service_predict_urls(chainlet_to_service)

    # Typed path: predict_url cleared.
    assert out["HelloWorld"].predict_url is None
    assert out["HelloWorld"].internal_url is not None

    # Public runtime API: both URLs preserved (different contract, same data).
    desc = runtime.get_service("Hello World!")
    assert desc.predict_url == "https://chain-x.api.baseten.co/.../run_remote"
    assert desc.internal_url is not None


def test_missing_chainlet_error_message_preserved(tmp_path, dynamic_config_mount_dir):
    """Existing test in test_utils.py matches on the substring
    "Chainlet 'X' not found". Project 1's refactor must keep this exact
    wording."""
    _write_config(tmp_path, {"OtherName": {"predict_url": "https://x"}})
    with pytest.raises(
        public_types.MissingDependencyError, match="Chainlet 'RandInt' not found"
    ):
        populate_chainlet_service_predict_urls(
            {
                "RandInt": private_types.ServiceDescriptor(
                    name="RandInt",
                    display_name="RandInt",
                    options=public_types.RPCOptions(),
                )
            }
        )


def test_missing_dynamic_config_error_message_preserved(
    tmp_path, dynamic_config_mount_dir
):
    """No file: typed path raises with the historical "Cannot override
    Chainlet configs" wording, distinct from the new runtime API's
    "not running inside a chain context" wording."""
    with pytest.raises(
        public_types.MissingDependencyError, match="Cannot override Chainlet configs"
    ):
        populate_chainlet_service_predict_urls(
            {
                "X": private_types.ServiceDescriptor(
                    name="X", display_name="X", options=public_types.RPCOptions()
                )
            }
        )


def test_descriptor_type_identity_across_imports():
    """Same Python type whether reached via public_types or as a re-export
    candidate — there's only one canonical class."""
    from truss_chains import DeployedServiceDescriptor as A
    from truss_chains.public_types import DeployedServiceDescriptor as B

    assert A is B

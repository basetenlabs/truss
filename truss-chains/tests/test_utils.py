import json

import pytest

from truss_chains import private_types, public_types
from truss_chains.remote_chainlet.utils import populate_chainlet_service_predict_urls

DYNAMIC_CHAINLET_CONFIG_VALUE = {
    "Hello World!": {
        "predict_url": "https://chain-232p81ql.api.baseten.co/environments/production/run_remote"
    }
}

DYNAMIC_CHAINLET_CONFIG_WITH_INTERNAL_URL = {
    "Hello World!": {
        "predict_url": "https://chain-232p81ql.api.baseten.co/environments/production/run_remote",
        "internal_url": {
            "gateway_run_remote_url": "https://aws-us-west-2-ai7.api.baseten.co/environments/production/run_remote",
            "hostname": "chain-232p81ql.api.baseten.co",
        },
    }
}

DYNAMIC_CHAINLET_CONFIG_INTERNAL_ONLY = {
    "InternalOnly": {
        "internal_url": {
            "gateway_run_remote_url": "https://aws-us-west-2-ai7.api.baseten.co/environments/production/run_remote",
            "hostname": "chain-232p81ql.api.baseten.co",
        }
    }
}


@pytest.fixture
def dynamic_config_mount_dir(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "truss.templates.shared.dynamic_config_resolver.DYNAMIC_CONFIG_MOUNT_DIR",
        str(tmp_path),
    )
    yield


def test_populate_chainlet_service_predict_urls(tmp_path, dynamic_config_mount_dir):
    with (tmp_path / private_types.DYNAMIC_CHAINLET_CONFIG_KEY).open("w") as f:
        f.write(json.dumps(DYNAMIC_CHAINLET_CONFIG_VALUE))

    chainlet_to_service = {
        "HelloWorld": private_types.ServiceDescriptor(
            name="HelloWorld",
            display_name="Hello World!",
            options=public_types.RPCOptions(),
        )
    }
    new_chainlet_to_service = populate_chainlet_service_predict_urls(
        chainlet_to_service
    )

    assert (
        new_chainlet_to_service["HelloWorld"].predict_url
        == DYNAMIC_CHAINLET_CONFIG_VALUE["Hello World!"]["predict_url"]
    )
    assert new_chainlet_to_service["HelloWorld"].internal_url is None


@pytest.mark.parametrize("config", [DYNAMIC_CHAINLET_CONFIG_VALUE, {}, ""])
def test_no_populate_chainlet_service_predict_urls(
    config, tmp_path, dynamic_config_mount_dir
):
    with (tmp_path / private_types.DYNAMIC_CHAINLET_CONFIG_KEY).open("w") as f:
        f.write(json.dumps(config))

    chainlet_to_service = {
        "RandInt": private_types.ServiceDescriptor(
            name="RandInt", display_name="RandInt", options=public_types.RPCOptions()
        )
    }

    with pytest.raises(
        public_types.MissingDependencyError, match="Chainlet 'RandInt' not found"
    ):
        populate_chainlet_service_predict_urls(chainlet_to_service)


def test_populate_chainlet_service_with_internal_url(
    tmp_path, dynamic_config_mount_dir
):
    """Test that internal_url is correctly parsed when present."""
    with (tmp_path / private_types.DYNAMIC_CHAINLET_CONFIG_KEY).open("w") as f:
        f.write(json.dumps(DYNAMIC_CHAINLET_CONFIG_WITH_INTERNAL_URL))

    chainlet_to_service = {
        "HelloWorld": private_types.ServiceDescriptor(
            name="HelloWorld",
            display_name="Hello World!",
            options=public_types.RPCOptions(),
        )
    }

    new_chainlet_to_service = populate_chainlet_service_predict_urls(
        chainlet_to_service
    )

    assert new_chainlet_to_service["HelloWorld"].predict_url is None
    assert (
        new_chainlet_to_service["HelloWorld"].internal_url.gateway_run_remote_url
        == "https://aws-us-west-2-ai7.api.baseten.co/environments/production/run_remote"
    )
    assert (
        new_chainlet_to_service["HelloWorld"].internal_url.hostname
        == "chain-232p81ql.api.baseten.co"
    )


def test_populate_chainlet_service_internal_only(tmp_path, dynamic_config_mount_dir):
    """Test case where only internal_url is provided (no predict_url)."""
    with (tmp_path / private_types.DYNAMIC_CHAINLET_CONFIG_KEY).open("w") as f:
        f.write(json.dumps(DYNAMIC_CHAINLET_CONFIG_INTERNAL_ONLY))

    chainlet_to_service = {
        "InternalService": private_types.ServiceDescriptor(
            name="InternalService",
            display_name="InternalOnly",
            options=public_types.RPCOptions(),
        )
    }

    new_chainlet_to_service = populate_chainlet_service_predict_urls(
        chainlet_to_service
    )

    assert new_chainlet_to_service["InternalService"].predict_url is None
    assert (
        new_chainlet_to_service["InternalService"].internal_url.gateway_run_remote_url
        == "https://aws-us-west-2-ai7.api.baseten.co/environments/production/run_remote"
    )
    assert (
        new_chainlet_to_service["InternalService"].internal_url.hostname
        == "chain-232p81ql.api.baseten.co"
    )

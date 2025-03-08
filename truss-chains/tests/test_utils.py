import json

import pytest

from truss_chains import private_types, public_types
from truss_chains.remote_chainlet.utils import populate_chainlet_service_predict_urls

DYNAMIC_CHAINLET_CONFIG_VALUE = {
    "Hello World!": {
        "predict_url": "https://model-diff_id.api.baseten.co/deployment/diff_deployment_id/predict"
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
